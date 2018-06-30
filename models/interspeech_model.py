#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Parcollet Titouan

# Imports

import complexnn
from   complexnn                             import *
import keras
from   keras.callbacks                       import Callback, ModelCheckpoint, LearningRateScheduler
from   keras.datasets                        import cifar10, cifar100
from   keras.initializers                    import Orthogonal
from   keras.layers                          import Layer, Dropout, AveragePooling1D, \
                                                    AveragePooling2D, AveragePooling3D, add, Add, concatenate, \
                                                    Concatenate, Input, Flatten, Dense, Convolution2D, BatchNormalization, \
                                                    Activation, Reshape, ConvLSTM2D, Conv2D, Lambda, Permute, TimeDistributed, \
                                                    SpatialDropout1D, PReLU
from   keras.models                          import Model, load_model, save_model
from   keras.optimizers                      import SGD, Adam, RMSprop
from   keras.preprocessing.image             import ImageDataGenerator
from   keras.regularizers                    import l2
from   keras.utils.np_utils                  import to_categorical
import keras.backend                         as     K
import keras.models                          as     KM
from keras.utils.training_utils              import multi_gpu_model
import logging                               as     L
import numpy                                 as     np
import os, pdb, socket, sys, time            as     T
from keras.backend.tensorflow_backend        import set_session
import tensorflow                            as tf


#
# CTC Loss
#

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

#
# Get Model
#

def getTimitModel2D(d):
    n             = d.num_layers
    sf            = d.start_filter
    activation    = d.act
    advanced_act  = d.aact
    drop_prob     = d.dropout
    inputShape    = (3,41,None)  
    filsize       = (3, 5)
    channelAxis   = 1
    
    if d.aact != "none":
        d.act = 'linear'

    convArgs      = {
            "activation":               d.act,
            "data_format":              "channels_first",
            "padding":                  "same",
            "bias_initializer":         "zeros",
            "kernel_regularizer":       l2(d.l2),
            "kernel_initializer":       "random_uniform",
            }
    denseArgs     = {
            "activation":               d.act,        
            "kernel_regularizer":       l2(d.l2),
            "kernel_initializer":       "random_uniform",
            "bias_initializer":         "zeros",
            "use_bias":                 True
            }
     
    if d.model == "quaternion":
        convArgs.update({"kernel_initializer": d.quat_init})

    #
    # Input Layer & CTC Parameters for TIMIT
    #
    if d.model == "quaternion":
        I    = Input(shape=(4,41,None))
    else:
        I = Input(shape=inputShape)

    labels = Input(name='the_labels', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    #
    # Input stage:
    #
    if d.model == "real":
        O = Conv2D(sf, filsize, name='conv', use_bias=True, **convArgs)(I)
        if d.aact == "prelu":
            O = PReLU(shared_axes=[1,0])(O)
    else:
        O = QuaternionConv2D(sf, filsize, name='conv', use_bias=True, **convArgs)(I)
        if d.aact == "prelu":
            O = PReLU(shared_axes=[1,0])(O)
    #
    # Pooling
    #
    O = keras.layers.MaxPooling2D(pool_size=(1, 3), padding='same')(O)


    #
    # Stage 1
    #
    for i in xrange(0,n/2):
        if d.model=="real":
            O = Conv2D(sf, filsize, name='conv'+str(i), use_bias=True,**convArgs)(O)
            if d.aact == "prelu":
                O = PReLU(shared_axes=[1,0])(O)
            O = Dropout(d.dropout)(O)
        else:
            O = QuaternionConv2D(sf, filsize, name='conv'+str(i), use_bias=True, **convArgs)(O)
            if d.aact == "prelu":
                O = PReLU(shared_axes=[1,0])(O)
            O = Dropout(d.dropout)(O)
    
    #
    # Stage 2
    #
    for i in xrange(0,n/2):
        if d.model=="real":
            O = Conv2D(sf*2, filsize, name='conv'+str(i+n/2), use_bias=True, **convArgs)(O)
            if d.aact == "prelu":
                O = PReLU(shared_axes=[1,0])(O)
            O = Dropout(d.dropout)(O)
        else:
            O = QuaternionConv2D(sf*2, filsize, name='conv'+str(i+n/2), use_bias=True, **convArgs)(O)
            if d.aact == "prelu":
                O = PReLU(shared_axes=[1,0])(O)
            O = Dropout(d.dropout)(O)
    
    #
    # Permutation for CTC 
    #
    
    O = Permute((3,1,2))(O)
    O = Lambda(lambda x: K.reshape(x, (K.shape(x)[0], K.shape(x)[1],
                                       K.shape(x)[2] * K.shape(x)[3])),
               output_shape=lambda x: (None, None, x[2] * x[3]))(O)
    
    #
    # Dense
    #
    if d.model== "quaternion":
        O = TimeDistributed( QuaternionDense(256,  **denseArgs))(O)
        if d.aact == "prelu":
            O = PReLU(shared_axes=[1,0])(O)
        O = Dropout(d.dropout)(O)
        O = TimeDistributed( QuaternionDense(256,  **denseArgs))(O)
        if d.aact == "prelu":
            O = PReLU(shared_axes=[1,0])(O)
        O = Dropout(d.dropout)(O)
        O = TimeDistributed( QuaternionDense(256,  **denseArgs))(O)
        if d.aact == "prelu":
            O = PReLU(shared_axes=[1,0])(O)
    else:
        O = TimeDistributed( Dense(1024,  **denseArgs))(O)
        if d.aact == "prelu":
            O = PReLU(shared_axes=[1,0])(O)
        O = Dropout(d.dropout)(O)
        O = TimeDistributed( Dense(1024, **denseArgs))(O)
        if d.aact == "prelu":
            O = PReLU(shared_axes=[1,0])(O)
        O = Dropout(d.dropout)(O)
        O = TimeDistributed( Dense(1024, **denseArgs))(O)
        if d.aact == "prelu":
            O = PReLU(shared_axes=[1,0])(O)

    pred = TimeDistributed( Dense(62,  activation='softmax', kernel_regularizer=l2(d.l2), use_bias=True, bias_initializer="zeros", kernel_initializer='random_uniform' ))(O)
      
    #
    # CTC For sequence labelling
    #
    O = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([pred, labels,input_length,label_length])
    
    # Return the model
    #
    # Creating a function for testing and validation purpose
    #
    val_function = K.function([I],[pred])
    return Model(inputs=[I, input_length, labels, label_length], outputs=O), val_function
    

