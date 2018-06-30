#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Parcollet Titouan
from   complexnn                             import *
import keras
from   keras.layers                          import *
from   keras.models                          import Model
import keras.backend                         as     K
import numpy                                 as     np

#
# ConvNet
#

def CNN(params):


    input_seq = Input((1000,1))

    if(params.model == 'QCNN'):

        # Conv
        conv    = QuaternionConv1D(32, 3, strides=1, activation='relu', padding="same")(input_seq)
        pool    = AveragePooling1D(2, padding='same')(conv)
        conv2   = QuaternionConv1D(64, 3, strides=1, activation='relu', padding="same")(pool)
        pool2   = AveragePooling1D(4, padding='same')(conv2)

        # Reducing dimension before Dense layer
        flat    = Flatten()(pool2)
        dense  = QuaternionDense(256, activation='relu')(flat)
    
    else:
        # Conv
        conv    = Conv1D(32, 3, strides=1, activation='relu', padding="same")(input_seq)
        pool    = AveragePooling1D(2, padding='same')(conv)
        conv2   = Conv1D(64, 3, strides=1, activation='relu', padding="same")(pool)
        pool2   = AveragePooling1D(4, padding='same')(conv2)

        # Reducing dimension before Dense layer
        flat    = Flatten()(pool2)
        dense  = Dense(256, activation='relu')(flat)

    output = Dense(8, activation='softmax')(dense)
    return Model(input_seq, output)

#
# DenseNet
#

def DNN(params):

    input_seq = Input((1000,))

    if(params.model == 'DNN'):
        h0 = Dense(512, activation='relu')(input_seq)
        d0 = Dropout(0.3)(h0)
        h1 = Dense(512, activation='relu')(d0)
        d1 = Dropout(0.3)(h1)
        h2 = Dense(512, activation='relu')(d1)
    elif(params.model == 'QDNN'):
        h0 = QuaternionDense(512, activation='relu')(input_seq)
        d0 = Dropout(0.3)(h0)
        h1 = QuaternionDense(512, activation='relu')(h0)
        d1 = Dropout(0.3)(h1)
        h2 = QuaternionDense(512, activation='relu')(h1)

    encoded = Dense(8, activation='softmax')(h2)
    
    return Model(input_seq, encoded);