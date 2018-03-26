#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Parcollet Titouan

# Imports
import editdistance
import h5py
import complexnn
from   complexnn                             import *
import h5py                                  as     H
import keras
from   keras.callbacks                       import Callback, ModelCheckpoint, LearningRateScheduler
from   keras.initializers                    import Orthogonal
from   keras.layers                          import (Layer, Dropout, AveragePooling1D, AveragePooling2D,  
                                                    AveragePooling3D, add, Add, concatenate, Concatenate, 
                                                    Input, Flatten, Dense, Convolution2D, BatchNormalization, 
                                                    Activation, Reshape, ConvLSTM2D, Conv2D, Lambda )
from   keras.models                          import Model, load_model, save_model 
from   keras.optimizers                      import SGD, Adam, RMSprop
from   keras.regularizers                    import l2
from   keras.utils.np_utils                  import to_categorical
import keras.backend                         as     K
import keras.models                          as     KM
from keras.utils.training_utils              import multi_gpu_model
import logging                               as     L
import numpy                                 as     np
import os, pdb, socket, sys, time
import theano                                as     T
from keras.backend.tensorflow_backend        import set_session
from models_timit                            import getTimitResnetModel2D,ctc_lambda_func
import tensorflow                            as tf
import itertools
import random


#######################
# Generator wrapper for timit
# You must write this generator in order to feat the Keras training loop
# the outputs should be:
# Input_data, Input_label, Input_length, Label_length
#
# Input_data: A batch of input sequence (Batch_size, Sequence_length, features(123))
# Input_label: Corresponding label (Batch_size, Sequence length)
# Input_length: Maximum sequence length (int32)
# Label_length: Maximum Label length (int32)

def timitGenerator(dataset):
    while True:
        Input_data   = 0
        Input_label  = 0
        Input_length = 0
        Label_length = 0
        yield Input_data, Input_label, Input_length, Label_length

#######################
# Custom metrics
# This metric has to be adapted to YOUR generator
# It's not functional since OUR generator is not provided
# It's provided as an example

#class EditDistance(Callback):
#    def __init__(self, func, dataset, quaternion, save_prefix):
	   

    # Map from labels to phonemes as text
#    def labels_to_text(self,labels):
        

    # Decode the prediction 
#    def decode_batch(self, out, mask):
        

    # Callback entering function 
    # Must contain a loop over the testing dataset
#    def on_epoch_end(self, epoch, logs={}):
#
########################     

#
# Callbacks:
#

class TrainLoss(Callback):
    def __init__(self, savedir):
        self.savedir = savedir
    def on_epoch_end(self, epoch, logs={}):
        f=open(str(self.savedir)+"_train_loss.txt",'ab')
        f2=open(str(self.savedir)+"_dev_loss.txt",'ab')
        value = float(logs['loss'])
        np.savetxt(f,np.array([value]))
        f.close()
        value = float(logs['val_loss'])
        np.savetxt(f2,np.array([value]))
        f2.close()
#
# Print a newline after each epoch, because Keras doesn't. Grumble.
#

class PrintNewlineAfterEpochCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        sys.stdout.write("\n")
#
# Save checkpoints.
#

class SaveLastModel(Callback):
    def __init__(self, workdir, save_prefix, model_mono,period=10):
        self.workdir          = workdir
        self.model_mono           = model_mono
        self.chkptsdir        = os.path.join(self.workdir, "chkpts")
        self.save_prefix = save_prefix
        if not os.path.isdir(self.chkptsdir):
            os.mkdir(self.chkptsdir)
        self.period_of_epochs = period
        self.linkFilename     = os.path.join(self.chkptsdir, str(save_prefix)+"ModelChkpt.hdf5")
        self.linkFilename_weight     = os.path.join(self.chkptsdir, str(save_prefix)+"ModelChkpt_weight.hdf5")

    def on_epoch_end(self, epoch, logs={}):
        if (epoch + 1) % self.period_of_epochs == 0:
            
            # Filenames
            baseHDF5Filename = str(self.save_prefix)+"ModelChkpt{:06d}.hdf5".format(epoch+1)
            baseHDF5Filename_weight = str(self.save_prefix)+"ModelChkpt{:06d}_weight.hdf5".format(epoch+1)
            baseYAMLFilename = str(self.save_prefix)+"ModelChkpt{:06d}.yaml".format(epoch+1)
            hdf5Filename     = os.path.join(self.chkptsdir, baseHDF5Filename)
            hdf5Filename_weight     = os.path.join(self.chkptsdir, baseHDF5Filename_weight)
            yamlFilename            = os.path.join(self.chkptsdir, baseYAMLFilename)

            # YAML
            yamlModel = self.model_mono.to_yaml()
            with open(yamlFilename, "w") as yamlFile:
                yamlFile.write(yamlModel)

            # HDF5
            KM.save_model(self.model_mono, hdf5Filename)
            self.model_mono.save_weights(hdf5Filename_weight)
            with H.File(hdf5Filename, "r+") as f:
                f.require_dataset("initialEpoch", (), "uint64", True)[...] = int(epoch+1)
                f.flush()
            with H.File(hdf5Filename_weight, "r+") as f:
                f.require_dataset("initialEpoch", (), "uint64", True)[...] = int(epoch+1)
                f.flush()


            # Symlink to new HDF5 file, then atomically rename and replace.
            os.symlink(baseHDF5Filename_weight, self.linkFilename_weight+".rename")
            os.rename (self.linkFilename_weight+".rename",
                    self.linkFilename_weight)


            # Symlink to new HDF5 file, then atomically rename and replace.
            os.symlink(baseHDF5Filename, self.linkFilename+".rename")
            os.rename (self.linkFilename+".rename",
                    self.linkFilename)

            # Print
            L.getLogger("train").info("Saved checkpoint to {:s} at epoch {:5d}".format(hdf5Filename, epoch+1))

#
# Summarize environment variable.
#

def summarizeEnvvar(var):
    if var in os.environ: return var+"="+os.environ.get(var)
    else:                 return var+" unset"

#
# TRAINING PROCESS
#

def train(d):
    
    #
    #
    # Log important data about how we were invoked.
    #
    L.getLogger("entry").info("INVOCATION:     "+" ".join(sys.argv))
    L.getLogger("entry").info("HOSTNAME:       "+socket.gethostname())
    L.getLogger("entry").info("PWD:            "+os.getcwd())
    L.getLogger("entry").info("CUDA DEVICE:            "+str(d.device))
    os.environ["CUDA_VISIBLE_DEVICES"]=str(d.device)
    
    #
    # Setup GPUs
    #
    config = tf.ConfigProto()
    
    # 
    # Don't pre-allocate memory; allocate as-needed
    #
    config.gpu_options.allow_growth = True
     
    #
    # Only allow a total of half the GPU memory to be allocated
    #
    config.gpu_options.per_process_gpu_memory_fraction = d.memory
    
    #
    # Create a session with the above options specified.
    #
    K.tensorflow_backend.set_session(tf.Session(config=config))
    
    summary  = "\n"
    summary += "Environment:\n"
    summary += summarizeEnvvar("THEANO_FLAGS")+"\n"
    summary += "\n"
    summary += "Software Versions:\n"
    summary += "Theano:                  "+T.__version__+"\n"
    summary += "Keras:                   "+keras.__version__+"\n"
    summary += "\n"
    summary += "Arguments:\n"
    summary += "Path to Datasets:        "+str(d.datadir)+"\n"
    summary += "Number of GPUs:          "+str(d.datadir)+"\n"
    summary += "Path to Workspace:       "+str(d.workdir)+"\n"
    summary += "Model:                   "+str(d.model)+"\n"
    summary += "Number of Epochs:        "+str(d.num_epochs)+"\n"
    summary += "Number of Start Filters: "+str(d.start_filter)+"\n"
    summary += "Number of Layers:        "+str(d.num_layers)+"\n"
    summary += "Optimizer:               "+str(d.optimizer)+"\n"
    summary += "Learning Rate:           "+str(d.lr)+"\n"
    summary += "Learning Rate Decay:     "+str(d.decay)+"\n"
    summary += "Clipping Norm:           "+str(d.clipnorm)+"\n"
    summary += "Clipping Value:          "+str(d.clipval)+"\n"
    summary += "Dropout Probability:     "+str(d.dropout)+"\n"
    if d.optimizer in ["adam"]:
        summary += "Beta 1:                  "+str(d.beta1)+"\n"
        summary += "Beta 2:                  "+str(d.beta2)+"\n"
    else:
        summary += "Momentum:                "+str(d.momentum)+"\n"
    summary += "Save Prefix:             "+str(d.save_prefix)+"\n"
    L.getLogger("entry").info(summary[:-1])

    #
    # Load dataset
    #
    L.getLogger("entry").info("Loading dataset {:s} ...".format(d.dataset))
    np.random.seed(d.seed % 2**32)

    #
    # Create training data generator
    #
    My_train_generator = timitGenerator("train")
   
    #
    # Create dev data generator
    #
    My_dev_generator = timitGenerator("dev")


    L.getLogger("entry").info("Training   set length: "+str(Timit('train').num_examples))
    L.getLogger("entry").info("Validation set length: "+str(Timit('dev').num_examples))
    L.getLogger("entry").info("Test       set length: "+str(Timit('test').num_examples))
    L.getLogger("entry").info("Loaded  dataset {:s}.".format(d.dataset))

    #
    # Optimizers
    #
    if   d.optimizer in ["sgd", "nag"]:
        opt = SGD    (lr       = d.lr,
                momentum = d.momentum,
                decay    = d.decay,
                nesterov = (d.optimizer=="nag"),
                clipnorm = d.clipnorm)
    elif d.optimizer == "rmsprop":
        opt = RMSProp(lr       = d.lr,
                decay    = d.decay,
                clipnorm = d.clipnorm)
    elif d.optimizer == "adam":
        opt = Adam   (lr       = d.lr,
                beta_1   = d.beta1,
                beta_2   = d.beta2,
                decay    = d.decay,
                clipnorm = d.clipnorm)
    else:
        raise ValueError("Unknown optimizer "+d.optimizer)


    #
    # Initial Entry or Resume ?
    #

    initialEpoch  = 0
    chkptFilename = os.path.join(d.workdir, "chkpts", str(d.save_prefix)+"ModelChkpt.hdf5")
    chkptFilename_weight = os.path.join(d.workdir, "chkpts", str(d.save_prefix)+"ModelChkpt_weight.hdf5")
    isResuming    = os.path.isfile(chkptFilename)
    isResuming_weight    = os.path.isfile(chkptFilename_weight)
    
    if isResuming or isResuming_weight:
        
        # Reload Model and Optimizer
        if d.dataset == "timit":
            L.getLogger("entry").info("Re-Creating the model from scratch.")
            model_mono,test_func = getTimitResnetModel2D(d)
            model_mono.load_weights(chkptFilename_weight)
            with H.File(chkptFilename_weight, "r") as f:
                initialEpoch = int(f["initialEpoch"][...])
            L.getLogger("entry").info("Training will restart at epoch {:5d}.".format(initialEpoch+1))
            L.getLogger("entry").info("Compilation Started.")

        else:
            
            L.getLogger("entry").info("Reloading a model from "+chkptFilename+" ...")
            np.random.seed(d.seed % 2**32)
            model = KM.load_model(chkptFilename, custom_objects={
                "QuaternionConv2D":          QuaternionConv2D,
                "QuaternionConv1D":          QuaternionConv1D,
                "GetIFirst":                   GetIFirst,
                "GetJFirst":                   GetJFirst,
                "GetKFirst":                   GetKFirst,
                "GetRFirst":                   GetRFirst,
                })
            L.getLogger("entry").info("... reloading complete.")
            with H.File(chkptFilename, "r") as f:
                initialEpoch = int(f["initialEpoch"][...])
            L.getLogger("entry").info("Training will restart at epoch {:5d}.".format(initialEpoch+1))
            L.getLogger("entry").info("Compilation Started.")
    else:
        model_mono,test_func = getTimitModel2D(d)
        
    L.getLogger("entry").info("Compilation Started.")
    
    #
    # Multi GPU: Can only save the model_mono because of keras bug
    #
    if d.gpus >1:
        model = multi_gpu_model(model_mono, gpus=d.gpus)
    else:
        model = model_mono
    
    #
    # Compile with CTC koss function
    #
    model.compile(opt, loss={'ctc': lambda y_true, y_pred: y_pred})
    

    #
    # Precompile several backend functions
    #
    if d.summary:
        model.summary()
    L.getLogger("entry").info("# of Parameters:              {:10d}".format(model.count_params()))
    L.getLogger("entry").info("Compiling Train   Function...")
    t =- time.time()
    model._make_train_function()
    t += time.time()
    L.getLogger("entry").info("                              {:10.3f}s".format(t))
    L.getLogger("entry").info("Compiling Predict Function...")
    t =- time.time()
    model._make_predict_function()
    t += time.time()
    L.getLogger("entry").info("                              {:10.3f}s".format(t))
    L.getLogger("entry").info("Compiling Test    Function...")
    t =- time.time()
    model._make_test_function()
    t += time.time()
    L.getLogger("entry").info("                              {:10.3f}s".format(t))
    L.getLogger("entry").info("Compilation Ended.")

    #
    # Create Callbacks
    #
    newLineCb      = PrintNewlineAfterEpochCallback()   
    saveLastCb     = SaveLastModel(d.workdir, d.save_prefix, model_mono, period=10)


    callbacks  = []

    #
    # End of line for better looking
    #
    callbacks += [newLineCb]
    if d.model=="quaternion":
        quaternion = True            
    else:
        quaternion = False
    
    if not os.path.exists(d.workdir+"/LOGS"):
        os.makedirs(d.workdir+"/LOGS")
    savedir = d.workdir+"/LOGS/"+d.save_prefix

    #
    # Save the Train loss
    #
    trainLoss = TrainLoss(savedir)
    
    

    ######################
    # Call your EditDistance metric
    #
    #editDistValCb  = EditDistance('dev')
    #editDistTestCb = EditDistance('test')
    #callbacks += [editDistValCb]
    #callbacks += [editDistTestCb]
    
    #
    # Compute losses
    #
    callbacks += [trainLoss]
    callbacks += [newLineCb]

    #
    # Save the model
    #
    callbacks += [saveLastCb]
    
    #
    # Enter training loop.
    #
    L               .getLogger("entry").info("**********************************************")
    if isResuming: L.getLogger("entry").info("*** Reentering Training Loop @ Epoch {:5d} ***".format(initialEpoch+1))
    else:          L.getLogger("entry").info("***  Entering Training Loop  @ First Epoch ***")
    L               .getLogger("entry").info("**********************************************")
    

    #
    # TRAIN
    #

    ########
    # Make sure to give the right number of mini_batch size
    # needed to complete ONE epoch (according to your data generator)
    ########

    epochs_train = 1
    epochs_dev   = 1

    model.fit_generator(generator        = My_train_generator,
                        steps_per_epoch  = epochs_train,
                        epochs           = d.num_epochs,
                        verbose          = 1,
                        validation_data  = My_dev_generator,
                        validation_steps = epochs_dev,
                        callbacks        = callbacks,
                        initial_epoch    = initialEpoch)
 
