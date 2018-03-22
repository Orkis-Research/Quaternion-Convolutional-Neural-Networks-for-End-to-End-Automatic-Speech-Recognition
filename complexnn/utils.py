#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Authors: Titouan Parcollet


import keras.backend as K
from keras.layers import Layer, Lambda

######
# Need to rewrite this part to have only one getter for each part
#
######################
#  Quaternions TIMIT #
######################

def get_rpart_first(x):
	ndim = K.ndim(x)
	input_shape = K.shape(x)
	data_format = 'channels_first'
	if (data_format == 'channels_first' and ndim != 3) or ndim == 2:
		input_dim = input_shape[1] // 4
		return x[:, :input_dim]

	input_dim = input_shape[-1] // 4
	if ndim == 3:
		return x[:, :, :input_dim]
	elif ndim == 4:
		return x[:, :, :, :input_dim]
	elif ndim == 5:
		return x[:, :, :, :, :input_dim]

def get_ipart_first(x):
	ndim = K.ndim(x)
	input_shape = K.shape(x)
	data_format = 'channels_first'
	if (data_format == 'channels_first' and ndim != 3) or ndim == 2:
		input_dim = input_shape[1] // 4
		return x[:, input_dim:input_dim*2]

	input_dim = input_shape[-1] // 4
	if ndim == 3:
		return x[:, :, input_dim:input_dim*2]
	elif ndim == 4:
		return x[:, :, :, input_dim:input_dim*2]
	elif ndim == 5:
		return x[:, :, :, :, input_dim:input_dim*2]

def get_jpart_first(x):
	ndim = K.ndim(x)
	input_shape = K.shape(x)
	data_format = 'channels_first'	
	if (data_format == 'channels_first' and ndim != 3) or ndim == 2:
		input_dim = input_shape[1] // 4
		return x[:, input_dim*2:input_dim*3]

	input_dim = input_shape[-1] // 4
	if ndim == 3:
		return x[:, :, input_dim*2:input_dim*3]
	elif ndim == 4:
		return x[:, :, :, input_dim*2:input_dim*3]
	elif ndim == 5:
		return x[:, :, :, :, input_dim*2:input_dim*3]

def get_kpart_first(x):
	ndim = K.ndim(x)
	input_shape = K.shape(x)
	data_format = 'channels_first'
	if (data_format == 'channels_first' and ndim != 3) or ndim == 2:
		input_dim = input_shape[1] // 4
		return x[:, input_dim*3:]

	input_dim = input_shape[-1] // 4
	if ndim == 3:
		return x[:, :, input_dim*3:]
	elif ndim == 4:
		return x[:, :, :, input_dim*3:]
	elif ndim == 5:
		return x[:, :, :, :, input_dim*3:]

class GetRFirst(Layer):
	def call(self, inputs):
		return get_rpart_first(inputs)
	def compute_output_shape(self, input_shape):
		return getpart_quaternion_output_shape_first(input_shape)
class GetIFirst(Layer):
	def call(self, inputs):
		return get_ipart_first(inputs)
	def compute_output_shape(self, input_shape):
		return getpart_quaternion_output_shape_first(input_shape)
class GetJFirst(Layer):
	def call(self, inputs):
		return get_jpart_first(inputs)
	def compute_output_shape(self, input_shape):
		return getpart_quaternion_output_shape_first(input_shape)
class GetKFirst(Layer):
	def call(self, inputs):
		return get_kpart_first(inputs)
	def compute_output_shape(self, input_shape):
		return getpart_quaternion_output_shape_first(input_shape)

def getpart_quaternion_output_shape_first(input_shape):
	returned_shape = list(input_shape[:])
	image_format = K.image_data_format()
	ndim = len(returned_shape)
	
	data_format = 'channels_first'
	if (data_format == 'channels_first' and ndim != 3) or ndim == 2:
		axis = 1
	else:
		axis = -1

	returned_shape[axis] = returned_shape[axis] // 4

	return tuple(returned_shape)


