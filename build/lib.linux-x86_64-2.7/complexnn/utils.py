#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Contributors: Titouan Parcollet
# Authors: Dmitriy Serdyuk, Olexa Bilaniuk, Chiheb Trabelsi

import keras.backend as K
from keras.layers import Layer, Lambda

################
#  Quaternions #
################

def get_rpart(x):
	image_format = K.image_data_format()
	ndim = K.ndim(x)
	input_shape = K.shape(x)

	if (image_format == 'channels_first' and ndim != 3) or ndim == 2:
		input_dim = input_shape[1] // 4
		return x[:, :input_dim]

	input_dim = input_shape[-1] // 4
	if ndim == 3:
		return x[:, :, :input_dim]
	elif ndim == 4:
		return x[:, :, :, :input_dim]
	elif ndim == 5:
		return x[:, :, :, :, :input_dim]

def get_ipart(x):
	image_format = K.image_data_format()
	ndim = K.ndim(x)
	input_shape = K.shape(x)

	if (image_format == 'channels_first' and ndim != 3) or ndim == 2:
		input_dim = input_shape[1] // 4
		return x[:, input_dim:input_dim*2]

	input_dim = input_shape[-1] // 4
	if ndim == 3:
		return x[:, :, input_dim:input_dim*2]
	elif ndim == 4:
		return x[:, :, :, input_dim:input_dim*2]
	elif ndim == 5:
		return x[:, :, :, :, input_dim:input_dim*2]

def get_jpart(x):
	image_format = K.image_data_format()
	ndim = K.ndim(x)
	input_shape = K.shape(x)

	if (image_format == 'channels_first' and ndim != 3) or ndim == 2:
		input_dim = input_shape[1] // 4
		return x[:, input_dim*2:input_dim*3]

	input_dim = input_shape[-1] // 4
	if ndim == 3:
		return x[:, :, input_dim*2:input_dim*3]
	elif ndim == 4:
		return x[:, :, :, input_dim*2:input_dim*3]
	elif ndim == 5:
		return x[:, :, :, :, input_dim*2:input_dim*3]

def get_kpart(x):
	image_format = K.image_data_format()
	ndim = K.ndim(x)
	input_shape = K.shape(x)

	if (image_format == 'channels_first' and ndim != 3) or ndim == 2:
		input_dim = input_shape[1] // 4
		return x[:, input_dim*3:]

	input_dim = input_shape[-1] // 4
	if ndim == 3:
		return x[:, :, input_dim*3:]
	elif ndim == 4:
		return x[:, :, :, input_dim*3:]
	elif ndim == 5:
		return x[:, :, :, :, input_dim*3:]

class GetR(Layer):
	def call(self, inputs):
		return get_rpart(inputs)
	def compute_output_shape(self, input_shape):
		return getpart_quaternion_output_shape(input_shape)
class GetI(Layer):
	def call(self, inputs):
		return get_ipart(inputs)
	def compute_output_shape(self, input_shape):
		return getpart_quaternion_output_shape(input_shape)
class GetJ(Layer):
	def call(self, inputs):
		return get_jpart(inputs)
	def compute_output_shape(self, input_shape):
		return getpart_quaternion_output_shape(input_shape)
class GetK(Layer):
	def call(self, inputs):
		return get_kpart(inputs)
	def compute_output_shape(self, input_shape):
		return getpart_quaternion_output_shape(input_shape)

def getpart_quaternion_output_shape(input_shape):
	returned_shape = list(input_shape[:])
	image_format = K.image_data_format()
	ndim = len(returned_shape)

	if (image_format == 'channels_first' and ndim != 3) or ndim == 2:
		axis = 1
	else:
		axis = -1

	returned_shape[axis] = returned_shape[axis] // 4

	return tuple(returned_shape)


#
# GetReal/GetImag Lambda layer Implementation
#


def get_realpart(x):
    image_format = K.image_data_format()
    ndim = K.ndim(x)
    input_shape = K.shape(x)

    if (image_format == 'channels_first' and ndim != 3) or ndim == 2:
        input_dim = input_shape[1] // 2
        return x[:, :input_dim]

    input_dim = input_shape[-1] // 2
    if ndim == 3:
        return x[:, :, :input_dim]
    elif ndim == 4:
        return x[:, :, :, :input_dim]
    elif ndim == 5:
        return x[:, :, :, :, :input_dim]


def get_imagpart(x):
    image_format = K.image_data_format()
    ndim = K.ndim(x)
    input_shape = K.shape(x)

    if (image_format == 'channels_first' and ndim != 3) or ndim == 2:
        input_dim = input_shape[1] // 2
        return x[:, input_dim:]

    input_dim = input_shape[-1] // 2
    if ndim == 3:
        return x[:, :, input_dim:]
    elif ndim == 4:
        return x[:, :, :, input_dim:]
    elif ndim == 5:
        return x[:, :, :, :, input_dim:]


def get_abs(x):
    real = get_realpart(x)
    imag = get_imagpart(x)

    return K.sqrt(real * real + imag * imag)


def getpart_output_shape(input_shape):
    returned_shape = list(input_shape[:])
    image_format = K.image_data_format()
    ndim = len(returned_shape)

    if (image_format == 'channels_first' and ndim != 3) or ndim == 2:
        axis = 1
    else:
        axis = -1

    returned_shape[axis] = returned_shape[axis] // 2

    return tuple(returned_shape)


class GetReal(Layer):
    def call(self, inputs):
        return get_realpart(inputs)
    def compute_output_shape(self, input_shape):
        return getpart_output_shape(input_shape)
class GetImag(Layer):
    def call(self, inputs):
        return get_imagpart(inputs)
    def compute_output_shape(self, input_shape):
        return getpart_output_shape(input_shape)
class GetAbs(Layer):
    def call(self, inputs):
        return get_abs(inputs)
    def compute_output_shape(self, input_shape):
        return getpart_output_shape(input_shape)

