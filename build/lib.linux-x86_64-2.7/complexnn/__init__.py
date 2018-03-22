#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Contributors: Titouan Parcollet
# Authors: Olexa Bilaniuk
#
# What this module includes by default:
import bn, conv, dense, fft, init, norm, pool

from   .bn    import ComplexBatchNormalization as ComplexBN
from   .bn    import QuaternionBatchNormalization as QuaternionBN
from   .conv  import (ComplexConv,
                      ComplexConv1D,
                      ComplexConv2D,
                      ComplexConv3D,
					  QuaternionConv,
					  QuaternionConv1D,
					  QuaternionConv2D,
					  QuaternionConv3D,
                      WeightNorm_Conv)
from   .dense import ComplexDense
from   .fft   import fft, ifft, fft2, ifft2, FFT, IFFT, FFT2, IFFT2
from   .init  import (ComplexIndependentFilters, IndependentFilters,
                      ComplexInit, SqrtInit, QuaternionInit, QuaternionIndependentFilters)
from   .norm  import LayerNormalization, ComplexLayerNorm
from   .pool  import SpectralPooling1D, SpectralPooling2D
from   .utils import (get_realpart, get_imagpart, getpart_output_shape,
                      GetImag, GetReal, GetAbs, getpart_quaternion_output_shape, get_rpart, get_ipart, get_jpart, get_kpart, GetR, GetI,GetJ,GetK)
