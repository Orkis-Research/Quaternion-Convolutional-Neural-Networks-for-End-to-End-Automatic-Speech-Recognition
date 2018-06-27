#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Contributors: Titouan Parcollet
# Authors: Olexa Bilaniuk
#
# What this module includes by default:
import conv, dense, fft, init

from   .conv  import (QuaternionConv,
					  QuaternionConv1D,
					  QuaternionConv2D,
					  QuaternionConv3D,
                      WeightNorm_Conv)
from   .dense import QuaternionDense
from   .init  import (SqrtInit, QuaternionInit)
from   .utils import (GetRFirst, GetIFirst, GetJFirst, GetKFirst, getpart_quaternion_output_shape, get_rpart_first, get_ipart_first, get_jpart_first,
		      get_kpart_first)
