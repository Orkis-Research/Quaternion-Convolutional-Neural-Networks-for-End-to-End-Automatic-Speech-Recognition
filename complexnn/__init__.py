#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Contributors: Titouan Parcollet
# Authors: Olexa Bilaniuk
#
# What this module includes by default:

from   .conv  import (QuaternionConv,
					  QuaternionConv1D,
					  QuaternionConv2D,
					  QuaternionConv3D)

from   .dense import QuaternionDense
from   .init  import (SqrtInit, QuaternionInit)
from   .utils import (GetRFirst, GetIFirst, GetJFirst, GetKFirst, getpart_quaternion_output_shape_first, get_rpart_first, get_ipart_first, get_jpart_first,
		      get_kpart_first)


