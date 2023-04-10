#    (C) Copyright 2018 Anthony D. Dutoi
# 
#    This file is part of Qode.
# 
#    Qode is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
# 
#    Qode is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
# 
#    You should have received a copy of the GNU General Public License
#    along with Qode.  If not, see <http://www.gnu.org/licenses/>.
#
# Use this command:
#    python3 setup.py build_ext --inplace
#

from distutils.core import setup, Extension
import numpy.distutils.misc_util

setup(
	name = 'transformVMat',
	version = '1.0',
    ext_modules=[ Extension("transformVMat", sources=["transformVMat.c"],extra_compile_args=['-std=c99']) ],
    include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs()
    )

