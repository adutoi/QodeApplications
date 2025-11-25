#    (C) Copyright 2025 Anthony D. Dutoi
# 
#    This file is part of QodeApplications.
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
import tensorly
import qode.math.tensornet as tensornet

# Just for testing, the thinnest imaginable wrapper for a tensornet.primitive_tensor (specifically tl_tensor)
class meta_wrapper(object):
    def __init__(self, internal):
        self.internal = internal

# These just reach in and do the tensor operations with the internal tensor
class meta_functions(object):
    @staticmethod
    def shape(tensor):
        return tensornet.shape(tensor.internal)
    @staticmethod
    def scalar_value(tensor):
        return tensornet.scalar_value(tensor.internal)
    @staticmethod
    def copy_data(tensor):
        return meta_wrapper(tensor.internal.copy())
    @staticmethod
    def zeros(shape):
        return meta_wrapper(tensornet.tl_tensor.zeros(shape))
    @staticmethod
    def scalar_tensor(scalar):
        return meta_wrapper(tensornet.tl_tensor.scalar_tensor(scalar))
    @staticmethod
    def increment(tensor, delta):
        # tensornet assumes that the incrementing of the raw tensor is truly in place,
        # but a tensornet tensor has no equivalent concept, so need to do this.
        tensor.internal._raw_tensor += delta.internal._raw_tensor
        return
    @staticmethod
    def mult(scalar, tensor):
        return meta_wrapper(scalar * tensor.internal)
    @staticmethod
    def element(tensor, indices):
        return tensor.internal[indices]
    @staticmethod
    def str(tensor):
        return str(tensor.internal)
    @staticmethod
    def contract(*tensor_factors):
        scalar = 1
        tensors = []
        for factor in tensor_factors:
            try:
                tensor, *indices = factor
            except:
                scalar *= factor
            else:
                tensors += [(tensor, indices)]
        return meta_wrapper(tensornet.evaluate(scalar * tensornet.contract(*(tensor.internal(*indices) for tensor,indices in tensors))))

# the factory for tensornet tensors backed by the meta backend
meta_tensor = tensornet.primitive_tensor_factory(meta_functions)
