#    (C) Copyright 2023 Anthony D. Dutoi
# 
#    This file is part of QodeApplications.
# 
#    QodeApplications is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
# 
#    QodeApplications is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
# 
#    You should have received a copy of the GNU General Public License
#    along with QodeApplications.  If not, see <http://www.gnu.org/licenses/>.
#

import numpy
import tensorly
from   qode.math.tensornet import tl_tensor, raw



def SVDapprox(M, indent, thresh=1e-6):
        # Works for last two dimensions of any tensor in "stacked" mode (see numpy docs).
	U, s, Vh = numpy.linalg.svd(M)
	d = 0
	D = s.shape[-1]
	for i in range(D):
		if numpy.max(numpy.abs(s[...,i]))>thresh:  d += 1
	print(indent, d/D)
	U     =  U[...,:d].swapaxes(-2,-1)
	Shalf = numpy.sqrt(s[...,:d])
	Vh    = Vh[...,:d,:]
	A = Shalf[..., None] * U
	B = Shalf[..., None] * Vh
	return A, B, d

def compress(nparray_Nd, free_indices, compression="none", _sum_indices=None, _indent="", _first=True):
	if _sum_indices is None:  _sum_indices = ["i"]
	if compression=="none":    # means none as in no compression, not as in None being no value given
		# default full tensors
		return tl_tensor(tensorly.tensor(nparray_Nd, dtype=tensorly.float64))
	elif compression=="SVD" or compression=="recur-SVD":
		# split indices to matricize and SVD
		if len(free_indices)==1:
			return tl_tensor(tensorly.tensor(nparray_Nd, dtype=tensorly.float64))
		else:
			if _first:
				all_free_indices = list(free_indices[0]) + list(free_indices[1])
				nparray_Nd = nparray_Nd.transpose(all_free_indices)
			tensor_shape = list(nparray_Nd.shape)
			outer_dims   = len(_sum_indices) - 1
			if outer_dims+len(free_indices[0])+len(free_indices[1])!=len(tensor_shape):  raise RuntimeError("dimension mismatch")
			outer_shape  = tensor_shape[:outer_dims]
			inner_shape  = tensor_shape[outer_dims:]
			inner_shape1 = inner_shape[:len(free_indices[0])]
			inner_shape2 = inner_shape[len(free_indices[0]):]
			last_letter  = _sum_indices[-1]
			#
			tensor_reshape = outer_shape + [numpy.prod(inner_shape1), numpy.prod(inner_shape2)]
			A, B, d = SVDapprox(nparray_Nd.reshape(tensor_reshape), _indent)
			A = A.reshape(outer_shape + [d] + inner_shape1)
			B = B.reshape(outer_shape + [d] + inner_shape2)
			#
			if compression=="recur-SVD" and len(free_indices[0])>1:
				split = len(free_indices[0]) // 2
				free_indices_new = free_indices[0][:split], free_indices[0][split:]
				_sum_indices_new = _sum_indices + [last_letter+"i"]
				A = compress(A, free_indices_new, compression="recur-SVD", _sum_indices=_sum_indices_new, _indent=_indent+"  ", _first=False)
			else:
				contract = _sum_indices + list(free_indices[0])
				A = tl_tensor(tensorly.tensor(A, dtype=tensorly.float64))
				A = A(*contract)
			if compression=="recur-SVD" and len(free_indices[1])>1:
				split = len(free_indices[1]) // 2
				free_indices_new = free_indices[1][:split], free_indices[1][split:]
				_sum_indices_new = _sum_indices + [last_letter+"j"]
				B = compress(B, free_indices_new, compression="recur-SVD", _sum_indices=_sum_indices_new, _indent=_indent+"  ", _first=False)
			else:
				contract = _sum_indices + list(free_indices[1])
				B = tl_tensor(tensorly.tensor(B, dtype=tensorly.float64))
				B = B(*contract)
			return A @ B
	elif compression=="CP":
		# Use tensorly for CP decomposition ... This is so out of date it certainly does not run
		if sum(n_ops)==1:
			print("Nothing to do for 1-D")
			return tl_tensor(tensorly.tensor(nparray_Nd, dtype=tensorly.float64))
		if sum(n_ops)==5:
			print("Fall back to SVD for 5-D")
			return compress(nparray_Nd, n_ops, compression="SVD")
		else:
			thresh = 1e-16
			M0 = tensorly.tensor(nparray_Nd, dtype=tensorly.float64)
			try:
				print("trying {}-D ...".format(sum(n_ops)))
				weights, factors = tensorly.decomposition.parafac(M0, rank=n_spin_orbs**(min(sum(n_ops)-1,2)), normalize_factors=True)
			except:
				print("Failed CP decomposition on {}-D tensor. Fall back to SVD.".format(sum(n_ops)))
				return compress(nparray_Nd, n_ops, compression="SVD")
			weights_ = []
			factors_ = [[] for factor in factors]
			for i in reversed(numpy.argsort(weights)):
				if abs(weights[i])>thresh:
					weights_ += [weights[i]]
					for factor_,factor in zip(factors_,factors):
						factor_ += [factor[:,i]]
			weights = tl_tensor(tensorly.tensor(weights_, dtype=tensorly.float64))
			factors = [tl_tensor(tensorly.tensor(factor_, dtype=tensorly.float64)) for factor_ in factors_]    # factors are transpose of tensorly convention
			expression = weights("i")
			for p,factor in enumerate(factors):
				expression @= factor("i",p)
			M1 = raw(expression)
			err = tensorly.norm(M0-M1)
			print(err)
			if abs(err)>1e-3:
				print("Bad fit. Fall back to SVD.".format(sum(n_ops)))
				return compress(nparray_Nd, n_ops, compression="SVD")
			else:
				return(tl_tensor(M1))
