#    (C) Copyright 2018, 2023 Anthony D. Dutoi and Yuhong Liu
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

class fci_space_traits_class(object):
	def __init__(self):
		self.field = numpy.float64
	def check_member(self,v):
		pass
	def check_lin_op(self,op):
		return False
	@staticmethod
	def copy(v):
		n, block, i, (I,dim) = v
		new = numpy.zeros((1,dim))
		new[0,:] = block[i,:]
		return (n, new, 0, (1,dim))
	@staticmethod
	def scale(c,v):
		n, block, i, (I,dim) = v
		block[i,:] *= c
	@staticmethod
	def add_to(v,w,c=1):
		nv, blockv, iv, (Iv,dimv) = v
		nw, blockw, iw, (Iw,dimw) = w
		if   nv!=nw:    raise Exception("this should not happen here ... true Fock-space vecs not allowed")
		if dimv!=dimw:  raise Exception("this should not happen here ... true Fock-space vecs not allowed")
		if c==1:  blockv[iv,:] +=   blockw[iw,:]
		else:     blockv[iv,:] += c*blockw[iw,:]
	@staticmethod
	def dot(v,w):
		nv, blockv, iv, (Iv,dimv) = v
		nw, blockw, iw, (Iw,dimw) = w
		if nv!=nw:  return 0.
		else:       return blockv[iv,:].dot(blockw[iw,:])
	@staticmethod
	def act_on_vec(op,v):
		return op(v)
	@staticmethod
	def back_act_on_vec(v,op):
		return op(v)
	@staticmethod
	def act_on_vec_block(op,v_block):
		return [ op(v) for v in v_block ]
	@staticmethod
	def back_act_on_vec_block(v_block,op):
		return [ op(v) for v in v_block ]
	@staticmethod
	def dot_vec_blocks(v_block,w_block):
		return numpy.array([[fci_space_traits_class.dot(v,w) for v in v_block] for w in w_block])

fci_space_traits = fci_space_traits_class()
