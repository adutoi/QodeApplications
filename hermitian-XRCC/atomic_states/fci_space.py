#    (C) Copyright 2018 Anthony D. Dutoi and Yuhong Liu
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
		n, block, i, (dim,I) = v
		new = numpy.zeros((dim,1))
		new[:,0] = block[:,i]
		return (n, new, 0, (dim,1))
	@staticmethod
	def scale(c,v):
		n, block, i, (dim,I) = v
		block[:,i] *= c
	@staticmethod
	def add_to(v,w,c=1):
		nv, blockv, iv, (dimv,Iv) = v
		nw, blockw, iw, (dimw,Iw) = w
		if   nv!=nw:    raise Exception("this should not happen here ... true Fock-space vecs not allowed")
		if dimv!=dimw:  raise Exception("this should not happen here ... true Fock-space vecs not allowed")
		if c==1:  blockv[:,iv] +=   blockw[:,iw]
		else:     blockv[:,iv] += c*blockw[:,iw]
	@staticmethod
	def dot(v,w):
		nv, blockv, iv, (dimv,Iv) = v
		nw, blockw, iw, (dimw,Iw) = w
		if nv!=nw:  return 0.
		else:       return blockv[:,iv].dot(blockw[:,iw])
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
		#
		# return [ op(v) for v in v_block ]
		#
		# unpack the blocks
		nn, bblock, ii, dims = zip(*v_block)
		ddim, II = zip(*dims)
		# check that they are homogeneous
		n, block, dim, Io = nn[0], bblock[0], ddim[0], II[0]
		for n,b,d,I in zip(nn,bblock,ddim,II):
			if (n,b,d,I)!=(n,block,dim,Io):  raise Exception() 
		# check that the vectors are all adjacent
		iA,iZ = ii[0], ii[-1]+1
		if list(ii)!=list(range(iA,iZ)):  raise Exception()
		num = len(ii)
		# ok then, do it!
		v_block = n, block, (iA,num), (dim,Io)
		u_block = op(v_block)
		n, block, (iA,num), (dim,Io) = u_block		# FYI, iA will be 0, and num will be the same as Io
		# repackage results
		return [ (n, block, iA+j, (dim,num)) for j in range(num) ]
	@staticmethod
	def dot_vec_blocks(v_block,w_block):
		# unpack the blocks
		nnv, bblockv, iiv, dimsv = zip(*v_block)
		nnw, bblockw, iiw, dimsw = zip(*w_block)
		ddimv, IIv = zip(*dimsv)
		ddimw, IIw = zip(*dimsw)
		# check that they are homogeneous
		nv, blockv, dimv, Iv = nnv[0], bblockv[0], ddimv[0], IIv[0]
		nw, blockw, dimw, Iw = nnw[0], bblockw[0], ddimw[0], IIw[0]
		for n,b,d,I in zip(nnv,bblockv,ddimv,IIv):
			if (n,b,d,I)!=(nv,blockv,dimv,Iv):  raise Exception() 
		for n,b,d,I in zip(nnw,bblockw,ddimw,IIw):
			if (n,b,d,I)!=(nw,blockw,dimw,Iw):  raise Exception() 
		# check that the vectors are all adjacent
		ivA,ivZ = iiv[0], iiv[-1]+1
		iwA,iwZ = iiw[0], iiw[-1]+1
		if list(iiv)!=list(range(ivA,ivZ)):  raise Exception()
		if list(iiw)!=list(range(iwA,iwZ)):  raise Exception()
		# ok then, do it!
		return blockv[:,ivA:ivZ].T.dot(blockw[:,iwA:iwZ]).tolist()

fci_space_traits = fci_space_traits_class()
