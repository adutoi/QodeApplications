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

class CI_space_traits_class(object):
	def __init__(self):
		self.field = numpy.float64
	def check_member(self,v):
		pass
	def check_lin_op(self,op):
		return False
	@staticmethod
	def copy(v):
		configs, vec = v
		return configs, vec.copy()
	@staticmethod
	def scale(c,v):
		configs, vec = v
		vec *= c
	@staticmethod
	def add_to(v,w,c=1):
		configsv, vecv = v
		configsw, vecw = w
		if configsv is not configsw:  raise RuntimeError("attempting to add CI vecs from different configuration spaces")
		if c==1:  vecv += vecw
		else:     vecv += c*vecw
	@staticmethod
	def dot(v,w):
		configsv, vecv = v
		configsw, vecw = w
		if configsv is not configsw:  raise RuntimeError("attempting to add CI vecs from different configuration spaces")
		return (vecv.dot(vecw.T)).item()    # Here we have to remember that vecs are actually 1xlen(configs) 2-tensors
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
		return numpy.array([[CI_space_traits_class.dot(v,w) for v in v_block] for w in w_block])

CI_space_traits = CI_space_traits_class()
