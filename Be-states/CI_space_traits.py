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
from qode.util.PyC import import_C, Double, BigInt
field_op = import_C("field_op", flags="-O2 -lm")
field_op.find_index.return_type(int)



class CI_space_traits(object):
	def __init__(self, configs):
		self.field = numpy.float64
		self.configs = numpy.array(configs, dtype=BigInt.numpy)
	def new_vec(self, config=None):
		v = numpy.zeros((1,len(self.configs)), dtype=Double.numpy)    # nominally a matrix bc Hamiltonian supports block action
		if config is not None:
			config = int(config, 2)    # config comes in as a string literal of 0 and 1 characters
			index  = field_op.find_index(config, self.configs, len(self.configs))
			v[0,index] = 1
		return v
	def check_member(self,v):
		pass
	def check_lin_op(self,op):
		return False
	@staticmethod
	def copy(v):
		return v.copy()
	@staticmethod
	def scale(c,v):
		v *= c
	@staticmethod
	def add_to(v,w,c=1):
		if c==1:  v += w
		else:     v += c*w
	@staticmethod
	def dot(v,w):
		return (v.dot(w.T)).item()    # Here we have to remember that vecs are actually 1xlen(configs) 2-tensors
	def act_on_vec(self, op, v):
		return op(v, self.configs)
	def back_act_on_vec(self, v, op):
		return op(v, self.configs)
	def act_on_vec_block(self, op, v_block):
		return [ op(v, self.configs) for v in v_block ]
	def back_act_on_vec_block(self, v_block, op):
		return [ op(v, self.configs) for v in v_block ]
	@staticmethod
	def dot_vec_blocks(v_block,w_block):
		return numpy.array([[CI_space_traits_class.dot(v,w) for v in v_block] for w in w_block])
