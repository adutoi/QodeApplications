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
import sys
import numpy

# def act_on_vec_block(op,v_block)

def list_to_numpy_block(list_vec):
    return ((numpy.array(list_vec)).T).copy()

def numpy_block_to_list(numpy_block):
    return [ numpy.array( numpy_block[:,i] ) for i in range(numpy_block.shape[1]) ]


class ham_traits(object):
    """ This code only deals with REAL matrices/vecotrs """
    def __init__(self): # num_elec, num_core_elec, num_spin_orbs, h_mat, V_mat):
        self.field = numpy.float64
    def check_member(self,v):  pass
    def check_lin_op(self,op):  pass
    @staticmethod
    def dot(v,w):
        return numpy.dot(v,w)
    @staticmethod
    def dot_vec_blocks(v_block,w_block):
        return [[numpy.dot(v,w) for w in w_block] for v in v_block]
    @staticmethod
    def add_to(v,w,c=1):
        if c==1:  v +=   w
        else:     v += c*w
    @staticmethod
    def scale(n,v):
        v *= n
    @staticmethod
    def copy(v):
        return v.copy()
    @staticmethod
    def act_on_vec(op,v):
        #print("running act_on_vec",flush=True)
        #sys.stdout.flush()
        return op(v)
    @staticmethod
    def back_act_on_vec(v,op):
        return op(v)
    @staticmethod
    def act_on_vec_block(op,v_block):
        #print("running act_on_vec_block",flush=True)
        #sys.stdout.flush()
        return numpy_block_to_list( op( list_to_numpy_block(v_block) ) )
