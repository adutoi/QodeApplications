#    (C) Copyright 2018, 2019, 2023 Yuhong Liu and Anthony Dutoi
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
from qode.util.PyC import import_C, Double
field_op = import_C("field_op", flags="-O2 -lm")



class Hamiltonian(object):
    def __init__(self, h, V, thresh=1e-10):
        self.h = h
        self.V = V
        self.thresh = thresh
    def __call__(self, Psi, configs, vec_0=0, num_vecs=1):    # The ability to act on blocks of consecutively stored vectors is not currently used (but it has been tested)
        num_spin_orbs = self.h.shape[0]
        HPsi = numpy.zeros((num_vecs,len(configs)), dtype=Double.numpy)
        field_op.opPsi_1e(self.h,           # tensor of matrix elements (integrals)
                          Psi,              # block of row vectors: input vectors to act on
                          HPsi,             # block of row vectors: incremented by output
                          configs,          # bitwise occupation strings stored as integers ... so, max 64 orbs for FCI ;-) [no checking here!]
                          num_spin_orbs,    # edge dimension of the integrals tensor.  cannot be bigger than the number of bits in a BigInt (64)
                          vec_0,            # index of first vector in block to act upon
                          num_vecs,         # how many vectors we are acting on simultaneously
                          len(configs),     # how many configurations are there (call signature is ok as long as PyInt not longer than BigInt)
                          self.thresh,      # threshold for ignoring integrals and coefficients (avoiding expensive index search)
                          0)                # number of OMP threads to spread the work over (not currently used)
        field_op.opPsi_2e(self.V,           # tensor of matrix elements (integrals)
                          Psi,              # block of row vectors: input vectors to act on
                          HPsi,             # block of row vectors: incremented by output
                          configs,          # bitwise occupation strings stored as integers ... so, max 64 orbs for FCI ;-) [no checking here!]
                          num_spin_orbs,    # edge dimension of the integrals tensor.  cannot be bigger than the number of bits in a BigInt (64)
                          vec_0,            # index of first vector in block to act upon
                          num_vecs,         # how many vectors we are acting on simultaneously
                          len(configs),     # how many configurations are there (call signature is ok as long as PyInt not longer than BigInt)
                          self.thresh,      # threshold for ignoring integrals and coefficients (avoiding expensive index search)
                          0)                # number of OMP threads to spread the work over (not currently used)
        return HPsi
