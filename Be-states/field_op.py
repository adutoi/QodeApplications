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
import math
import numpy
from qode.util.PyC import import_C, Double, BigInt

# Import the C module in a python wrapper for external aesthetics and to avoid having compile
# flags in multiple places (which points to a weakness in PyC that changing these does not
# force a recompile and defining them inconsistently will silently just use the first one
# imported.

field_op = import_C("field_op", flags="-O3 -lm -fopenmp")
field_op.orbs_per_configint.return_type(int)
field_op.bisect_search.return_type(int)



orbs_per_configint = field_op.orbs_per_configint()

class packed_configs(object):
    def __init__(self, configs):
        self.length = len(configs)
        max_orbs  = math.floor(1 + math.log(configs[-1],2))
        self.size = 1 + int(max_orbs)//orbs_per_configint
        self.packed = numpy.zeros(self.length * self.size, dtype=BigInt.numpy)
        reduction = 2**orbs_per_configint
        for i,config in enumerate(configs):
            reduced = config
            for n in range(self.size):
                self.packed[i*self.size + n] = reduced % reduction
                reduced //= reduction
    def __len__(self):
        return self.length

def find_index(config, configs):
    reduction = 2**orbs_per_configint
    packed_config = numpy.zeros(configs.size, dtype=BigInt.numpy)
    reduced = config
    for n in range(configs.size):
        packed_config[n] = reduced % reduction
        reduced //= reduction
    return field_op.bisect_search(packed_config, configs.packed, configs.size, 0, len(configs)-1)

def opPsi_1e(HPsi, Psi, h, vec_0, num_vecs, configs, thresh, n_threads):
    field_op.opPsi(1,                 # electron order of the operator
                   h,                 # tensor of matrix elements (integrals)
                   Psi,               # block of row vectors: input vectors to act on
                   HPsi,              # block of row vectors: incremented by output
                   configs.packed,    # bitwise occupation strings stored as arrays of integers (see packed_configs above)
                   configs.size,      # the number of BigInts needed to store a configuration
                   h.shape[0],        # edge dimension of the integrals tensor
                   vec_0,             # index of first vector in block to act upon
                   num_vecs,          # how many vectors we are acting on simultaneously
                   len(configs),      # how many configurations are there (call signature is ok as long as PyInt not longer than BigInt)
                   thresh,            # threshold for ignoring integrals and coefficients (avoiding expensive index search)
                   n_threads)         # number of OMP threads to spread the work over

def opPsi_2e(HPsi, Psi, V, vec_0, num_vecs, configs, thresh, n_threads):
    field_op.opPsi(2,                 # electron order of the operator
                   V,                 # tensor of matrix elements (integrals), assumed antisymmetrized
                   Psi,               # block of row vectors: input vectors to act on
                   HPsi,              # block of row vectors: incremented by output
                   configs.packed,    # bitwise occupation strings stored as arrays of integers (see packed_configs above)
                   configs.size,      # the number of BigInts needed to store a configuration
                   V.shape[0],        # edge dimension of the integrals tensor
                   vec_0,             # index of first vector in block to act upon
                   num_vecs,          # how many vectors we are acting on simultaneously
                   len(configs),      # how many configurations are there (call signature is ok as long as PyInt not longer than BigInt)
                   thresh,            # threshold for ignoring integrals and coefficients (avoiding expensive index search)
                   n_threads)         # number of OMP threads to spread the work over
