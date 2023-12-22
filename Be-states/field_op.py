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

antisymm = import_C("antisymm", flags="-O3")



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

def opPsi_1e(HPsi, Psi, h, configs, thresh, n_threads):
    field_op.op_Psi(1,                 # electron order of the operator
                    h,                 # tensor of matrix elements (integrals), assumed antisymmetrized
                    [Psi],             # block of row vectors: input vectors to act on
                    [HPsi],            # block of row vectors: incremented by output
                    configs.packed,    # bitwise occupation strings stored as arrays of integers (see packed_configs above)
                    configs.size,      # the number of BigInts needed to store a configuration
                    h.shape[0],        # edge dimension of the integrals tensor
                    0,                 # index of first vector in block to act upon
                    1,                 # how many vectors we are acting on simultaneously
                    len(configs),      # how many configurations are there (call signature is ok as long as PyInt not longer than BigInt)
                    thresh,            # threshold for ignoring integrals and coefficients (avoiding expensive index search)
                    n_threads)         # number of OMP threads to spread the work over

def opPsi_2e(HPsi, Psi, V, configs, thresh, n_threads):
    field_op.op_Psi(2,                 # electron order of the operator
                    V,                 # tensor of matrix elements (integrals), assumed antisymmetrized
                    [Psi],             # block of row vectors: input vectors to act on
                    [HPsi],            # block of row vectors: incremented by output
                    configs.packed,    # bitwise occupation strings stored as arrays of integers (see packed_configs above)
                    configs.size,      # the number of BigInts needed to store a configuration
                    V.shape[0],        # edge dimension of the integrals tensor
                    0,                 # index of first vector in block to act upon
                    1,                 # how many vectors we are acting on simultaneously
                    len(configs),      # how many configurations are there (call signature is ok as long as PyInt not longer than BigInt)
                    thresh,            # threshold for ignoring integrals and coefficients (avoiding expensive index search)
                    n_threads)         # number of OMP threads to spread the work over

def build_densities(op_string, n_orbs, bras, kets, bra_configs, ket_configs, thresh, n_threads):
    n_create  = op_string.count("c")
    n_destroy = op_string.count("a")
    if (op_string != "c"*n_create + "a"*n_destroy):  raise ValueError("density operator string is not vacuum normal ordered")
    shape = [n_orbs] * (n_create + n_destroy)
    rho = []
    for _ in range(len(bras) * len(kets)):
        rho += [numpy.array(shape, dtype=Double.numpy)]
    field_op.densities(n_create,           # number of creation operators
                       n_destroy,          # number of destruction operators
                       rho,                # storage for density tensor for each pair of states in linear list
                       bras,               # block of row vectors: bra states
                       kets,               # block of row vectors: ket states
                       bra_configs.packed,     # bitwise occupation strings stored as arrays of integers (packed in one contiguous block, per global comments above)
                       ket_configs.packed,     # bitwise occupation strings stored as arrays of integers (packed in one contiguous block, per global comments above)
                       bra_configs.size,       # the number of BigInts needed to store a configuration
                       ket_configs.size,       # the number of BigInts needed to store a configuration
                       n_orbs,             # edge dimension of the density tensors
                       len(bras),          # how many bra states
                       len(kets),          # how many ket states
                       len(bra_configs),       # how many configurations are there (call signature is ok as long as PyInt not longer than BigInt)
                       len(ket_configs),       # how many configurations are there (call signature is ok as long as PyInt not longer than BigInt)
                       thresh,             # threshold for ignoring coefficients (avoiding expensive index search)
                       n_threads)          # number of OMP threads to spread the work over
    antisymm.antisymmetrize(rho,    # the density to antisymmetrize
                            len(rho),
                            n_orbs,     # the number of orbitals
                            n_create,
                            n_destroy)       # the respective number of creation and annihilation operators
    return [rho[i*len(kets):(i+1)*len(kets)] for i in range(len(bras))]






