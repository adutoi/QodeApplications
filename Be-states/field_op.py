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
from qode.util.PyC import import_C, Int, Double, BigInt

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

def find_index_by_occ(occupied, configs):
    config = 0
    for p in occupied:
        config += 2**p
    return find_index(config, configs)



class det_densities(object):
    def __init__(self, n_elec_right):
        self._n_elec_right  = n_elec_right
        self._occupied      = None
        self._det_indices   = None
        self._configs_left  = None
        self._configs_right = None
        self._scal_params   = None
        self._initialized    = False
    def _initialize(self, n_orbs, n_create, n_annihil, configs_left, configs_right):
        self._configs_left, self._configs_right = configs_left, configs_right
        self._scal_params = (n_orbs, n_create, n_annihil)
        size = (n_orbs - self._n_elec_right + n_annihil)**n_create * self._n_elec_right**n_annihil
        self._occupied    = [numpy.zeros((self._n_elec_right,), dtype=Int.numpy)    for _ in range(len(configs_right))]
        self._det_indices = [numpy.zeros((size,),               dtype=BigInt.numpy) for _ in range(len(configs_right))]
        self._initialized = True
    #def check(self, configs_left, configs_right):
    #    if ((configs_left is not self._configs_left) or (configs_right is not self._configs_right)):
    #        raise ValueError("inapplicable wisdom given to field_op engine")
    def check_initialization(self, n_orbs, n_create, n_annihil, configs_left, configs_right):
        unpopulated = False
        if not self._initialized:
            self._initialize(n_orbs, n_create, n_annihil, configs_left, configs_right)
            unpopulated = True
        if ((configs_left is not self._configs_left) or (configs_right is not self._configs_right)):
            raise ValueError("inapplicable wisdom given to field_op engine")
        if ((n_orbs, n_create, n_annihil) != self._scal_params):
            raise ValueError("inapplicable wisdom given to field_op engine")
        return self._occupied, self._det_indices, unpopulated

def opPsi_1e(HPsi, Psi, h, configs, thresh, wisdom, n_threads):
    generate_wisdom = 0
    wisdom_occupied = [numpy.zeros((1,),    dtype=Int.numpy)]
    wisdom_det_idx  = [numpy.zeros((1,), dtype=BigInt.numpy)]
    if wisdom is not None:
        wisdom_occupied, wisdom_det_idx, unpopulated = wisdom.check_initialization(h.shape[0], 1, 1, configs, configs)
        if unpopulated:  generate_wisdom = 1
        else:            generate_wisdom = 2
    field_op.op_Psi(1,                 # electron order of the operator
                    h,                 # tensor of matrix elements (integrals), assumed antisymmetrized
                    h.shape[0],        # edge dimension of the integrals tensor
                    [HPsi],            # array of row vectors: incremented by output
                    [Psi],             # array of row vectors: input vectors to act on
                    1,                 # how many vectors we are acting on and producing simultaneously in Psi and opPsi
                    configs.packed,    # configuration strings representing the basis for the states in Psi and opPsi (see packed_configs above)
                    len(configs),      # number of configurations in the configs basis (call signature ok if PyInt not longer than BigInt)
                    configs.size,      # number of BigInts needed to store a single configuration in configs
                    thresh,            # perform no further work if result will be smaller than this
                    n_threads,         # number of threads to spread the work over
                    generate_wisdom, wisdom_occupied, wisdom_det_idx)

def opPsi_2e(HPsi, Psi, V, configs, thresh, wisdom, n_threads):
    generate_wisdom = 0
    wisdom_occupied = [numpy.zeros((1,),    dtype=Int.numpy)]
    wisdom_det_idx  = [numpy.zeros((1,), dtype=BigInt.numpy)]
    if wisdom is not None:
        wisdom_occupied, wisdom_det_idx, unpopulated = wisdom.check_initialization(V.shape[0], 2, 2, configs, configs)
        if unpopulated:  generate_wisdom = 1
        else:            generate_wisdom = 0
    generate_wisdom = 0
    field_op.op_Psi(2,                 # electron order of the operator
                    V,                 # tensor of matrix elements (integrals), assumed antisymmetrized
                    V.shape[0],        # edge dimension of the integrals tensor
                    [HPsi],            # array of row vectors: incremented by output
                    [Psi],             # array of row vectors: input vectors to act on
                    1,                 # how many vectors we are acting on and producing simultaneously in Psi and opPsi
                    configs.packed,    # configuration strings representing the basis for the states in Psi and opPsi (see packed_configs above)
                    len(configs),      # number of configurations in the configs basis (call signature ok if PyInt not longer than BigInt)
                    configs.size,      # number of BigInts needed to store a single configuration in configs
                    thresh,            # perform no further work if result will be smaller than this
                    n_threads,         # number of threads to spread the work over
                    generate_wisdom, wisdom_occupied, wisdom_det_idx)

def build_densities(op_string, n_orbs, bras, kets, bra_configs, ket_configs, thresh, wisdom, n_threads):
    n_create  = op_string.count("c")
    n_annihil = op_string.count("a")
    if (op_string != "c"*n_create + "a"*n_annihil):  raise ValueError("density operator string is not vacuum normal ordered")
    shape = [n_orbs] * (n_create + n_annihil)
    print("####", op_string, "->", shape, "x", len(bras)*len(kets))
    rho = [numpy.zeros(shape, dtype=Double.numpy) for _ in range(len(bras)*len(kets))]
    generate_wisdom = 0
    wisdom_occupied = [numpy.zeros((1,),    dtype=Int.numpy)]
    wisdom_det_idx  = [numpy.zeros((1,), dtype=BigInt.numpy)]
    if wisdom is not None:
        wisdom_occupied, wisdom_det_idx, unpopulated = wisdom.check_initialization(n_orbs, n_create, n_annihil, bra_configs, ket_configs)
        if unpopulated:  generate_wisdom = 1
        else:            generate_wisdom = 0
    generate_wisdom = 0
    field_op.densities(n_create,              # number of creation operators
                       n_annihil,             # number of annihilation operators
                       rho,                   # array of storage for density tensors (for each bra-ket pair in linear list)
                       n_orbs,                # edge dimension of each density tensor
                       bras,                  # array of row vectors: bras for transition-density tensors
                       len(bras),             # number of bras
                       bra_configs.packed,    # configuration strings representing the basis for the bras (see packed_configs above)
                       len(bra_configs),      # number of configurations in the bra basis (call signature ok if PyInt not longer than BigInt)
                       bra_configs.size,      # number of BigInts needed to store a single configuration in the bra basis
                       kets,                  # array of row vectors: kets for transition-density tensors
                       len(kets),             # number of kets
                       ket_configs.packed,    # configuration strings representing the basis for the kets (see packed_configs above)
                       len(ket_configs),      # number of configurations in the ket basis (call signature ok if PyInt not longer than BigInt)
                       ket_configs.size,      # number of BigInts needed to store a single configuration in the ket basis
                       thresh,                # perform no further work if result will be smaller than this
                       n_threads,             # number of threads to spread the work over
                       generate_wisdom, wisdom_occupied, wisdom_det_idx)
    antisymm.antisymmetrize(rho,          # linear array of density tensors to antisymmetrize
                            len(rho),     # number of density tensors to antisymmetrize
                            n_orbs,       # number of orbitals
                            n_create,     # number of creation operators
                            n_annihil)    # number of annihilation operators
    return {(i,j):rho[i*len(kets)+j] for i in range(len(bras)) for j in range(len(kets))}
