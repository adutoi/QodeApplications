#    (C) Copyright 2024 Anthony D. Dutoi
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
import tensorly
from qode.util.dynamic_array import dynamic_array, cached
from qode.math.tensornet     import evaluate, raw, tl_tensor



#"cca#pqX_U#pq##"
#cca(p,q,0) @ V(p,q,1,2)

def precontract(densities, integrals, timings):
    n_frag = len(densities)

    def mother_rule(label):
        n_indices = label.count("#")

        def contract_rho_int(*indices):
            rho_label, int_label = label.split("_")
            rho_type, rho_idxstr = rho_label.split("#")
            int_type, int_idxstr = int_label[0], int_label[1:]
            rho_indices = []
            int_indices = []
            int_blocks  = []
            block_idx = 0
            free_idx  = 2
            if int_type=="U":
                block_idx += 1
                int_blocks += [indices[block_idx]]
                int_idxstr = int_idxstr[1:]
            for idx in rho_idxstr:
                if idx=="X":
                    rho_indices += [free_idx]
                    free_idx += 1
                else:
                    rho_indices += [idx]
            for idx in int_idxstr:
                if idx=="#":
                    int_indices += [free_idx]
                    free_idx += 1
                    block_idx += 1
                    int_blocks += [indices[block_idx]]
                else:
                    int_indices += [idx]
                    int_blocks += [indices[0]]
                #try:
                #    ints = integrals.T[int_blocks]
                #except:
                #    print(int_blocks)
            int_blocks = tuple(int_blocks)
            if int_type=="S":  ints = integrals[int_blocks]    # makes the calling signature inhomogeneous to force error otherwise
            if int_type=="T":  ints = integrals.T[int_blocks]
            if int_type=="U":  ints = integrals.U[int_blocks]
            if int_type=="V":  ints = integrals.V[int_blocks]
            densities_m = densities[indices[0]]
            Dchg = rho_type.count("a") - rho_type.count("c")    # get rid of this and allow the exception

            def contract_rho_int_m(chg_i,chg_j):
                if chg_i-chg_j==Dchg:
                    rho = densities_m[rho_type][chg_i,chg_j]
                    timings.start()
                    result = evaluate(rho(*([0,1]+rho_indices)) @ ints(*int_indices))
                    timings.record(label)
                    return result
                else:
                    return None
            return dynamic_array(cached(contract_rho_int_m), [densities_m["n_states"].keys()]*2)  # no specification between bra and ket needed, because only keys are required

        def contract_rho_rho_int(*indices):
            raise NotImplementedError

        if label.count("_")==1:    # good enough for now
            return dynamic_array(cached(contract_rho_int), [range(n_frag)]*n_indices)

    class _hack(object):    # wraps single string index as 1-tuple to not confuse dynamic_array
        def __init__(self, dyn_array):
            self.dyn_array = dyn_array
        def __getitem__(self, key):
            return self.dyn_array[(key,)]

    return _hack(dynamic_array(cached(mother_rule), [None]))
