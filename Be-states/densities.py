#    (C) Copyright 2018, 2019, 2023 Anthony D. Dutoi and Yuhong Liu
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
from qode.math.tensornet import tl_tensor
from qode.math           import svd_decomposition
import field_op



def tens_wrap(tensor):
    return tl_tensor(tensorly.tensor(tensor, dtype=tensorly.float64))



# states[n].coeffs  = [numpy.array, numpy.array, . . .]   One (effectively 1D) array of coefficients per n-electron state
# states[n].configs = [int, int, . . . ]                  Each int represents a configuration (has the same length as arrays in list above)

def build_tensors(states, n_orbs, n_elec_0, thresh=1e-10, n_threads=1):
    densities = {}
    densities["n_elec"]   = {chg:(n_elec_0-chg)          for chg in states}
    densities["n_states"] = {chg:len(states[chg].coeffs) for chg in states}

    op_strings = {2:["aa", "caaa"], 1:["a", "caa", "ccaaa"], 0:["ca", "ccaa"], -1:["c", "cca", "cccaa"], -2:["cc", "ccca"]}
    for bra_chg in states:
        print(bra_chg)
        #print(states[bra_chg].coeffs)
        #print(states[bra_chg].configs)
        bra_coeffs  = states[bra_chg].coeffs
        bra_configs = field_op.packed_configs(states[bra_chg].configs)
        for ket_chg in states:
            print("  ", ket_chg)
            #print("  ", states[ket_chg].coeffs)
            #print("  ", states[ket_chg].configs)
            ket_coeffs  = states[ket_chg].coeffs
            ket_configs = field_op.packed_configs(states[ket_chg].configs)
            chg_diff = bra_chg - ket_chg
            if chg_diff in op_strings:
                for op_string in op_strings[chg_diff]:
                    if op_string not in densities:  densities[op_string] = {}
                    print(op_string, bra_chg, ket_chg)
                    rho = field_op.build_densities(op_string, n_orbs, bra_coeffs, ket_coeffs, bra_configs, ket_configs, thresh, n_threads)
                    for i in range(len(bra_coeffs)):
                        for j in range(len(ket_coeffs)):
                            indices = list(range(len(op_string)))
                            c_count = op_string.count("c")
                            rho[i,j] = svd_decomposition(rho[i,j], indices[:c_count], indices[c_count:], wrapper=tens_wrap)
                    densities[op_string][bra_chg,ket_chg] = rho

    return densities
