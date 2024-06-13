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
import numpy
import tensorly
from qode.math.tensornet import tl_tensor, tensor_sum
from qode.math           import svd_decomposition
import field_op

# states[n].coeffs  = [numpy.array, numpy.array, . . .]   One (effectively 1D) array of coefficients per n-electron state
# states[n].configs = [int, int, . . . ]                  Each int represents a configuration (has the same length as arrays in list above)



def _tens_wrap(tensor):
    return tl_tensor(tensorly.tensor(tensor, dtype=tensorly.float64))

def _vec(i, length):
    v = numpy.zeros((length,))
    v[i] = 1
    return _tens_wrap(v)

def build_tensors(states, n_orbs, n_elec_0, thresh=1e-10, compress=True, n_threads=1):
    densities = {}
    densities["n_elec"]   = {chg:(n_elec_0-chg)          for chg in states}
    densities["n_states"] = {chg:len(states[chg].coeffs) for chg in states}

    op_strings = {2:["aa", "caaa"], 1:["a", "caa", "ccaaa"], 0:["ca", "ccaa"], -1:["c", "cca", "cccaa"], -2:["cc", "ccca"]}
    for bra_chg in states:
        print(bra_chg)
        bra_coeffs  = states[bra_chg].coeffs
        bra_configs = field_op.packed_configs(states[bra_chg].configs)
        for ket_chg in states:
            print("  ", ket_chg)
            ket_coeffs  = states[ket_chg].coeffs
            ket_configs = field_op.packed_configs(states[ket_chg].configs)
            chg_diff = bra_chg - ket_chg
            if chg_diff in op_strings:
                for op_string in op_strings[chg_diff]:
                    if op_string not in densities:  densities[op_string] = {}
                    print(op_string, bra_chg, ket_chg)
                    rho = field_op.build_densities(op_string, n_orbs, bra_coeffs, ket_coeffs, bra_configs, ket_configs, thresh, wisdom=None, n_threads=n_threads)
                    temp = tensor_sum()
                    for i in range(len(bra_coeffs)):
                        for j in range(len(ket_coeffs)):
                            indices = list(range(len(op_string)))
                            if compress:
                                c_count = op_string.count("c")
                                rho_ij = svd_decomposition(rho[i,j], indices[:c_count], indices[c_count:], wrapper=_tens_wrap)
                            else:
                                rho_ij = _tens_wrap(rho[i,j])
                            temp +=  _vec(i,len(bra_coeffs))(0) @ _vec(j,len(ket_coeffs))(1) @ rho_ij(*(p+2 for p in indices))
                    densities[op_string][bra_chg,ket_chg] = temp
    return densities
