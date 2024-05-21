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
import numpy as np



def tens_wrap(tensor):
    return tl_tensor(tensorly.tensor(tensor, dtype=tensorly.float64))



# states[n].coeffs  = [numpy.array, numpy.array, . . .]   One (effectively 1D) array of coefficients per n-electron state
# states[n].configs = [int, int, . . . ]                  Each int represents a configuration (has the same length as arrays in list above)

def build_tensors(states, n_orbs, n_elec_0, thresh=1e-10, n_threads=1, bra_det=False):
    densities = {}
    densities["n_elec"]   = {chg:(n_elec_0-chg)          for chg in states}
    densities["n_states_ket"] = {chg:len(states[chg].coeffs) for chg in states}
    if bra_det == False:
        densities["n_states_bra"] = {chg:len(states[chg].coeffs) for chg in states}
    else:
        densities["n_states_bra"] = {chg:len(states[chg].configs) for chg in states}
    print(densities["n_states_bra"], densities["n_states_ket"])

    #op_strings = {2:["aa", "caaa"], 1:["a", "caa", "ccaaa"], 1:["ca", "ccaa"], -1:["c", "cca", "cccaa"], -2:["cc", "ccca"]}
    op_strings = {2:["aa"], 1:["a", "caa"], 0:["ca", "ccaa"], -1:["c", "cca"], -2:["cc"]}
    for bra_chg in states:
        print(bra_chg)
        #print(states[bra_chg].coeffs)
        #print(states[bra_chg].configs)
        if bra_det == False:
            bra_coeffs  = states[bra_chg].coeffs
        else:
            n_configs = (len(states[bra_chg].coeffs[0]))  # better set (len(states[bra_chg].configs)) for more transparency
            bra_coeffs = [i for i in np.eye(n_configs)]
            #op_strings = {2:["aa"], 1:["a", "caa"], 0:["ca"], -1:["c", "cca"], -2:["cc"]}
            #op_strings = {2:["aa"], 1:["a"], 0:["ca"], -1:["c"], -2:["cc"]}
        bra_configs = field_op.packed_configs(states[bra_chg].configs)
        for ket_chg in states:
            print("  ", ket_chg)
            #print("  ", states[ket_chg].coeffs)
            #print("  ", states[ket_chg].configs)
            ket_coeffs  = states[ket_chg].coeffs
            ket_configs = field_op.packed_configs(states[ket_chg].configs)
            chg_diff = bra_chg - ket_chg
            if chg_diff not in op_strings:
                continue
            for op_string in op_strings[chg_diff]:
                if op_string not in densities:  densities[op_string] = {}
                print(op_string, bra_chg, ket_chg)
                #if op_string == "ccaa" and (bra_chg == -1 and ket_chg == -1):
                #    densities[op_string][bra_chg,ket_chg] = np.zeros(())
                rho = field_op.build_densities(op_string, n_orbs, bra_coeffs, ket_coeffs, bra_configs, ket_configs, thresh, n_threads)
                for i in range(len(bra_coeffs)):
                    for j in range(len(ket_coeffs)):
                        indices = list(range(len(op_string)))
                        c_count = op_string.count("c")
                        try:
                            if len(bra_coeffs) > 200 and op_string == "ccaa":
                                print("this density is not SVD'd, because it's already in a compressed form")
                                #rho[i,j] = tens_wrap(rho[i,j])  # this density is already a tl_tensor object
                                pass
                            else:
                                rho[i,j] = svd_decomposition(rho[i,j], indices[:c_count], indices[c_count:], wrapper=tens_wrap)
                        except np.linalg.LinAlgError:
                            print("this density is not SVD'd, because svd didn't converge")
                            #rho[i,j] = tens_wrap(rho[i,j])  # this sometimes leaves NaNs or infs, which the eigensolver cannot handle
                            raise ValueError("SVD didnt converge")
                densities[op_string][bra_chg,ket_chg] = rho

    return densities
