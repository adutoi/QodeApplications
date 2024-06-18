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
from qode.util           import sort_eigen
from qode.util.PyC       import Double
from qode.math.tensornet import tl_tensor, tensor_sum, raw, evaluate
from qode.math           import svd_decomposition
import field_op

# states[n].coeffs  = [numpy.array, numpy.array, . . .]   One (effectively 1D) array of coefficients per n-electron state
# states[n].configs = [int, int, . . . ]                  Each int represents a configuration (has the same length as arrays in list above)



def _token_parser(options):
    parsed_options = {}
    if options is None:  options = []
    for option in options:
        if "=" in option:
            key, value = option.split("=")
            if value=="True":   value = True
            if value=="False":  value = False
            parsed_options[key] = value
        else:
            parsed_options[option] = True
    def check(option, value=True):
        answer = False
        if option in parsed_options:
            if parsed_options[option]==value:
                answer = True
        return answer
    return check

def _tens_wrap(tensor):
    return tl_tensor(tensorly.tensor(tensor, dtype=Double.tensorly))

def _vec(i, length):
    v = numpy.zeros((length,), dtype=Double.numpy, order="C")
    v[i] = 1
    return _tens_wrap(v)



def build_tensors(states, n_orbs, n_elec_0, thresh=1e-10, options=None, n_threads=1):
    op_strings = {2:["aa", "caaa"], 1:["a", "caa", "ccaaa"], 0:["ca", "ccaa"], -1:["c", "cca", "cccaa"], -2:["cc", "ccca"]}
    permutations = {
        "aa":    {+1:[(0,1)], -1:[(1,0)]},
        "cc":    {+1:[(0,1)], -1:[(1,0)]},
        "caa":   {+1:[(0,1,2)], -1:[(0,2,1)]},
        "cca":   {+1:[(0,1,2)], -1:[(1,0,2)]},
        "ccaa":  {+1:[(0,1,2,3), (1,0,3,2)], -1:[(0,1,3,2), (1,0,2,3)]},
        "caaa":  {+1:[(0,1,2,3), (0,2,3,1), (0,3,1,2)], -1:[(0,1,3,2), (0,2,1,3), (0,3,2,1)]},
        "ccca":  {+1:[(0,1,2,3), (1,2,0,3), (2,0,1,3)], -1:[(0,2,1,3), (1,0,2,3), (2,1,0,3)]},
        "ccaaa": {+1:[(0,1,2,3,4), (0,1,3,4,2), (0,1,4,2,3), (1,0,2,4,3), (1,0,3,2,4), (1,0,4,3,2)], -1:[(0,1,2,4,3), (0,1,3,2,4), (0,1,4,3,2), (1,0,2,3,4), (1,0,3,4,2), (1,0,4,2,3)]},
        "cccaa": {+1:[(0,1,2,3,4), (1,2,0,3,4), (2,0,1,3,4), (0,2,1,4,3), (1,0,2,4,3), (2,1,0,4,3)], -1:[(0,2,1,3,4), (1,0,2,3,4), (2,1,0,3,4), (0,1,2,4,3), (1,2,0,4,3), (2,0,1,4,3)]},
    }
    densities = {}

    options = _token_parser(options)
    use_natural_orbs   = options("nat-orbs")                            # do compression in natural orbital rep? (default: no)
    antisymm_abstract  = options("abs-anti")                            # antisymmetry abstract in final rep, which might be original? (default: no)
    antisymm_numerical = (not antisymm_abstract) or use_natural_orbs    # numerically antisymmetrize in original rep? (default: yes)

    print("Computing densities ...")

    for bra_chg in states:
        bra_coeffs  = states[bra_chg].coeffs
        bra_configs = field_op.packed_configs(states[bra_chg].configs)
        for ket_chg in states:
            ket_coeffs  = states[ket_chg].coeffs
            ket_configs = field_op.packed_configs(states[ket_chg].configs)
            chg_diff = bra_chg - ket_chg
            if chg_diff in op_strings:
                for op_string in op_strings[chg_diff]:
                    if op_string not in densities:  densities[op_string] = {}
                    print(bra_chg, ket_chg, op_string)
                    rho = field_op.build_densities(op_string, n_orbs, bra_coeffs, ket_coeffs, bra_configs, ket_configs, thresh, wisdom=None, antisymmetrize=antisymm_numerical, n_threads=n_threads)
                    densities[op_string][bra_chg,ket_chg] = [[_tens_wrap(rho_ij) for rho_ij in rho_i] for rho_i in rho]

    print("Postprocessing ...")

    if use_natural_orbs:
        natural_orbs = {}
        for chg,_ in densities["ca"]:
            rho = densities["ca"][chg,chg]              # bra/ket charges must be the same for this string ...
            natural_orbs_chg = []
            for i in range(len(rho)):                   # ... which means the number of bras and kets are the same
                rho_ii = numpy.array(raw(rho[i][i]), dtype=Double.numpy, order="C")
                #print(chg, i, "deviation from symmetric:", numpy.linalg.norm(rho_ii - rho_ii.T))
                evals, evecs = sort_eigen(numpy.linalg.eigh(rho_ii), order="descending")
                natural_orbs_chg += [_tens_wrap(evecs)]
            natural_orbs[chg] = natural_orbs_chg

    for op_string in densities:
        c_count = op_string.count("c")
        a_count = op_string.count("a")
        indices = list(range(len(op_string)))
        for bra_chg,ket_chg in densities[op_string]:
            print(op_string, bra_chg, ket_chg)
            rho = densities[op_string][bra_chg,ket_chg]
            temp = tensor_sum()
            #
            n_bras = len(rho)
            for i,rho_i in enumerate(rho):
                n_kets = len(rho_i)
                for j,rho_ij in enumerate(rho_i):
                    #
                    if use_natural_orbs:
                        p = 0
                        for _ in range(c_count):
                            indices_p = list(indices)
                            indices_p[p] = "p"
                            rho_ij = rho_ij(*indices_p) @ natural_orbs[bra_chg][i]("p",p)
                            p += 1
                        for _ in range(a_count):
                            indices_p = list(indices)
                            indices_p[p] = "p"
                            rho_ij = rho_ij(*indices_p) @ natural_orbs[ket_chg][j]("p",p)
                            p += 1
                        rho_ij = numpy.array(raw(rho_ij), dtype=Double.numpy, order="C")
                        if antisymm_abstract:
                            field_op.asymmetrize(op_string, rho_ij)
                        rho_ij = _tens_wrap(rho_ij)
                    if options("compress", "cc-aa-svd"):    # SVD-compress the densities, separating creation from annihilation indices
                        rho_ij = svd_decomposition(numpy.array(raw(rho_ij), dtype=Double.numpy, order="C"), indices[:c_count], indices[c_count:], wrapper=_tens_wrap)
                    if use_natural_orbs:
                        p = 0
                        for _ in range(c_count):
                            indices_p = list(indices)
                            indices_p[p] = "p"
                            rho_ij = rho_ij(*indices_p) @ natural_orbs[bra_chg][i](p,"p")
                            p += 1
                        for _ in range(a_count):
                            indices_p = list(indices)
                            indices_p[p] = "p"
                            rho_ij = rho_ij(*indices_p) @ natural_orbs[ket_chg][j](p,"p")
                            p += 1
                    if antisymm_abstract:
                        if op_string in permutations:
                            temp_ij = tensor_sum()
                            for permutation in permutations[op_string][+1]:
                                temp_ij += rho_ij(*permutation)
                            for permutation in permutations[op_string][-1]:
                                temp_ij -= rho_ij(*permutation)
                            rho_ij = temp_ij
                    #
                    temp += _vec(i,n_bras)(0) @ _vec(j,n_kets)(1) @ rho_ij(*(p+2 for p in indices))
            #
            densities[op_string][bra_chg,ket_chg] = temp

    densities["n_elec"]   = {chg:(n_elec_0-chg)          for chg in states}
    densities["n_states"] = {chg:len(states[chg].coeffs) for chg in states}

    print("Writing to hard drive ...")
    return densities
