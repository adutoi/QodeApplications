#    (C) Copyright 2018, 2019, 2023, 2024 Anthony D. Dutoi and Yuhong Liu
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
from qode.util.PyC       import Double
from qode.math.tensornet import tl_tensor, raw
from qode.math           import svd_decomposition
from qode.many_body.fermion_field import field_op
from multi_term import multi_term

_permutations = {
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

def compress(rho_ij, op_string, bra_chg, ket_chg, i, j, compress, natural_orbs, antisymm_abstract, tens_wrap):
    c_count = op_string.count("c")
    a_count = op_string.count("a")
    indices = list(range(len(op_string)))
    if natural_orbs is not None:
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
        rho_ij = tens_wrap(rho_ij)
    if c_count>0 and a_count>0 and compress[0]=="SVD":
        thresh = 1e-6
        if len(compress)>2:
            thresh = compress[2]
            if len(compress)>3 and op_string=="ccaaa" and bra_chg==0:
                thresh = compress[3]
        if compress[1]=="cc-aa":      # SVD-compress the densities, separating creation from annihilation indices
            left_indices = indices[:c_count]
        if compress[1]=="ca-ca-F":    # SVD-compress the densities, separating one c an a index from the others
            left_indices = [0, c_count]
        if compress[1]=="ca-ca-O":    # SVD-compress the densities, separating one c an a index from the others
            left_indices = [0, len(indices)-1]
        if compress[1]=="ca-ca-I":    # SVD-compress the densities, separating one c an a index from the others
            left_indices = [c_count-1, c_count]
        if compress[1]=="ca-ca-L":    # SVD-compress the densities, separating one c an a index from the others
            left_indices = [c_count-1, len(indices)-1]
        right_indices = [i for i in indices if i not in left_indices]
        try:
            ret_rho = svd_decomposition(numpy.array(raw(rho_ij), dtype=Double.numpy, order="C"), left_indices, right_indices, thresh=thresh, wrapper=tens_wrap)
        except: # numpy.linalg.LinAlgError:
            ret_rho = rho_ij
        rho_ij = ret_rho
    elif c_count+a_count>1 and compress[0]=="multi":
        rho_ij = multi_term(rho_ij, c_count, a_count, compress[1], tens_wrap)
    if natural_orbs is not None:
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
        if op_string in _permutations:
            temp = tl_tensor.zeros()    # takes its shape from summed terms
            for permutation in _permutations[op_string][+1]:
                temp += rho_ij(*permutation)
            for permutation in _permutations[op_string][-1]:
                temp -= rho_ij(*permutation)
            rho_ij = temp
    return rho_ij
