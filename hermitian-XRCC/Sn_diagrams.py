#    (C) Copyright 2023 Anthony D. Dutoi and Marco Bauer
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

from tendot import tendot
from qode.math.tensornet import scalar_value

p, q, r, s, t, u, v, w = "pqrstuvw"



class _empty(object):  pass    # Basically just a dictionary

def _parameters(densities, integrals, subsystem, charges, permutation=(0,)):
    # helper function to do repetitive manipulations of data passed from above
    densities = [densities[m] for m in subsystem]
    S, N = integrals
    S = {(m0_,m1_):S[m0,m1] for m1_,m1 in enumerate(subsystem) for m0_,m0 in enumerate(subsystem)}
    N = {(m0_,m1_):N[m0,m1] for m1_,m1 in enumerate(subsystem) for m0_,m0 in enumerate(subsystem)}
    #
    data = _empty()
    data.P = 0 if permutation==(0,1) else 1    # This line of code is specific to two fragments (needs to be generalized for >=3).
    #
    Dchg_rhos = {+2:["aa", "caaa"], +1:["a","caa","ccaaa"], 0:["ca","ccaa"], -1:["c","cca","cccaa"], -2:["cc", "ccca"]}
    n_i = 0
    n_i_label = ""
    for m0,m0_ in reversed(list(enumerate(permutation))):
        m0_str = str(m0)
        n_i_label = m0_str + n_i_label
        chg_i_m0 , chg_j_m0  = charges[m0]
        chg_i_m0_, chg_j_m0_ = charges[m0_]
        Dchg_m0_ = chg_i_m0_ - chg_j_m0_
        n_i += densities[m0]['n_elec'][chg_i_m0]    # this is not an error!
        data.__dict__["Dchg_"+m0_str] = Dchg_m0_
        data.__dict__["n_i"+n_i_label] = n_i%2
        for Dchg,rhos in Dchg_rhos.items():
            if Dchg==Dchg_m0_:
                for rho in rhos:
                    data.__dict__[rho+"_"+m0_str] = densities[m0_][rho][chg_i_m0_,chg_j_m0_]
        for m1,m1_ in enumerate(permutation):
            m01_str = m0_str + str(m1)
            data.__dict__["S_"+m01_str] = S[m0_,m1_]
            data.__dict__["N_"+m01_str] = N[m0_,m1_]
    return data



##########
# Here are the implementations of the actual diagrams.
# They take the arguments (densities, integrals, subsystem, charges), but after that, it is up to you.
# It should return a list of kernels that takes state indices (for specified fragment charges) their relevant permutations.
# Don't forget to update the "catalog" dictionary at the end.
##########

# monomer diagram

# N_00
def n00(densities, integrals, subsystem, charges):
    X = _parameters(densities, integrals, subsystem, charges)
    if X.Dchg_0==0:
        def diagram(i0,j0):
            if i0==j0:  return X.N_00
            else:       return 0
        return [(diagram, (0,))]
    else:
        return [(None, None)]



# dimer diagrams

# N_01
def n01(densities, integrals, subsystem, charges):
    X = _parameters(densities, integrals, subsystem, charges, permutation=(0,1))
    if X.Dchg_0==0 and X.Dchg_1==0:
        def diagram(i0,i1,j0,j1):
            if i0==j0 and i1==j1:  return X.N_01
            else:                  return 0
        return [(diagram, (0,1))]
    else:
        return [(None, None)]

# pq,p,q-> :  S_01  N_00  c_0  a_1
def s01n00(densities, integrals, subsystem, charges):
    result01 = _s01n00(densities, integrals, subsystem, charges, permutation=(0,1))
    result10 = _s01n00(densities, integrals, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01n00(densities, integrals, subsystem, charges, permutation):
    X = _parameters(densities, integrals, subsystem, charges, permutation)
    if X.Dchg_0==-1 and X.Dchg_1==+1:
        prefactor = (-1)**(X.n_i1 + X.P)
        def diagram(i0,i1,j0,j1):
            return X.N_00 * scalar_value( prefactor * X.c_0[i0][j0](p) @ X.S_01(p,q) @ X.a_1[i1][j1](q) )
        return diagram, permutation
    else:
        return None, None

# pq,p,q-> :  S_01  N_11  c_0  a_1
def s01n11(densities, integrals, subsystem, charges):
    result01 = _s01n11(densities, integrals, subsystem, charges, permutation=(0,1))
    result10 = _s01n11(densities, integrals, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01n11(densities, integrals, subsystem, charges, permutation):
    X = _parameters(densities, integrals, subsystem, charges, permutation)
    if X.Dchg_0==-1 and X.Dchg_1==+1:
        prefactor = (-1)**(X.n_i1 + X.P)
        def diagram(i0,i1,j0,j1):
            return X.N_11 * scalar_value( prefactor * X.c_0[i0][j0](p) @ X.S_01(p,q) @ X.a_1[i1][j1](q) )
        return diagram, permutation
    else:
        return None, None

# pq,p,q-> :  S_01  N_01  c_0  a_1
def s01n01(densities, integrals, subsystem, charges):
    result01 = _s01n01(densities, integrals, subsystem, charges, permutation=(0,1))
    result10 = _s01n01(densities, integrals, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01n01(densities, integrals, subsystem, charges, permutation):
    X = _parameters(densities, integrals, subsystem, charges, permutation)
    if X.Dchg_0==-1 and X.Dchg_1==+1:
        prefactor = (-1)**(X.n_i1 + X.P)
        def diagram(i0,i1,j0,j1):
            return X.N_01 * scalar_value( prefactor * X.c_0[i0][j0](p) @ X.S_01(p,q) @ X.a_1[i1][j1](q) )
        return diagram, permutation
    else:
        return None, None



##########
# A dictionary catalog.  the string association lets users specify active diagrams at the top level.
##########

catalog = {}

catalog[1] = {
    "n00":   n00
}

catalog[2] = {
    "n01":      n01,
    "s01n00":   s01n00,
    "s01n11":   s01n11,
    "s01n01":   s01n01
}
