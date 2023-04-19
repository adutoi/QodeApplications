#    (C) Copyright 2023 Anthony D. Dutoi
# 
#    This file is part of Qode.
# 
#    Qode is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
# 
#    Qode is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
# 
#    You should have received a copy of the GNU General Public License
#    along with Qode.  If not, see <http://www.gnu.org/licenses/>.
#

import tensorly
from tendot_wrapper import tendot

class _empty(object):  pass    # Basically just a dictionary

def _parameters1(densities, integrals, subsystem, charges):
    # helper functions to do repetitive manipulations of data passed from above
    # needs to be generalized (should not be hard) and have "2" removed from its name ... or maybe it is better this way
    densities = [densities[m] for m in subsystem]
    S = {(m1_,m2_):integrals.S[m1,m2] for m2_,m2 in enumerate(subsystem) for m1_,m1 in enumerate(subsystem)}
    T = {(m1_,m2_):integrals.T[m1,m2] for m2_,m2 in enumerate(subsystem) for m1_,m1 in enumerate(subsystem)}
    U = {(m1_,m2_,m3_):integrals.U[m1,m2,m3] for m3_,m3 in enumerate(subsystem) for m2_,m2 in enumerate(subsystem) for m1_,m1 in enumerate(subsystem)}
    V = {(m1_,m2_,m3_,m4_):integrals.V[m1,m2,m3,m4] for m4_,m4 in enumerate(subsystem) for m3_,m3 in enumerate(subsystem) for m2_,m2 in enumerate(subsystem) for m1_,m1 in enumerate(subsystem)}
    m1 = 0
    (chg_i1,chg_j1), = charges
    data = _empty()
    data.Dchg_1 = chg_i1 - chg_j1
    if data.Dchg_1==0:
        data.ca_1    = tensorly.tensor(densities[m1]['ca'  ][chg_i1,chg_j1])
        data.ccaa_1  = tensorly.tensor(densities[m1]['ccaa'][chg_i1,chg_j1])
    if data.Dchg_1==-1:
        data.c_1     = tensorly.tensor(densities[m1]['c'   ][chg_i1,chg_j1])
        data.cca_1   = tensorly.tensor(densities[m1]['cca' ][chg_i1,chg_j1])
    if data.Dchg_1==+1:
        data.a_1     = tensorly.tensor(densities[m1]['a'   ][chg_i1,chg_j1])
        data.caa_1   = tensorly.tensor(densities[m1]['caa' ][chg_i1,chg_j1])
    if data.Dchg_1==-2:
        data.cc_1    = tensorly.tensor(densities[m1]['cc'  ][chg_i1,chg_j1])
    if data.Dchg_1==+2:
        data.aa_1    = tensorly.tensor(densities[m1]['aa'  ][chg_i1,chg_j1])
    data.T_11   = tensorly.tensor(T[m1,m1])
    data.U_1_11 = tensorly.tensor(U[m1,m1,m1])
    data.V_1111 = tensorly.tensor(V[m1,m1,m1,m1])
    return data

def _parameters2(densities, integrals, subsystem, charges, permutation):
    # helper functions to do repetitive manipulations of data passed from above
    # needs to be generalized (should not be hard) and have "2" removed from its name ... or maybe it is better this way
    densities = [densities[m] for m in subsystem]
    S = {(m1_,m2_):integrals.S[m1,m2] for m2_,m2 in enumerate(subsystem) for m1_,m1 in enumerate(subsystem)}
    T = {(m1_,m2_):integrals.T[m1,m2] for m2_,m2 in enumerate(subsystem) for m1_,m1 in enumerate(subsystem)}
    U = {(m1_,m2_,m3_):integrals.U[m1,m2,m3] for m3_,m3 in enumerate(subsystem) for m2_,m2 in enumerate(subsystem) for m1_,m1 in enumerate(subsystem)}
    V = {(m1_,m2_,m3_,m4_):integrals.V[m1,m2,m3,m4] for m4_,m4 in enumerate(subsystem) for m3_,m3 in enumerate(subsystem) for m2_,m2 in enumerate(subsystem) for m1_,m1 in enumerate(subsystem)}
    m1, m2 = 0, 1
    (chg_i1,chg_j1), (chg_i2,chg_j2) = charges
    n_i2 = densities[m2]['n_elec'][chg_i2]
    p = 0
    if permutation==(1,0):
        m1, m2 = 1, 0
        (chg_i2,chg_j2), (chg_i1,chg_j1) = (chg_i1,chg_j1), (chg_i2,chg_j2)
        p = 1
    data = _empty()
    data.n_i2   = n_i2%2
    data.p      = p
    data.Dchg_1 = chg_i1 - chg_j1
    data.Dchg_2 = chg_i2 - chg_j2
    if data.Dchg_1==0:
        data.ca_1    = tensorly.tensor(densities[m1]['ca'  ][chg_i1,chg_j1])
        data.ccaa_1  = tensorly.tensor(densities[m1]['ccaa'][chg_i1,chg_j1])
    if data.Dchg_1==-1:
        data.c_1     = tensorly.tensor(densities[m1]['c'   ][chg_i1,chg_j1])
        data.cca_1   = tensorly.tensor(densities[m1]['cca' ][chg_i1,chg_j1])
    if data.Dchg_1==+1:
        data.a_1     = tensorly.tensor(densities[m1]['a'   ][chg_i1,chg_j1])
        data.caa_1   = tensorly.tensor(densities[m1]['caa' ][chg_i1,chg_j1])
    if data.Dchg_1==-2:
        data.cc_1    = tensorly.tensor(densities[m1]['cc'  ][chg_i1,chg_j1])
    if data.Dchg_1==+2:
        data.aa_1    = tensorly.tensor(densities[m1]['aa'  ][chg_i1,chg_j1])
    if data.Dchg_2==0:
        data.ca_2    = tensorly.tensor(densities[m2]['ca'  ][chg_i2,chg_j2])
        data.ccaa_2  = tensorly.tensor(densities[m2]['ccaa'][chg_i2,chg_j2])
    if data.Dchg_2==-1:
        data.c_2     = tensorly.tensor(densities[m2]['c'   ][chg_i2,chg_j2])
        data.cca_2   = tensorly.tensor(densities[m2]['cca' ][chg_i2,chg_j2])
    if data.Dchg_2==+1:
        data.a_2     = tensorly.tensor(densities[m2]['a'   ][chg_i2,chg_j2])
        data.caa_2   = tensorly.tensor(densities[m2]['caa' ][chg_i2,chg_j2])
    if data.Dchg_2==-2:
        data.cc_2    = tensorly.tensor(densities[m2]['cc'  ][chg_i2,chg_j2])
    if data.Dchg_2==+2:
        data.aa_2    = tensorly.tensor(densities[m2]['aa'  ][chg_i2,chg_j2])
    data.S_12   = tensorly.tensor(S[m1,m2])
    data.S_21   = tensorly.tensor(S[m2,m1])
    data.T_11   = tensorly.tensor(T[m1,m1])
    data.T_12   = tensorly.tensor(T[m1,m2])
    data.T_21   = tensorly.tensor(T[m2,m1])
    data.T_22   = tensorly.tensor(T[m2,m2])
    data.U_1_11 = tensorly.tensor(U[m1,m1,m1])
    data.U_1_12 = tensorly.tensor(U[m1,m1,m2])
    data.U_1_21 = tensorly.tensor(U[m1,m2,m1])
    data.U_1_22 = tensorly.tensor(U[m1,m2,m2])
    data.U_2_11 = tensorly.tensor(U[m2,m1,m1])
    data.U_2_12 = tensorly.tensor(U[m2,m1,m2])
    data.U_2_21 = tensorly.tensor(U[m2,m2,m1])
    data.U_2_22 = tensorly.tensor(U[m2,m2,m2])
    data.V_1111 = tensorly.tensor(V[m1,m1,m1,m1])
    data.V_1112 = tensorly.tensor(V[m1,m1,m1,m2])
    data.V_1121 = tensorly.tensor(V[m1,m1,m2,m1])
    data.V_1122 = tensorly.tensor(V[m1,m1,m2,m2])
    data.V_1211 = tensorly.tensor(V[m1,m2,m1,m1])
    data.V_1212 = tensorly.tensor(V[m1,m2,m1,m2])
    data.V_1221 = tensorly.tensor(V[m1,m2,m2,m1])
    data.V_1222 = tensorly.tensor(V[m1,m2,m2,m2])
    data.V_2111 = tensorly.tensor(V[m2,m1,m1,m1])
    data.V_2112 = tensorly.tensor(V[m2,m1,m1,m2])
    data.V_2121 = tensorly.tensor(V[m2,m1,m2,m1])
    data.V_2122 = tensorly.tensor(V[m2,m1,m2,m2])
    data.V_2211 = tensorly.tensor(V[m2,m2,m1,m1])
    data.V_2212 = tensorly.tensor(V[m2,m2,m1,m2])
    data.V_2221 = tensorly.tensor(V[m2,m2,m2,m1])
    data.V_2222 = tensorly.tensor(V[m2,m2,m2,m2])
    return data



##########
# Here are the implementations of the actual diagrams.
# The public @staticmethods must take the arguments (densities, integrals, subsystem, charges), but after that, it is up to you.
# It should return a list of kernels that takes state indices (for specified fragment charges) their relevant permutations.
# Don't forget to update the "catalog" dictionary at the end.
##########

class body_1(object):

    @staticmethod
    def order1(densities, integrals, subsystem, charges):
        X = _parameters1(densities, integrals, subsystem, charges)
        if X.Dchg_1==0:
            prefactor = 1
            def diagram(i1,j1):
                return prefactor * tendot((X.T_11+X.U_1_11), X.ca_1[i1][j1], axes=([0,1],[0,1]))
            return [(diagram, (0,))]
        else:
            return [(None, None)]

    @staticmethod
    def order2(densities, integrals, subsystem, charges):
        X = _parameters1(densities, integrals, subsystem, charges)
        if X.Dchg_1==0:
            prefactor = 1
            def diagram(i1,j1):
                return prefactor * tendot(X.V_1111, X.ccaa_1[i1][j1], axes=([0,1,2,3],[0,1,3,2]))
            return [(diagram, (0,))]
        else:
            return [(None, None)]



class body_2(object):

    @staticmethod
    def order1_CT0(densities, integrals, subsystem, charges):
        result01 = body_2._order1_CT0(densities, integrals, subsystem, charges, permutation=(0,1))
        result10 = body_2._order1_CT0(densities, integrals, subsystem, charges, permutation=(1,0))
        return [result01, result10]
    @staticmethod
    def _order1_CT0(densities, integrals, subsystem, charges, permutation):
        X = _parameters2(densities, integrals, subsystem, charges, permutation)
        if X.Dchg_1==0 and X.Dchg_2==0:
            prefactor = 1
            def diagram(i1,i2,j1,j2):
                if j1==j2:
                    return prefactor * tendot(X.U_2_11, X.ca1[i1][j1], axes=([0,1],[0,1]))
                else:
                    return 0    # This is inefficient, but we need the identity of a second fragment ... (how to do as 1-body term?)
            return diagram, permutation
        else:
            return None, None

    @staticmethod
    def order1_CT1(densities, integrals, subsystem, charges):
        result01 = body_2._order1_CT1(densities, integrals, subsystem, charges, permutation=(0,1))
        result10 = body_2._order1_CT1(densities, integrals, subsystem, charges, permutation=(1,0))
        return [result01, result10]
    @staticmethod
    def _order1_CT1(densities, integrals, subsystem, charges, permutation):
        X = _parameters2(densities, integrals, subsystem, charges, permutation)
        if X.Dchg_1==-1 and X.Dchg_2==+1:
            prefactor = (-1)**(X.n_i2 + X.p)
            def diagram(i1,i2,j1,j2):
                partial =          tendot((X.T_12 + X.U_1_12 + X.U_2_12), X.c1[i1][j1], axes=([0],[0]))
                return prefactor * tendot(partial,                        X.a2[i2][j2], axes=([0],[0]))
            return diagram, permutation
        else:
            return None, None



##########
# A dictionary catalog.  the string association lets users specify active diagrams at the top level.
# would like to build automatically, but more difficult than expected to get function references correct
# e.g., does not work
#catalog[2] = {}
#for k,v in body_2.__dict__.items():
#    catalog[2][k] = v
##########

catalog = {}

catalog[1] = {
"order1": body_1.order1,
"order2": body_1.order2
}

catalog[2] = {
"order1_CT1": body_2.order1_CT1,
}
