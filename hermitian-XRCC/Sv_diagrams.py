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



class _empty(object):  pass    # Basically just a dictionary

def _parameters(densities, integrals, subsystem, charges, permutation=(0,)):
    # helper function to do repetitive manipulations of data passed from above
    densities = [densities[m] for m in subsystem]
    S, V = integrals
    S = {(m0_,m1_):S[m0,m1] for m1_,m1 in enumerate(subsystem) for m0_,m0 in enumerate(subsystem)}
    V = {(m0_,m1_,m2_,m3_):V[m0,m1,m2,m3] for m3_,m3 in enumerate(subsystem)
         for m2_,m2 in enumerate(subsystem) for m1_,m1 in enumerate(subsystem) for m0_,m0 in enumerate(subsystem)}
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
            for m2,m2_ in enumerate(permutation):
                m012_str  = m01_str + str(m2)
                for m3,m3_ in enumerate(permutation):
                    m0123_str = m012_str + str(m3)
                    data.__dict__["V_"+m0123_str] = V[m0_,m1_,m2_,m3_]
    return data



##########
# Here are the implementations of the actual diagrams.
# They must take the arguments (densities, integrals, subsystem, charges), but after that, it is up to you.
# It should return a list of kernels that takes state indices (for specified fragment charges) their relevant permutations.
# Don't forget to update the "catalog" dictionary at the end.
##########

# monomer diagram

# pqrs,pqsr-> :  V_0000  ccaa_0
def v0000(densities, integrals, subsystem, charges):
    X = _parameters(densities, integrals, subsystem, charges)
    if X.Dchg_0==0:
        prefactor = 1
        def diagram(i0,j0):
            return prefactor * tendot(X.V_0000, X.ccaa_0[i0][j0], axes=([0, 1, 2, 3], [0, 1, 3, 2]))
        return [(diagram, (0,))]
    else:
        return [(None, None)]



# dimer diagrams

# prqs,pq,rs-> :  V_0101  ca_0  ca_1
def v0101(densities, integrals, subsystem, charges):
    X = _parameters(densities, integrals, subsystem, charges, permutation=(0,1))
    if X.Dchg_0==0 and X.Dchg_1==0:
        prefactor = 4
        def diagram(i0,i1,j0,j1):
            partial =          tendot(X.V_0101, X.ca_0[i0][j0], axes=([0, 2], [0, 1]))
            return prefactor * tendot(partial,  X.ca_1[i1][j1], axes=([0, 1], [0, 1]))
        return [(diagram, (0,1))]
    else:
        return [(None, None)]

# pqsr,pqr,s-> :  V_0010  cca_0  a_1
def v0010(densities, integrals, subsystem, charges):
    result01 = _v0010(  densities, integrals, subsystem, charges, permutation=(0,1))
    result10 = _v0010(  densities, integrals, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _v0010(densities, integrals, subsystem, charges, permutation):
    X = _parameters(densities, integrals, subsystem, charges, permutation)
    if X.Dchg_0==-1 and X.Dchg_1==+1:
        prefactor = 2 * (-1)**(X.n_i1 + X.P)
        def diagram(i0,i1,j0,j1):
            partial =          tendot(X.V_0010, X.a_1[i1][j1],   axes=([2], [0]))
            return prefactor * tendot(partial,  X.cca_0[i0][j0], axes=([0, 1, 2], [0, 1, 2]))
        return diagram, permutation
    else:
        return None, None

# psrq,pqr,s-> :  V_0100  caa_0  c_1
def v0100(densities, integrals, subsystem, charges):
    result01 = _v0100(  densities, integrals, subsystem, charges, permutation=(0,1))
    result10 = _v0100(  densities, integrals, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _v0100(densities, integrals, subsystem, charges, permutation):
    X = _parameters(densities, integrals, subsystem, charges, permutation)
    if X.Dchg_0==+1 and X.Dchg_1==-1:
        prefactor = 2 * (-1)**(X.n_i1 + X.P)
        def diagram(i0,i1,j0,j1):
            partial =          tendot(X.V_0100, X.c_1[i1][j1],   axes=([1], [0]))
            return prefactor * tendot(partial,  X.caa_0[i0][j0], axes=([0, 1, 2], [0, 2, 1]))
        return diagram, permutation
    else:
        return None, None

# pqsr,pq,rs-> :  V_0011  cc_0  aa_1
def v0011(densities, integrals, subsystem, charges):
    result01 = _v0011(  densities, integrals, subsystem, charges, permutation=(0,1))
    result10 = _v0011(  densities, integrals, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _v0011(densities, integrals, subsystem, charges, permutation):
    X = _parameters(densities, integrals, subsystem, charges, permutation)
    if X.Dchg_0==-2 and X.Dchg_1==+2:
        prefactor = 1
        def diagram(i0,i1,j0,j1):
            partial =          tendot(X.V_0011, X.cc_0[i0][j0], axes=([0, 1], [0, 1]))
            return prefactor * tendot(partial,  X.aa_1[i1][j1], axes=([0, 1], [1, 0]))
        return diagram, permutation
    else:
        return None, None

# ij,pqsr,pqjr,is-> :  S_10  V_0010  ccaa_0  ca_1
def s10v0010(densities, integrals, subsystem, charges):
    result01 = _s10v0010(densities, integrals, subsystem, charges, permutation=(0,1))
    result10 = _s10v0010(densities, integrals, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s10v0010(densities, integrals, subsystem, charges, permutation):
    X = _parameters(densities, integrals, subsystem, charges, permutation)
    if X.Dchg_0==0 and X.Dchg_1==0:
        prefactor = 2
        def diagram(i0,i1,j0,j1):
            partial =          tendot(X.S_10,            X.ca_1[i1][j1], axes=([0], [0]))
            partial =          tendot(X.ccaa_0[i0][j0],  partial,        axes=([2], [0]))
            return prefactor * tendot(X.V_0010,          partial,        axes=([0, 1, 2, 3], [0, 1, 3, 2]))
        return diagram, permutation
    else:
        return None, None

# ij,psrq,pirq,sj-> :  S_01  V_0100  ccaa_0  ca_1
def s01v0100(densities, integrals, subsystem, charges):
    result01 = _s01v0100(densities, integrals, subsystem, charges, permutation=(0,1))
    result10 = _s01v0100(densities, integrals, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01v0100(densities, integrals, subsystem, charges, permutation):
    X = _parameters(densities, integrals, subsystem, charges, permutation)
    if X.Dchg_0==0 and X.Dchg_1==0:
        prefactor = 2
        def diagram(i0,i1,j0,j1):
            partial =          tendot(X.S_01,           X.ca_1[i1][j1], axes=([1], [1]))
            partial =          tendot(X.ccaa_0[i0][j0], partial,        axes=([1], [0]))
            return prefactor * tendot(X.V_0100,         partial,        axes=([0, 2, 3, 1], [0, 1, 2, 3]))
        return diagram, permutation
    else:
        return None, None

# ij,prqs,piq,rjs-> :  S_01  V_0101  cca_0  caa_1
def s01v0101(densities, integrals, subsystem, charges):
    result01 = _s01v0101(densities, integrals, subsystem, charges, permutation=(0,1))
    result10 = _s01v0101(densities, integrals, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01v0101(densities, integrals, subsystem, charges, permutation):
    X = _parameters(densities, integrals, subsystem, charges, permutation)
    if X.Dchg_0==-1 and X.Dchg_1==+1:
        prefactor = 4 * (-1)**(X.n_i1 + X.P)
        def diagram(i0,i1,j0,j1):
            partial =          tendot(X.V_0101,        X.cca_0[i0][j0], axes=([0, 2], [0, 2]))
            partial =          tendot(X.caa_1[i1][j1], partial,         axes=([0, 2], [0, 1]))
            return prefactor * tendot(X.S_01,          partial,         axes=([0, 1], [1, 0]))
        return diagram, permutation
    else:
        return None, None

# ij,pqsr,qpj,irs-> :  S_10  V_0011  cca_0  caa_1
def s10v0011(densities, integrals, subsystem, charges):
    result01 = _s10v0011(densities, integrals, subsystem, charges, permutation=(0,1))
    result10 = _s10v0011(densities, integrals, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s10v0011(densities, integrals, subsystem, charges, permutation):
    X = _parameters(densities, integrals, subsystem, charges, permutation)
    if X.Dchg_0==-1 and X.Dchg_1==+1:
        prefactor = (-1)**(X.n_i1 + X.P)
        def diagram(i0,i1,j0,j1):
            partial =          tendot(X.V_0011,        X.cca_0[i0][j0], axes=([0, 1], [1, 0]))
            partial =          tendot(X.caa_1[i1][j1], partial,         axes=([1, 2], [1, 0]))
            return prefactor * tendot(X.S_10,          partial,         axes=([0, 1], [0, 1]))
        return diagram, permutation
    else:
        return None, None

# ij,pqrs,pqisr,j-> :  S_01  V_0000  cccaa_0  a_1
def s01v0000(densities, integrals, subsystem, charges):
    result01 = _s01v0000(densities, integrals, subsystem, charges, permutation=(0,1))
    result10 = _s01v0000(densities, integrals, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01v0000(densities, integrals, subsystem, charges, permutation):
    X = _parameters(densities, integrals, subsystem, charges, permutation)
    if X.Dchg_0==-1 and X.Dchg_1==+1:
        prefactor = (-1)**(X.n_i1 + X.P)
        def diagram(i0,i1,j0,j1):
            partial =          tendot(X.S_01,            X.a_1[i1][j1], axes=([1], [0]))
            partial =          tendot(X.cccaa_0[i0][j0], partial,       axes=([2], [0]))
            return prefactor * tendot(X.V_0000,          partial,       axes=([0, 1, 2, 3], [0, 1, 3, 2]))
        return diagram, permutation
    else:
        return None, None

# ij,pqrs,pqjrs,i-> :  S_10  V_0000  ccaaa_0  c_1
def s10v0000(densities, integrals, subsystem, charges):
    result01 = _s10v0000(densities, integrals, subsystem, charges, permutation=(0,1))
    result10 = _s10v0000(densities, integrals, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s10v0000(densities, integrals, subsystem, charges, permutation):
    X = _parameters(densities, integrals, subsystem, charges, permutation)
    if X.Dchg_0==+1 and X.Dchg_1==-1:
        prefactor = (-1)**(X.n_i1 + X.P)
        def diagram(i0,i1,j0,j1):
            partial =          tendot(X.S_10,            X.c_1[i1][j1], axes=([0], [0]))
            partial =          tendot(X.ccaaa_0[i0][j0], partial,       axes=([2], [0]))
            return prefactor * tendot(X.V_0000,          partial,       axes=([0, 1, 2, 3], [0, 1, 2, 3]))
        return diagram, permutation
    else:
        return None, None

# ij,pqsr,qpir,js-> :  S_01  V_0010  ccca_0  aa_1
def s01v0010(densities, integrals, subsystem, charges):
    result01 = _s01v0010(densities, integrals, subsystem, charges, permutation=(0,1))
    result10 = _s01v0010(densities, integrals, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01v0010(densities, integrals, subsystem, charges, permutation):
    X = _parameters(densities, integrals, subsystem, charges, permutation)
    if X.Dchg_0==-2 and X.Dchg_1==+2:
        prefactor = 2
        def diagram(i0,i1,j0,j1):
            partial =          tendot(X.S_01,           X.aa_1[i1][j1], axes=([1], [0]))
            partial =          tendot(X.ccca_0[i0][j0], partial,        axes=([2], [0]))
            return prefactor * tendot(X.V_0010,         partial,        axes=([0, 1, 2, 3], [0, 1, 2, 3]))
        return diagram, permutation
    else:
        return None, None

# ij,psrq,pjqr,si-> :  S_10  V_0100  caaa_0  cc_1
def s10v0100(densities, integrals, subsystem, charges):
    result01 = _s10v0100(densities, integrals, subsystem, charges, permutation=(0,1))
    result10 = _s10v0100(densities, integrals, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s10v0100(densities, integrals, subsystem, charges, permutation):
    X = _parameters(densities, integrals, subsystem, charges, permutation)
    if X.Dchg_0==+2 and X.Dchg_1==-2:
        prefactor = 2
        def diagram(i0,i1,j0,j1):
            partial =          tendot(X.S_10,           X.cc_1[i1][j1], axes=([0], [1]))
            partial =          tendot(X.caaa_0[i0][j0], partial,        axes=([1], [0]))
            return prefactor * tendot(X.V_0100,         partial,        axes=([0, 1, 2, 3], [0, 1, 3, 2]))
        return diagram, permutation
    else:
        return None, None

# ij,pqsr,pqi,jrs-> :  S_01  V_0011  ccc_0  aaa_1
def s01v0011(densities, integrals, subsystem, charges):
    result01 = _s01v0011(densities, integrals, subsystem, charges, permutation=(0,1))
    result10 = _s01v0011(densities, integrals, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01v0011(densities, integrals, subsystem, charges, permutation):
    X = _parameters(densities, integrals, subsystem, charges, permutation)
    if X.Dchg_0==-3 and X.Dchg_1==+3:
        prefactor = (-1)**(X.n_i1 + X.P)
        def diagram(i0,i1,j0,j1):
            partial =          tendot(X.V_0011,        X.ccc_0[i0][j0], axes=([0, 1], [0, 1]))
            partial =          tendot(X.aaa_1[i1][j1], partial,         axes=([1, 2], [1, 0]))
            return prefactor * tendot(X.S_01,          partial,         axes=([0, 1], [1, 0]))
        return diagram, permutation
    else:
        return None, None



##########
# A dictionary catalog.  the string association lets users specify active diagrams at the top level.
##########

catalog = {}

catalog[1] = {
    "v0000": v0000
}

catalog[2] = {
    "v0101":    v0101,
    "v0010":    v0010,
    "v0100":    v0100,
    "v0011":    v0011,
    "s10v0010": s10v0010,
    "s01v0100": s01v0100,
    "s01v0101": s01v0101,
    "s10v0011": s10v0011,
    "s10v0000": s10v0000,
    "s01v0000": s01v0000,
    "s01v0010": s01v0010,
    "s10v0100": s10v0100,
    "s01v0011": s01v0011
}
