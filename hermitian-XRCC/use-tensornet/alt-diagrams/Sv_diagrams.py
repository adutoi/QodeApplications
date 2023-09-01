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
from qode.math.tensornet import tl_tensor, scalar_value

p, q, r, s, t, u, v, w = "pqrstuvw"

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
            V_0000 = tl_tensor(X.V_0000)
            ccaa_0 = tl_tensor(X.ccaa_0[i0][j0])
            return scalar_value( prefactor * V_0000(p,q,r,s) @ ccaa_0(p,q,s,r) )
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
            V_0101 = tl_tensor(X.V_0101)
            ca_0   = tl_tensor(X.ca_0[i0][j0])
            ca_1   = tl_tensor(X.ca_1[i1][j1])
            return scalar_value( prefactor * V_0101(p,q,r,s) @ ca_0(p,r) @ ca_1(q,s) )
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
            V_0010 = tl_tensor(X.V_0010)
            a_1    = tl_tensor(X.a_1[i1][j1])
            cca_0  = tl_tensor(X.cca_0[i0][j0])
            return scalar_value( prefactor * V_0010(p,q,r,s) @ a_1(r) @ cca_0(p,q,s) )
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
            V_0100 = tl_tensor(X.V_0100)
            c_1    = tl_tensor(X.c_1[i1][j1])
            caa_0  = tl_tensor(X.caa_0[i0][j0])
            return scalar_value( prefactor * V_0100(p,q,r,s) @ c_1(q) @ caa_0(p,s,r) )
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
            V_0011 = tl_tensor(X.V_0011)
            cc_0   = tl_tensor(X.cc_0[i0][j0])
            aa_1   = tl_tensor(X.aa_1[i1][j1])
            return scalar_value( prefactor * V_0011(p,q,r,s) @ cc_0(p,q) @ aa_1(s,r) )
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
            S_10   = tl_tensor(X.S_10)
            V_0010 = tl_tensor(X.V_0010)
            ca_1   = tl_tensor(X.ca_1[i1][j1])
            ccaa_0 = tl_tensor(X.ccaa_0[i0][j0])
            return scalar_value( prefactor * V_0010(s,t,r,u) @ ccaa_0(s,t,q,u) @ S_10(p,q) @ ca_1(p,r) )
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
            S_01   = tl_tensor(X.S_01)
            V_0100 = tl_tensor(X.V_0100)
            ca_1   = tl_tensor(X.ca_1[i1][j1])
            ccaa_0 = tl_tensor(X.ccaa_0[i0][j0])
            return scalar_value( prefactor * V_0100(s,r,t,u) @ ccaa_0(s,p,t,u) @ S_01(p,q) @ ca_1(r,q) )
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
            S_01   = tl_tensor(X.S_01)
            V_0101 = tl_tensor(X.V_0101)
            cca_0  = tl_tensor(X.cca_0[i0][j0])
            caa_1  = tl_tensor(X.caa_1[i1][j1])
            return scalar_value( prefactor * S_01(t,u) @ caa_1(q,u,s) @ V_0101(p,q,r,s) @ cca_0(p,t,r) )
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
            S_10   = tl_tensor(X.S_10)
            V_0011 = tl_tensor(X.V_0011)
            cca_0  = tl_tensor(X.cca_0[i0][j0])
            caa_1  = tl_tensor(X.caa_1[i1][j1])
            return scalar_value( prefactor * S_10(u,t) @ caa_1(u,s,r) @ V_0011(p,q,r,s) @ cca_0(q,p,t) )
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
            S_01    = tl_tensor(X.S_01)
            V_0000  = tl_tensor(X.V_0000)
            a_1     = tl_tensor(X.a_1[i1][j1])
            cccaa_0 = tl_tensor(X.cccaa_0[i0][j0])
            return scalar_value( prefactor * V_0000(r,s,u,t) @ cccaa_0(r,s,p,t,u) @ S_01(p,q) @ a_1(q) )
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
            S_10    = tl_tensor(X.S_10)
            V_0000  = tl_tensor(X.V_0000)
            c_1     = tl_tensor(X.c_1[i1][j1])
            ccaaa_0 = tl_tensor(X.ccaaa_0[i0][j0])
            return scalar_value( prefactor * V_0000(r,s,t,u) @ ccaaa_0(r,s,q,t,u) @ S_10(p,q) @ c_1(p) )
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
            S_01   = tl_tensor(X.S_01)
            V_0010 = tl_tensor(X.V_0010)
            aa_1   = tl_tensor(X.aa_1[i1][j1])
            ccca_0 = tl_tensor(X.ccca_0[i0][j0])
            return scalar_value( prefactor * V_0010(s,t,u,r) @ ccca_0(s,t,p,u) @ S_01(p,q) @ aa_1(q,r) )
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
            S_10   = tl_tensor(X.S_10)
            V_0100 = tl_tensor(X.V_0100)
            cc_1   = tl_tensor(X.cc_1[i1][j1])
            caaa_0 = tl_tensor(X.caaa_0[i0][j0])
            return scalar_value( prefactor * V_0100(s,t,r,u) @ caaa_0(s,q,t,u) @ S_10(p,q) @ cc_1(r,p) )
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
            S_01   = tl_tensor(X.S_01)
            V_0011 = tl_tensor(X.V_0011)
            ccc_0  = tl_tensor(X.ccc_0[i0][j0])
            aaa_1  = tl_tensor(X.aaa_1[i1][j1])
            return scalar_value( prefactor * S_01(t,u) @ aaa_1(u,s,r) @ V_0011(p,q,r,s) @ ccc_0(p,q,t) )
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
