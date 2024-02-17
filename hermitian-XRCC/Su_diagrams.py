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
from Sx_precontract import precontract

p, q, r, s, t, u, v, w = "pqrstuvw"

class _empty(object):  pass    # Basically just a dictionary



def _parameters(supersys_info, subsystem, charges, permutation=(0,)):
    # helper function to do repetitive manipulations of data passed from above
    # Version in Sv_diagrams.py has comments (and an updated internal structure that should eventually be replicated here)
    densities, integrals = supersys_info.densities, supersys_info.integrals
    integrals = integrals.S, integrals.U
    #
    densities = [densities[m] for m in subsystem]
    S, U = integrals
    S = {(m0_,m1_):S[m0,m1] for m1_,m1 in enumerate(subsystem) for m0_,m0 in enumerate(subsystem)}
    U = {(m0_,m1_,m2_):U[m0,m1,m2] for m2_,m2 in enumerate(subsystem) for m1_,m1 in enumerate(subsystem) for m0_,m0 in enumerate(subsystem)}
    #
    data = _empty()
    data.P = 0 if permutation==(0,1) else 1    # This line of code is specific to two fragments (needs to be generalized for >=3).
    #
    Dchg_rhos = {+2:["aa", "caaa"], +1:["a","caa"], 0:["ca","ccaa"], -1:["c","cca"], -2:["cc", "ccca"]}
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
                m2_01_str = str(m2) + "_" + m01_str
                data.__dict__["U_"+m2_01_str] = U[m2_,m0_,m1_]
    return data



##########
# Here are the implementations of the actual diagrams.
# They must take the arguments (supersys_info, subsystem, charges), but after that, it is up to you.
# It should return a list of kernels that takes state indices (for specified fragment charges) their relevant permutations.
# Don't forget to update the "catalog" dictionary at the end.
##########

# monomer diagram

# pq,pq-> :  ca0  u0_00
def u000(supersys_info, subsystem, charges):
    X = _parameters(supersys_info, subsystem, charges)
    prefactor = 1
    def diagram(i0,j0):
        return scalar_value( prefactor * X.ca_0[i0,j0](p,q) @ X.U_0_00(p,q) )
    if X.Dchg_0==0:
        return [(diagram, (0,))]
    else:
        return [(None, None)]



# dimer diagrams

# pq,pq-> :  ca0  u1_00
def u100(supersys_info, subsystem, charges):
    result01 = _u100(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _u100(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _u100(supersys_info, subsystem, charges, permutation):
    X = _parameters(supersys_info, subsystem, charges, permutation)
    prefactor = 1
    def diagram(i0,i1,j0,j1):
        if i1==j1:  return scalar_value( prefactor * X.ca_0[i0,j0](p,q) @ X.U_1_00(p,q) )
        else:       return 0
    if X.Dchg_0==0 and X.Dchg_1==0:
        return diagram, permutation
    else:
        return None, None

# p,q,pq-> :  c0  a1  u0_01
def u001(supersys_info, subsystem, charges):
    result01 = _u001(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _u001(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _u001(supersys_info, subsystem, charges, permutation):
    X = _parameters(supersys_info, subsystem, charges, permutation)
    prefactor = (-1)**(X.n_i1 + X.P)
    def diagram(i0,i1,j0,j1):
        return scalar_value( prefactor * X.c_0[i0,j0](p) @ X.a_1[i1,j1](q) @ X.U_0_01(p,q) )
    if X.Dchg_0==-1 and X.Dchg_1==+1:
        return diagram, permutation
    else:
        return None, None

# p,q,pq-> :  c_0  a_1  u1_01
def u101(supersys_info, subsystem, charges):
    result01 = _u101(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _u101(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _u101(supersys_info, subsystem, charges, permutation):
    X = _parameters(supersys_info, subsystem, charges, permutation)
    prefactor = (-1)**(X.n_i1 + X.P)
    def diagram(i0,i1,j0,j1):
        return scalar_value( prefactor * X.c_0[i0,j0](p) @ X.a_1[i1,j1](q) @ X.U_1_01(p,q) )
    if X.Dchg_0==-1 and X.Dchg_1==+1:
        return diagram, permutation
    else:
        return None, None

# tq,pu,tu,pq-> :  ca0  ca1  s01  u0_10
def s01u010(supersys_info, subsystem, charges):
    result01 = _s01u010(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _s01u010(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01u010(supersys_info, subsystem, charges, permutation):
    X = _parameters(supersys_info, subsystem, charges, permutation)
    prefactor = -1
    def diagram(i0,i1,j0,j1):
        return scalar_value( prefactor * X.ca_0[i0,j0](t,q) @ X.ca_1[i1,j1](p,u) @ X.S_01(t,u) @ X.U_0_10(p,q) )
    if X.Dchg_0==0 and X.Dchg_1==0:
        return diagram, permutation
    else:
        return None, None

# tq,pu,tu,pq-> :  ca0  ca1  s01  u1_10
def s01u110(supersys_info, subsystem, charges):
    result01 = _s01u110(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _s01u110(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01u110(supersys_info, subsystem, charges, permutation):
    X = _parameters(supersys_info, subsystem, charges, permutation)
    prefactor = -1
    def diagram(i0,i1,j0,j1):
        return scalar_value( prefactor * X.ca_0[i0,j0](t,q) @ X.ca_1[i1,j1](p,u) @ X.S_01(t,u) @ X.U_1_10(p,q) )
    if X.Dchg_0==0 and X.Dchg_1==0:
        return diagram, permutation
    else:
        return None, None

# ptq,u,tu,pq-> :  cca0  a1  s01  u0_00
def s01u000(supersys_info, subsystem, charges):
    result01 = _s01u000(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _s01u000(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01u000(supersys_info, subsystem, charges, permutation):
    X = _parameters(supersys_info, subsystem, charges, permutation)
    prefactor = (-1)**(X.n_i1 + X.P + 1)
    def diagram(i0,i1,j0,j1):
        return scalar_value( prefactor * X.cca_0[i0,j0](p,t,q) @ X.a_1[i1,j1](u) @ X.S_01(t,u) @ X.U_0_00(p,q) )
    if X.Dchg_0==-1 and X.Dchg_1==+1:
        return diagram, permutation
    else:
        return None, None

# ptq,u,tu,pq-> :  cca0  a1  s01  u1_00
def s01u100(supersys_info, subsystem, charges):
    result01 = _s01u100(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _s01u100(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01u100(supersys_info, subsystem, charges, permutation):
    X = _parameters(supersys_info, subsystem, charges, permutation)
    prefactor = (-1)**(X.n_i1 + X.P + 1)
    def diagram(i0,i1,j0,j1):
        return scalar_value( prefactor * X.cca_0[i0,j0](p,t,q) @ X.a_1[i1,j1](u) @ X.S_01(t,u) @ X.U_1_00(p,q) )
    if X.Dchg_0==-1 and X.Dchg_1==+1:
        return diagram, permutation
    else:
        return None, None

# t,puq,tu,pq-> :  c0  caa1  s01  u0_11
def s01u011(supersys_info, subsystem, charges):
    result01 = _s01u011(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _s01u011(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01u011(supersys_info, subsystem, charges, permutation):
    X = _parameters(supersys_info, subsystem, charges, permutation)
    prefactor = (-1)**(X.n_i1 + X.P + 1)
    def diagram(i0,i1,j0,j1):
        return scalar_value( prefactor * X.c_0[i0,j0](t) @ X.caa_1[i1,j1](p,u,q) @ X.S_01(t,u) @ X.U_0_11(p,q) )
    if X.Dchg_0==-1 and X.Dchg_1==+1:
        return diagram, permutation
    else:
        return None, None

# t,puq,tu,pq-> :  c0  caa1  s01  u1_11
def s01u111(supersys_info, subsystem, charges):
    result01 = _s01u111(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _s01u111(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01u111(supersys_info, subsystem, charges, permutation):
    X = _parameters(supersys_info, subsystem, charges, permutation)
    prefactor = (-1)**(X.n_i1 + X.P + 1)
    def diagram(i0,i1,j0,j1):
        return scalar_value( prefactor * X.c_0[i0,j0](t) @ X.caa_1[i1,j1](p,u,q) @ X.S_01(t,u) @ X.U_1_11(p,q) )
    if X.Dchg_0==-1 and X.Dchg_1==+1:
        return diagram, permutation
    else:
        return None, None

# pt,uq,tu,pq-> :  cc_0  aa_1  s01  u0_01
def s01u001(supersys_info, subsystem, charges):
    result01 = _s01u001(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _s01u001(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01u001(supersys_info, subsystem, charges, permutation):
    X = _parameters(supersys_info, subsystem, charges, permutation)
    prefactor = 1
    def diagram(i0,i1,j0,j1):
        return scalar_value( prefactor * X.cc_0[i0,j0](p,t) @ X.aa_1[i1,j1](u,q) @ X.S_01(t,u) @ X.U_0_01(p,q) )
    if X.Dchg_0==-2 and X.Dchg_1==+2:
        return diagram, permutation
    else:
        return None, None

# pt,uq,tu,pq-> :  cc0  aa1  s01  u1_01
def s01u101(supersys_info, subsystem, charges):
    result01 = _s01u101(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _s01u101(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01u101(supersys_info, subsystem, charges, permutation):
    X = _parameters(supersys_info, subsystem, charges, permutation)
    prefactor = 1
    def diagram(i0,i1,j0,j1):
        return scalar_value( prefactor * X.cc_0[i0,j0](p,t) @ X.aa_1[i1,j1](u,q) @ X.S_01(t,u) @ X.U_1_01(p,q) )
    if X.Dchg_0==-2 and X.Dchg_1==+2:
        return diagram, permutation
    else:
        return None, None



##########
# A dictionary catalog.  the string association lets users specify active diagrams at the top level.
##########

catalog = {}

catalog[1] = {
    "u000":  u000
}

catalog[2] = {
    "u100":     u100,
    "u001":     u001,
    "u101":     u101,
    "s01u010":  s01u010,
    "s01u110":  s01u110,
    "s01u000":  s01u000,
    "s01u100":  s01u100,
    "s01u011":  s01u011,
    "s01u111":  s01u111,
    "s01u001":  s01u001,
    "s01u101":  s01u101
}
