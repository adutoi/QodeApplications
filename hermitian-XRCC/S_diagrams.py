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
from Sx_precontract import precontract

p, q, r, s, t, u, v, w = "pqrstuvw"

class _empty(object):  pass    # Basically just a dictionary



def _parameters(supersys_info, subsystem, charges, permutation=(0,)):
    # helper functions to do repetitive manipulations of data passed from above
    # Version in Sv_diagrams.py has comments (and an updated internal structure that should eventually be replicated here)
    # needs to be generalized (should not be hard) and have "2" removed from its name ... or maybe it is better this way
    densities, overlaps = supersys_info.densities, supersys_info.integrals
    #
    densities = [densities[m] for m in subsystem]
    overlaps = {(m0_,m1_):overlaps[m0,m1]    for m1_,m1 in enumerate(subsystem) for m0_,m0 in enumerate(subsystem)}
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
            data.__dict__["S_"+m01_str] = overlaps[m0_,m1_]
    return data



##########
# Here are the implementations of the actual diagrams.
# The public @staticmethods must take the arguments (supersys_info, subsystem, charges), but after that, it is up to you.
# It should return a list of kernels that takes state indices (for specified fragment charges) their relevant permutations.
# Don't forget to update the "catalog" dictionary at the end.
##########

# 0-mer diagram

# -> :
def identity(supersys_info, subsystem, charges):
    # Identity
    def diagram():
        return 1
    return [(diagram, [])]



# dimer diagrams

# p,q,pq-> :  c0  a1  s01
def s01(supersys_info, subsystem, charges):
    result01 = _s01(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _s01(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01(supersys_info, subsystem, charges, permutation):
    X = _parameters(supersys_info, subsystem, charges, permutation)
    prefactor = (-1)**(X.n_i1 + X.P)
    def diagram(i0,i1,j0,j1):
        return scalar_value( prefactor * X.c_0[i0,j0](p) @ X.a_1[i1,j1](q) @ X.S_01(p,q))
    if X.Dchg_0==-1 and X.Dchg_1==+1:
        return diagram, permutation
    else:
        return None, None

# ps,rq,pq,rs-> :  ca0  ca1  s01  s10
def s01s10(supersys_info, subsystem, charges):
    X = _parameters(supersys_info, subsystem, charges, permutation=(0,1))
    prefactor = -1
    def diagram(i0,i1,j0,j1):
        return scalar_value( prefactor * X.ca_0[i0,j0](p,s) @ X.ca_1[i1,j1](r,q) @ X.S_01(p,q) @ X.S_10(r,s))
    if X.Dchg_0==0 and X.Dchg_1==0:
        return [(diagram, (0,1))]
    else:
        return [(None, None)]

# pr,sq,pq,rs-> :  cc0  aa1  s01  s01
def s01s01(supersys_info, subsystem, charges):
    result01 = _s01s01(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _s01s01(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01s01(supersys_info, subsystem, charges, permutation):
    X = _parameters(supersys_info, subsystem, charges, permutation)
    prefactor = 1/2.
    def diagram(i0,i1,j0,j1):
        return scalar_value( prefactor * X.cc_0[i0,j0](p,r) @ X.aa_1[i1,j1](s,q) @ X.S_01(p,q) @ X.S_01(r,s))
    if X.Dchg_0==-2 and X.Dchg_1==+2:
        return diagram, permutation
    else:
        return None, None

# pru,tsq,pq,rs,tu-> :  cca0  caa1  s01  s01  s10
def s01s01s10(supersys_info, subsystem, charges):
    result01 = _s01s01s10(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _s01s01s10(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01s01s10(supersys_info, subsystem, charges, permutation):
    X = _parameters(supersys_info, subsystem, charges, permutation)
    prefactor = (-1)**(X.n_i1 + X.P + 1) / 2.
    def diagram(i0,i1,j0,j1):
        return scalar_value( prefactor * X.cca_0[i0,j0](p,r,u) @ X.caa_1[i1,j1](t,s,q) @ X.S_01(p,q) @ X.S_01(r,s) @ X.S_10(t,u))
    if X.Dchg_0==-1 and X.Dchg_1==+1:
        return diagram, permutation
    else:
        return None, None

# prwu,tvsq,pq,rs,tu,vw-> :  ccaa0  ccaa1  s01  s01  s10  s10
def s01s01s10s10(supersys_info, subsystem, charges):
    X = _parameters(supersys_info, subsystem, charges, permutation=(0,1))
    prefactor = 1/4.
    def diagram(i0,i1,j0,j1):
        return scalar_value( prefactor * X.ccaa_0[i0,j0](p,r,w,u) @ X.ccaa_1[i1,j1](t,v,s,q) @ X.S_01(p,q) @ X.S_01(r,s) @ X.S_10(t,u) @ X.S_10(v,w))
    if X.Dchg_0==0 and X.Dchg_1==0:
        return [(diagram, (0,1))]
    else:
        return [(None, None)]

# prtw,vusq,pq,rs,tu,vw-> :  ccca0  caaa1  s01  s01  s01  s10
def s01s01s01s10(supersys_info, subsystem, charges):
    result01 = _s01s01s01s10(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _s01s01s01s10(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01s01s01s10(supersys_info, subsystem, charges, permutation):
    X = _parameters(supersys_info, subsystem, charges, permutation)
    prefactor = -1 / 6.
    def diagram(i0,i1,j0,j1):
        return scalar_value( prefactor * X.ccca_0[i0,j0](p,r,t,w) @ X.caaa_1[i1,j1](v,u,s,q) @ X.S_01(p,q) @ X.S_01(r,s) @ X.S_01(t,u) @ X.S_10(v,w))
    if X.Dchg_0==-2 and X.Dchg_1==+2:
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

catalog[0] = {
    "identity": identity
}

catalog[2] = {
    "s01":          s01,
    "s01s10":       s01s10,
    "s01s01":       s01s01,
    "s01s01s10":    s01s01s10,
    "s01s01s10s10": s01s01s10s10,
    "s01s01s01s10": s01s01s01s10
}
