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

from qode.math.tensornet import tl_tensor, scalar_value



# helper function to do repetitive manipulations of data passed from above
#    densities and integrals are the full arrays for the supersystem
#    subsystem is always in acending order, naming the fragment indices of interest here
#    charges gives the bra and ket charges (as a 2-tuple) for each such respective fragment in the subsystem
#    permutation gives a potential reordering of the fragments in the subsystem
def _parameters(densities, integrals, subsystem, charges, permutation=(0,)):

    class _empty(object):  pass    # Basically just a dictionary

    def _Dchg_rho(rho):
        for chg_index in rho:    # only need to test one of the dict keys, so return just breaks the loop
            try:
                chg_i,chg_j = chg_index    # this will work if and only if the entry is actually a density (as opposed to number of electrons)
            except:
                return None                # if it is not a density ...
            else:
                return chg_i-chg_j         # ... but if it is, this is what we want to know.

    # First cull full arrays of densities and integrals to those of the fragments in the subsystem
    # Their indexing here is now that 0 is the first fragment in the subsystem, and so forth
    densities = [densities[m] for m in subsystem]
    S, V = integrals
    S = {(m0_,m1_):S[m0,m1] for m0_,m0 in enumerate(subsystem) for m1_,m1 in enumerate(subsystem)}
    V = {(m0_,m1_,m2_,m3_):V[m0,m1,m2,m3] for m0_,m0 in enumerate(subsystem)
         for m1_,m1 in enumerate(subsystem) for m2_,m2 in enumerate(subsystem) for m3_,m3 in enumerate(subsystem)}
    # Initialize the "X" structure and set the permutation flag
    data = _empty()
    data.P = 0 if permutation==(0,1) else 1    # needs to be generalized for n>=3.
    # Some diagrams need to know the number of e- in the ket for the combined "latter" frags of the un(!)permuted subsystem
    n_i_label = ""    # explicitly label which are included in the "latter" frags (to store all possibilities)
    n_i = 0           # start at 0 before looping backwards over frags in outer loop below (order of that loop otherwise irrelevant)
    # Loop over all combinations of up to 4 fragments, reversal of first loop explained above
    for m0,m0_ in reversed(list(enumerate(permutation))):    # m0 is fragment index in permuted subsystem, m0_ is its index in unpermuted subsystem
        m0_str = str(m0)                            # first digit of the "_0010"-like subscript string for X.V_#### (etc)
        #
        chg_i_m0_, chg_j_m0_ = charges[m0_]         # bra and ket charge that has been requested for this fragment
        Dchg_m0_ = chg_i_m0_ - chg_j_m0_            # charge change of this fragment (bra relative to the ket)
        data.__dict__["Dchg_"+m0_str] = Dchg_m0_    # include the charge change for this fragment under its permuted label
        #
        chg_i_m0 , chg_j_m0  = charges[m0]          # bra and ket charge of m0-th frag (double duty for m0, looping backwards over all frags in physical order)
        n_i_label = m0_str + n_i_label              # incrementally build the label for electron count of (variable) "latter" frags
        n_i += densities[m0]['n_elec'][chg_i_m0]    # number of electrons in "latter" frags (so far) in physical (unpermuted) order
        data.__dict__["n_i"+n_i_label] = n_i%2      # include only the parity of the number of electrons in the "latter" frags of the un(!)permuted subsystem
        #
        for rho in densities[m0_]:                  # include all fragment density tensors that match the charge change for this fragment
            if _Dchg_rho(densities[m0_][rho])==Dchg_m0_:
                data.__dict__[rho+"_"+m0_str] = densities[m0_][rho][chg_i_m0_,chg_j_m0_]
        # Continue loops over combinations of up to 4 fragments, building labels and storing quantities under permuted lables
        for m1,m1_ in enumerate(permutation):                # m1 is fragment index in permuted subsystem, m1_ is its index in unpermuted subsystem
            m01_str = m0_str + str(m1)
            data.__dict__["S_"+m01_str] = S[m0_,m1_]
            for m2,m2_ in enumerate(permutation):            # m2 is fragment index in permuted subsystem, m2_ is its index in unpermuted subsystem
                m012_str  = m01_str + str(m2)
                for m3,m3_ in enumerate(permutation):        # m3 is fragment index in permuted subsystem, m3_ is its index in unpermuted subsystem
                    m0123_str = m012_str + str(m3)
                    data.__dict__["V_"+m0123_str] = V[m0_,m1_,m2_,m3_]
    return data



##########
# Here are the implementations of the actual diagrams.
# They must take the arguments (densities, integrals, subsystem, charges), but after that, it is up to you.
# It should return a list of kernels that takes state indices (for specified fragment charges) their relevant permutations.
# Don't forget to update the "catalog" dictionary at the end.
##########

p, q, r, s, t, u, v, w = "pqrstuvw"    # some contraction indices for easier reading

# monomer diagram

# pqrs,pqsr-> :  V_0000  ccaa_0
def v0000(densities, integrals, subsystem, charges):
    X = _parameters(densities, integrals, subsystem, charges)
    prefactor = 1
    def diagram(i0,j0):
        return scalar_value( prefactor * X.V_0000(p,q,r,s) @ X.ccaa_0[i0,j0](p,q,s,r) )
        #return scalar_value( prefactor * X.V_0[i0,j0] )
    if X.Dchg_0==0:
        return [(diagram, (0,))]
    else:
        return [(None, None)]



# dimer diagrams

# prqs,pq,rs-> :  V_0101  ca_0  ca_1
def v0101(densities, integrals, subsystem, charges):
    X = _parameters(densities, integrals, subsystem, charges, permutation=(0,1))
    prefactor = 4
    def diagram(i0,i1,j0,j1):
        return scalar_value( prefactor * X.V_0101(p,q,r,s) @ X.ca_0[i0,j0](p,r) @ X.ca_1[i1,j1](q,s) )
    if X.Dchg_0==0 and X.Dchg_1==0:
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
    prefactor = 2 * (-1)**(X.n_i1 + X.P)
    def diagram(i0,i1,j0,j1):
        return scalar_value( prefactor * X.V_0010(p,q,r,s) @ X.a_1[i1,j1](r) @ X.cca_0[i0,j0](p,q,s) )
    if X.Dchg_0==-1 and X.Dchg_1==+1:
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
    prefactor = 2 * (-1)**(X.n_i1 + X.P)
    def diagram(i0,i1,j0,j1):
        return scalar_value( prefactor * X.V_0100(p,q,r,s) @ X.c_1[i1,j1](q) @ X.caa_0[i0,j0](p,s,r) )
    if X.Dchg_0==+1 and X.Dchg_1==-1:
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
    prefactor = 1
    def diagram(i0,i1,j0,j1):
        return scalar_value( prefactor * X.V_0011(p,q,r,s) @ X.cc_0[i0,j0](p,q) @ X.aa_1[i1,j1](s,r) )
    if X.Dchg_0==-2 and X.Dchg_1==+2:
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
    prefactor = 2
    def diagram(i0,i1,j0,j1):
        return scalar_value( prefactor * X.V_0010(s,t,r,u) @ X.ccaa_0[i0,j0](s,t,q,u) @ X.S_10(p,q) @ X.ca_1[i1,j1](p,r) )
    if X.Dchg_0==0 and X.Dchg_1==0:
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
    prefactor = 2
    def diagram(i0,i1,j0,j1):
        return scalar_value( prefactor * X.V_0100(s,r,t,u) @ X.ccaa_0[i0,j0](s,p,t,u) @ X.S_01(p,q) @ X.ca_1[i1,j1](r,q) )
    if X.Dchg_0==0 and X.Dchg_1==0:
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
    prefactor = 4 * (-1)**(X.n_i1 + X.P)
    def diagram(i0,i1,j0,j1):
        return scalar_value( prefactor * X.S_01(t,u) @ X.caa_1[i1,j1](q,u,s) @ X.V_0101(p,q,r,s) @ X.cca_0[i0,j0](p,t,r) )
    if X.Dchg_0==-1 and X.Dchg_1==+1:
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
    prefactor = (-1)**(X.n_i1 + X.P)
    def diagram(i0,i1,j0,j1):
        return scalar_value( prefactor * X.S_10(u,t) @ X.caa_1[i1,j1](u,s,r) @ X.V_0011(p,q,r,s) @ X.cca_0[i0,j0](q,p,t) )
    if X.Dchg_0==-1 and X.Dchg_1==+1:
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
    prefactor = (-1)**(X.n_i1 + X.P)
    def diagram(i0,i1,j0,j1):
        #return scalar_value( prefactor * X.cV_0[i0,j0](p) @ X.S_01(p,q) @ X.a_1[i1,j1](q) )
        return scalar_value( prefactor * X.V_0000(r,s,u,t) @ X.cccaa_0[i0,j0](r,s,p,t,u) @ X.S_01(p,q) @ X.a_1[i1,j1](q) )
    if X.Dchg_0==-1 and X.Dchg_1==+1:
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
    prefactor = (-1)**(X.n_i1 + X.P)
    def diagram(i0,i1,j0,j1):
        #return scalar_value( prefactor * X.Va_0[i0,j0](q) @ X.S_10(p,q) @ X.c_1[i1,j1](p) )
        return scalar_value( prefactor * X.V_0000(r,s,t,u) @ X.ccaaa_0[i0,j0](r,s,q,t,u) @ X.S_10(p,q) @ X.c_1[i1,j1](p) )
    if X.Dchg_0==+1 and X.Dchg_1==-1:
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
    prefactor = 2
    def diagram(i0,i1,j0,j1):
        return scalar_value( prefactor * X.V_0010(s,t,u,r) @ X.ccca_0[i0,j0](s,t,p,u) @ X.S_01(p,q) @ X.aa_1[i1,j1](q,r) )
    if X.Dchg_0==-2 and X.Dchg_1==+2:
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
    prefactor = 2
    def diagram(i0,i1,j0,j1):
        return scalar_value( prefactor * X.V_0100(s,t,r,u) @ X.caaa_0[i0,j0](s,q,t,u) @ X.S_10(p,q) @ X.cc_1[i1,j1](r,p) )
    if X.Dchg_0==+2 and X.Dchg_1==-2:
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
    prefactor = (-1)**(X.n_i1 + X.P)
    def diagram(i0,i1,j0,j1):
        return scalar_value( prefactor * X.S_01(t,u) @ X.aaa_1[i1,j1](u,s,r) @ X.V_0011(p,q,r,s) @ X.ccc_0[i0,j0](p,q,t) )
    if X.Dchg_0==-3 and X.Dchg_1==+3:
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
