#    (C) Copyright 2023, 2024 Anthony D. Dutoi and Marco Bauer
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
import time
from qode.math.tensornet import scalar_value
from frag_resolve import frag_resolve

p, q, r, s, t, u, v, w = "pqrstuvw"    # some contraction indices for easier reading



##########
# Here are the implementations of the actual diagrams.
# They must take the arguments (supersys_info, subsystem, charges), but after that, it is up to you.
# It should return a list of kernels that takes state indices (for specified fragment charges) their relevant permutations.
# Don't forget to update the "catalog" dictionary at the end.
##########


# monomer diagram

# pqsr,pqrs-> :  ccaa0  v0000
def v0000(supersys_info, subsystem, charges):
    if "v0000" not in supersys_info.timings:  supersys_info.timings["v0000"] = 0.
    X = frag_resolve(supersys_info, zip(subsystem, charges))
    prefactor = 1
    def diagram(i0,j0):
        t0 = time.time()
        result = scalar_value( prefactor * X.ccaa0pqsr_Vpqrs[i0,j0] )
        #result = scalar_value( prefactor * X.ccaa0[i0,j0](p,q,s,r) @ X.v0000(p,q,r,s) )
        supersys_info.timings["v0000"] += (time.time() - t0)
        return result
    if X.Dchg0==0:
        return [(diagram, (0,))]
    else:
        return [(None, None)]



# dimer diagrams

# pr,qs,pqrs-> :  ca0  ca1  v0101
def v0101(supersys_info, subsystem, charges):
    if "v0101" not in supersys_info.timings:  supersys_info.timings["v0101"] = 0.
    X = frag_resolve(supersys_info, zip(subsystem, charges), permutation=(0,1))
    prefactor = 4
    def diagram(i0,i1,j0,j1):
        t0 = time.time()
        result = scalar_value( prefactor * X.ca1[i1,j1](q,s) @ X.ca0pr_Vp1r1[i0,j0](q,s) )
        #result = scalar_value( prefactor * X.ca0[i0,j0](p,r) @ X.ca1[i1,j1](q,s) @ X.v0101(p,q,r,s) )
        supersys_info.timings["v0101"] += (time.time() - t0)
        return result
    if X.Dchg0==0 and X.Dchg1==0:
        return [(diagram, (0,1))]
    else:
        return [(None, None)]

# pqs,r,pqrs-> :  cca0  a1  v0010
def v0010(supersys_info, subsystem, charges):
    if "v0010" not in supersys_info.timings:  supersys_info.timings["v0010"] = 0.
    result01 = _v0010(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _v0010(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _v0010(supersys_info, subsystem, charges, permutation):
    X = frag_resolve(supersys_info, zip(subsystem, charges), permutation)
    prefactor = 2 * (-1)**(X.n_i1 + X.P)
    def diagram(i0,i1,j0,j1):
        t0 = time.time()
        result = scalar_value( prefactor * X.a1[i1,j1](r) @ X.cca0pqs_Vpq1s[i0,j0](r) )
        #result = scalar_value( prefactor * X.cca0[i0,j0](p,q,s) @ X.a1[i1,j1](r) @ X.v0010(p,q,r,s) )
        supersys_info.timings["v0010"] += (time.time() - t0)
        return result
    if X.Dchg0==-1 and X.Dchg1==+1:
        return diagram, permutation
    else:
        return None, None

# p,qsr,pqrs-> :  c0  caa1  v0111
def v0111(supersys_info, subsystem, charges):
    if "v0111" not in supersys_info.timings:  supersys_info.timings["v0111"] = 0.
    result01 = _v0111(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _v0111(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _v0111(supersys_info, subsystem, charges, permutation):
    X = frag_resolve(supersys_info, zip(subsystem, charges), permutation)
    prefactor = 2 * (-1)**(X.n_i1 + X.P)
    def diagram(i0,i1,j0,j1):
        t0 = time.time()
        result = scalar_value( prefactor * X.c0[i0,j0](p) @ X.caa1qsr_V0qrs[i1,j1](p) )
        #result = scalar_value( prefactor * X.c0[i0,j0](p) @ X.caa1[i1,j1](q,s,r) @ X.v0111(p,q,r,s) )
        supersys_info.timings["v0111"] += (time.time() - t0)
        return result
    if X.Dchg0==-1 and X.Dchg1==+1:
        return diagram, permutation
    else:
        return None, None

# pq,sr,pqrs-> :  cc0  aa1  v0011
def v0011(supersys_info, subsystem, charges):
    if "v0011" not in supersys_info.timings:  supersys_info.timings["v0011"] = 0.
    result01 = _v0011(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _v0011(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _v0011(supersys_info, subsystem, charges, permutation):
    X = frag_resolve(supersys_info, zip(subsystem, charges), permutation)
    prefactor = 1
    def diagram(i0,i1,j0,j1):
        t0 = time.time()
        result = scalar_value( prefactor * X.aa1[i1,j1](s,r) @ X.cc0pq_Vpq11[i0,j0](r,s) )
        #result = scalar_value( prefactor * X.cc0[i0,j0](p,q) @ X.aa1[i1,j1](s,r) @ X.v0011(p,q,r,s) )
        supersys_info.timings["v0011"] += (time.time() - t0)
        return result
    if X.Dchg0==-2 and X.Dchg1==+2:
        return diagram, permutation
    else:
        return None, None

# qtsr,pu,tu,pqrs-> :  ccaa0  ca1  s01  v1000
def s01v1000(supersys_info, subsystem, charges):
    if "s01v1000" not in supersys_info.timings:  supersys_info.timings["s01v1000"] = 0.
    result01 = _s01v1000(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _s01v1000(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01v1000(supersys_info, subsystem, charges, permutation):
    X = frag_resolve(supersys_info, zip(subsystem, charges), permutation)
    prefactor = 2
    def diagram(i0,i1,j0,j1):
        t0 = time.time()
        result = scalar_value( prefactor * X.ca1Xu_S0u[i1,j1](p,t) @ X.ccaa0qXsr_V1qrs[i0,j0](t,p) )
        #result = scalar_value( prefactor * X.ca1[i1,j1](p,u) @ X.s01(t,u) @ X.ccaa0qXsr_V1qrs[i0,j0](t,p) )
        #result = scalar_value( prefactor * X.ccaa0[i0,j0](q,t,s,r) @ X.ca1[i1,j1](p,u) @ X.s01(t,u) @ X.v1000(p,q,r,s) )
        supersys_info.timings["s01v1000"] += (time.time() - t0)
        return result
    if X.Dchg0==0 and X.Dchg1==0:
        return diagram, permutation
    else:
        return None, None

# tr,pqus,tu,pqrs-> :  ca0  ccaa1  s01  v1101
def s01v1101(supersys_info, subsystem, charges):
    if "s01v1101" not in supersys_info.timings:  supersys_info.timings["s01v1101"] = 0.
    result01 = _s01v1101(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _s01v1101(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01v1101(supersys_info, subsystem, charges, permutation):
    X = frag_resolve(supersys_info, zip(subsystem, charges), permutation)
    prefactor = 2
    def diagram(i0,i1,j0,j1):
        t0 = time.time()
        result = scalar_value( prefactor * X.ca0tX_St1[i0,j0](r,u) @ X.ccaa1pqXs_Vpq0s[i1,j1](u,r) )
        #result = scalar_value( prefactor * X.ca0[i0,j0](t,r) @ X.s01(t,u) @ X.ccaa1pqXs_Vpq0s[i1,j1](u,r) )
        #result = scalar_value( prefactor * X.ca0[i0,j0](t,r) @ X.ccaa1[i1,j1](p,q,u,s) @ X.s01(t,u) @ X.v1101(p,q,r,s) )
        supersys_info.timings["s01v1101"] += (time.time() - t0)
        return result
    if X.Dchg0==0 and X.Dchg1==0:
        return diagram, permutation
    else:
        return None, None

# pqtsr,u,tu,pqrs-> :  cccaa0  a1  s01  v0000
def s01v0000(supersys_info, subsystem, charges):
    if "s01v0000" not in supersys_info.timings:  supersys_info.timings["s01v0000"] = 0.
    result01 = _s01v0000(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _s01v0000(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01v0000(supersys_info, subsystem, charges, permutation):
    X = frag_resolve(supersys_info, zip(subsystem, charges), permutation)
    prefactor = (-1)**(X.n_i1 + X.P)
    def diagram(i0,i1,j0,j1):
        t0 = time.time()
        result = scalar_value( prefactor * X.a1u_S0u[i1,j1](t) @ X.cccaa0pqXsr_Vpqrs[i0,j0](t) )
        #result = scalar_value( prefactor * X.a1[i1,j1](u) @ X.s01(t,u) @ X.cccaa0pqXsr_Vpqrs[i0,j0](t) )
        #result = scalar_value( prefactor * X.cccaa0[i0,j0](p,q,t,s,r) @ X.a1[i1,j1](u) @ X.s01(t,u) @ X.v0000(p,q,r,s) )
        supersys_info.timings["s01v0000"] += (time.time() - t0)
        return result
    if X.Dchg0==-1 and X.Dchg1==+1:
        return diagram, permutation
    else:
        return None, None

# t,pqusr,tu,pqrs-> :  c0  ccaaa1  s01  v1111
def s01v1111(supersys_info, subsystem, charges):
    if "s01v1111" not in supersys_info.timings:  supersys_info.timings["s01v1111"] = 0.
    result01 = _s01v1111(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _s01v1111(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01v1111(supersys_info, subsystem, charges, permutation):
    X = frag_resolve(supersys_info, zip(subsystem, charges), permutation)
    prefactor = (-1)**(X.n_i1 + X.P)
    def diagram(i0,i1,j0,j1):
        t0 = time.time()
        result = scalar_value( prefactor * X.c0t_St1[i0,j0](u) @ X.ccaaa1pqXsr_Vpqrs[i1,j1](u) )
        #result = scalar_value( prefactor * X.c0[i0,j0](t) @ X.s01(t,u) @ X.ccaaa1pqXsr_Vpqrs[i1,j1](u) )
        #result = scalar_value( prefactor * X.c0[i0,j0](t) @ X.ccaaa1[i1,j1](p,q,u,s,r) @ X.s01(t,u) @ X.v1111(p,q,r,s) )
        supersys_info.timings["s01v1111"] += (time.time() - t0)
        return result
    if X.Dchg0==-1 and X.Dchg1==+1:
        return diagram, permutation
    else:
        return None, None

# ptr,qus,tu,pqrs-> :  cca0  caa1  s01  v0101
def s01v0101(supersys_info, subsystem, charges):
    if "s01v0101" not in supersys_info.timings:  supersys_info.timings["s01v0101"] = 0.
    result01 = _s01v0101(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _s01v0101(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01v0101(supersys_info, subsystem, charges, permutation):
    X = frag_resolve(supersys_info, zip(subsystem, charges), permutation)
    prefactor = 4 * (-1)**(X.n_i1 + X.P)
    def diagram(i0,i1,j0,j1):
        t0 = time.time()
        result = scalar_value( prefactor * X.caa1XuX_S0u[i1,j1](q,s,t) @ X.cca0pXr_Vp1r1[i0,j0](t,q,s) )
        #result = scalar_value( prefactor * X.caa1[i1,j1](q,u,s) @ X.s01(t,u) @ X.cca0pXr_Vp1r1[i0,j0](t,q,s) )
        #result = scalar_value( prefactor * X.cca0[i0,j0](p,t,r) @ X.caa1[i1,j1](q,u,s) @ X.s01(t,u) @ X.v0101(p,q,r,s) )
        supersys_info.timings["s01v0101"] += (time.time() - t0)
        return result
    if X.Dchg0==-1 and X.Dchg1==+1:
        return diagram, permutation
    else:
        return None, None

# tsr,pqu,tu,pqrs-> :  caa0  cca1  s01  v1100
def s01v1100(supersys_info, subsystem, charges):
    if "s01v1100" not in supersys_info.timings:  supersys_info.timings["s01v1100"] = 0.
    result01 = _s01v1100(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _s01v1100(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01v1100(supersys_info, subsystem, charges, permutation):
    X = frag_resolve(supersys_info, zip(subsystem, charges), permutation)
    prefactor = (-1)**(X.n_i1 + X.P)
    def diagram(i0,i1,j0,j1):
        t0 = time.time()
        result = scalar_value( prefactor * X.caa0tXX_St1[i0,j0](s,r,u) @ X.cca1pqX_Vpq00[i1,j1](u,r,s) )
        #result = scalar_value( prefactor * X.caa0[i0,j0](t,s,r) @ X.s01(t,u) @ X.cca1pqX_Vpq00[i1,j1](u,r,s) )
        #result = scalar_value( prefactor * X.caa0[i0,j0](t,s,r) @ X.cca1[i1,j1](p,q,u) @ X.s01(t,u) @ X.v1100(p,q,r,s) )
        supersys_info.timings["s01v1100"] += (time.time() - t0)
        return result
    if X.Dchg0==+1 and X.Dchg1==-1:
        return diagram, permutation
    else:
        return None, None

# pqts,ur,tu,pqrs-> :  ccca0  aa1  s01  v0010
def s01v0010(supersys_info, subsystem, charges):
    if "s01v0010" not in supersys_info.timings:  supersys_info.timings["s01v0010"] = 0.
    result01 = _s01v0010(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _s01v0010(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01v0010(supersys_info, subsystem, charges, permutation):
    X = frag_resolve(supersys_info, zip(subsystem, charges), permutation)
    prefactor = -2
    def diagram(i0,i1,j0,j1):
        t0 = time.time()
        result = scalar_value( prefactor * X.aa1uX_S0u[i1,j1](r,t) @ X.ccca0pqXs_Vpq1s[i0,j0](t,r) )
        #result = scalar_value( prefactor * X.aa1[i1,j1](u,r) @ X.s01(t,u) @ X.ccca0pqXs_Vpq1s[i0,j0](t,r) )
        #result = scalar_value( prefactor * X.ccca0[i0,j0](p,q,t,s) @ X.aa1[i1,j1](u,r) @ X.s01(t,u) @ X.v0010(p,q,r,s) )
        supersys_info.timings["s01v0010"] += (time.time() - t0)
        return result
    if X.Dchg0==-2 and X.Dchg1==+2:
        return diagram, permutation
    else:
        return None, None

# pt,qusr,tu,pqrs-> :  cc0  caaa1  s01  v0111
def s01v0111(supersys_info, subsystem, charges):
    if "s01v0111" not in supersys_info.timings:  supersys_info.timings["s01v0111"] = 0.
    result01 = _s01v0111(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _s01v0111(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01v0111(supersys_info, subsystem, charges, permutation):
    X = frag_resolve(supersys_info, zip(subsystem, charges), permutation)
    prefactor = -2
    def diagram(i0,i1,j0,j1):
        t0 = time.time()
        result = scalar_value( prefactor * X.cc0Xt_St1[i0,j0](p,u) @ X.caaa1qXsr_V0qrs[i1,j1](u,p) )
        #result = scalar_value( prefactor * X.cc0[i0,j0](p,t) @ X.s01(t,u) @ X.caaa1qXsr_V0qrs[i1,j1](u,p) )
        #result = scalar_value( prefactor * X.cc0[i0,j0](p,t) @ X.caaa1[i1,j1](q,u,s,r) @ X.s01(t,u) @ X.v0111(p,q,r,s) )
        supersys_info.timings["s01v0111"] += (time.time() - t0)
        return result
    if X.Dchg0==-2 and X.Dchg1==+2:
        return diagram, permutation
    else:
        return None, None

# pqt,usr,tu,pqrs-> :  ccc0  aaa1  s01  v0011
def s01v0011(supersys_info, subsystem, charges):
    if "s01v0011" not in supersys_info.timings:  supersys_info.timings["s01v0011"] = 0.
    result01 = _s01v0011(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _s01v0011(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01v0011(supersys_info, subsystem, charges, permutation):
    X = frag_resolve(supersys_info, zip(subsystem, charges), permutation)
    prefactor = (-1)**(X.n_i1 + X.P)
    def diagram(i0,i1,j0,j1):
        t0 = time.time()
        result = scalar_value( prefactor * X.ccc0[i0,j0](p,q,t) @ X.aaa1[i1,j1](u,s,r) @ X.s01(t,u) @ X.v0011(p,q,r,s) )
        supersys_info.timings["s01v0011"] += (time.time() - t0)
        return result
    if X.Dchg0==-3 and X.Dchg1==+3:
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
    "v0111":    v0111,
    "v0011":    v0011,
    "s01v1101": s01v1101,
    "s01v1000": s01v1000,
    "s01v0101": s01v0101,
    "s01v1100": s01v1100,
    "s01v1111": s01v1111,
    "s01v0000": s01v0000,
    "s01v0010": s01v0010,
    "s01v0111": s01v0111,
    "s01v0011": s01v0011
}
