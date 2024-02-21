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
import time
from timer import timer
from qode.math.tensornet import scalar_value
from frag_resolve import frag_resolve

p, q, r, s, t, u, v, w = "pqrstuvw"



##########
# Here are the implementations of the actual diagrams.
# The public @staticmethods must take the arguments (supersys_info, subsystem, charges), but after that, it is up to you.
# It should return a list of kernels that takes state indices (for specified fragment charges) their relevant permutations.
# Don't forget to update the "catalog" dictionary at the end.
##########

# 0-mer diagram

# -> :
def identity(supersys_info, subsystem, charges):
    if "identity" not in supersys_info.timings:  supersys_info.timings["identity"] = timer()
    # Identity
    def diagram():
        t0 = time.time()
        result = 1
        supersys_info.timings["identity"] += (time.time() - t0)
        return result
    return [(diagram, [])]



# dimer diagrams

# p,q,pq-> :  c0  a1  s01
def s01(supersys_info, subsystem, charges):
    if "s01" not in supersys_info.timings:  supersys_info.timings["s01"] = timer()
    result01 = _s01(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _s01(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01(supersys_info, subsystem, charges, permutation):
    X = frag_resolve(supersys_info, zip(subsystem, charges), permutation)
    prefactor = (-1)**(X.n_i1 + X.P)
    def diagram(i0,i1,j0,j1):
        t0 = time.time()
        result = scalar_value( prefactor * X.c0[i0,j0](p) @ X.a1q_S0q[i1,j1](p) )
        #result = scalar_value( prefactor * X.c0[i0,j0](p) @ X.a1[i1,j1](q) @ X.s01(p,q))
        supersys_info.timings["s01"] += (time.time() - t0)
        return result
    if X.Dchg0==-1 and X.Dchg1==+1:
        return diagram, permutation
    else:
        return None, None

# ps,rq,pq,rs-> :  ca0  ca1  s01  s10
def s01s10(supersys_info, subsystem, charges):
    if "s01s10" not in supersys_info.timings:  supersys_info.timings["s01s10"] = timer()
    X = frag_resolve(supersys_info, zip(subsystem, charges), permutation=(0,1))
    prefactor = -1
    def diagram(i0,i1,j0,j1):
        t0 = time.time()
        result = scalar_value( prefactor * X.ca0[i0,j0](p,s) @ X.ca1[i1,j1](r,q) @ X.s01(p,q) @ X.s10(r,s))
        supersys_info.timings["s01s10"] += (time.time() - t0)
        return result
    if X.Dchg0==0 and X.Dchg1==0:
        return [(diagram, (0,1))]
    else:
        return [(None, None)]

# pr,sq,pq,rs-> :  cc0  aa1  s01  s01
def s01s01(supersys_info, subsystem, charges):
    if "s01s01" not in supersys_info.timings:  supersys_info.timings["s01s01"] = timer()
    result01 = _s01s01(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _s01s01(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01s01(supersys_info, subsystem, charges, permutation):
    X = frag_resolve(supersys_info, zip(subsystem, charges), permutation)
    prefactor = 1/2.
    def diagram(i0,i1,j0,j1):
        t0 = time.time()
        result = scalar_value( prefactor * X.cc0[i0,j0](p,r) @ X.aa1[i1,j1](s,q) @ X.s01(p,q) @ X.s01(r,s))
        supersys_info.timings["s01s01"] += (time.time() - t0)
        return result
    if X.Dchg0==-2 and X.Dchg1==+2:
        return diagram, permutation
    else:
        return None, None

# pru,tsq,pq,rs,tu-> :  cca0  caa1  s01  s01  s10
def s01s01s10(supersys_info, subsystem, charges):
    if "s01s01s10" not in supersys_info.timings:  supersys_info.timings["s01s01s10"] = timer()
    result01 = _s01s01s10(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _s01s01s10(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01s01s10(supersys_info, subsystem, charges, permutation):
    X = frag_resolve(supersys_info, zip(subsystem, charges), permutation)
    prefactor = (-1)**(X.n_i1 + X.P + 1) / 2.
    def diagram(i0,i1,j0,j1):
        t0 = time.time()
        result = scalar_value( prefactor * X.cca0[i0,j0](p,r,u) @ X.caa1[i1,j1](t,s,q) @ X.s01(p,q) @ X.s01(r,s) @ X.s10(t,u))
        supersys_info.timings["s01s01s10"] += (time.time() - t0)
        return result
    if X.Dchg0==-1 and X.Dchg1==+1:
        return diagram, permutation
    else:
        return None, None

# prwu,tvsq,pq,rs,tu,vw-> :  ccaa0  ccaa1  s01  s01  s10  s10
def s01s01s10s10(supersys_info, subsystem, charges):
    if "s01s01s10s10" not in supersys_info.timings:  supersys_info.timings["s01s01s10s10"] = timer()
    X = frag_resolve(supersys_info, zip(subsystem, charges), permutation=(0,1))
    prefactor = 1/4.
    def diagram(i0,i1,j0,j1):
        t0 = time.time()
        result = scalar_value( prefactor * X.ccaa0[i0,j0](p,r,w,u) @ X.ccaa1[i1,j1](t,v,s,q) @ X.s01(p,q) @ X.s01(r,s) @ X.s10(t,u) @ X.s10(v,w))
        supersys_info.timings["s01s01s10s10"] += (time.time() - t0)
        return result
    if X.Dchg0==0 and X.Dchg1==0:
        return [(diagram, (0,1))]
    else:
        return [(None, None)]

# prtw,vusq,pq,rs,tu,vw-> :  ccca0  caaa1  s01  s01  s01  s10
def s01s01s01s10(supersys_info, subsystem, charges):
    if "s01s01s01s10" not in supersys_info.timings:  supersys_info.timings["s01s01s01s10"] = timer()
    result01 = _s01s01s01s10(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _s01s01s01s10(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01s01s01s10(supersys_info, subsystem, charges, permutation):
    X = frag_resolve(supersys_info, zip(subsystem, charges), permutation)
    prefactor = -1 / 6.
    def diagram(i0,i1,j0,j1):
        t0 = time.time()
        result = scalar_value( prefactor * X.ccca0[i0,j0](p,r,t,w) @ X.caaa1[i1,j1](v,u,s,q) @ X.s01(p,q) @ X.s01(r,s) @ X.s01(t,u) @ X.s10(v,w))
        supersys_info.timings["s01s01s01s10"] += (time.time() - t0)
        return result
    if X.Dchg0==-2 and X.Dchg1==+2:
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
