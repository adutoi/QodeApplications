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
from qode.math.tensornet import scalar_value
from frag_resolve import frag_resolve

p, q, r, s, t, u, v, w = "pqrstuvw"



##########
# Here are the implementations of the actual diagrams.
# They must take the arguments (supersys_info, subsystem, charges), but after that, it is up to you.
# It should return a list of kernels that takes state indices (for specified fragment charges) their relevant permutations.
# Don't forget to update the "catalog" dictionary at the end.
##########

# monomer diagram

# pq,pq-> :  ca0  t00
def t00(supersys_info, subsystem, charges):
    if "t00" not in supersys_info.timings:  supersys_info.timings["t00"] = 0.
    X = frag_resolve(supersys_info, zip(subsystem, charges))
    prefactor = 1
    def diagram(i0,j0):
        t0 = time.time()
        result = scalar_value( prefactor * X.ca0pq_Tpq[i0,j0] )
        #result = scalar_value( prefactor * X.ca0[i0,j0](p,q) @ X.t00(p,q) )
        supersys_info.timings["t00"] += (time.time() - t0)
        return result
    if X.Dchg0==0:
        return [(diagram, (0,))]
    else:
        return [(None, None)]



# dimer diagrams

# p,q,pq-> :  c0  a1  t01
def t01(supersys_info, subsystem, charges):
    if "t01" not in supersys_info.timings:  supersys_info.timings["t01"] = 0.
    result01 = _t01(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _t01(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _t01(supersys_info, subsystem, charges, permutation):
    X = frag_resolve(supersys_info, zip(subsystem, charges), permutation)
    prefactor = (-1)**(X.n_i1 + X.P)
    def diagram(i0,i1,j0,j1):
        t0 = time.time()
        result = scalar_value( prefactor * X.c0[i0,j0](p) @ X.a1q_T0q[i1,j1](p) )
        #result = scalar_value( prefactor * X.c0[i0,j0](p) @ X.a1[i1,j1](q) @ X.t01(p,q) )
        supersys_info.timings["t01"] += (time.time() - t0)
        return result
    if X.Dchg0==-1 and X.Dchg1==+1:
        return diagram, permutation
    else:
        return None, None

# tq,pu,tu,pq-> :  ca0  ca1  s01  t10
def s01t10(supersys_info, subsystem, charges):
    if "s01t10" not in supersys_info.timings:  supersys_info.timings["s01t10"] = 0.
    result01 = _s01t10(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _s01t10(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01t10(supersys_info, subsystem, charges, permutation):
    X = frag_resolve(supersys_info, zip(subsystem, charges), permutation)
    prefactor = -1
    def diagram(i0,i1,j0,j1):
        t0 = time.time()
        result = scalar_value( prefactor * X.ca0tX_St1[i0,j0](q,u) @ X.ca1pX_Tp0[i1,j1](u,q) )
        #result = scalar_value( prefactor * X.ca0[i0,j0](t,q) @ X.ca1[i1,j1](p,u) @ X.s01(t,u) @ X.t10(p,q) )
        supersys_info.timings["s01t10"] += (time.time() - t0)
        return result
    if X.Dchg0==0 and X.Dchg1==0:
        return diagram, permutation
    else:
        return None, None

# ptq,u,tu,pq-> :  cca0  a1  s01  t00
def s01t00(supersys_info, subsystem, charges):
    if "s01t00" not in supersys_info.timings:  supersys_info.timings["s01t00"] = 0.
    result01 = _s01t00(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _s01t00(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01t00(supersys_info, subsystem, charges, permutation):
    X = frag_resolve(supersys_info, zip(subsystem, charges), permutation)
    prefactor = (-1)**(X.n_i1 + X.P + 1)
    def diagram(i0,i1,j0,j1):
        t0 = time.time()
        result = scalar_value( prefactor * X.a1u_S0u[i1,j1](t) @ X.cca0pXq_Tpq[i0,j0](t) )
        #result = scalar_value( prefactor * X.cca0[i0,j0](p,t,q) @ X.a1[i1,j1](u) @ X.s01(t,u) @ X.t00(p,q) )
        supersys_info.timings["s01t00"] += (time.time() - t0)
        return result
    if X.Dchg0==-1 and X.Dchg1==+1:
        return diagram, permutation
    else:
        return None, None

# t,puq,tu,pq-> :  c0  caa1  s01  t11
def s01t11(supersys_info, subsystem, charges):
    if "s01t11" not in supersys_info.timings:  supersys_info.timings["s01t11"] = 0.
    result01 = _s01t11(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _s01t11(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01t11(supersys_info, subsystem, charges, permutation):
    X = frag_resolve(supersys_info, zip(subsystem, charges), permutation)
    prefactor = (-1)**(X.n_i1 + X.P + 1)
    def diagram(i0,i1,j0,j1):
        t0 = time.time()
        result = scalar_value( prefactor * X.c0t_St1[i0,j0](u) @ X.caa1pXq_Tpq[i1,j1](u) )
        #result = scalar_value( prefactor * X.c0[i0,j0](t) @ X.caa1[i1,j1](p,u,q) @ X.s01(t,u) @ X.t11(p,q) )
        supersys_info.timings["s01t11"] += (time.time() - t0)
        return result
    if X.Dchg0==-1 and X.Dchg1==+1:
        return diagram, permutation
    else:
        return None, None

# pt,uq,tu,pq-> :  cc0  aa1  s01  t01
def s01t01(supersys_info, subsystem, charges):
    if "s01t01" not in supersys_info.timings:  supersys_info.timings["s01t01"] = 0.
    result01 = _s01t01(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _s01t01(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01t01(supersys_info, subsystem, charges, permutation):
    X = frag_resolve(supersys_info, zip(subsystem, charges), permutation)
    prefactor = 1
    def diagram(i0,i1,j0,j1):
        t0 = time.time()
        result = scalar_value( prefactor * X.cc0Xt_St1[i0,j0](p,u) @ X.aa1Xq_T0q[i1,j1](u,p) )
        #result = scalar_value( prefactor * X.cc0[i0,j0](p,t) @ X.aa1[i1,j1](u,q) @ X.s01(t,u) @ X.t01(p,q) )
        supersys_info.timings["s01t01"] += (time.time() - t0)
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

catalog[1] = {
    "t00":   t00
}

catalog[2] = {
    "t01":      t01,
    "s01t10":   s01t10,
    "s01t00":   s01t00,
    "s01t11":   s01t11,
    "s01t01":   s01t01
}
