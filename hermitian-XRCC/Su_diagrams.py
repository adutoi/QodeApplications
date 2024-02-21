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
# They must take the arguments (supersys_info, subsys_chgs), but after that, it is up to you.
# It should return a list of kernels that takes state indices (for specified fragment charges) their relevant permutations.
# Don't forget to update the "catalog" dictionary at the end.
##########

# monomer diagram

# pq,pq-> :  ca0  u0_00
def u000(supersys_info, subsys_chgs):
    if "u000" not in supersys_info.timings:  supersys_info.timings["u000"] = timer()
    X = frag_resolve(supersys_info, subsys_chgs)
    prefactor = 1
    def diagram(i0,j0):
        t0 = time.time()
        result = scalar_value( prefactor * X.ca0pq_U0pq[i0,j0] )
        #result = scalar_value( prefactor * X.ca0[i0,j0](p,q) @ X.u0_00(p,q) )
        supersys_info.timings["u000"] += (time.time() - t0)
        return result
    if X.Dchg0==0:
        return [(diagram, (0,))]
    else:
        return [None]



# dimer diagrams

# pq,pq-> :  ca0  u1_00
def u100(supersys_info, subsys_chgs):
    if "u100" not in supersys_info.timings:  supersys_info.timings["u100"] = timer()
    result01 = _u100(supersys_info, subsys_chgs, permutation=(0,1))
    result10 = _u100(supersys_info, subsys_chgs, permutation=(1,0))
    return [result01, result10]
def _u100(supersys_info, subsys_chgs, permutation):
    X = frag_resolve(supersys_info, subsys_chgs, permutation)
    prefactor = 1
    def diagram(i0,i1,j0,j1):
        t0 = time.time()
        if i1==j1:
            result = scalar_value( prefactor * X.ca0pq_U1pq[i0,j0] )
            #result = scalar_value( prefactor * X.ca0[i0,j0](p,q) @ X.u1_00(p,q) )
        else:
            return 0
        supersys_info.timings["u100"] += (time.time() - t0)
        return result
    if X.Dchg0==0 and X.Dchg1==0:
        return diagram, permutation
    else:
        return None

# p,q,pq-> :  c0  a1  u0_01
def u001(supersys_info, subsys_chgs):
    if "u001" not in supersys_info.timings:  supersys_info.timings["u001"] = timer()
    result01 = _u001(supersys_info, subsys_chgs, permutation=(0,1))
    result10 = _u001(supersys_info, subsys_chgs, permutation=(1,0))
    return [result01, result10]
def _u001(supersys_info, subsys_chgs, permutation):
    X = frag_resolve(supersys_info, subsys_chgs, permutation)
    prefactor = (-1)**(X.n_i1 + X.P)
    def diagram(i0,i1,j0,j1):
        t0 = time.time()
        result = scalar_value( prefactor * X.c0[i0,j0](p) @ X.a1q_U00q[i1,j1](p) )
        #result = scalar_value( prefactor * X.c0[i0,j0](p) @ X.a1[i1,j1](q) @ X.u0_01(p,q) )
        supersys_info.timings["u001"] += (time.time() - t0)
        return result
    if X.Dchg0==-1 and X.Dchg1==+1:
        return diagram, permutation
    else:
        return None

# p,q,pq-> :  c0  a1  u1_01
def u101(supersys_info, subsys_chgs):
    if "u101" not in supersys_info.timings:  supersys_info.timings["u101"] = timer()
    result01 = _u101(supersys_info, subsys_chgs, permutation=(0,1))
    result10 = _u101(supersys_info, subsys_chgs, permutation=(1,0))
    return [result01, result10]
def _u101(supersys_info, subsys_chgs, permutation):
    X = frag_resolve(supersys_info, subsys_chgs, permutation)
    prefactor = (-1)**(X.n_i1 + X.P)
    def diagram(i0,i1,j0,j1):
        t0 = time.time()
        result = scalar_value( prefactor * X.c0[i0,j0](p) @ X.a1q_U10q[i1,j1](p) )
        #result = scalar_value( prefactor * X.c0[i0,j0](p) @ X.a1[i1,j1](q) @ X.u1_01(p,q) )
        supersys_info.timings["u101"] += (time.time() - t0)
        return result
    if X.Dchg0==-1 and X.Dchg1==+1:
        return diagram, permutation
    else:
        return None

# tq,pu,tu,pq-> :  ca0  ca1  s01  u0_10
def s01u010(supersys_info, subsys_chgs):
    if "s01u010" not in supersys_info.timings:  supersys_info.timings["s01u010"] = timer()
    result01 = _s01u010(supersys_info, subsys_chgs, permutation=(0,1))
    result10 = _s01u010(supersys_info, subsys_chgs, permutation=(1,0))
    return [result01, result10]
def _s01u010(supersys_info, subsys_chgs, permutation):
    X = frag_resolve(supersys_info, subsys_chgs, permutation)
    prefactor = -1
    def diagram(i0,i1,j0,j1):
        t0 = time.time()
        result = scalar_value( prefactor * X.ca0tX_St1[i0,j0](q,u) @ X.ca1pX_U0p0[i1,j1](u,q) )
        #result = scalar_value( prefactor * X.ca0[i0,j0](t,q) @ X.ca1[i1,j1](p,u) @ X.s01(t,u) @ X.u0_10(p,q) )
        supersys_info.timings["s01u010"] += (time.time() - t0)
        return result
    if X.Dchg0==0 and X.Dchg1==0:
        return diagram, permutation
    else:
        return None

# tq,pu,tu,pq-> :  ca0  ca1  s01  u1_10
def s01u110(supersys_info, subsys_chgs):
    if "s01u110" not in supersys_info.timings:  supersys_info.timings["s01u110"] = timer()
    result01 = _s01u110(supersys_info, subsys_chgs, permutation=(0,1))
    result10 = _s01u110(supersys_info, subsys_chgs, permutation=(1,0))
    return [result01, result10]
def _s01u110(supersys_info, subsys_chgs, permutation):
    X = frag_resolve(supersys_info, subsys_chgs, permutation)
    prefactor = -1
    def diagram(i0,i1,j0,j1):
        t0 = time.time()
        result = scalar_value( prefactor * X.ca0tX_St1[i0,j0](q,u) @ X.ca1pX_U1p0[i1,j1](u,q) )
        #result = scalar_value( prefactor * X.ca0[i0,j0](t,q) @ X.ca1[i1,j1](p,u) @ X.s01(t,u) @ X.u1_10(p,q) )
        supersys_info.timings["s01u110"] += (time.time() - t0)
        return result
    if X.Dchg0==0 and X.Dchg1==0:
        return diagram, permutation
    else:
        return None

# ptq,u,tu,pq-> :  cca0  a1  s01  u0_00
def s01u000(supersys_info, subsys_chgs):
    if "s01u000" not in supersys_info.timings:  supersys_info.timings["s01u000"] = timer()
    result01 = _s01u000(supersys_info, subsys_chgs, permutation=(0,1))
    result10 = _s01u000(supersys_info, subsys_chgs, permutation=(1,0))
    return [result01, result10]
def _s01u000(supersys_info, subsys_chgs, permutation):
    X = frag_resolve(supersys_info, subsys_chgs, permutation)
    prefactor = (-1)**(X.n_i1 + X.P + 1)
    def diagram(i0,i1,j0,j1):
        t0 = time.time()
        result = scalar_value( prefactor * X.a1u_S0u[i1,j1](t) @ X.cca0pXq_U0pq[i0,j0](t) )
        #result = scalar_value( prefactor * X.cca0[i0,j0](p,t,q) @ X.a1[i1,j1](u) @ X.s01(t,u) @ X.u0_00(p,q) )
        supersys_info.timings["s01u000"] += (time.time() - t0)
        return result
    if X.Dchg0==-1 and X.Dchg1==+1:
        return diagram, permutation
    else:
        return None

# ptq,u,tu,pq-> :  cca0  a1  s01  u1_00
def s01u100(supersys_info, subsys_chgs):
    if "s01u100" not in supersys_info.timings:  supersys_info.timings["s01u100"] = timer()
    result01 = _s01u100(supersys_info, subsys_chgs, permutation=(0,1))
    result10 = _s01u100(supersys_info, subsys_chgs, permutation=(1,0))
    return [result01, result10]
def _s01u100(supersys_info, subsys_chgs, permutation):
    X = frag_resolve(supersys_info, subsys_chgs, permutation)
    prefactor = (-1)**(X.n_i1 + X.P + 1)
    def diagram(i0,i1,j0,j1):
        t0 = time.time()
        result = scalar_value( prefactor * X.a1u_S0u[i1,j1](t) @ X.cca0pXq_U1pq[i0,j0](t) )
        #result = scalar_value( prefactor * X.cca0[i0,j0](p,t,q) @ X.a1[i1,j1](u) @ X.s01(t,u) @ X.u1_00(p,q) )
        supersys_info.timings["s01u100"] += (time.time() - t0)
        return result
    if X.Dchg0==-1 and X.Dchg1==+1:
        return diagram, permutation
    else:
        return None

# t,puq,tu,pq-> :  c0  caa1  s01  u0_11
def s01u011(supersys_info, subsys_chgs):
    if "s01u011" not in supersys_info.timings:  supersys_info.timings["s01u011"] = timer()
    result01 = _s01u011(supersys_info, subsys_chgs, permutation=(0,1))
    result10 = _s01u011(supersys_info, subsys_chgs, permutation=(1,0))
    return [result01, result10]
def _s01u011(supersys_info, subsys_chgs, permutation):
    X = frag_resolve(supersys_info, subsys_chgs, permutation)
    prefactor = (-1)**(X.n_i1 + X.P + 1)
    def diagram(i0,i1,j0,j1):
        t0 = time.time()
        result = scalar_value( prefactor * X.c0t_St1[i0,j0](u) @ X.caa1pXq_U0pq[i1,j1](u) )
        #result = scalar_value( prefactor * X.c0[i0,j0](t) @ X.caa1[i1,j1](p,u,q) @ X.s01(t,u) @ X.u0_11(p,q) )
        supersys_info.timings["s01u011"] += (time.time() - t0)
        return result
    if X.Dchg0==-1 and X.Dchg1==+1:
        return diagram, permutation
    else:
        return None

# t,puq,tu,pq-> :  c0  caa1  s01  u1_11
def s01u111(supersys_info, subsys_chgs):
    if "s01u111" not in supersys_info.timings:  supersys_info.timings["s01u111"] = timer()
    result01 = _s01u111(supersys_info, subsys_chgs, permutation=(0,1))
    result10 = _s01u111(supersys_info, subsys_chgs, permutation=(1,0))
    return [result01, result10]
def _s01u111(supersys_info, subsys_chgs, permutation):
    X = frag_resolve(supersys_info, subsys_chgs, permutation)
    prefactor = (-1)**(X.n_i1 + X.P + 1)
    def diagram(i0,i1,j0,j1):
        t0 = time.time()
        result = scalar_value( prefactor * X.c0t_St1[i0,j0](u) @ X.caa1pXq_U1pq[i1,j1](u) )
        #result = scalar_value( prefactor * X.c0[i0,j0](t) @ X.caa1[i1,j1](p,u,q) @ X.s01(t,u) @ X.u1_11(p,q) )
        supersys_info.timings["s01u111"] += (time.time() - t0)
        return result
    if X.Dchg0==-1 and X.Dchg1==+1:
        return diagram, permutation
    else:
        return None

# pt,uq,tu,pq-> :  cc0  aa1  s01  u0_01
def s01u001(supersys_info, subsys_chgs):
    if "s01u001" not in supersys_info.timings:  supersys_info.timings["s01u001"] = timer()
    result01 = _s01u001(supersys_info, subsys_chgs, permutation=(0,1))
    result10 = _s01u001(supersys_info, subsys_chgs, permutation=(1,0))
    return [result01, result10]
def _s01u001(supersys_info, subsys_chgs, permutation):
    X = frag_resolve(supersys_info, subsys_chgs, permutation)
    prefactor = 1
    def diagram(i0,i1,j0,j1):
        t0 = time.time()
        result = scalar_value( prefactor * X.cc0Xt_St1[i0,j0](p,u) @ X.aa1Xq_U00q[i1,j1](u,p) )
        #result = scalar_value( prefactor * X.cc0[i0,j0](p,t) @ X.aa1[i1,j1](u,q) @ X.s01(t,u) @ X.u0_01(p,q) )
        supersys_info.timings["s01u001"] += (time.time() - t0)
        return result
    if X.Dchg0==-2 and X.Dchg1==+2:
        return diagram, permutation
    else:
        return None

# pt,uq,tu,pq-> :  cc0  aa1  s01  u1_01
def s01u101(supersys_info, subsys_chgs):
    if "s01u101" not in supersys_info.timings:  supersys_info.timings["s01u101"] = timer()
    result01 = _s01u101(supersys_info, subsys_chgs, permutation=(0,1))
    result10 = _s01u101(supersys_info, subsys_chgs, permutation=(1,0))
    return [result01, result10]
def _s01u101(supersys_info, subsys_chgs, permutation):
    X = frag_resolve(supersys_info, subsys_chgs, permutation)
    prefactor = 1
    def diagram(i0,i1,j0,j1):
        t0 = time.time()
        result = scalar_value( prefactor * X.cc0Xt_St1[i0,j0](p,u) @ X.aa1Xq_U10q[i1,j1](u,p) )
        #result = scalar_value( prefactor * X.cc0[i0,j0](p,t) @ X.aa1[i1,j1](u,q) @ X.s01(t,u) @ X.u1_01(p,q) )
        supersys_info.timings["s01u101"] += (time.time() - t0)
        return result
    if X.Dchg0==-2 and X.Dchg1==+2:
        return diagram, permutation
    else:
        return None



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
