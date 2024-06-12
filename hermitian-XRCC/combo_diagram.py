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
import numpy
from qode.math.tensornet import scalar_value, evaluate, raw
from build_diagram       import build_diagram

p, q, r, s, t, u, v, w = "pqrstuvw"    # some contraction indices for easier reading

i0, i1, j0, j1 = 0, 1, 2, 3

# tq,pu,tu,pq-> :        ca0  ca1  s01    t10
# tq,pu,tu,pq-> :        ca0  ca1  s01  u0_10
# tq,pu,tu,pq-> :        ca0  ca1  s01  u1_10
# qtsr,pu,tu,pqrs-> :  ccaa0  ca1  s01  v1000
def combo(X, i0s,i1s,j0s,j1s):
    temp = evaluate(2 * X.ccaa0qXsr_V1qrs - X.ca0Xq_T1q - X.ca0Xq_U01q - X.ca0Xq_U11q )
    return raw( X.ca1Xu_S0u(i1,j1,p,t) @ temp(i0,j0,t,p) )
    ##temp = evaluate(2 * X.ccaa0qXsr_V1qrs[i0,j0] - X.ca0Xq_T1q[i0,j0] - X.ca0Xq_U01q[i0,j0] - X.ca0Xq_U11q[i0,j0] )
    ##return scalar_value( X.ca1Xu_S0u[i1,j1](p,t) @ temp(t,p) )
    #value  =  2 * scalar_value( X.ca1Xu_S0u[i1,j1](p,t) @ X.ccaa0qXsr_V1qrs[i0,j0](t,p) )
    #value += -1 * scalar_value( X.ca1Xu_S0u[i1,j1](p,t) @ X.ca0Xq_T1q[i0,j0](t,p) )
    #value += -1 * scalar_value( X.ca1Xu_S0u[i1,j1](p,t) @ X.ca0Xq_U01q[i0,j0](t,p) )
    #value += -1 * scalar_value( X.ca1Xu_S0u[i1,j1](p,t) @ X.ca0Xq_U11q[i0,j0](t,p) )
    #value += -1 * scalar_value( X.ca0[i0,j0](t,q) @ X.ca1[i1,j1](p,u) @ X.s01(t,u) @ X.t10(p,q) )
    #value += -1 * scalar_value( X.ca0[i0,j0](t,q) @ X.ca1[i1,j1](p,u) @ X.s01(t,u) @ X.u0_10(p,q) )
    #value += -1 * scalar_value( X.ca0[i0,j0](t,q) @ X.ca1[i1,j1](p,u) @ X.s01(t,u) @ X.u1_10(p,q) )
    #value += -1 * scalar_value( X.ca0tX_St1[i0,j0](q,u) @ X.ca1pX_Tp0[i1,j1](u,q) )
    #value += -1 * scalar_value( X.ca0tX_St1[i0,j0](q,u) @ X.ca1pX_U0p0[i1,j1](u,q) )
    #value += -1 * scalar_value( X.ca0tX_St1[i0,j0](q,u) @ X.ca1pX_U1p0[i1,j1](u,q) )
    #return value



##########
# A dictionary catalog.  The string association lets users specify active diagrams at the top level
# for each kind of integral (symmetric, biorthogonal, ...).  The first argument to build_diagram() is
# one of the functions found above, which performs the relevant contraction for the given diagram for
# a fixed permutation (adjusted externally).
##########

catalog = {}

catalog[2] = {
    "combo": build_diagram(combo, Dchgs=(0,0), permutations=[(0,1),(1,0)]),
}
