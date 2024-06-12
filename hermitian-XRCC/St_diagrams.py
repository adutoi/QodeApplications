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
import numpy
from qode.math.tensornet import scalar_value, raw
from build_diagram       import build_diagram

p, q, r, s, t, u, v, w = "pqrstuvw"    # some contraction indices for easier reading



##########
# Here are the implementations of the actual diagrams.
# They must take the arguments (X, i0,i1,...,j0,j1,...) where and instance of frag_resolve (see build_diagram),
# which provides all of the input tensors and/or intermediate contractions.  These functions then return a scalar
# which is the evaluated diagram.  Don't forget to update the "catalog" dictionary at the end.
##########

# monomer diagram

# pq,pq-> :  ca0  t00
def t00(X, i0s,j0s):
    return 1 * raw( X.ca0pq_Tpq )
    #return 1 * scalar_value( X.ca0pq_Tpq[i0,j0] )
    #return 1 * scalar_value( X.ca0[i0,j0](p,q) @ X.t00(p,q) )

# dimer diagrams

i0, i1, j0, j1 = 0, 1, 2, 3

# p,q,pq-> :  c0  a1  t01
def t01(X, i0s,i1s,j0s,j1s):
    return (-1)**(X.n_i1 + X.P) * raw( X.c0(i0,j0,p) @ X.a1q_T0q(i1,j1,p) )
    #return (-1)**(X.n_i1 + X.P) * scalar_value( X.c0[i0,j0](p) @ X.a1q_T0q[i1,j1](p) )
    #return (-1)**(X.n_i1 + X.P) * scalar_value( X.c0[i0,j0](p) @ X.a1[i1,j1](q) @ X.t01(p,q) )

# tq,pu,tu,pq-> :  ca0  ca1  s01  t10
def s01t10(X, i0s,i1s,j0s,j1s):
    return -1 * raw( X.ca0tX_St1(i0,j0,q,u) @ X.ca1pX_Tp0(i1,j1,u,q) )
    #return -1 * scalar_value( X.ca0tX_St1[i0,j0](q,u) @ X.ca1pX_Tp0[i1,j1](u,q) )
    #return -1 * scalar_value( X.ca0[i0,j0](t,q) @ X.ca1[i1,j1](p,u) @ X.s01(t,u) @ X.t10(p,q) )

# ptq,u,tu,pq-> :  cca0  a1  s01  t00
def s01t00(X, i0s,i1s,j0s,j1s):
    return (-1)**(X.n_i1 + X.P + 1) * raw( X.a1u_S0u(i1,j1,t) @ X.cca0pXq_Tpq(i0,j0,t) )
    #return (-1)**(X.n_i1 + X.P + 1) * scalar_value( X.a1u_S0u[i1,j1](t) @ X.cca0pXq_Tpq[i0,j0](t) )
    #return (-1)**(X.n_i1 + X.P + 1) * scalar_value( X.cca0[i0,j0](p,t,q) @ X.a1[i1,j1](u) @ X.s01(t,u) @ X.t00(p,q) )

# t,puq,tu,pq-> :  c0  caa1  s01  t11
def s01t11(X, i0s,i1s,j0s,j1s):
    return (-1)**(X.n_i1 + X.P + 1) * raw( X.c0t_St1(i0,j0,u) @ X.caa1pXq_Tpq(i1,j1,u) )
    #return (-1)**(X.n_i1 + X.P + 1) * scalar_value( X.c0t_St1[i0,j0](u) @ X.caa1pXq_Tpq[i1,j1](u) )
    #return (-1)**(X.n_i1 + X.P + 1) * scalar_value( X.c0[i0,j0](t) @ X.caa1[i1,j1](p,u,q) @ X.s01(t,u) @ X.t11(p,q) )

# pt,uq,tu,pq-> :  cc0  aa1  s01  t01
def s01t01(X, i0s,i1s,j0s,j1s):
    return 1 * raw( X.cc0Xt_St1(i0,j0,p,u) @ X.aa1Xq_T0q(i1,j1,u,p) )
    #return 1 * scalar_value( X.cc0Xt_St1[i0,j0](p,u) @ X.aa1Xq_T0q[i1,j1](u,p) )
    #return 1 * scalar_value( X.cc0[i0,j0](p,t) @ X.aa1[i1,j1](u,q) @ X.s01(t,u) @ X.t01(p,q) )



##########
# A dictionary catalog.  The string association lets users specify active diagrams at the top level
# for each kind of integral (symmetric, biorthogonal, ...).  The first argument to build_diagram() is
# one of the functions found above, which performs the relevant contraction for the given diagram for
# a fixed permutation (adjusted externally).
##########

catalog = {}

catalog[1] = {
    "t00":   build_diagram(t00)
}
catalog[2] = {
    "t01":      build_diagram(t01,    Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "s01t10":   build_diagram(s01t10, Dchgs=( 0, 0), permutations=[(0,1),(1,0)]),
    "s01t00":   build_diagram(s01t00, Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "s01t11":   build_diagram(s01t11, Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "s01t01":   build_diagram(s01t01, Dchgs=(-2,+2), permutations=[(0,1),(1,0)])
}
