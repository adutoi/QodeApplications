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
from diagram_hack        import state_indices, no_result

p, q, r, s, t, u, v, w = "pqrstuvw"    # some contraction indices for easier reading



##########
# Here are the implementations of the actual diagrams.
# They must take the arguments (X, i0,i1,...,j0,j1,...) where and instance of frag_resolve (see build_diagram),
# which provides all of the input tensors and/or intermediate contractions.  These functions then return a scalar
# which is the evaluated diagram.  Don't forget to update the "catalog" dictionary at the end.
##########

# 0-mer diagram

# -> :
def identity(X):
        return 1

# dimer diagrams

# p,q,pq-> :  c0  a1  s01
def s01(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (-1)**(X.n_i1 + X.P) * raw( X.c0(i0,j0,p) @ X.a1q_S0q(i1,j1,p) )
    #return (-1)**(X.n_i1 + X.P) * scalar_value( X.c0[i0,j0](p) @ X.a1q_S0q[i1,j1](p) )
    #return (-1)**(X.n_i1 + X.P) * scalar_value( X.c0[i0,j0](p) @ X.a1[i1,j1](q) @ X.s01(p,q))

# ps,rq,pq,rs-> :  ca0  ca1  s01  s10
def s01s10(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return -1 * scalar_value( X.ca0[i0,j0](p,s) @ X.ca1[i1,j1](r,q) @ X.s01(p,q) @ X.s10(r,s))

# pr,sq,pq,rs-> :  cc0  aa1  s01  s01
def s01s01(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (1/2.) * scalar_value( X.cc0[i0,j0](p,r) @ X.aa1[i1,j1](s,q) @ X.s01(p,q) @ X.s01(r,s))

# pru,tsq,pq,rs,tu-> :  cca0  caa1  s01  s01  s10
def s01s01s10(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (1/2.) * (-1)**(X.n_i1 + X.P + 1) * scalar_value( X.cca0[i0,j0](p,r,u) @ X.caa1[i1,j1](t,s,q) @ X.s01(p,q) @ X.s01(r,s) @ X.s10(t,u))

# prwu,tvsq,pq,rs,tu,vw-> :  ccaa0  ccaa1  s01  s01  s10  s10
def s01s01s10s10(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (1/4.) * scalar_value( X.ccaa0[i0,j0](p,r,w,u) @ X.ccaa1[i1,j1](t,v,s,q) @ X.s01(p,q) @ X.s01(r,s) @ X.s10(t,u) @ X.s10(v,w))

# prtw,vusq,pq,rs,tu,vw-> :  ccca0  caaa1  s01  s01  s01  s10
def s01s01s01s10(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (-1/6.) * scalar_value( X.ccca0[i0,j0](p,r,t,w) @ X.caaa1[i1,j1](v,u,s,q) @ X.s01(p,q) @ X.s01(r,s) @ X.s01(t,u) @ X.s10(v,w))



##########
# A dictionary catalog.  The string association lets users specify active diagrams at the top level
# for each kind of integral (symmetric, biorthogonal, ...).  The first argument to build_diagram() is
# one of the functions found above, which performs the relevant contraction for the given diagram for
# a fixed permutation (adjusted externally).
##########

catalog = {}

catalog[0] = {
    "identity": build_diagram(identity, Dchgs=None, permutations=None)
}
catalog[2] = {
    "s01":          build_diagram(s01,          Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "s01s10":       build_diagram(s01s10,       Dchgs=( 0, 0), permutations=[(0,1)]),
    "s01s01":       build_diagram(s01s01,       Dchgs=(-2,+2), permutations=[(0,1),(1,0)]),
    "s01s01s10":    build_diagram(s01s01s10,    Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "s01s01s10s10": build_diagram(s01s01s10s10, Dchgs=( 0, 0), permutations=[(0,1)]),
    "s01s01s01s10": build_diagram(s01s01s01s10, Dchgs=(-2,+2), permutations=[(0,1),(1,0)]),
}
