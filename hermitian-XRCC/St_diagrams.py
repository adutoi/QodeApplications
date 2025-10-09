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

# monomer diagram

def t00(X):
    i0, j0 = 0, 1
    return 1 * raw(
        #  X.ca0(i0,j0,p,q)
        #@ X.t00(p,q)
        X.ca0pq_Tpq
        )

# dimer diagrams

def t01(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (-1)**(X.n_j0 + X.P) * raw(
        #  X.c0(i0,j0,p)
        #@ X.a1(i1,j1,q)
        #@ X.t01(p,q)
          X.c0(i0,j0,p)
        @ X.a1q_T0q(i1,j1,p)
        )

def s01t10(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return -1 * raw(
        #  X.ca0(i0,j0,t,q)
        #@ X.ca1(i1,j1,p,u)
        #@ X.s01(t,u)
        #@ X.t10(p,q)
          X.ca0tX_St1(i0,j0,q,u)
        @ X.ca1pX_Tp0(i1,j1,u,q)
        )

def s01t00(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (-1)**(X.n_j0 + X.P + 1) * raw(
        #  X.cca0(i0,j0,p,t,q)
        #@ X.a1(i1,j1,u)
        #@ X.s01(t,u)
        #@ X.t00(p,q)
          X.cca0pXq_Tpq(i0,j0,t)
        @ X.a1u_S0u(i1,j1,t)
        )

def s01t11(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (-1)**(X.n_j0 + X.P + 1) * raw(
        #  X.c0(i0,j0,t)
        #@ X.caa1(i1,j1,p,u,q)
        #@ X.s01(t,u)
        #@ X.t11(p,q)
          X.c0t_St1(i0,j0,u)
        @ X.caa1pXq_Tpq(i1,j1,u)
        )

def s01t01(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return 1 * raw(
        #  X.cc0(i0,j0,p,t)
        #@ X.aa1(i1,j1,u,q)
        #@ X.s01(t,u)
        #@ X.t01(p,q)
          X.cc0Xt_St1(i0,j0,p,u)
        @ X.aa1Xq_T0q(i1,j1,u,p)
        )

def s01s10t00(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return -1 * raw(
        #  X.ccaa0(i0,j0,p,t,w,q)
        #@ X.ca1(i1,j1,v,u)
        #@ X.s01(t,u)
        #@ X.s10(v,w)
        #@ X.t00(p,q)
          X.ccaa0pXXq_Tpq(i0,j0,t,w)
        @ X.ca1Xu_S0u(i1,j1,v,t)
        @ X.s10(v,w)
        )

def s01s01t10(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (1/2) * (-1)**(X.n_j0 + X.P + 1) * raw(
        #  X.cca0(i0,j0,t,v,q)
        #@ X.caa1(i1,j1,p,w,u)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.t10(p,q)
          X.cca0tXX_St1(i0,j0,v,q,u)
        @ X.caa1XwX_S0w(i1,j1,p,u,v)
        @ X.t10(p,q)
        )

def s01s10t01(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (-1)**(X.n_j0 + X.P + 1) * raw(
        #  X.cca0(i0,j0,p,t,w)
        #@ X.caa1(i1,j1,v,u,q)
        #@ X.s01(t,u)
        #@ X.s10(v,w)
        #@ X.t01(p,q)
          X.cca0XtX_St1(i0,j0,p,w,u)
        @ X.caa1vXX_Sv0(i1,j1,u,q,w)
        @ X.t01(p,q)
        )

def s01s01t00(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (1/2) * raw(
        #  X.ccca0(i0,j0,p,t,v,q)
        #@ X.aa1(i1,j1,w,u)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.t00(p,q)
          X.ccca0pXXq_Tpq(i0,j0,t,v)
        @ X.aa1Xu_S0u(i1,j1,w,t)
        @ X.s01(v,w)
        )

def s01s01t11(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (1/2) * raw(
        #  X.cc0(i0,j0,t,v)
        #@ X.caaa1(i1,j1,p,w,u,q)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.t11(p,q)
          X.cc0tX_St1(i0,j0,v,u)
        @ X.caaa1pXXq_Tpq(i1,j1,w,u)
        @ X.s01(v,w)
        )

def s01s01t01(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (1/2) * (-1)**(X.n_j0 + X.P) * raw(
          X.ccc0(i0,j0,p,t,v)
        @ X.aaa1(i1,j1,w,u,q)
        @ X.s01(t,u)
        @ X.s01(v,w)
        @ X.t01(p,q)
        )



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
    "t01":           build_diagram(t01,          Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "s01t10":        build_diagram(s01t10,       Dchgs=(0,0),   permutations=[(0,1),(1,0)]),
    "s01t00":        build_diagram(s01t00,       Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "s01t11":        build_diagram(s01t11,       Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "s01t01":        build_diagram(s01t01,       Dchgs=(-2,+2), permutations=[(0,1),(1,0)]),
    "s01s10t00":     build_diagram(s01s10t00,    Dchgs=(0,0),   permutations=[(0,1),(1,0)]),
    "s01s01t10":     build_diagram(s01s01t10,    Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "s01s10t01":     build_diagram(s01s10t01,    Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "s01s01t00":     build_diagram(s01s01t00,    Dchgs=(-2,+2), permutations=[(0,1),(1,0)]),
    "s01s01t11":     build_diagram(s01s01t11,    Dchgs=(-2,+2), permutations=[(0,1),(1,0)]),
    "s01s01t01":     build_diagram(s01s01t01,    Dchgs=(-3,+3), permutations=[(0,1),(1,0)]),
}
