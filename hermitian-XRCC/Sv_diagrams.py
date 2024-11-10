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

def v0000(X):
    i0, j0 = 0, 1
    return 1 * raw(
        #  X.ccaa0(i0,j0,p,q,s,r)
        #@ X.v0000(p,q,r,s)
          X.ccaa0pqsr_Vpqrs
        )

# dimer diagrams

def v0110(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return -4 * raw(
        #  X.ca0(i0,j0,p,s)
        #@ X.ca1(i1,j1,q,r)
        #@ X.v0110(p,q,r,s)
          X.ca0ps_Vp11s(i0,j0,q,r)
        @ X.ca1(i1,j1,q,r)
        )

def v0010(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return 2 * (-1)**(X.n_i1 + X.P) * raw(
        #  X.cca0(i0,j0,p,q,s)
        #@ X.a1(i1,j1,r)
        #@ X.v0010(p,q,r,s)
          X.cca0pqs_Vpq1s(i0,j0,r)
        @ X.a1(i1,j1,r) 
        )

def v0100(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return 2 * (-1)**(X.n_i1 + X.P) * raw(
        #  X.caa0(i0,j0,p,s,r)
        #@ X.c1(i1,j1,q)
        #@ X.v0100(p,q,r,s)
          X.caa0psr_Vp1rs(i0,j0,q)
        @ X.c1(i1,j1,q)
        )

def v0011(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return 1 * raw(
        #  X.cc0(i0,j0,p,q)
        #@ X.aa1(i1,j1,s,r)
        #@ X.v0011(p,q,r,s)
        ##  X.cc0pq_Vpq11(i0,j0,r,s)
        ##@ X.aa1(i1,j1,s,r)
          X.cc0(i0,j0,p,q)
        @ X.aa1sr_V00rs(i1,j1,p,q)
        )

def s01v0100(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return -2 * raw(
        #  X.ccaa0(i0,j0,p,t,s,r)
        #@ X.ca1(i1,j1,q,u)
        #@ X.s01(t,u)
        #@ X.v0100(p,q,r,s)
          X.ccaa0pXsr_Vp1rs(i0,j0,t,q)
        @ X.ca1Xu_S0u(i1,j1,q,t)
        )

def s01v1110(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return -2 * raw(
        #  X.ca0(i0,j0,t,s)
        #@ X.ccaa1(i1,j1,p,q,u,r)
        #@ X.s01(t,u)
        #@ X.v1110(p,q,r,s)
          X.ca0tX_St1(i0,j0,s,u)
        @ X.ccaa1pqXr_Vpqr0(i1,j1,u,s)
        )

def s01v0000(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (-1)**(X.n_i1 + X.P) * raw(
        #  X.cccaa0(i0,j0,p,q,t,s,r)
        #@ X.a1(i1,j1,u)
        #@ X.s01(t,u)
        #@ X.v0000(p,q,r,s)
          X.cccaa0pqXsr_Vpqrs(i0,j0,t)
        @ X.a1u_S0u(i1,j1,t)
        )

def s01v0110(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return -4 * (-1)**(X.n_i1 + X.P) * raw(
        #  X.cca0(i0,j0,p,t,s)
        #@ X.caa1(i1,j1,q,u,r)
        #@ X.s01(t,u)
        #@ X.v0110(p,q,r,s)
          X.cca0pXs_Vp11s(i0,j0,t,q,r)
        @ X.caa1XuX_S0u(i1,j1,q,r,t)
        )

def s01v1100(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (-1)**(X.n_i1 + X.P) * raw(
        #  X.caa0(i0,j0,t,s,r)
        #@ X.cca1(i1,j1,p,q,u)
        #@ X.s01(t,u)
        #@ X.v1100(p,q,r,s)
          X.caa0Xsr_V11rs(i0,j0,t,p,q)
        @ X.cca1XXu_S0u(i1,j1,p,q,t)
        )

def s01v1111(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (-1)**(X.n_i1 + X.P) * raw(
        #  X.c0(i0,j0,t)
        #@ X.ccaaa1(i1,j1,p,q,u,s,r)
        #@ X.s01(t,u)
        #@ X.v1111(p,q,r,s)
          X.c0t_St1(i0,j0,u)
        @ X.ccaaa1pqXsr_Vpqrs(i1,j1,u)
        )

def s01v0010(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return -2 * raw(
        #  X.ccca0(i0,j0,p,q,t,s)
        #@ X.aa1(i1,j1,u,r)
        #@ X.s01(t,u)
        #@ X.v0010(p,q,r,s)
          X.ccca0pqXs_Vpq1s(i0,j0,t,r)
        @ X.aa1uX_S0u(i1,j1,r,t)
        )

def s01v0111(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return -2 * raw(
        #  X.cc0(i0,j0,p,t)
        #@ X.caaa1(i1,j1,q,u,s,r)
        #@ X.s01(t,u)
        #@ X.v0111(p,q,r,s)
          X.cc0Xt_St1(i0,j0,p,u)
        @ X.caaa1qXsr_V0qrs(i1,j1,u,p)
        )

# pqt,usr,tu,pqrs-> :  ccc0  aaa1  s01  v0011
#def s01v0011(X, contract_last=False):
#    if no_result(X, contract_last):  return []
#    i0, i1, j0, j1 = state_indices(contract_last)
#    return (-1)**(X.n_i1 + X.P) * raw( X.ccc0(i0,j0,p,q,t) @ X.aaa1(i1,j1,u,s,r) @ X.s01(t,u) @ X.v0011(p,q,r,s) )
#    #return (-1)**(X.n_i1 + X.P) * scalar_value( X.ccc0[i0,j0](p,q,t) @ X.aaa1[i1,j1](u,s,r) @ X.s01(t,u) @ X.v0011(p,q,r,s) )



##########
# A dictionary catalog.  The string association lets users specify active diagrams at the top level
# for each kind of integral (symmetric, biorthogonal, ...).  The first argument to build_diagram() is
# one of the functions found above, which performs the relevant contraction for the given diagram for
# a fixed permutation (adjusted externally).
##########

catalog = {}

catalog[1] = {
    "v0000":    build_diagram(v0000)
}
catalog[2] = {
    "v0110":    build_diagram(v0110,    Dchgs=( 0, 0), permutations=[(0,1)]),
    "v0010":    build_diagram(v0010,    Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "v0100":    build_diagram(v0100,    Dchgs=(+1,-1), permutations=[(0,1),(1,0)]),
    "v0011":    build_diagram(v0011,    Dchgs=(-2,+2), permutations=[(0,1),(1,0)]),
    "s01v0100": build_diagram(s01v0100, Dchgs=( 0, 0), permutations=[(0,1),(1,0)]),
    "s01v1110": build_diagram(s01v1110, Dchgs=( 0, 0), permutations=[(0,1),(1,0)]),
    "s01v0000": build_diagram(s01v0000, Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "s01v0110": build_diagram(s01v0110, Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "s01v1100": build_diagram(s01v1100, Dchgs=(+1,-1), permutations=[(0,1),(1,0)]),
    "s01v1111": build_diagram(s01v1111, Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "s01v0010": build_diagram(s01v0010, Dchgs=(-2,+2), permutations=[(0,1),(1,0)]),
    "s01v0111": build_diagram(s01v0111, Dchgs=(-2,+2), permutations=[(0,1),(1,0)]),
    #"s01v0011": build_diagram(s01v0011, Dchgs=(-3,+3), permutations=[(0,1),(1,0)])
}
