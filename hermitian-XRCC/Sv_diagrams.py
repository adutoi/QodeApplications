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

def v0101(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return 4 * raw(
        #  X.ca0(i0,j0,p,r)
        #@ X.ca1(i1,j1,q,s)
        #@ X.v0101(p,q,r,s)
          X.ca0(i0,j0,p,r)
        @ X.ca1qs_V0q0s(i1,j1,p,r)
        )

def v0001(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return 2 * (-1)**(X.n_j0 + X.P + 1) * raw(
        #  X.cca0(i0,j0,p,q,r)
        #@ X.a1(i1,j1,s)
        #@ X.v0001(p,q,r,s)
          X.cca0pqr_Vpqr1(i0,j0,s)
        @ X.a1(i1,j1,s)
        )

def v0100(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return 2 * (-1)**(X.n_j0 + X.P) * raw(
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

def s01v1101(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return 2 * raw(
        #  X.ca0(i0,j0,t,r)
        #@ X.ccaa1(i1,j1,p,q,u,s)
        #@ X.s01(t,u)
        #@ X.v1101(p,q,r,s)
          X.ca0tX_St1(i0,j0,r,u)
        @ X.ccaa1pqXs_Vpq0s(i1,j1,u,r)
        )

def s01v0000(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (-1)**(X.n_j0 + X.P) * raw(
        #  X.cccaa0(i0,j0,p,q,t,s,r)
        #@ X.a1(i1,j1,u)
        #@ X.s01(t,u)
        #@ X.v0000(p,q,r,s)
          X.cccaa0pqXsr_Vpqrs(i0,j0,t)
        @ X.a1u_S0u(i1,j1,t)
        )

def s01v0101(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return 4 * (-1)**(X.n_j0 + X.P) * raw(
        #  X.cca0(i0,j0,p,t,r)
        #@ X.caa1(i1,j1,q,u,s)
        #@ X.s01(t,u)
        #@ X.v0101(p,q,r,s)
          X.cca0pXr_Vp1r1(i0,j0,t,q,s)
        @ X.caa1XuX_S0u(i1,j1,q,s,t)
        )

def s01v1100(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (-1)**(X.n_j0 + X.P) * raw(
        #  X.caa0(i0,j0,t,s,r)
        #@ X.cca1(i1,j1,p,q,u)
        #@ X.s01(t,u)
        #@ X.v1100(p,q,r,s)
          X.caa0tXX_St1(i0,j0,s,r,u)
        @ X.cca1pqX_Vpq00(i1,j1,u,r,s)
        )

def s01v1111(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (-1)**(X.n_j0 + X.P) * raw(
        #  X.c0(i0,j0,t)
        #@ X.ccaaa1(i1,j1,p,q,u,s,r)
        #@ X.s01(t,u)
        #@ X.v1111(p,q,r,s)
          X.c0t_St1(i0,j0,u)
        @ X.ccaaa1pqXsr_Vpqrs(i1,j1,u)
        )

def s01v0001(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return 2 * raw(
        #  X.ccca0(i0,j0,p,q,t,r)
        #@ X.aa1(i1,j1,u,s)
        #@ X.s01(t,u)
        #@ X.v0001(p,q,r,s)
          X.ccca0pqXr_Vpqr1(i0,j0,t,s)
        @ X.aa1uX_S0u(i1,j1,s,t)
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

def s01v0011(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (-1)**(X.n_j0 + X.P) * raw(
          X.ccc0(i0,j0,p,q,t)
        @ X.aaa1(i1,j1,u,s,r)
        @ X.s01(t,u)
        @ X.v0011(p,q,r,s)
        #  X.ccc0pqX_Vpq11(i0,j0,t,r,s)    # should be right, ...
        #@ X.aaa1uXX_S0u(i1,j1,s,r,t)      # ... but 3xCT not yet tested
        )

def s01s01v1100(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (1/2) * raw(
        #  X.ccaa0(i0,j0,t,v,s,r)
        #@ X.ccaa1(i1,j1,p,q,w,u)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.v1100(p,q,r,s)
          X.ccaa0tXXX_St1(i0,j0,v,s,r,u)
        @ X.ccaa1pqXX_Vpq00(i1,j1,w,u,r,s)
        @ X.s01(v,w)
        )

def s01s10v0000(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return -1 * raw(
        #  X.cccaaa0(i0,j0,p,q,t,w,s,r)
        #@ X.ca1(i1,j1,v,u)
        #@ X.s01(t,u)
        #@ X.s10(v,w)
        #@ X.v0000(p,q,r,s)
          X.cccaaa0pqXXsr_Vpqrs(i0,j0,t,w)
        @ X.ca1Xu_S0u(i1,j1,v,t)
        @ X.s10(v,w)
        )

def s01s10v0101(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return -4 * raw(
        #  X.ccaa0(i0,j0,p,t,w,r)
        #@ X.ccaa1(i1,j1,q,v,u,s)
        #@ X.s01(t,u)
        #@ X.s10(v,w)
        #@ X.v0101(p,q,r,s)
          X.ccaa0XtXX_St1(i0,j0,p,w,r,u)
        @ X.ccaa1qXXs_V0q0s(i1,j1,v,u,p,r)
        @ X.s10(v,w)
        )

def s01s01v0100(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (-1)**(X.n_j0 + X.P) * raw(
        #  X.cccaa0(i0,j0,p,t,v,s,r)
        #@ X.caa1(i1,j1,q,w,u)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.v0100(p,q,r,s)
          X.cccaa0pXXsr_Vp1rs(i0,j0,t,v,q)
        @ X.caa1XXu_S0u(i1,j1,q,w,t)
        @ X.s01(v,w)
        )

def s01s01v1101(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (-1)**(X.n_j0 + X.P + 1) * raw(
        #  X.cca0(i0,j0,t,v,r)
        #@ X.ccaaa1(i1,j1,p,q,w,u,s)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.v1101(p,q,r,s)
          X.cca0tXX_St1(i0,j0,v,r,u)
        @ X.ccaaa1pqXXs_Vpq0s(i1,j1,w,u,r)
        @ X.s01(v,w)
        )

def s01s10v0001(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return 2 * (-1)**(X.n_j0 + X.P) * raw(
        #  X.cccaa0(i0,j0,p,q,t,w,r)
        #@ X.caa1(i1,j1,v,u,s)
        #@ X.s01(t,u)
        #@ X.s10(v,w)
        #@ X.v0001(p,q,r,s)
          X.cccaa0pqXXr_Vpqr1(i0,j0,t,w,s)
        @ X.caa1XuX_S0u(i1,j1,v,s,t)
        @ X.s10(v,w)
        )

def s01s10v0100(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return 2 * (-1)**(X.n_j0 + X.P + 1) * raw(
        #  X.ccaaa0(i0,j0,p,t,w,s,r)
        #@ X.cca1(i1,j1,q,v,u)
        #@ X.s01(t,u)
        #@ X.s10(v,w)
        #@ X.v0100(p,q,r,s)
          X.ccaaa0pXXsr_Vp1rs(i0,j0,t,w,q)
        @ X.cca1XXu_S0u(i1,j1,q,v,t)
        @ X.s10(v,w)
        )

def s01s01v0000(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (1/2) * raw(
          X.ccccaa0(i0,j0,p,q,t,v,s,r)
        @ X.aa1(i1,j1,w,u)
        @ X.s01(t,u)
        @ X.s01(v,w)
        @ X.v0000(p,q,r,s)
        )

def s01s01v0101(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return 2 * raw(
        #  X.ccca0(i0,j0,p,t,v,r)
        #@ X.caaa1(i1,j1,q,w,u,s)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.v0101(p,q,r,s)
          X.ccca0XtXX_St1(i0,j0,p,v,r,u)
        @ X.caaa1qXXs_V0q0s(i1,j1,w,u,p,r)
        @ X.s01(v,w)
        )

def s01s01v1111(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (1/2) * raw(
          X.cc0(i0,j0,t,v)
        @ X.ccaaaa1(i1,j1,p,q,w,u,s,r)
        @ X.s01(t,u)
        @ X.s01(v,w)
        @ X.v1111(p,q,r,s)
        )

def s01s10v0011(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return -1 * raw(
        #  X.ccca0(i0,j0,p,q,t,w)
        #@ X.caaa1(i1,j1,v,u,s,r)
        #@ X.s01(t,u)
        #@ X.s10(v,w)
        #@ X.v0011(p,q,r,s)
          X.ccca0XXtX_St1(i0,j0,p,q,w,u)
        @ X.caaa1XXsr_V00rs(i1,j1,v,u,p,q)
        @ X.s10(v,w)
        )

def s01s01v0001(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (-1)**(X.n_j0 + X.P + 1) * raw(
          X.cccca0(i0,j0,p,q,t,v,r)
        @ X.aaa1(i1,j1,w,u,s)
        @ X.s01(t,u)
        @ X.s01(v,w)
        @ X.v0001(p,q,r,s)
        )

def s01s01v0111(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (-1)**(X.n_j0 + X.P) * raw(
          X.ccc0(i0,j0,p,t,v)
        @ X.caaaa1(i1,j1,q,w,u,s,r)
        @ X.s01(t,u)
        @ X.s01(v,w)
        @ X.v0111(p,q,r,s)
        )

def s01s01v0011(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (1/2) * raw(
          X.cccc0(i0,j0,p,q,t,v)
        @ X.aaaa1(i1,j1,w,u,s,r)
        @ X.s01(t,u)
        @ X.s01(v,w)
        @ X.v0011(p,q,r,s)
        )



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
    "v0101":         build_diagram(v0101,        Dchgs=(0,0),   permutations=[(0,1)]),
    "v0001":         build_diagram(v0001,        Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "v0100":         build_diagram(v0100,        Dchgs=(+1,-1), permutations=[(0,1),(1,0)]),
    "v0011":         build_diagram(v0011,        Dchgs=(-2,+2), permutations=[(0,1),(1,0)]),
    "s01v0100":      build_diagram(s01v0100,     Dchgs=(0,0),   permutations=[(0,1),(1,0)]),
    "s01v1101":      build_diagram(s01v1101,     Dchgs=(0,0),   permutations=[(0,1),(1,0)]),
    "s01v0000":      build_diagram(s01v0000,     Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "s01v0101":      build_diagram(s01v0101,     Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "s01v1100":      build_diagram(s01v1100,     Dchgs=(+1,-1), permutations=[(0,1),(1,0)]),
    "s01v1111":      build_diagram(s01v1111,     Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "s01v0001":      build_diagram(s01v0001,     Dchgs=(-2,+2), permutations=[(0,1),(1,0)]),
    "s01v0111":      build_diagram(s01v0111,     Dchgs=(-2,+2), permutations=[(0,1),(1,0)]),
    "s01v0011":      build_diagram(s01v0011,     Dchgs=(-3,+3), permutations=[(0,1),(1,0)]),
    "s01s01v1100":   build_diagram(s01s01v1100,  Dchgs=(0,0),   permutations=[(0,1),(1,0)]),
    "s01s10v0000":   build_diagram(s01s10v0000,  Dchgs=(0,0),   permutations=[(0,1),(1,0)]),
    "s01s10v0101":   build_diagram(s01s10v0101,  Dchgs=(0,0),   permutations=[(0,1)]),
    "s01s01v0100":   build_diagram(s01s01v0100,  Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "s01s01v1101":   build_diagram(s01s01v1101,  Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "s01s10v0001":   build_diagram(s01s10v0001,  Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "s01s10v0100":   build_diagram(s01s10v0100,  Dchgs=(+1,-1), permutations=[(0,1),(1,0)]),
    "s01s01v0000":   build_diagram(s01s01v0000,  Dchgs=(-2,+2), permutations=[(0,1),(1,0)]),
    "s01s01v0101":   build_diagram(s01s01v0101,  Dchgs=(-2,+2), permutations=[(0,1),(1,0)]),
    "s01s01v1111":   build_diagram(s01s01v1111,  Dchgs=(-2,+2), permutations=[(0,1),(1,0)]),
    "s01s10v0011":   build_diagram(s01s10v0011,  Dchgs=(-2,+2), permutations=[(0,1),(1,0)]),
    "s01s01v0001":   build_diagram(s01s01v0001,  Dchgs=(-3,+3), permutations=[(0,1),(1,0)]),
    "s01s01v0111":   build_diagram(s01s01v0111,  Dchgs=(-3,+3), permutations=[(0,1),(1,0)]),
    "s01s01v0011":   build_diagram(s01s01v0011,  Dchgs=(-4,+4), permutations=[(0,1),(1,0)]),
}
