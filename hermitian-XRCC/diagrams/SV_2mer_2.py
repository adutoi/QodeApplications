#    (C) Copyright 2023, 2024, 2025 Anthony D. Dutoi and Marco Bauer
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
from qode.math.tensornet import raw
from .diagram_hack import state_indices, no_result

p, q, r, s, t, u, v, w = "pqrstuvw"    # some contraction indices for easier reading



def s01s01v1100(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
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
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
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
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
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
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
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
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
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
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
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
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
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
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/2) * raw(
          X.ccccaa0(i0,j0,p,q,t,v,s,r)
        @ X.aa1(i1,j1,w,u)
        @ X.s01(t,u)
        @ X.s01(v,w)
        @ X.v0000(p,q,r,s)
        )

def s01s01v0101(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
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
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/2) * raw(
          X.cc0(i0,j0,t,v)
        @ X.ccaaaa1(i1,j1,p,q,w,u,s,r)
        @ X.s01(t,u)
        @ X.s01(v,w)
        @ X.v1111(p,q,r,s)
        )

def s01s10v0011(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
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
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (-1)**(X.n_j0 + X.P + 1) * raw(
          X.cccca0(i0,j0,p,q,t,v,r)
        @ X.aaa1(i1,j1,w,u,s)
        @ X.s01(t,u)
        @ X.s01(v,w)
        @ X.v0001(p,q,r,s)
        )

def s01s01v0111(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (-1)**(X.n_j0 + X.P) * raw(
          X.ccc0(i0,j0,p,t,v)
        @ X.caaaa1(i1,j1,q,w,u,s,r)
        @ X.s01(t,u)
        @ X.s01(v,w)
        @ X.v0111(p,q,r,s)
        )

def s01s01v0011(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/2) * raw(
          X.cccc0(i0,j0,p,q,t,v)
        @ X.aaaa1(i1,j1,w,u,s,r)
        @ X.s01(t,u)
        @ X.s01(v,w)
        @ X.v0011(p,q,r,s)
        )
