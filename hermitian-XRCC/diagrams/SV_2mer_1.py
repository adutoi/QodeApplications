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
from XR_tensor import raw
from .diagram_hack import state_indices, no_result

p, q, r, s, t, u, v, w = "pqrstuvw"    # some contraction indices for easier reading



def s01v0100(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
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
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
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
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (-1)**(X.n_j0) * raw(
        #  X.cccaa0(i0,j0,p,q,t,s,r)
        #@ X.a1(i1,j1,u)
        #@ X.s01(t,u)
        #@ X.v0000(p,q,r,s)
          X.cccaa0pqXsr_Vpqrs(i0,j0,t)
        @ X.a1u_S0u(i1,j1,t)
        )

def s01v0101(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return 4 * (-1)**(X.n_j0) * raw(
        #  X.cca0(i0,j0,p,t,r)
        #@ X.caa1(i1,j1,q,u,s)
        #@ X.s01(t,u)
        #@ X.v0101(p,q,r,s)
          X.cca0pXr_Vp1r1(i0,j0,t,q,s)
        @ X.caa1XuX_S0u(i1,j1,q,s,t)
        )

def s01v1100(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (-1)**(X.n_j0) * raw(
        #  X.caa0(i0,j0,t,s,r)
        #@ X.cca1(i1,j1,p,q,u)
        #@ X.s01(t,u)
        #@ X.v1100(p,q,r,s)
          X.caa0Xsr_V11rs(i0,j0,t,p,q)
        @ X.cca1XXu_S0u(i1,j1,p,q,t)
        )

def s01v1111(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (-1)**(X.n_j0) * raw(
        #  X.c0(i0,j0,t)
        #@ X.ccaaa1(i1,j1,p,q,u,s,r)
        #@ X.s01(t,u)
        #@ X.v1111(p,q,r,s)
          X.c0t_St1(i0,j0,u)
        @ X.ccaaa1pqXsr_Vpqrs(i1,j1,u)
        )

def s01v0001(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
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
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
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
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (-1)**(X.n_j0) * raw(
        #  X.ccc0(i0,j0,p,q,t)
        #@ X.aaa1(i1,j1,u,s,r)
        #@ X.s01(t,u)
        #@ X.v0011(p,q,r,s)
          X.ccc0pqX_Vpq11(i0,j0,t,r,s)
        @ X.aaa1uXX_S0u(i1,j1,s,r,t)
        )
