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



def s01u010(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return -1 * raw(
        #  X.ca0(i0,j0,t,q)
        #@ X.ca1(i1,j1,p,u)
        #@ X.s01(t,u)
        #@ X.u0_10(p,q)
          X.ca0tX_St1(i0,j0,q,u)
        @ X.ca1pX_U0p0(i1,j1,u,q)
        )

def s01u000(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (-1)**(X.n_j0 + X.P + 1) * raw(
        #  X.cca0(i0,j0,p,t,q)
        #@ X.a1(i1,j1,u)
        #@ X.s01(t,u)
        #@ X.u0_00(p,q)
          X.cca0pXq_U0pq(i0,j0,t)
        @ X.a1u_S0u(i1,j1,t)
        )

def s01u011(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (-1)**(X.n_j0 + X.P + 1) * raw(
        #  X.c0(i0,j0,t)
        #@ X.caa1(i1,j1,p,u,q)
        #@ X.s01(t,u)
        #@ X.u0_11(p,q)
          X.c0t_St1(i0,j0,u)
        @ X.caa1pXq_U0pq(i1,j1,u)
        )

def s01u001(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return 1 * raw(
        #  X.cc0(i0,j0,p,t)
        #@ X.aa1(i1,j1,u,q)
        #@ X.s01(t,u)
        #@ X.u0_01(p,q)
          X.cc0Xt_St1(i0,j0,p,u)
        @ X.aa1Xq_U00q(i1,j1,u,p)
        )

def s01u110(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return -1 * raw(
        #  X.ca0(i0,j0,t,q)
        #@ X.ca1(i1,j1,p,u)
        #@ X.s01(t,u)
        #@ X.u1_10(p,q)
          X.ca0tX_St1(i0,j0,q,u)
        @ X.ca1pX_U1p0(i1,j1,u,q)
        )

def s01u100(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (-1)**(X.n_j0 + X.P + 1) * raw(
        #  X.cca0(i0,j0,p,t,q)
        #@ X.a1(i1,j1,u)
        #@ X.s01(t,u)
        #@ X.u1_00(p,q)
          X.cca0pXq_U1pq(i0,j0,t)
        @ X.a1u_S0u(i1,j1,t)
        )

def s01u111(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (-1)**(X.n_j0 + X.P + 1) * raw(
        #  X.c0(i0,j0,t)
        #@ X.caa1(i1,j1,p,u,q)
        #@ X.s01(t,u)
        #@ X.u1_11(p,q)
          X.c0t_St1(i0,j0,u)
        @ X.caa1pXq_U1pq(i1,j1,u)
        )

def s01u101(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return 1 * raw(
        #  X.cc0(i0,j0,p,t)
        #@ X.aa1(i1,j1,u,q)
        #@ X.s01(t,u)
        #@ X.u1_01(p,q)
          X.cc0Xt_St1(i0,j0,p,u)
        @ X.aa1Xq_U10q(i1,j1,u,p)
        )
