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



def s01s10t00(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
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
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
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
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
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
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
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
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
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
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/2) * (-1)**(X.n_j0 + X.P) * raw(
          X.ccc0(i0,j0,p,t,v)
        @ X.aaa1(i1,j1,w,u,q)
        @ X.s01(t,u)
        @ X.s01(v,w)
        @ X.t01(p,q)
        )
