#    (C) Copyright 2025 Anthony D. Dutoi and Marco Bauer
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

p, q, r, s, t, u, v, w, x, y = "pqrstuvwxy"    # some contraction indices for easier reading



def s01s01s10t10(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/2) * raw(
        #  X.ccaa0(i0,j0,t,v,y,q)
        #@ X.ccaa1(i1,j1,p,x,w,u)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s10(x,y)
        #@ X.t10(p,q)
          X.ccaa0XXXq_T1q(i0,j0,t,v,y,p)
        @ X.ccaa1XXXu_S0u(i1,j1,p,x,w,t)
        @ X.s01(v,w)
        @ X.s10(x,y)
        )

def s01s01s10t00(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/2) * (-1)**(X.n_j0) * raw(
        #  X.cccaa0(i0,j0,p,t,v,y,q)
        #@ X.caa1(i1,j1,x,w,u)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s10(x,y)
        #@ X.t00(p,q)
          X.cccaa0pXXXq_Tpq(i0,j0,t,v,y)
        @ X.caa1XXu_S0u(i1,j1,x,w,t)
        @ X.s01(v,w)
        @ X.s10(x,y)
        )

def s01s01s10t11(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/2) * (-1)**(X.n_j0) * raw(
        #  X.cca0(i0,j0,t,v,y)
        #@ X.ccaaa1(i1,j1,p,x,w,u,q)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s10(x,y)
        #@ X.t11(p,q)
          X.cca0tXX_St1(i0,j0,v,y,u)
        @ X.ccaaa1pXXXq_Tpq(i1,j1,x,w,u)
        @ X.s01(v,w)
        @ X.s10(x,y)
        )

def s01s01s01t10(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return -(1/6) * raw(
        #  X.ccca0(i0,j0,t,v,x,q)
        #@ X.caaa1(i1,j1,p,y,w,u)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.t10(p,q)
          X.ccca0XXXq_T1q(i0,j0,t,v,x,p)
        @ X.caaa1XXXu_S0u(i1,j1,p,y,w,t)
        @ X.s01(v,w)
        @ X.s01(x,y)
        )

def s01s01s10t01(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return -(1/2) * raw(
        #  X.ccca0(i0,j0,p,t,v,y)
        #@ X.caaa1(i1,j1,x,w,u,q)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s10(x,y)
        #@ X.t01(p,q)
          X.ccca0pXXX_Tp1(i0,j0,t,v,y,q)
        @ X.caaa1XXuX_S0u(i1,j1,x,w,q,t)
        @ X.s01(v,w)
        @ X.s10(x,y)
        )

def s01s01s01t00(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/6) * (-1)**(X.n_j0 + 1) * raw(
        #  X.cccca0(i0,j0,p,t,v,x,q)
        #@ X.aaa1(i1,j1,y,w,u)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.t00(p,q)
          X.cccca0pXXXq_Tpq(i0,j0,t,v,x)
        @ X.aaa1XXu_S0u(i1,j1,y,w,t)
        @ X.s01(v,w)
        @ X.s01(x,y)
        )

def s01s01s01t11(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/6) * (-1)**(X.n_j0 + 1) * raw(
        #  X.ccc0(i0,j0,t,v,x)
        #@ X.caaaa1(i1,j1,p,y,w,u,q)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.t11(p,q)
          X.ccc0tXX_St1(i0,j0,v,x,u)
        @ X.caaaa1pXXXq_Tpq(i1,j1,y,w,u)
        @ X.s01(v,w)
        @ X.s01(x,y)
        )

def s01s01s01t01(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/6) * raw(
        #  X.cccc0(i0,j0,p,t,v,x)
        #@ X.aaaa1(i1,j1,y,w,u,q)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.t01(p,q)
          X.cccc0pXXX_Tp1(i0,j0,t,v,x,q)
        @ X.aaaa1XXuX_S0u(i1,j1,y,w,q,t)
        @ X.s01(v,w)
        @ X.s01(x,y)
        )
