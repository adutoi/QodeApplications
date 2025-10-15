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



def s01s01s10s10u000(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/4) * raw(
        #  X.cccaaa0(i0,j0,p,t,v,a,y,q)
        #@ X.ccaa1(i1,j1,x,z,w,u)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s10(x,y)
        #@ X.s10(z,a)
        #@ X.u0_00(p,q)
          X.cccaaa0pXXXXq_U0pq(i0,j0,t,v,a,y)
        @ X.ccaa1XXXu_S0u(i1,j1,x,z,w,t)
        @ X.s01(v,w)
        @ X.s10(x,y)
        @ X.s10(z,a)
        )

def s01s01s01s10u010(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/6) * (-1)**(X.n_j0) * raw(
        #  X.cccaa0(i0,j0,t,v,x,a,q)
        #@ X.ccaaa1(i1,j1,p,z,y,w,u)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.s10(z,a)
        #@ X.u0_10(p,q)
          X.cccaa0XXXXq_U01q(i0,j0,t,v,x,a,p)
        @ X.ccaaa1XXXXu_S0u(i1,j1,p,z,y,w,t)
        @ X.s01(v,w)
        @ X.s01(x,y)
        @ X.s10(z,a)
        )

def s01s01s10s10u001(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/4) * (-1)**(X.n_j0) * raw(
        #  X.cccaa0(i0,j0,p,t,v,a,y)
        #@ X.ccaaa1(i1,j1,x,z,w,u,q)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s10(x,y)
        #@ X.s10(z,a)
        #@ X.u0_01(p,q)
          X.cccaa0pXXXX_U0p1(i0,j0,t,v,a,y,q)
        @ X.ccaaa1XXXuX_S0u(i1,j1,x,z,w,q,t)
        @ X.s01(v,w)
        @ X.s10(x,y)
        @ X.s10(z,a)
        )

def s01s01s01s10u000(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return -(1/6) * raw(
        #  X.ccccaa0(i0,j0,p,t,v,x,a,q)
        #@ X.caaa1(i1,j1,z,y,w,u)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.s10(z,a)
        #@ X.u0_00(p,q)
          X.ccccaa0pXXXXq_U0pq(i0,j0,t,v,x,a)
        @ X.caaa1XXXu_S0u(i1,j1,z,y,w,t)
        @ X.s01(v,w)
        @ X.s01(x,y)
        @ X.s10(z,a)
        )

def s01s01s01s10u011(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return -(1/6) * raw(
        #  X.ccca0(i0,j0,t,v,x,a)
        #@ X.ccaaaa1(i1,j1,p,z,y,w,u,q)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.s10(z,a)
        #@ X.u0_11(p,q)
          X.ccca0tXXX_St1(i0,j0,v,x,a,u)
        @ X.ccaaaa1pXXXXq_U0pq(i1,j1,z,y,w,u)
        @ X.s01(v,w)
        @ X.s01(x,y)
        @ X.s10(z,a)
        )

def s01s01s01s01u010(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/24) * (-1)**(X.n_j0 + 1) * raw(
        #  X.cccca0(i0,j0,t,v,x,z,q)
        #@ X.caaaa1(i1,j1,p,a,y,w,u)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.s01(z,a)
        #@ X.u0_10(p,q)
          X.cccca0XXXXq_U01q(i0,j0,t,v,x,z,p)
        @ X.caaaa1XXXXu_S0u(i1,j1,p,a,y,w,t)
        @ X.s01(v,w)
        @ X.s01(x,y)
        @ X.s01(z,a)
        )

def s01s01s01s10u001(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/6) * (-1)**(X.n_j0 + 1) * raw(
        #  X.cccca0(i0,j0,p,t,v,x,a)
        #@ X.caaaa1(i1,j1,z,y,w,u,q)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.s10(z,a)
        #@ X.u0_01(p,q)
          X.cccca0pXXXX_U0p1(i0,j0,t,v,x,a,q)
        @ X.caaaa1XXXuX_S0u(i1,j1,z,y,w,q,t)
        @ X.s01(v,w)
        @ X.s01(x,y)
        @ X.s10(z,a)
        )

def s01s01s01s01u000(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/24) * raw(
        #  X.ccccca0(i0,j0,p,t,v,x,z,q)
        #@ X.aaaa1(i1,j1,a,y,w,u)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.s01(z,a)
        #@ X.u0_00(p,q)
          X.ccccca0pXXXXq_U0pq(i0,j0,t,v,x,z)
        @ X.aaaa1XXXu_S0u(i1,j1,a,y,w,t)
        @ X.s01(v,w)
        @ X.s01(x,y)
        @ X.s01(z,a)
        )

def s01s01s01s01u011(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/24) * raw(
        #  X.cccc0(i0,j0,t,v,x,z)
        #@ X.caaaaa1(i1,j1,p,a,y,w,u,q)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.s01(z,a)
        #@ X.u0_11(p,q)
          X.cccc0tXXX_St1(i0,j0,v,x,z,u)
        @ X.caaaaa1pXXXXq_U0pq(i1,j1,a,y,w,u)
        @ X.s01(v,w)
        @ X.s01(x,y)
        @ X.s01(z,a)
        )

def s01s01s01s01u001(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/24) * (-1)**(X.n_j0) * raw(
        #  X.ccccc0(i0,j0,p,t,v,x,z)
        #@ X.aaaaa1(i1,j1,a,y,w,u,q)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.s01(z,a)
        #@ X.u0_01(p,q)
          X.ccccc0pXXXX_U0p1(i0,j0,t,v,x,z,q)
        @ X.aaaaa1XXXuX_S0u(i1,j1,a,y,w,q,t)
        @ X.s01(v,w)
        @ X.s01(x,y)
        @ X.s01(z,a)
        )

def s01s01s10s10u100(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/4) * raw(
        #  X.cccaaa0(i0,j0,p,t,v,a,y,q)
        #@ X.ccaa1(i1,j1,x,z,w,u)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s10(x,y)
        #@ X.s10(z,a)
        #@ X.u1_00(p,q)
          X.cccaaa0pXXXXq_U1pq(i0,j0,t,v,a,y)
        @ X.ccaa1XXXu_S0u(i1,j1,x,z,w,t)
        @ X.s01(v,w)
        @ X.s10(x,y)
        @ X.s10(z,a)
        )

def s01s01s01s10u110(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/6) * (-1)**(X.n_j0) * raw(
        #  X.cccaa0(i0,j0,t,v,x,a,q)
        #@ X.ccaaa1(i1,j1,p,z,y,w,u)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.s10(z,a)
        #@ X.u1_10(p,q)
          X.cccaa0XXXXq_U11q(i0,j0,t,v,x,a,p)
        @ X.ccaaa1XXXXu_S0u(i1,j1,p,z,y,w,t)
        @ X.s01(v,w)
        @ X.s01(x,y)
        @ X.s10(z,a)
        )

def s01s01s10s10u101(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/4) * (-1)**(X.n_j0) * raw(
        #  X.cccaa0(i0,j0,p,t,v,a,y)
        #@ X.ccaaa1(i1,j1,x,z,w,u,q)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s10(x,y)
        #@ X.s10(z,a)
        #@ X.u1_01(p,q)
          X.cccaa0pXXXX_U1p1(i0,j0,t,v,a,y,q)
        @ X.ccaaa1XXXuX_S0u(i1,j1,x,z,w,q,t)
        @ X.s01(v,w)
        @ X.s10(x,y)
        @ X.s10(z,a)
        )

def s01s01s01s10u100(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return -(1/6) * raw(
        #  X.ccccaa0(i0,j0,p,t,v,x,a,q)
        #@ X.caaa1(i1,j1,z,y,w,u)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.s10(z,a)
        #@ X.u1_00(p,q)
          X.ccccaa0pXXXXq_U1pq(i0,j0,t,v,x,a)
        @ X.caaa1XXXu_S0u(i1,j1,z,y,w,t)
        @ X.s01(v,w)
        @ X.s01(x,y)
        @ X.s10(z,a)
        )

def s01s01s01s10u111(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return -(1/6) * raw(
        #  X.ccca0(i0,j0,t,v,x,a)
        #@ X.ccaaaa1(i1,j1,p,z,y,w,u,q)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.s10(z,a)
        #@ X.u1_11(p,q)
          X.ccca0tXXX_St1(i0,j0,v,x,a,u)
        @ X.ccaaaa1pXXXXq_U1pq(i1,j1,z,y,w,u)
        @ X.s01(v,w)
        @ X.s01(x,y)
        @ X.s10(z,a)
        )

def s01s01s01s01u110(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/24) * (-1)**(X.n_j0 + 1) * raw(
        #  X.cccca0(i0,j0,t,v,x,z,q)
        #@ X.caaaa1(i1,j1,p,a,y,w,u)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.s01(z,a)
        #@ X.u1_10(p,q)
          X.cccca0XXXXq_U11q(i0,j0,t,v,x,z,p)
        @ X.caaaa1XXXXu_S0u(i1,j1,p,a,y,w,t)
        @ X.s01(v,w)
        @ X.s01(x,y)
        @ X.s01(z,a)
        )

def s01s01s01s10u101(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/6) * (-1)**(X.n_j0 + 1) * raw(
        #  X.cccca0(i0,j0,p,t,v,x,a)
        #@ X.caaaa1(i1,j1,z,y,w,u,q)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.s10(z,a)
        #@ X.u1_01(p,q)
          X.cccca0pXXXX_U1p1(i0,j0,t,v,x,a,q)
        @ X.caaaa1XXXuX_S0u(i1,j1,z,y,w,q,t)
        @ X.s01(v,w)
        @ X.s01(x,y)
        @ X.s10(z,a)
        )

def s01s01s01s01u100(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/24) * raw(
        #  X.ccccca0(i0,j0,p,t,v,x,z,q)
        #@ X.aaaa1(i1,j1,a,y,w,u)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.s01(z,a)
        #@ X.u1_00(p,q)
          X.ccccca0pXXXXq_U1pq(i0,j0,t,v,x,z)
        @ X.aaaa1XXXu_S0u(i1,j1,a,y,w,t)
        @ X.s01(v,w)
        @ X.s01(x,y)
        @ X.s01(z,a)
        )

def s01s01s01s01u111(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/24) * raw(
        #  X.cccc0(i0,j0,t,v,x,z)
        #@ X.caaaaa1(i1,j1,p,a,y,w,u,q)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.s01(z,a)
        #@ X.u1_11(p,q)
          X.cccc0tXXX_St1(i0,j0,v,x,z,u)
        @ X.caaaaa1pXXXXq_U1pq(i1,j1,a,y,w,u)
        @ X.s01(v,w)
        @ X.s01(x,y)
        @ X.s01(z,a)
        )

def s01s01s01s01u101(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/24) * (-1)**(X.n_j0) * raw(
        #  X.ccccc0(i0,j0,p,t,v,x,z)
        #@ X.aaaaa1(i1,j1,a,y,w,u,q)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.s01(z,a)
        #@ X.u1_01(p,q)
          X.ccccc0pXXXX_U1p1(i0,j0,t,v,x,z,q)
        @ X.aaaaa1XXXuX_S0u(i1,j1,a,y,w,q,t)
        @ X.s01(v,w)
        @ X.s01(x,y)
        @ X.s01(z,a)
        )
