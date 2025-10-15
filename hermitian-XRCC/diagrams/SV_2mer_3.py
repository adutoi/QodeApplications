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



def s01s01s10v0100(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return 1 * raw(
        #  X.cccaaa0(i0,j0,p,t,v,y,s,r)
        #@ X.ccaa1(i1,j1,q,x,w,u)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s10(x,y)
        #@ X.v0100(p,q,r,s)
          X.cccaaa0pXXXsr_Vp1rs(i0,j0,t,v,y,q)
        @ X.ccaa1XXXu_S0u(i1,j1,q,x,w,t)
        @ X.s01(v,w)
        @ X.s10(x,y)
        )

def s01s01s10v1101(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return -1 * raw(
        #  X.ccaa0(i0,j0,t,v,y,r)
        #@ X.cccaaa1(i1,j1,p,q,x,w,u,s)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s10(x,y)
        #@ X.v1101(p,q,r,s)
          X.ccaa0tXXX_St1(i0,j0,v,y,r,u)
        @ X.cccaaa1pqXXXs_Vpq0s(i1,j1,x,w,u,r)
        @ X.s01(v,w)
        @ X.s10(x,y)
        )

def s01s01s01v1100(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/6) * (-1)**(X.n_j0) * raw(
        #  X.cccaa0(i0,j0,t,v,x,s,r)
        #@ X.ccaaa1(i1,j1,p,q,y,w,u)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.v1100(p,q,r,s)
          X.cccaa0XXXsr_V11rs(i0,j0,t,v,x,p,q)
        @ X.ccaaa1XXXXu_S0u(i1,j1,p,q,y,w,t)
        @ X.s01(v,w)
        @ X.s01(x,y)
        )

def s01s01s10v0000(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/2) * (-1)**(X.n_j0 + 1) * raw(
        #  X.ccccaaa0(i0,j0,p,q,t,v,y,s,r)
        #@ X.caa1(i1,j1,x,w,u)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s10(x,y)
        #@ X.v0000(p,q,r,s)
          X.ccccaaa0pqXXXsr_Vpqrs(i0,j0,t,v,y)
        @ X.caa1XXu_S0u(i1,j1,x,w,t)
        @ X.s01(v,w)
        @ X.s10(x,y)
        )

def s01s01s10v0101(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return 2 * (-1)**(X.n_j0 + 1) * raw(
        #  X.cccaa0(i0,j0,p,t,v,y,r)
        #@ X.ccaaa1(i1,j1,q,x,w,u,s)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s10(x,y)
        #@ X.v0101(p,q,r,s)
          X.cccaa0pXXXr_Vp1r1(i0,j0,t,v,y,q,s)
        @ X.ccaaa1XXXuX_S0u(i1,j1,q,x,w,s,t)
        @ X.s01(v,w)
        @ X.s10(x,y)
        )

def s01s01s10v1100(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/2) * (-1)**(X.n_j0 + 1) * raw(
        #  X.ccaaa0(i0,j0,t,v,y,s,r)
        #@ X.cccaa1(i1,j1,p,q,x,w,u)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s10(x,y)
        #@ X.v1100(p,q,r,s)
          X.ccaaa0XXXsr_V11rs(i0,j0,t,v,y,p,q)
        @ X.cccaa1XXXXu_S0u(i1,j1,p,q,x,w,t)
        @ X.s01(v,w)
        @ X.s10(x,y)
        )

def s01s01s10v1111(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/2) * (-1)**(X.n_j0 + 1) * raw(
        #  X.cca0(i0,j0,t,v,y)
        #@ X.cccaaaa1(i1,j1,p,q,x,w,u,s,r)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s10(x,y)
        #@ X.v1111(p,q,r,s)
          X.cca0tXX_St1(i0,j0,v,y,u)
        @ X.cccaaaa1pqXXXsr_Vpqrs(i1,j1,x,w,u)
        @ X.s01(v,w)
        @ X.s10(x,y)
        )

def s01s01s01v0100(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return -(1/3) * raw(
        #  X.ccccaa0(i0,j0,p,t,v,x,s,r)
        #@ X.caaa1(i1,j1,q,y,w,u)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.v0100(p,q,r,s)
          X.ccccaa0pXXXsr_Vp1rs(i0,j0,t,v,x,q)
        @ X.caaa1XXXu_S0u(i1,j1,q,y,w,t)
        @ X.s01(v,w)
        @ X.s01(x,y)
        )

def s01s01s01v1101(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/3) * raw(
        #  X.ccca0(i0,j0,t,v,x,r)
        #@ X.ccaaaa1(i1,j1,p,q,y,w,u,s)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.v1101(p,q,r,s)
          X.ccca0tXXX_St1(i0,j0,v,x,r,u)
        @ X.ccaaaa1pqXXXs_Vpq0s(i1,j1,y,w,u,r)
        @ X.s01(v,w)
        @ X.s01(x,y)
        )

def s01s01s10v0001(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return -1 * raw(
        #  X.ccccaa0(i0,j0,p,q,t,v,y,r)
        #@ X.caaa1(i1,j1,x,w,u,s)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s10(x,y)
        #@ X.v0001(p,q,r,s)
          X.ccccaa0pqXXXr_Vpqr1(i0,j0,t,v,y,s)
        @ X.caaa1XXuX_S0u(i1,j1,x,w,s,t)
        @ X.s01(v,w)
        @ X.s10(x,y)
        )

def s01s01s10v0111(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return 1 * raw(
        #  X.ccca0(i0,j0,p,t,v,y)
        #@ X.ccaaaa1(i1,j1,q,x,w,u,s,r)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s10(x,y)
        #@ X.v0111(p,q,r,s)
          X.ccca0XtXX_St1(i0,j0,p,v,y,u)
        @ X.ccaaaa1qXXXsr_V0qrs(i1,j1,x,w,u,p)
        @ X.s01(v,w)
        @ X.s10(x,y)
        )

def s01s01s01v0000(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/6) * (-1)**(X.n_j0) * raw(
        #  X.cccccaa0(i0,j0,p,q,t,v,x,s,r)
        #@ X.aaa1(i1,j1,y,w,u)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.v0000(p,q,r,s)
          X.cccccaa0pqXXXsr_Vpqrs(i0,j0,t,v,x)
        @ X.aaa1XXu_S0u(i1,j1,y,w,t)
        @ X.s01(v,w)
        @ X.s01(x,y)
        )

def s01s01s01v0101(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (2/3) * (-1)**(X.n_j0) * raw(
        #  X.cccca0(i0,j0,p,t,v,x,r)
        #@ X.caaaa1(i1,j1,q,y,w,u,s)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.v0101(p,q,r,s)
          X.cccca0pXXXr_Vp1r1(i0,j0,t,v,x,q,s)
        @ X.caaaa1XXXuX_S0u(i1,j1,q,y,w,s,t)
        @ X.s01(v,w)
        @ X.s01(x,y)
        )

def s01s01s01v1111(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/6) * (-1)**(X.n_j0) * raw(
        #  X.ccc0(i0,j0,t,v,x)
        #@ X.ccaaaaa1(i1,j1,p,q,y,w,u,s,r)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.v1111(p,q,r,s)
          X.ccc0tXX_St1(i0,j0,v,x,u)
        @ X.ccaaaaa1pqXXXsr_Vpqrs(i1,j1,y,w,u)
        @ X.s01(v,w)
        @ X.s01(x,y)
        )

def s01s01s10v0011(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/2) * (-1)**(X.n_j0 + 1) * raw(
        #  X.cccca0(i0,j0,p,q,t,v,y)
        #@ X.caaaa1(i1,j1,x,w,u,s,r)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s10(x,y)
        #@ X.v0011(p,q,r,s)
          X.cccca0pqXXX_Vpq11(i0,j0,t,v,y,r,s)
        @ X.caaaa1XXuXX_S0u(i1,j1,x,w,s,r,t)
        @ X.s01(v,w)
        @ X.s10(x,y)
        )

def s01s01s01v0001(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/3) * raw(
        #  X.ccccca0(i0,j0,p,q,t,v,x,r)
        #@ X.aaaa1(i1,j1,y,w,u,s)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.v0001(p,q,r,s)
          X.ccccca0pqXXXr_Vpqr1(i0,j0,t,v,x,s)
        @ X.aaaa1XXuX_S0u(i1,j1,y,w,s,t)
        @ X.s01(v,w)
        @ X.s01(x,y)
        )

def s01s01s01v0111(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return -(1/3) * raw(
        #  X.cccc0(i0,j0,p,t,v,x)
        #@ X.caaaaa1(i1,j1,q,y,w,u,s,r)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.v0111(p,q,r,s)
          X.cccc0XtXX_St1(i0,j0,p,v,x,u)
        @ X.caaaaa1qXXXsr_V0qrs(i1,j1,y,w,u,p)
        @ X.s01(v,w)
        @ X.s01(x,y)
        )

def s01s01s01v0011(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/6) * (-1)**(X.n_j0) * raw(
        #  X.ccccc0(i0,j0,p,q,t,v,x)
        #@ X.aaaaa1(i1,j1,y,w,u,s,r)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.v0011(p,q,r,s)
          X.ccccc0pqXXX_Vpq11(i0,j0,t,v,x,r,s)
        @ X.aaaaa1XXuXX_S0u(i1,j1,y,w,s,r,t)
        @ X.s01(v,w)
        @ X.s01(x,y)
        )
