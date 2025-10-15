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

p, q, r, s, t, u, v, w, x, y, z, a = "pqrstuvwxyza"    # some contraction indices for easier reading



def s01s01s01s10v1100(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return -(1/6) * raw(
        #  X.cccaaa0(i0,j0,t,v,x,a,s,r)
        #@ X.cccaaa1(i1,j1,p,q,z,y,w,u)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.s10(z,a)
        #@ X.v1100(p,q,r,s)
          X.cccaaa0XXXXsr_V11rs(i0,j0,t,v,x,a,p,q)
        @ X.cccaaa1XXXXXu_S0u(i1,j1,p,q,z,y,w,t)
        @ X.s01(v,w)
        @ X.s01(x,y)
        @ X.s10(z,a)
        )

def s01s01s10s10v0000(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/4) * raw(
        #  X.ccccaaaa0(i0,j0,p,q,t,v,a,y,s,r)
        #@ X.ccaa1(i1,j1,x,z,w,u)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s10(x,y)
        #@ X.s10(z,a)
        #@ X.v0000(p,q,r,s)
          X.ccccaaaa0pqXXXXsr_Vpqrs(i0,j0,t,v,a,y)
        @ X.ccaa1XXXu_S0u(i1,j1,x,z,w,t)
        @ X.s01(v,w)
        @ X.s10(x,y)
        @ X.s10(z,a)
        )

def s01s01s10s10v0101(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return 1 * raw(
        #  X.cccaaa0(i0,j0,p,t,v,a,y,r)
        #@ X.cccaaa1(i1,j1,q,x,z,w,u,s)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s10(x,y)
        #@ X.s10(z,a)
        #@ X.v0101(p,q,r,s)
          X.cccaaa0pXXXXr_Vp1r1(i0,j0,t,v,a,y,q,s)
        @ X.cccaaa1XXXXuX_S0u(i1,j1,q,x,z,w,s,t)
        @ X.s01(v,w)
        @ X.s10(x,y)
        @ X.s10(z,a)
        )

def s01s01s01s10v0100(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/3) * (-1)**(X.n_j0 + 1) * raw(
        #  X.ccccaaa0(i0,j0,p,t,v,x,a,s,r)
        #@ X.ccaaa1(i1,j1,q,z,y,w,u)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.s10(z,a)
        #@ X.v0100(p,q,r,s)
          X.ccccaaa0pXXXXsr_Vp1rs(i0,j0,t,v,x,a,q)
        @ X.ccaaa1XXXXu_S0u(i1,j1,q,z,y,w,t)
        @ X.s01(v,w)
        @ X.s01(x,y)
        @ X.s10(z,a)
        )

def s01s01s01s10v1101(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/3) * (-1)**(X.n_j0) * raw(
        #  X.cccaa0(i0,j0,t,v,x,a,r)
        #@ X.cccaaaa1(i1,j1,p,q,z,y,w,u,s)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.s10(z,a)
        #@ X.v1101(p,q,r,s)
          X.cccaa0tXXXX_St1(i0,j0,v,x,a,r,u)
        @ X.cccaaaa1pqXXXXs_Vpq0s(i1,j1,z,y,w,u,r)
        @ X.s01(v,w)
        @ X.s01(x,y)
        @ X.s10(z,a)
        )

def s01s01s10s10v0001(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/2) * (-1)**(X.n_j0 + 1) * raw(
        #  X.ccccaaa0(i0,j0,p,q,t,v,a,y,r)
        #@ X.ccaaa1(i1,j1,x,z,w,u,s)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s10(x,y)
        #@ X.s10(z,a)
        #@ X.v0001(p,q,r,s)
          X.ccccaaa0pqXXXXr_Vpqr1(i0,j0,t,v,a,y,s)
        @ X.ccaaa1XXXuX_S0u(i1,j1,x,z,w,s,t)
        @ X.s01(v,w)
        @ X.s10(x,y)
        @ X.s10(z,a)
        )

def s01s01s10s10v0100(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/2) * (-1)**(X.n_j0) * raw(
        #  X.cccaaaa0(i0,j0,p,t,v,a,y,s,r)
        #@ X.cccaa1(i1,j1,q,x,z,w,u)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s10(x,y)
        #@ X.s10(z,a)
        #@ X.v0100(p,q,r,s)
          X.cccaaaa0pXXXXsr_Vp1rs(i0,j0,t,v,a,y,q)
        @ X.cccaa1XXXXu_S0u(i1,j1,q,x,z,w,t)
        @ X.s01(v,w)
        @ X.s10(x,y)
        @ X.s10(z,a)
        )

def s01s01s01s01v1100(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/24) * raw(
        #  X.ccccaa0(i0,j0,t,v,x,z,s,r)
        #@ X.ccaaaa1(i1,j1,p,q,a,y,w,u)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.s01(z,a)
        #@ X.v1100(p,q,r,s)
          X.ccccaa0XXXXsr_V11rs(i0,j0,t,v,x,z,p,q)
        @ X.ccaaaa1XXXXXu_S0u(i1,j1,p,q,a,y,w,t)
        @ X.s01(v,w)
        @ X.s01(x,y)
        @ X.s01(z,a)
        )

def s01s01s01s10v0000(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return -(1/6) * raw(
        #  X.cccccaaa0(i0,j0,p,q,t,v,x,a,s,r)
        #@ X.caaa1(i1,j1,z,y,w,u)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.s10(z,a)
        #@ X.v0000(p,q,r,s)
          X.cccccaaa0pqXXXXsr_Vpqrs(i0,j0,t,v,x,a)
        @ X.caaa1XXXu_S0u(i1,j1,z,y,w,t)
        @ X.s01(v,w)
        @ X.s01(x,y)
        @ X.s10(z,a)
        )

def s01s01s01s10v0101(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return -(2/3) * raw(
        #  X.ccccaa0(i0,j0,p,t,v,x,a,r)
        #@ X.ccaaaa1(i1,j1,q,z,y,w,u,s)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.s10(z,a)
        #@ X.v0101(p,q,r,s)
          X.ccccaa0pXXXXr_Vp1r1(i0,j0,t,v,x,a,q,s)
        @ X.ccaaaa1XXXXuX_S0u(i1,j1,q,z,y,w,s,t)
        @ X.s01(v,w)
        @ X.s01(x,y)
        @ X.s10(z,a)
        )

def s01s01s01s10v1111(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return -(1/6) * raw(
        #  X.ccca0(i0,j0,t,v,x,a)
        #@ X.cccaaaaa1(i1,j1,p,q,z,y,w,u,s,r)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.s10(z,a)
        #@ X.v1111(p,q,r,s)
          X.ccca0tXXX_St1(i0,j0,v,x,a,u)
        @ X.cccaaaaa1pqXXXXsr_Vpqrs(i1,j1,z,y,w,u)
        @ X.s01(v,w)
        @ X.s01(x,y)
        @ X.s10(z,a)
        )

def s01s01s10s10v0011(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/4) * raw(
        #  X.ccccaa0(i0,j0,p,q,t,v,a,y)
        #@ X.ccaaaa1(i1,j1,x,z,w,u,s,r)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s10(x,y)
        #@ X.s10(z,a)
        #@ X.v0011(p,q,r,s)
          X.ccccaa0pqXXXX_Vpq11(i0,j0,t,v,a,y,r,s)
        @ X.ccaaaa1XXXuXX_S0u(i1,j1,x,z,w,s,r,t)
        @ X.s01(v,w)
        @ X.s10(x,y)
        @ X.s10(z,a)
        )

def s01s01s01s01v0100(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/12) * (-1)**(X.n_j0) * raw(
        #  X.cccccaa0(i0,j0,p,t,v,x,z,s,r)
        #@ X.caaaa1(i1,j1,q,a,y,w,u)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.s01(z,a)
        #@ X.v0100(p,q,r,s)
          X.cccccaa0pXXXXsr_Vp1rs(i0,j0,t,v,x,z,q)
        @ X.caaaa1XXXXu_S0u(i1,j1,q,a,y,w,t)
        @ X.s01(v,w)
        @ X.s01(x,y)
        @ X.s01(z,a)
        )

def s01s01s01s01v1101(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/12) * (-1)**(X.n_j0 + 1) * raw(
        #  X.cccca0(i0,j0,t,v,x,z,r)
        #@ X.ccaaaaa1(i1,j1,p,q,a,y,w,u,s)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.s01(z,a)
        #@ X.v1101(p,q,r,s)
          X.cccca0tXXXX_St1(i0,j0,v,x,z,r,u)
        @ X.ccaaaaa1pqXXXXs_Vpq0s(i1,j1,a,y,w,u,r)
        @ X.s01(v,w)
        @ X.s01(x,y)
        @ X.s01(z,a)
        )

def s01s01s01s10v0001(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/3) * (-1)**(X.n_j0) * raw(
        #  X.cccccaa0(i0,j0,p,q,t,v,x,a,r)
        #@ X.caaaa1(i1,j1,z,y,w,u,s)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.s10(z,a)
        #@ X.v0001(p,q,r,s)
          X.cccccaa0pqXXXXr_Vpqr1(i0,j0,t,v,x,a,s)
        @ X.caaaa1XXXuX_S0u(i1,j1,z,y,w,s,t)
        @ X.s01(v,w)
        @ X.s01(x,y)
        @ X.s10(z,a)
        )

def s01s01s01s10v0111(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/3) * (-1)**(X.n_j0 + 1) * raw(
        #  X.cccca0(i0,j0,p,t,v,x,a)
        #@ X.ccaaaaa1(i1,j1,q,z,y,w,u,s,r)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.s10(z,a)
        #@ X.v0111(p,q,r,s)
          X.cccca0XtXXX_St1(i0,j0,p,v,x,a,u)
        @ X.ccaaaaa1qXXXXsr_V0qrs(i1,j1,z,y,w,u,p)
        @ X.s01(v,w)
        @ X.s01(x,y)
        @ X.s10(z,a)
        )

def s01s01s01s01v0000(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/24) * raw(
        #  X.ccccccaa0(i0,j0,p,q,t,v,x,z,s,r)
        #@ X.aaaa1(i1,j1,a,y,w,u)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.s01(z,a)
        #@ X.v0000(p,q,r,s)
          X.ccccccaa0pqXXXXsr_Vpqrs(i0,j0,t,v,x,z)
        @ X.aaaa1XXXu_S0u(i1,j1,a,y,w,t)
        @ X.s01(v,w)
        @ X.s01(x,y)
        @ X.s01(z,a)
        )

def s01s01s01s01v0101(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/6) * raw(
        #  X.ccccca0(i0,j0,p,t,v,x,z,r)
        #@ X.caaaaa1(i1,j1,q,a,y,w,u,s)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.s01(z,a)
        #@ X.v0101(p,q,r,s)
          X.ccccca0pXXXXr_Vp1r1(i0,j0,t,v,x,z,q,s)
        @ X.caaaaa1XXXXuX_S0u(i1,j1,q,a,y,w,s,t)
        @ X.s01(v,w)
        @ X.s01(x,y)
        @ X.s01(z,a)
        )

def s01s01s01s01v1111(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/24) * raw(
        #  X.cccc0(i0,j0,t,v,x,z)
        #@ X.ccaaaaaa1(i1,j1,p,q,a,y,w,u,s,r)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.s01(z,a)
        #@ X.v1111(p,q,r,s)
          X.cccc0tXXX_St1(i0,j0,v,x,z,u)
        @ X.ccaaaaaa1pqXXXXsr_Vpqrs(i1,j1,a,y,w,u)
        @ X.s01(v,w)
        @ X.s01(x,y)
        @ X.s01(z,a)
        )

def s01s01s01s10v0011(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return -(1/6) * raw(
        #  X.ccccca0(i0,j0,p,q,t,v,x,a)
        #@ X.caaaaa1(i1,j1,z,y,w,u,s,r)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.s10(z,a)
        #@ X.v0011(p,q,r,s)
          X.ccccca0pqXXXX_Vpq11(i0,j0,t,v,x,a,r,s)
        @ X.caaaaa1XXXuXX_S0u(i1,j1,z,y,w,s,r,t)
        @ X.s01(v,w)
        @ X.s01(x,y)
        @ X.s10(z,a)
        )

def s01s01s01s01v0001(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/12) * (-1)**(X.n_j0 + 1) * raw(
        #  X.cccccca0(i0,j0,p,q,t,v,x,z,r)
        #@ X.aaaaa1(i1,j1,a,y,w,u,s)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.s01(z,a)
        #@ X.v0001(p,q,r,s)
          X.cccccca0pqXXXXr_Vpqr1(i0,j0,t,v,x,z,s)
        @ X.aaaaa1XXXuX_S0u(i1,j1,a,y,w,s,t)
        @ X.s01(v,w)
        @ X.s01(x,y)
        @ X.s01(z,a)
        )

def s01s01s01s01v0111(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/12) * (-1)**(X.n_j0) * raw(
        #  X.ccccc0(i0,j0,p,t,v,x,z)
        #@ X.caaaaaa1(i1,j1,q,a,y,w,u,s,r)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.s01(z,a)
        #@ X.v0111(p,q,r,s)
          X.ccccc0XtXXX_St1(i0,j0,p,v,x,z,u)
        @ X.caaaaaa1qXXXXsr_V0qrs(i1,j1,a,y,w,u,p)
        @ X.s01(v,w)
        @ X.s01(x,y)
        @ X.s01(z,a)
        )

def s01s01s01s01v0011(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/24) * raw(
        #  X.cccccc0(i0,j0,p,q,t,v,x,z)
        #@ X.aaaaaa1(i1,j1,a,y,w,u,s,r)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.s01(x,y)
        #@ X.s01(z,a)
        #@ X.v0011(p,q,r,s)
          X.cccccc0pqXXXX_Vpq11(i0,j0,t,v,x,z,r,s)
        @ X.aaaaaa1XXXuXX_S0u(i1,j1,a,y,w,s,r,t)
        @ X.s01(v,w)
        @ X.s01(x,y)
        @ X.s01(z,a)
        )
