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



def s01s01s10s10(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/4) * raw(
        #  X.ccaa0(i0,j0,p,r,w,u)
        #@ X.ccaa1(i1,j1,t,v,s,q)
        #@ X.s01(p,q)
        #@ X.s01(r,s)
        #@ X.s10(t,u)
        #@ X.s10(v,w)
          X.ccaa0pXXX_Sp1(i0,j0,r,w,u,q)
        @ X.ccaa1XXsX_S0s(i1,j1,t,v,q,r)
        @ X.s10(t,u)
        @ X.s10(v,w)
        )

def s01s01s01s10(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return -(1/6) * raw(
        #  X.ccca0(i0,j0,p,r,t,w)
        #@ X.caaa1(i1,j1,v,u,s,q)
        #@ X.s01(p,q)
        #@ X.s01(r,s)
        #@ X.s01(t,u)
        #@ X.s10(v,w)
          X.ccca0pXXX_Sp1(i0,j0,r,t,w,q)
        @ X.caaa1XXsX_S0s(i1,j1,v,u,q,r)
        @ X.s01(t,u)
        @ X.s10(v,w)
        )

def s01s01s01s01(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/24) * raw(
        #  X.cccc0(i0,j0,p,r,t,v)
        #@ X.aaaa1(i1,j1,w,u,s,q)
        #@ X.s01(p,q)
        #@ X.s01(r,s)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
          X.cccc0pXXX_Sp1(i0,j0,r,t,v,q)
        @ X.aaaa1XXsX_S0s(i1,j1,w,u,q,r)
        @ X.s01(t,u)
        @ X.s01(v,w)
        )
