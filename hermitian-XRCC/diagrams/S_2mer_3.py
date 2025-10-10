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



def s01s01s10(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/2) * (-1)**(X.n_j0 + X.P + 1) * raw(
          X.cca0(i0,j0,p,r,u)
        @ X.caa1(i1,j1,t,s,q)
        @ X.s01(p,q)
        @ X.s01(r,s)
        @ X.s10(t,u)
        )

def s01s01s01(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (1/6) * (-1)**(X.n_j0 + X.P) * raw(
          X.ccc0(i0,j0,p,r,t)
        @ X.aaa1(i1,j1,u,s,q)
        @ X.s01(p,q)
        @ X.s01(r,s)
        @ X.s01(t,u)
        )
