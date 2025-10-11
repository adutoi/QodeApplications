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



def v0101(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return 4 * raw(
        #  X.ca0(i0,j0,p,r)
        #@ X.ca1(i1,j1,q,s)
        #@ X.v0101(p,q,r,s)
          X.ca0pr_Vp1r1(i0,j0,q,s)
        @ X.ca1(i1,j1,q,s)
        )

def v0001(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return 2 * (-1)**(X.n_j0 + 1) * raw(
        #  X.cca0(i0,j0,p,q,r)
        #@ X.a1(i1,j1,s)
        #@ X.v0001(p,q,r,s)
          X.cca0pqr_Vpqr1(i0,j0,s)
        @ X.a1(i1,j1,s)
        )

def v0100(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return 2 * (-1)**(X.n_j0) * raw(
        #  X.caa0(i0,j0,p,s,r)
        #@ X.c1(i1,j1,q)
        #@ X.v0100(p,q,r,s)
          X.caa0psr_Vp1rs(i0,j0,q)
        @ X.c1(i1,j1,q)
        )

def v0011(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return 1 * raw(
        #  X.cc0(i0,j0,p,q)
        #@ X.aa1(i1,j1,s,r)
        #@ X.v0011(p,q,r,s)
          X.cc0pq_Vpq11(i0,j0,r,s)
        @ X.aa1(i1,j1,s,r)
        )
