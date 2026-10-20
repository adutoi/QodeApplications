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


"""
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
"""
#import numpy as np

def v0101(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return 2 * raw(
        #  X.ca0(i0,j0,p,r)
        #@ X.ca1(i1,j1,q,s)
        #@ X.v0101(p,q,r,s)
          X.ca0pr_Vp1r1(i0,j0,q,s)
        @ X.ca1(i1,j1,q,s)
        )
    #return np.zeros_like(ret)

# exchange-type split of v (p,s on fragment 0, q,r on fragment 1): unlike in v0101 the c and a
# of a fragment belong to different spin summations, so rank-1 ca tensors contribute as well
# (spin weights 1/2 and -sqrt(3)/2, see spin_structure.py)

def v0110(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return -2 * raw(
        #  X.ca0(i0,j0,p,s)
        #@ X.ca1(i1,j1,q,r)
        #@ X.v0101(p,q,r,s)
          X.ca0ps_Vp11s(i0,j0,q,r)  # for RI decomp hummr dens antisymm doesn't hold
        @ X.ca1(i1,j1,q,r)
        )
    #return np.zeros_like(ret)

def v0001(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    # The two fragment assignments (s on frag 1 and r on frag 1) contribute equally due to
    # v_pqrs = v_qpsr.  hummr's cca_pqr has the rank-0 restricted pair (q,r), which is the
    # operator-local pair of the r-on-frag-1 assignment, so only that one is evaluated (the
    # s-on-frag-1 assignment would need the pair (p,r)).
    return -2 * (-1)**(X.n_j0 + 1) * raw(
        #  X.cca0(i0,j0,p,q,s)
        #@ X.a1(i1,j1,r)
        #@ X.v0010(p,q,r,s)
          X.cca0pqs_Vpq1s(i0,j0,r)
        @ X.a1(i1,j1,r)
        )
    #return np.zeros_like(ret)

def v0100(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    # The two fragment assignments (q on frag 1 and p on frag 1) contribute equally due to
    # v_pqrs = v_qpsr.  hummr's caa_pqr has the rank-0 restricted pair (p,q), which is the
    # operator-local pair of the p-on-frag-1 assignment, so only that one is evaluated (the
    # q-on-frag-1 assignment would need the pair (p,r) of caa_psr).
    return -2 * (-1)**(X.n_j0) * raw(
        #  X.caa0(i0,j0,q,s,r)
        #@ X.c1(i1,j1,p)
        #@ X.v1000(p,q,r,s)
          X.caa0qsr_V1qrs(i0,j0,p)
        @ X.c1(i1,j1,p)
        )
    #return np.zeros_like(ret)

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
    #return np.zeros_like(ret)
