#    (C) Copyright 2024 Marco Bauer
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

p, q, r, s, i0, i1, j0, j1 = "p", "q", "r", "s", "i", "j", "k", "l"

import numpy as np
from qode.math.tensornet import raw, tl_tensor, evaluate, backend_contract_path
from qode.math import precise_numpy_inverse
import tensorly as tl
import time

tl.plugins.use_opt_einsum()
backend_contract_path(True)

def get_prefactor_exponent(n_occ, subsys_chgs):
    return (n_occ[1] - subsys_chgs[1][0])%2

###################################################
# one particle XR density
###################################################

def one_p_diagonal(dl, dr, dens, perm):  # 00
    #return 1 * scalar_value( X.ca0[i0,j0](p,q) @ X.t00(p,q) )
    #return 1 * raw(dl(i0, i1) @ dens(i0, j0, 0, 1) @ delta(i1, i1) @ dr(j0, i1))
    int1 = evaluate(dl(0, i1) @ dr(1, i1))  # i0, j0
    #return 1 * raw(dl(i0, i1) @ dens(i0, j0, 0, 1) @ dr(j0, i1))
    return 1 * raw(int1(i0, j0) @ dens(i0, j0, 0, 1))

#def one_p_diagonal_11(dl, dr, dens, perm):  # 11
    #return 1 * scalar_value( X.ca0[i0,j0](p,q) @ X.t00(p,q) )
    #return 1 * raw(dl(i0, i1) @ delta(i0, i0) @ dens(i1, j1, 0, 1) @ dr(i0, j1))
#    return 1 * raw(dl(i0, i1) @ dens(i1, j1, 0, 1) @ dr(i0, j1))

def one_p_off_diagonal(dl, dr, dens0, dens1, n_i1, perm):  # 01
    #return (-1)**(n_i1 + perm) * scalar_value( X.c0[i0,j0](p) @ X.a1[i1,j1](q) @ X.t01(p,q) )
    int1 = evaluate(dl(i0, 1) @ dens0(i0, j0, 0) @ dr(j0, 2))  # 0, i1, j1
    #return (-1)**(n_i1 + perm) * raw(dl(i0, i1) @ dens0(i0, j0, 0) @ dens1(i1, j1, 1) @ dr(j0, j1))
    return (-1)**(n_i1 + perm) * raw(int1(0, i1, j1) @ dens1(i1, j1, 1))

###################################################
# V diagrams
###################################################
# these are explicit, to prevent building the full 2p XR density
# note that the prefactors are different too sometimes, due to additional
# terms from the derivatives

def v0000(dl, dr, v_bior, v_halfbior1, v_halfbior2, s_inv_2, dens, perm, n_occ, sl0, sl1):
    int1 = evaluate(dl(0, i1) @ dr(1, i1))  # i0, j0
    int2 = evaluate(dens(i0, j0, 0, 1, 2, 3) @ int1(i0, j0))  # p, q, s, r
    #return 1 * raw( X.ccaa0pqsr_Vpqrs )
    #return 1 * scalar_value( X.ccaa0pqsr_Vpqrs[i0,j0] )
    #return 1 * scalar_value( X.ccaa0[i0,j0](p,q,s,r) @ X.v0000(p,q,r,s) )
    return 1 * raw(v_bior[:,:,sl1,:](p, q, 1, s) @ int2[:,:,sl0,:](p, q, s, 0)
                 + v_bior[:,:,:,sl1](p, q, r, 1) @ int2[:,:,:,sl0](p, q, 0, r)
                 - s_inv_2[:,sl1](p, 1) @ v_halfbior2[sl0,:,:,:](0, q, r, s) @ int2(p, q, s, r)
                 - s_inv_2[:,sl1](q, 1) @ v_halfbior1[:,sl0,:,:](p, 0, r, s) @ int2(p, q, s, r))

def v0000_s(dl, dr, v_bior, v_halfbior1, v_halfbior2, s_inv_2, dens, perm, n_occ, sl0, sl1):
    int1 = evaluate(dl(0, i1) @ dr(1, i1))  # i0, j0
    int2 = evaluate(dens(i0, j0, 0, 1, 2, 3) @ int1(i0, j0))  # p, q, r, s
    #return 1 * raw( X.ccaa0pqsr_Vpqrs )
    #return 1 * scalar_value( X.ccaa0pqsr_Vpqrs[i0,j0] )
    #return 1 * scalar_value( X.ccaa0[i0,j0](p,q,s,r) @ X.v0000(p,q,r,s) )
    return -1 * raw(s_inv_2[:,sl1](p, 1) @ v_halfbior2[sl0,:,:,:](0, q, r, s) @ int2(p, q, s, r)
                  + s_inv_2[:,sl1](q, 1) @ v_halfbior1[:,sl0,:,:](p, 0, r, s) @ int2(p, q, s, r))

#def v1111(dl, dr, v_bior, v_halfbior, s_inv_2, dens, perm):
#    return 2 * raw(dl(i0, i1) @ v_bior(p, q, 1, s) @ dens(i1, j1, p, q, 0, s) @ dr(i0, j1)
#                   + dl(i0, i1) @ s_inv_2(p, 0) @ v_halfbior(1, q, r, s) @ dens(i1, j1, p, q, r, s) @ dr(i0, j1))

# pr,qs,pqrs-> :  ca0  ca1  v0101
def v0101(dl, dr, v_bior, v_halfbior1, v_halfbior2, s_inv_2, dens0, dens1, n_i1, perm, n_occ, sl0, sl1):
    #return 4 * raw( X.ca1(i1,j1,q,s) @ X.ca0pr_Vp1r1(i0,j0,q,s) )
    #return 4 * scalar_value( X.ca1[i1,j1](q,s) @ X.ca0pr_Vp1r1[i0,j0](q,s) )
    #return 4 * scalar_value( X.ca0[i0,j0](p,r) @ X.ca1[i1,j1](q,s) @ X.v0101(p,q,r,s) )
    #int1 = evaluate(dl(i0, 0) @ dens0[:,:,:,sl0](i0, j0, 2, 3) @ dr(j0, 1))  # i1, j1, p, 0
    int1 = evaluate(dl(0, i1) @ dens1(i1, j1, 2, 3) @ dr(1, j1))  # i0, j0, q, s  # this is also independent of the comb loop
    #int2 = evaluate(int1(i0, j0, 0, 1) @ dens0[:,:,:,sl0](i0, j0, 2, 3))  # q, s, p, 0
    #int3 = evaluate(dl(0, i1) @ dens1(i1, j1, 2, 3) @ dr(1, j1))  # i0, j0, q, s  # better also use int 3 above
    int4 = evaluate(int1(i0, j0, 0, 1) @ dens0(i0, j0, 2, 3))  # q, s, p, r
    #return 4 * raw(dl(i0, i1) @ v_bior[:,:,sl1,:](p, q, 1, s) @ dens0[:,:,:,sl0](i0, j0, p, 0) @ dens1(i1, j1, q, s) @ dr(j0, j1)
    #                 + dl(i0, i1) @ s_inv_2[:,sl0](p, 0) @ v_halfbior[sl1,:,:,:](1, q, r, s) @ dens0(i0, j0, p, r) @ dens1(i1, j1, q, s) @ dr(j0, j1))
    return 2 * raw(2 * v_bior[:,:,sl1,:](p, q, 1, s) @ int4[:,:,:,sl0](q, s, p, 0)  # int2(q, s, p, 0)
                   #- v_bior[:,:,sl1,:](p, q, 1, r) @ int4[:,:,:,sl0](p, r, q, 0)
                     - (s_inv_2[:,sl1](p, 1) @ v_halfbior2[sl0,:,:,:](0, q, r, s) @ int4(q, s, p, r)))
                              #- s_inv_2[:,sl1](q, 1) @ v_halfbior[sl0,:,:,:](0, p, r, s) @ int4(q, s, p, r))

def v1010(dl, dr, v_bior, v_halfbior1, v_halfbior2, s_inv_2, dens0, dens1, n_i1, perm, n_occ, sl0, sl1):
    int1 = evaluate(dl(0, i1) @ dens1(i1, j1, 2, 3) @ dr(1, j1))  # i0, j0, p, r  # this is also independent of the comb loop
    int4 = evaluate(int1(i0, j0, 0, 1) @ dens0(i0, j0, 2, 3))  # p, r, q, s
    return -2 * raw(#- v_bior[:,:,:,sl1](p, q, r, 1) @ int4[:,:,:,sl0](p, r, q, 0)
                    s_inv_2[:,sl1](q, 1) @ v_halfbior1[:,sl0,:,:](p, 0, r, s) @ int4(p, r, q, s))

# pqs,r,pqrs-> :  cca0  a1  v0010
def v0010(dl, dr, v_bior, v_halfbior, s_inv_2, dens0, dens1, n_i1, perm, n_occ, sl0, sl1):
    #return 2 * (-1)**(X.n_i1 + X.P) * raw( X.a1(i1,j1,r) @ X.cca0pqs_Vpq1s(i0,j0,r) )
    #return 2 * (-1)**(X.n_i1 + X.P) * scalar_value( X.a1[i1,j1](r) @ X.cca0pqs_Vpq1s[i0,j0](r) )
    #return 2 * (-1)**(X.n_i1 + X.P) * scalar_value( X.cca0[i0,j0](p,q,s) @ X.a1[i1,j1](r) @ X.v0010(p,q,r,s) )
    int1 = evaluate(dl(0, i1) @ dens1(i1, j1, 2) @ dr(1, j1))  # i0, j0, r
    #int2 = evaluate(int1(i0, j0, 0) @ dens0[:,:,:,:,sl0](i0, j0, 1, 2, 3))  # r, q, p, 0
    int3 = evaluate(dens0(i0, j0, 0, 1, 2) @ int1(i0, j0, 3))  # p, q, s, r
    #return 2 * (-1)**(n_i1 + perm) * raw(
    #    # note that in the first of the following two lines v_bior is used with (p, q, r, 1) instead of (p, q, 1, s),
    #    # because this way the target fragment for both terms is the same
    #    dl(i0, i1) @ v_bior[:,:,:,sl1](p, q, r, 1) @ dens0[:,:,:,:,sl0](i0, j0, q, p, 0) @ dens1(i1, j1, r) @ dr(j0, j1)
    #  + 2 * dl(i0, i1) @ s_inv_2[:,sl0](p, 0) @ v_halfbior[sl1,:,:,:](1, q, r, s) @ dens0(i0, j0, p, q, s) @ dens1(i1, j1, r) @ dr(j0, j1)
    #)
    return 2 * (-1)**(n_i1 + perm) * raw(
        # note that in the first of the following two lines v_bior is used with (p, q, r, 1) instead of (p, q, 1, s),
        # because this way the target fragment for both terms is the same
        v_bior[:,:,:,sl1](p, q, r, 1) @ int3[:,:,sl0,:](q, p, 0, r)  #int2(r, q, p, 0)
      - 2 * s_inv_2[:,sl1](p, 1) @ v_halfbior[sl0,:,:,:](0, q, r, s) @ int3(p, q, s, r)
    )

# p,qsr,pqrs-> :  c0  caa1  v0111
#def v0111(dl, dr, v_bior, v_halfbior, s_inv_2, dens0, dens1, n_i1, perm, n_occ):
    #return 2 * (-1)**(X.n_i1 + X.P) * raw( X.c0(i0,j0,p) @ X.caa1qsr_V0qrs(i1,j1,p) )
    #return 2 * (-1)**(X.n_i1 + X.P) * scalar_value( X.c0[i0,j0](p) @ X.caa1qsr_V0qrs[i1,j1](p) )
    #return 2 * (-1)**(X.n_i1 + X.P) * scalar_value( X.c0[i0,j0](p) @ X.caa1[i1,j1](q,s,r) @ X.v0111(p,q,r,s) )
#    return 2 * 2 * (-1)**(n_i1 + perm) * raw(
#        dl(i0, i1) @ v_bior[:,:,sl1,:](p, q, 1, s) @ dens0(i0, j0, p) @ dens1[:,:,:,:,sl0](i1, j1, q, s, 0) @ dr(j0, j1)
#      + dl(i0, i1) @ s_inv_2[:,sl0](p, 0) @ v_halfbior[sl1,:,:,:](1, q, r, s) @ dens0(i0, j0, p) @ dens1(i1, j1, q, s, r) @ dr(j0, j1)
#    )

def v0111(dl, dr, v_bior, v_halfbior, s_inv_2, dens0, dens1, n_i1, perm, n_occ, sl0, sl1):
    int1 = evaluate(v_halfbior[sl0,:,:,:](0, q, r, s) @ dens1(1, 2, q, s, r))  # 1, i1, j1
    #return 2 * (-1)**(n_i1 + perm) * raw(
    #    dl(i0, i1) @ s_inv_2[:,sl0](p, 0) @ v_halfbior[sl1,:,:,:](1, q, r, s) @ dens0(i0, j0, p) @ dens1(i1, j1, q, s, r) @ dr(j0, j1)
    #)
    return 2 * (-1)**(n_i1 + perm) * raw(
        dl(i0, i1) @ s_inv_2[:,sl1](p, 1) @ dens0(i0, j0, p) @ int1(0, i1, j1) @ dr(j0, j1)
    )

def v1000(dl, dr, v_bior, v_halfbior, s_inv_2, dens0, dens1, n_i1, perm, n_occ, sl0, sl1):
    int1 = evaluate(dl(0, i1) @ dens1(i1, j1, 2) @ dr(1, j1))  # i0, j0, p
    int2 = evaluate(int1(i0, j0, 0) @ dens0[:,:,:,:,sl0](i0, j0, 1, 2, 3))  # p, q, s, 0
    # for this term to have fragment 0 as target index the permutation (frag0<-->frag1) needs to be applied
    #return -2 * 2 * (-1)**(n_i1 + perm) * raw(
    #    dl(i0, i1) @ v_bior[:,:,sl1,:](p, q, 1, s) @ dens1(i1, j1, p) @ dens0[:,:,:,:,sl0](i0, j0, q, s, 0) @ dr(j0, j1)
    #)
    return -2 * 2 * (-1)**(n_i1 + perm) * raw(
        v_bior[:,:,sl1,:](p, q, 1, s) @ int2(p, q, s, 0)
    )

# pq,sr,pqrs-> :  cc0  aa1  v0011
#def v0011(dl, dr, v_bior, v_halfbior, s_inv_2, dens0, dens1, n_i1, perm, n_occ):
    #return 1 * raw( X.aa1(i1,j1,s,r) @ X.cc0pq_Vpq11(i0,j0,r,s) )
    #return 1 * scalar_value( X.aa1[i1,j1](s,r) @ X.cc0pq_Vpq11[i0,j0](r,s) )
    #return 1 * scalar_value( X.cc0[i0,j0](p,q) @ X.aa1[i1,j1](s,r) @ X.v0011(p,q,r,s) )
#    return 2 * raw(dl(i0, i1) @ v_bior[:,:,sl1,:](p, q, 1, s) @ dens0(i0, j0, p, q) @ dens1[:,:,:,sl0](i1, j1, s, 0) @ dr(j0, j1)
#                 + dl(i0, i1) @ s_inv_2[:,sl0](p, 0) @ v_halfbior[sl1,:,:,:](1, q, r, s) @ dens0(i0, j0, p, q) @ dens1(i1, j1, s, r) @ dr(j0, j1))

def v0011(dl, dr, v_bior, v_halfbior, s_inv_2, dens0, dens1, n_i1, perm, n_occ, sl0, sl1):
    int1 = evaluate(dl(0, i1) @ dens1(i1, j1, 2, 3) @ dr(1, j1))  # i0, j0, s, r
    int2 = evaluate(int1(i0, j0, 0, 1) @ dens0(i0, j0, 2, 3))  # s, r, p, q
    #return 2 * raw(dl(i0, i1) @ s_inv_2[:,sl0](p, 0) @ v_halfbior[sl1,:,:,:](1, q, r, s) @ dens0(i0, j0, p, q) @ dens1(i1, j1, s, r) @ dr(j0, j1))
    return -2 * raw(s_inv_2[:,sl1](p, 1) @ v_halfbior[sl0,:,:,:](0, q, r, s) @ int2(s, r, p, q))

def v1100(dl, dr, v_bior, v_halfbior, s_inv_2, dens0, dens1, n_i1, perm, n_occ, sl0, sl1):
    int1 = evaluate(dl(i0, 0) @ dens0[:,:,:,sl0](i0, j0, 2, 3) @ dr(j0, 1))  # i1, j1, s, 0
    int2 = evaluate(int1(i1, j1, 0, 1) @ dens1(i1, j1, 2, 3))  # s, 0, p, q
    # for this term to have fragment 0 as target index the permutation (frag0<-->frag1) needs to be applied
    #return 2 * raw(dl(i0, i1) @ v_bior[:,:,sl1,:](p, q, 1, s) @ dens1(i1, j1, p, q) @ dens0[:,:,:,sl0](i0, j0, s, 0) @ dr(j0, j1))
    return 2 * raw(v_bior[:,:,sl1,:](p, q, 1, s) @ int2(s, 0, p, q))

###################################################
# V diagrams for the hessian
###################################################

def v0000_hess(dl, dr, v_bior, v_halfbior1, v_halfbior2, v_symm, s_inv_2, dens, perm, n_occ, sl0, sl1, sl2, sl3):
    int1 = evaluate(dl(0, i1) @ dr(1, i1))  # i0, j0
    int2 = evaluate(int1(i0, j0) @ dens(i0, j0, 0, 1, 2, 3))  # p, q, s, r
    #return 2 * raw(dl(i0, i1) @ v_bior[:,:,sl1,sl3](p, q, 1, 3) @ dens[:,:,:,:,sl0,sl2](i0, j0, p, q, 0, 2) @ dr(j0, i1)
    #             + 2 * dl(i0, i1) @ s_inv_2[:,sl0](p, 0) @ v_halfbior[sl1,:,sl3,:](1, q, 3, s) @ dens[:,:,:,:,sl2,:](i0, j0, p, q, 2, s) @ dr(j0, i1)
    #             + dl(i0, i1) @ s_inv_2[:,sl0](p, 0) @ s_inv_2[:,sl2](q,2) @ v_symm[sl1,sl3,:,:](1, 3, r, s) @ dens(i0, j0, p, q, r, s) @ dr(j0, i1))
    return 1 * raw(v_bior[:,:,sl1,sl3](p, q, 1, 3) @ int2[:,:,sl0,sl2](p, q, 2, 0)
                 - s_inv_2[:,sl1](p, 1) @ v_halfbior2[sl0,:,sl3,:](0, q, 3, s) @ int2[:,:,sl2,:](p, q, s, 2)
                 - s_inv_2[:,sl1](q, 1) @ v_halfbior1[:,sl0,sl3,:](p, 0, 3, s) @ int2[:,:,sl2,:](p, q, s, 2)
                 + s_inv_2[:,sl1](p, 1) @ s_inv_2[:,sl3](q,3) @ v_symm[sl0,sl2,:,:](0, 2, r, s) @ int2(p, q, s, r))

def v0101_hess(dl, dr, v_bior, v_halfbior1, v_halfbior2, v_symm, s_inv_2, dens0, dens1, n_i1, perm, n_occ, sl0, sl1, sl2, sl3):
    int1 = evaluate(v_halfbior2[sl0,:,sl3,:](0, q, 1, s) @ dens1(2, 3, q, s))  # 1, 3, i1, j1
    int2 = evaluate(dl(0, i1) @ dr(1, j1) @ int1(2, 3, i1, j1))  # i0, j0, 1, 3
    #return 4 * raw(dl(i0, i1) @ s_inv_2[:,sl0](p, 0) @ v_halfbior[sl1,:,sl3,:](1, q, 3, s) @ dens0[:,:,:,sl2](i0, j0, p, 2) @ dens1(i1, j1, q, s) @ dr(j0, j1))
    return -2 * raw(s_inv_2[:,sl1](p, 1) @ dens0[:,:,:,sl2](i0, j0, p, 2) @ int2(i0, j0, 0, 3))

def v1010_hess(dl, dr, v_bior, v_halfbior1, v_halfbior2, v_symm, s_inv_2, dens0, dens1, n_i1, perm, n_occ, sl0, sl1, sl2, sl3):
    int1 = evaluate(v_halfbior1[:,sl0,:,sl3](p, 0, r, 1) @ dens1(2, 3, p, r))  # 1, 3, i1, j1
    int2 = evaluate(dl(0, i1) @ dr(1, j1) @ int1(2, 3, i1, j1))  # i0, j0, 1, 3
    #return 4 * raw(dl(i0, i1) @ s_inv_2[:,sl0](p, 0) @ v_halfbior[sl1,:,sl3,:](1, q, 3, s) @ dens0[:,:,:,sl2](i0, j0, p, 2) @ dens1(i1, j1, q, s) @ dr(j0, j1))
    return -2 * raw(s_inv_2[:,sl1](q, 1) @ dens0[:,:,:,sl2](i0, j0, q, 2) @ int2(i0, j0, 0, 3))

def v0010_hess(dl, dr, v_bior, v_halfbior, v_symm, s_inv_2, dens0, dens1, n_i1, perm, n_occ, sl0, sl1, sl2, sl3):
    int1 = evaluate(dl(0, i1) @ dens1(i1, j1, 2) @ dr(1, j1))  # i0, j0, r
    #int2 = evaluate(int1(i0, j0, 0) @ dens0[:,:,:,:,sl2](i0, j0, 1, 2, 3))  # r, q, p, 2
    int3 = evaluate(int1(i0, j0, 0) @ dens0(i0, j0, 1, 2, 3))  # r, p, q, s
    #return 2 * 2 * (-1)**(n_i1 + perm) * raw(
    #    # note that in the first of the following two lines v_bior is used with (p, q, r, 1) instead of (p, q, 1, s),
    #    # because this way the target fragment for both terms is the same
    #    dl(i0, i1) @ s_inv_2[:,sl0](p, 0) @ v_halfbior[sl1,:,:,sl3](1, q, r, 3) @ dens0[:,:,:,:,sl2](i0, j0, q, p, 2) @ dens1(i1, j1, r) @ dr(j0, j1)
    #  + dl(i0, i1) @ s_inv_2[:,sl0](p, 0) @ s_inv_2[:,sl2](q,2) @ v_symm[sl1,sl3,:,:](1, 3, r, s) @ dens0(i0, j0, p, q, s) @ dens1(i1, j1, r) @ dr(j0, j1)
    #)
    return -2 * 2 * (-1)**(n_i1 + perm) * raw(
        # note that in the first of the following two lines v_bior is used with (p, q, r, 1) instead of (p, q, 1, s),
        # because this way the target fragment for both terms is the same
        s_inv_2[:,sl1](p, 1) @ v_halfbior[sl0,:,:,sl3](0, q, r, 3) @ int3[:,:,:,sl2](r, q, p, 2)  #int2(r, q, p, 2)
      - s_inv_2[:,sl1](p, 1) @ s_inv_2[:,sl3](q,3) @ v_symm[sl0,sl2,:,:](0, 2, r, s) @ int3(r, p, q, s)
    )

#def v0111_hess(dl, dr, v_bior, v_halfbior, s_inv_2, dens0, dens1, n_i1, perm, n_occ, sl0, sl1):
#    return 2 * 2 * (-1)**(n_i1 + perm) * raw(
#        dl(i0, i1) @ s_inv_2[:,sl0](p, 0) @ v_halfbior[sl1,:,:,:](1, q, r, s) @ dens0(i0, j0, p) @ dens1(i1, j1, q, s, r) @ dr(j0, j1)
#    )

def v1000_hess(dl, dr, v_bior, v_halfbior, v_symm, s_inv_2, dens0, dens1, n_i1, perm, n_occ, sl0, sl1, sl2, sl3):
    int1 = evaluate(dl(0, i1) @ dens1(i1, j1, 2) @ dr(1, j1))  # i0, j0, p
    int2 = evaluate(int1(i0, j0, 0) @ dens0[:,:,:,sl2,:](i0, j0, 1, 2, 3))  # p, q, 2, s
    # for this term to have fragment 0 as target index the permutation (frag0<-->frag1) needs to be applied
    #return -2 * 2 * (-1)**(n_i1 + perm) * raw(
    #    dl(i0, i1) @ v_bior[:,:,sl1,sl3](p, q, 1, 3) @ dens1(i1, j1, p) @ dens0[:,:,:,sl1,sl0](i0, j0, q, 2, 0) @ dr(j0, j1)
    #    # the following line again switches 1000 to 0100 in v
    #    + dl(i0, i1) @ s_inv_2[:,sl0](p, 0) @ v_halfbior[:,:,sl3,:](q, 1, 3, s) @ dens1(i1, j1, p) @ dens0[:,:,:,sl2,:](i0, j0, q, 2, s) @ dr(j0, j1)
    #)
    return -2 * 2 * (-1)**(n_i1 + perm) * raw(
        v_bior[:,:,sl1,sl3](p, q, 1, 3) @ int2[:,:,:,sl0](p, q, 2, 0)
        # the following line again switches 1000 to 0100 in v
        - s_inv_2[:,sl1](p, 1) @ v_halfbior[:,sl0,sl3,:](q, 0, 3, s) @ int2(q, p, 2, s)
    )

def v0011_hess(dl, dr, v_bior, v_halfbior, v_symm, s_inv_2, dens0, dens1, n_i1, perm, n_occ, sl0, sl1, sl2, sl3):
    int1 = evaluate(v_symm[sl0,sl2,:,:](0, 1, r, s) @ dens1(2, 3, s, r))  # 1, 3, i1, j1
    int2 = evaluate(dl(0, i1) @ dr(1, j1) @ int1(2, 3, i1, j1))  # i0, j0, 1, 3
    #return 2 * raw(dl(i0, i1) @ s_inv_2[:,sl0](p, 0) @ s_inv_2[:,sl2](q, 2) @ v_symm[sl1,sl3,:,:](1, 3, r, s) @ dens0(i0, j0, p, q) @ dens1(i1, j1, s, r) @ dr(j0, j1))
    return 2 * raw(s_inv_2[:,sl1](p, 1) @ s_inv_2[:,sl3](q, 3) @ dens0(i0, j0, p, q) @ int2(i0, j0, 0, 2))

def v1100_hess(dl, dr, v_bior, v_halfbior, v_symm, s_inv_2, dens0, dens1, n_i1, perm, n_occ, sl0, sl1, sl2, sl3):
    int1 = evaluate(v_bior[:,:,sl1,sl3](p, q, 0, 1) @ dens1(2, 3, p, q))  # 1, 3, i1, j1
    # for this term to have fragment 0 as target index the permutation (frag0<-->frag1) needs to be applied
    #return 2 * raw(dl(i0, i1) @ v_bior[:,:,sl1,sl3](p, q, 1, 3) @ dens1(i1, j1, p, q) @ dens0[:,:,sl2,sl0](i0, j0, 2, 0) @ dr(j0, j1))
    return 2 * raw(dl(i0, i1) @ int1(1, 3, i1, j1) @ dens0[:,:,sl2,sl0](i0, j0, 2, 0) @ dr(j0, j1))

#####################################
# 2p density
#####################################

def two_p_0000(dl, dr, dens, perm):
    int1 = evaluate(dl(0, i1) @ dr(1, i1))  # i0, j0
    return 1 * raw(dens(i0, j0, 0, 1, 3, 2) @ int1(i0, j0))  # p,q,r,s

# pr,qs,pqrs-> :  ca0  ca1  v0101
def two_p_0101(dl, dr, dens0, dens1, n_i1, perm):
    int1 = evaluate(dl(0, i1) @ dens1(i1, j1, 2, 3) @ dr(1, j1))  # i0, j0, q, s
    return 1 * raw(int1(i0, j0, 1, 3) @ dens0(i0, j0, 0, 2))  # p,q,r,s

def two_p_0110(dl, dr, dens0, dens1, n_i1, perm):
    int1 = evaluate(dl(0, i1) @ dens1(i1, j1, 2, 3) @ dr(1, j1))  # i0, j0, q, r
    return -1 * raw(int1(i0, j0, 1, 2) @ dens0(i0, j0, 0, 3))  # p,q,r,s

one_p_dens_catalog = {
    # (frag0, frag1): [Dchgs, dens_types, diagram]
    (0, 0): [(0, 0), ["ca"], one_p_diagonal]#,
    #(0, 1): [(-1, +1), 0, ["c", "a"], one_p_off_diagonal],
}

two_p_dens_catalog = {
    # (frag0, frag1): [Dchgs, permutation, dens_types, diagram]
    (0, 0, 0, 0): [(0, 0), ["ccaa"], two_p_0000],
    (0, 1, 0, 1): [(0, 0), ["ca", "ca"], two_p_0101],
    (0, 1, 1, 0): [(0, 0), ["ca", "ca"], two_p_0110]#,  # can also be obtained from permutation of the upper one, but that is not yet implemented
}

"""
v_catalog = {
    # (frag0, frag1, frag2, frag3): [Dchgs, permutation, dens_types, diagram]
    (0, 0, 0, 0): [(0, 0), 0, ["ccaa"], v0000],
    (1, 1, 1, 1): [(0, 0), 1, ["ccaa"], v0000],
    (0, 1, 0, 1): [(0, 0), 0, ["ca", "ca"], v0101],
    (1, 0, 1, 0): [(0, 0), 1, ["ca", "ca"], v0101],
    (0, 0, 1, 0): [(-1, +1), 0, ["cca", "a"], v0010],
    (1, 1, 0, 1): [(+1, -1), 1, ["a", "cca"], v0010],
    (0, 1, 1, 1): [(-1, +1), 0, ["c", "caa"], v0111],
    (1, 0, 0, 0): [(+1, -1), 1, ["caa", "c"], v0111],
    (0, 0, 1, 1): [(-2, +2), 0, ["cc", "aa"], v0011],
    (1, 1, 0, 0): [(+2, -2), 1, ["aa", "cc"], v0011]
}
"""

v_catalog = [  # target fragment is 0 ... for target fragment 1 simply put perm = 1
    [(0, 0), ["ccaa"], "v0000"],
    [(0, 0), ["ca", "ca"], "v0101"],
    [(0, 0), ["ca", "ca"], "v1010"]
    #[(-1, +1), ["cca", "a"], "v0010"],
    #[(-1, +1), ["c", "caa"], "v0111"],
    #[(+1, -1), ["caa", "c"], "v1000"],
    #[(-2, +2), ["cc", "aa"], "v0011"],
    #[(+2, -2), ["aa", "cc"], "v1100"]
]

v_catalog_hess = [  # target fragment is 0 ... for target fragment 1 simply put perm = 1
    [(0, 0), ["ccaa"], "v0000"],
    [(0, 0), ["ca", "ca"], "v0101"],
    [(0, 0), ["ca", "ca"], "v1010"]#,
    #[(-1, +1), ["cca", "a"], "v0010"],
    #[(+1, -1), ["caa", "c"], "v1000"],
    #[(-2, +2), ["cc", "aa"], "v0011"],
    #[(+2, -2), ["aa", "cc"], "v1100"]
]

def t_diagram(ret, h_dens, ints_bior, ov_slice, mat_ov_slice, mat_slice):
    combs = [(1, [x, y]) for x in ["a", "i"] for y in ["a", "i"]]
    def t_bior(i, j):
        return ints_bior.T[i,j]
    for comb in combs:
        pl, ql = comb[1]
        for s1 in range(2):
            for s2 in range(2):
                for i in range(2):
                    for j in range(2):
                        for l in range(2):
                            if l != j:
                                continue
                            ret[mat_ov_slice[(j, pl, s1)], mat_ov_slice[(l, ql, s2)]] +=\
                                comb[0] * raw(t_bior(i, l)[:, ov_slice[(l, ql, s2)]](r, 1) @ h_dens[mat_slice[i], mat_ov_slice[(j, pl, s1)]](r, 0))
                                            #- t_bior(j, i)[ov_slice[(j, pl, s1)],:](0, r) @ h_dens[mat_ov_slice[(l, ql, s2)], mat_slice[i]](1, r))
                            
def t_diagram2(ret, h_dens, ints_bior, ov_slice, mat_ov_slice, mat_slice):
    combs = [(1, [x, y]) for x in ["a", "i"] for y in ["a", "i"]]
    def t_bior(i, j):
        return ints_bior.T[i,j]
    for comb in combs:
        pl, ql = comb[1]
        for s1 in range(2):
            for s2 in range(2):
                for i in range(2):
                    for j in range(2):
                        for l in range(2):
                            if l != j:
                                continue
                            ret[mat_ov_slice[(j, pl, s1)], mat_ov_slice[(l, ql, s2)]] -=\
                                comb[0] * raw(t_bior(j, i)[ov_slice[(j, pl, s1)],:](0, r) @ h_dens[mat_ov_slice[(l, ql, s2)], mat_slice[i]](1, r))
                        

def t_diagram_hess(ret, h_dens, ints_bior, ov_slice, mat_ov_slice, mat_slice):
    #combs = [(1, [x, y]) for x in ["a", "i"] for y in ["a", "i"]]
    #combs = [(1, [x, y, z, z1]) for x in ["a", "i"] for y in ["a", "i"] for z in ["a", "i"] for z1 in ["a", "i"]]
    def t_bior(i, j):
        return ints_bior.T[i,j]
    for k in range(2):
        ret[mat_slice[k], mat_slice[k], mat_slice[k], mat_slice[k]] -=\
            1 * raw(t_bior(k, k)[:,:](2, 1) @ h_dens[mat_slice[k], mat_slice[k]](3, 0))
        """
        for comb in combs:
            macrocomb = [comb[1]]
            #if comb[1][0] == comb[1][1]:
            #    macrocomb = [comb[1] + comb[1]]
            #else:
            #    macrocomb = [comb[1] + comb[1]]#, comb[1] + list(reversed(comb[1]))]
            for mac in macrocomb:
                pl, ql, pr, qr = mac
                for s1 in range(2):
                    for s2 in range(2):
                        #if s2 != s1:
                        #    continue
                        s3 = s1
                        s4 = s2
                        ret[mat_ov_slice[(k, pl, s1)], mat_ov_slice[(k, ql, s2)], mat_ov_slice[(k, pr, s3)], mat_ov_slice[(k, qr, s4)]] -=\
                            comb[0] * raw(t_bior(k, k)[ov_slice[(k, pr, s3)], ov_slice[(k, ql, s2)]](2, 1) @ h_dens[mat_ov_slice[(k, qr, s4)], mat_ov_slice[(k, pl, s1)]](3, 0))
        """
        """
        # for building a 2d matrix
        for i in range(*ov_slice_range[(k, "i")]):
            i_ = i + mat_ov_slice_range[(k, "i")][0]
            for a in range(*ov_slice_range[(k, "a")]):
                a_ = a + mat_ov_slice_range[(k, "i")][0]
                ret[a_, i_] +=\
                    raw(t_symm(k, k)[i, i] * s_inv_2(j, k)[:, a](r) @ h_dens[mat_slice[j], a_](r))
        """


def u_diagram(ret, h_dens, ints_bior, ov_slice, mat_ov_slice, mat_slice):
    combs = [(1, [x, y]) for x in ["a", "i"] for y in ["a", "i"]]
    def u_bior(n, i, j):
        return ints_bior.U[n,i,j]
    for comb in combs:
        pl, ql = comb[1]
        for s1 in range(2):
            for s2 in range(2):
                for n in range(2):
                    for i in range(2):
                        for j in range(2):
                            for l in range(2):
                                if l != j:
                                    continue
                                ret[mat_ov_slice[(j, pl, s1)], mat_ov_slice[(l, ql, s2)]] +=\
                                    comb[0] * raw(u_bior(n, i, l)[:, ov_slice[(l, ql, s2)]](r, 1) @ h_dens[mat_slice[i], mat_ov_slice[(j, pl, s1)]](r, 0))
                                                #- u_bior(n, j, i)[ov_slice[(j, pl, s1)],:](0, r) @ h_dens[mat_ov_slice[(l, ql, s2)], mat_slice[i]](1, r))
                                
def u_diagram2(ret, h_dens, ints_bior, ov_slice, mat_ov_slice, mat_slice):
    combs = [(1, [x, y]) for x in ["a", "i"] for y in ["a", "i"]]
    def u_bior(n, i, j):
        return ints_bior.U[n,i,j]
    for comb in combs:
        pl, ql = comb[1]
        for s1 in range(2):
            for s2 in range(2):
                for n in range(2):
                    for i in range(2):
                        for j in range(2):
                            for l in range(2):
                                if l != j:
                                    continue
                                ret[mat_ov_slice[(j, pl, s1)], mat_ov_slice[(l, ql, s2)]] -=\
                                    comb[0] * raw(u_bior(n, j, i)[ov_slice[(j, pl, s1)],:](0, r) @ h_dens[mat_ov_slice[(l, ql, s2)], mat_slice[i]](1, r))                                
                                

def u_diagram_hess(ret, h_dens, ints_bior, ov_slice, mat_ov_slice, mat_slice):
    #combs = [(1, [x, y]) for x in ["a", "i"] for y in ["a", "i"]]
    #combs = [(1, [x, y, z, z1]) for x in ["a", "i"] for y in ["a", "i"] for z in ["a", "i"] for z1 in ["a", "i"]]
    def u_bior(n, i, j):
        return ints_bior.U[n,i,j]
    for n in range(2):
        for k in range(2):
            ret[mat_slice[k], mat_slice[k], mat_slice[k], mat_slice[k]] -=\
                1 * raw(u_bior(n, k, k)[:,:](2, 1) @ h_dens[mat_slice[k], mat_slice[k]](3, 0))
            """
            for comb in combs:
                macrocomb = [comb]
                #if comb[1][0] == comb[1][1]:
                #    macrocomb = [comb[1] + comb[1]]
                #else:
                #    macrocomb = [comb[1] + comb[1]]#, comb[1] + list(reversed(comb[1]))]
                for mac in macrocomb:
                    pl, ql, pr, qr = mac
                    for s1 in range(2):
                        for s2 in range(2):
                            #if s2 != s1:
                            #    continue
                            s3 = s1
                            s4 = s2
                            ret[mat_ov_slice[(k, pl, s1)], mat_ov_slice[(k, ql, s2)], mat_ov_slice[(k, pr, s3)], mat_ov_slice[(k, qr, s4)]] -=\
                                comb[0] * raw(u_bior(n, k, k)[ov_slice[(k, pr, s3)], ov_slice[(k, ql, s2)]](2, 1) @ h_dens[mat_ov_slice[(k, qr, s4)], mat_ov_slice[(k, pl, s1)]](3, 0))
            """

def v_diagram_without_2p_dens(ret, dl_, dr_, dens, ints_bior, ints_symm, s_inv_2_, d_slices, n_occ, mat_ov_slice, ov_slice):
    # TODO: check the convention for v_halfbior, because the convention used here is symm bior symm symm
    #combs = [(1, ["a", "i"]), (-1, ["i", "a"])]
    combs = [(1, [x, y]) for x in ["a", "i"] for y in ["a", "i"]]
    for instruction in v_catalog:
        Dchgs, field_ops, int_diag = instruction
        diagram = globals()[int_diag]
        print(int_diag)
        for target_frag in range(2):
            perm = target_frag  # this is only true if targets are normalized to 0 and 1
            if perm:
                dl = tl_tensor(tl.tensor(dl_.T, dtype=tl.float64))
                dr = tl_tensor(tl.tensor(dr_.T, dtype=tl.float64))
                int_frags = [1 - int(i) for i in int_diag[1:]]
            else:
                dl = tl_tensor(tl.tensor(dl_, dtype=tl.float64))
                dr = tl_tensor(tl.tensor(dr_, dtype=tl.float64))
                int_frags = [int(i) for i in int_diag[1:]]
            def get_int(id):
                z0, z1, z2, z3 = int_frags
                if id == "V":
                    return ints_bior.V[z0,z1,z2,z3]
                elif id == "V_half":
                    return ints_bior.V_half[z0,z1,z2,z3]
                elif id == "V_half1":
                    return ints_bior.V_half1[z0,z1,z2,z3]
                elif id == "V_half2":
                    return ints_bior.V_half2[z0,z1,z2,z3]
                elif id == "s_inv_2":
                    return s_inv_2_[(z0,target_frag)]  # actually for the terms including s z0 is also target_frag 
            #for i in range(2):
            i = target_frag
            if len(field_ops) == 1:
                # do monomer stuff
                for chg in d_slices[i]:
                    #for frag, op in enumerate(field_ops):
                        #if op == "delta":
                        #    delta = tl_tensor(tl.eye(n_occ[frag] + n_virt[frag], dtype=tl.float64))
                        #else:
                    dens0 = dens[i][field_ops[0]][(chg, chg)]
                    #if i == 0:
                    for comb in combs:
                        pl, ql = comb[1]
                        for s1 in range(2):
                            for s2 in range(2):
                                ret[mat_ov_slice[(i, pl, s1)], mat_ov_slice[(i, ql, s2)]] +=\
                                    comb[0] * diagram(dl[d_slices[i][chg], :], dr[d_slices[i][chg], :],
                                                    get_int("V"), get_int("V_half1"), get_int("V_half2"), get_int("s_inv_2"),
                                                    dens0, perm, n_occ, ov_slice[(i, pl, s1)], ov_slice[(i, ql, s2)])
                        #diagram(dl[d_slices[i][chg], :], dr[d_slices[i][chg], :], dens0, perm)
            else:
                #pass  # do dimer stuff
                #for j in range(2):
                j = 1 - target_frag
                # do dimer stuff
                for bra_chgi in d_slices[i]:
                    for ket_chgi in d_slices[i]:
                        if bra_chgi - ket_chgi != Dchgs[0]:
                            continue
                        for bra_chgj in d_slices[j]:
                            for ket_chgj in d_slices[j]:
                                if bra_chgj - ket_chgj != Dchgs[1]:
                                    continue
                                n_i1 = get_prefactor_exponent([sum(n_occ), sum(n_occ)], [(bra_chgi, ket_chgi), (bra_chgj, ket_chgj)])
                                densi = dens[i][field_ops[0]][(bra_chgi, ket_chgi)]
                                densj = dens[j][field_ops[1]][(bra_chgj, ket_chgj)]
                                #ret[mat_slice[i], mat_slice[j]] += diagram(dl[d_slices[i][bra_chgi], d_slices[j][bra_chgj]],
                                #                                        dr[d_slices[i][ket_chgi], d_slices[j][ket_chgj]],
                                #                                        densi, densj, n_i1, perm)
                                for comb in combs:
                                    pl, ql = comb[1]
                                    for s1 in range(2):
                                        for s2 in range(2):
                                            ret[mat_ov_slice[(i, pl, s1)], mat_ov_slice[(i, ql, s2)]] +=\
                                                comb[0] * diagram(dl[d_slices[i][bra_chgi], d_slices[j][bra_chgj]], dr[d_slices[i][ket_chgi], d_slices[j][ket_chgj]],
                                                                get_int("V"), get_int("V_half1"), get_int("V_half2"), get_int("s_inv_2"),
                                                                densi, densj, n_i1, perm, n_occ, ov_slice[(i, pl, s1)], ov_slice[(i, ql, s2)])
    #return tl_tensor(tl.tensor(mat, dtype=tl.float64))

def v_diagram_hess_without_2p_dens(ret, dl_, dr_, dens, ints_bior, ints_symm, s_inv_2_, d_slices, n_occ, mat_ov_slice, ov_slice):
    # TODO: check the convention for v_halfbior, because the convention used here is symm bior symm symm
    target_inds_pre = [(0, ["a", "i"]), (1, ["i", "a"])]
    combs = [((-1)**(target_inds_pre[x1][0] + target_inds_pre[x2][0]), [*target_inds_pre[x1][1], *target_inds_pre[x2][1]])
             for x1 in range(2) for x2 in range(2)]
    #combs = [(1, ["i", "a", "i", "a"])]
    for instruction in v_catalog_hess:
        Dchgs, field_ops, int_diag = instruction
        diagram = globals()[int_diag + "_hess"]
        print(int_diag + "_hess")
        for target_frag in range(2):
            perm = target_frag  # this is only true if targets are normalized to 0 and 1
            if perm:
                dl = tl_tensor(tl.tensor(dl_.T, dtype=tl.float64))
                dr = tl_tensor(tl.tensor(dr_.T, dtype=tl.float64))
                int_frags = [1 - int(i) for i in int_diag[1:]]
            else:
                dl = tl_tensor(tl.tensor(dl_, dtype=tl.float64))
                dr = tl_tensor(tl.tensor(dr_, dtype=tl.float64))
                int_frags = [int(i) for i in int_diag[1:]]
            def get_int(id):
                z0, z1, z2, z3 = int_frags
                #print(ints_bior._data_dict.keys())
                if id == "V":
                    return ints_bior.V[z0,z1,z2,z3]
                elif id == "V_half":
                    return ints_bior.V_half[z0,z1,z2,z3]
                elif id == "V_half1":
                    return ints_bior.V_half1[z0,z1,z2,z3]
                elif id == "V_half2":
                    return ints_bior.V_half2[z0,z1,z2,z3]
                elif id == "V_symm":
                    return ints_symm.V[z0,z1,z2,z3]
                elif id == "s_inv_2":
                    return s_inv_2_[(z0,target_frag)]  # actually for the terms including s z0 is also target_frag 
                else:
                    raise ValueError(f"id {id} unknown")
            i = target_frag
            if len(field_ops) == 1:
                # do monomer stuff
                for chg in d_slices[i]:
                    dens0 = dens[i][field_ops[0]][(chg, chg)]
                    for comb in combs:
                        #if comb[0] == -1:
                        #    continue
                        pl, ql, pr, qr = comb[1]
                        for s1 in range(2):
                            for s2 in range(2):
                                if s2 != s1:
                                    continue
                                #for s3 in range(2):
                                #    for s4 in range(2):
                                s3 = s1
                                s4 = s2
                                ret[mat_ov_slice[(i, pl, s1)], mat_ov_slice[(i, ql, s2)], mat_ov_slice[(i, pr, s3)], mat_ov_slice[(i, qr, s4)]] +=\
                                    comb[0] * diagram(dl[d_slices[i][chg], :], dr[d_slices[i][chg], :],
                                            get_int("V"), get_int("V_half1"), get_int("V_half2"), get_int("V_symm"), get_int("s_inv_2"), dens0, perm, n_occ,
                                            ov_slice[(i, pl, s1)], ov_slice[(i, ql, s2)], ov_slice[(i, pr, s3)], ov_slice[(i, qr, s4)])
            else:
                # do dimer stuff
                j = 1 - target_frag
                for bra_chgi in d_slices[i]:
                    for ket_chgi in d_slices[i]:
                        if bra_chgi - ket_chgi != Dchgs[0]:
                            continue
                        for bra_chgj in d_slices[j]:
                            for ket_chgj in d_slices[j]:
                                if bra_chgj - ket_chgj != Dchgs[1]:
                                    continue
                                n_i1 = get_prefactor_exponent([sum(n_occ), sum(n_occ)], [(bra_chgi, ket_chgi), (bra_chgj, ket_chgj)])
                                densi = dens[i][field_ops[0]][(bra_chgi, ket_chgi)]
                                densj = dens[j][field_ops[1]][(bra_chgj, ket_chgj)]
                                for comb in combs:
                                    #if comb[0] == -1:
                                    #    continue
                                    pl, ql, pr, qr = comb[1]
                                    for s1 in range(2):
                                        for s2 in range(2):
                                            if s2 != s1:
                                                continue
                                            #for s3 in range(2):
                                            #    for s4 in range(2):
                                            s3 = s1
                                            s4 = s2
                                            ret[mat_ov_slice[(i, pl, s1)], mat_ov_slice[(i, ql, s2)], mat_ov_slice[(i, pr, s3)], mat_ov_slice[(i, qr, s4)]] +=\
                                                comb[0] * diagram(dl[d_slices[i][bra_chgi], d_slices[j][bra_chgj]], dr[d_slices[i][ket_chgi], d_slices[j][ket_chgj]],
                                                        get_int("V"), get_int("V_half1"), get_int("V_half2"), get_int("V_symm"), get_int("s_inv_2"), densi, densj, n_i1, perm, n_occ,
                                                        ov_slice[(i, pl, s1)], ov_slice[(i, ql, s2)], ov_slice[(i, pr, s3)], ov_slice[(i, qr, s4)])

def v_diagram(ret, v_dens, ints_bior, ov_slice, mat_ov_slice, mat_slice):
    combs = [(1, [x, y]) for x in ["a", "i"] for y in ["a", "i"]]
    def v_bior(i, j, k, l):
        return ints_bior.V[i,j,k,l]
    for comb in combs:
        pl, ql = comb[1]
        for s1 in range(2):
            for s2 in range(2):
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            for l in range(2):
                                # further restrictions on the loops should be possible
                                for m in range(2):
                                    if m != l:
                                        continue
                                    ret[mat_ov_slice[(l, pl, s1)], mat_ov_slice[(m, ql, s2)]] +=\
                                        comb[0] * 1 * raw(v_bior(i, j, k, m)[:,:,:,ov_slice[(m, ql, s2)]](p, q, r, 1)
                                                        @ v_dens[mat_slice[i],mat_slice[j],mat_slice[k],mat_ov_slice[(l, pl, s1)]](p, q, r, 0)
                                                        + v_bior(i, j, m, k)[:,:,ov_slice[(m, ql, s2)],:](p, q, 1, s)
                                                        @ v_dens[mat_slice[i],mat_slice[j],mat_ov_slice[(l, pl, s1)],mat_slice[k]](p, q, 0, s))
                                                        #- v_bior(l, i, j, k)[ov_slice[(l, pl, s1)],:,:,:](0, q, r, s)
                                                        #@ v_dens[mat_ov_slice[(m, ql, s2)],mat_slice[i],mat_slice[j],mat_slice[k]](1, q, r, s)
                                                        #- v_bior(i, l, j, k)[:,ov_slice[(l, pl, s1)],:,:](p, 0, r, s)
                                                        #@ v_dens[mat_slice[i],mat_ov_slice[(m, ql, s2)],mat_slice[j],mat_slice[k]](p, 1, r, s))

def v_diagram2(ret, v_dens, ints_bior, ov_slice, mat_ov_slice, mat_slice):
    combs = [(1, [x, y]) for x in ["a", "i"] for y in ["a", "i"]]
    def v_bior(i, j, k, l):
        return ints_bior.V[i,j,k,l]
    for comb in combs:
        pl, ql = comb[1]
        for s1 in range(2):
            for s2 in range(2):
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            for l in range(2):
                                # further restrictions on the loops should be possible
                                for m in range(2):
                                    if m != l:
                                        continue
                                    ret[mat_ov_slice[(l, pl, s1)], mat_ov_slice[(m, ql, s2)]] -=\
                                        comb[0] * 1 * raw(v_bior(l, i, j, k)[ov_slice[(l, pl, s1)],:,:,:](0, q, r, s)
                                                        @ v_dens[mat_ov_slice[(m, ql, s2)],mat_slice[i],mat_slice[j],mat_slice[k]](1, q, r, s)
                                                        + v_bior(i, l, j, k)[:,ov_slice[(l, pl, s1)],:,:](p, 0, r, s)
                                                        @ v_dens[mat_slice[i],mat_ov_slice[(m, ql, s2)],mat_slice[j],mat_slice[k]](p, 1, r, s))

def v_diagram_hess(ret, v_dens, ints_bior, ov_slice, mat_ov_slice, mat_slice):
    #combs = [(1, [x, y]) for x in ["a", "i"] for y in ["a", "i"]]
    #combs = [(1, [x, y, z, z1]) for x in ["a", "i"] for y in ["a", "i"] for z in ["a", "i"] for z1 in ["a", "i"]]
    def v_bior(i, j, k, l):
        return ints_bior.V[i,j,k,l]
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for m in range(2):
                    if m != k:
                        continue
                    ret[mat_slice[k], mat_slice[m], mat_slice[k], mat_slice[m]] +=\
                        1 * raw(v_bior(i, j, m, m)[:,:,:,:](p, q, 1, 3)
                            @ v_dens[mat_slice[i],mat_slice[j],mat_slice[k],mat_slice[k]](p, q, 0, 2)  # both right
                            + v_bior(k, k, i, j)[:,:,:,:](0, 2, p, q)
                            @ v_dens[mat_slice[m],mat_slice[m],mat_slice[i],mat_slice[j]](1, 3, p, q)  # both left ... both left on the same index is handled on higher level
                            - v_bior(i, k, j, m)[:,:,:,:](p, 0, q, 3)
                            @ v_dens[mat_slice[i],mat_slice[m],mat_slice[j],mat_slice[k]](p, 1, q, 2)  # one right one left
                            - v_bior(k, i, j, m)[:,:,:,:](0, p, q, 3)
                            @ v_dens[mat_slice[m],mat_slice[i],mat_slice[j],mat_slice[k]](1, p, q, 2)  # one right one left
                            - v_bior(i, k, m, j)[:,:,:,:](p, 0, 3, q)
                            @ v_dens[mat_slice[i],mat_slice[m],mat_slice[k],mat_slice[j]](p, 1, 2, q)  # one right one left
                            - v_bior(k, i, m, j)[:,:,:,:](0, p, 3, q)
                            @ v_dens[mat_slice[m],mat_slice[i],mat_slice[k],mat_slice[j]](1, p, 2, q))  # one right one left
    """
    for comb in combs:
        macrocomb = [comb]
        #pl, ql = comb[1]
        #if comb[1][0] == comb[1][1]:
        #    macrocomb = [comb[1] + comb[1]]
        #else:
        #    macrocomb = [comb[1] + comb[1]]#, comb[1] + list(reversed(comb[1]))]
        for mac in macrocomb:
            pl, ql, pr, qr = mac
            for s1 in range(2):
                for s2 in range(2):
                    for i in range(2):
                        for j in range(2):
                            for k in range(2):
                                # further restrictions on the loops should be possible
                                for m in range(2):
                                    if m != k:
                                        continue
                                    s3 = s1
                                    s4 = s2
                                    ret[mat_ov_slice[(k, pl, s1)], mat_ov_slice[(m, ql, s2)], mat_ov_slice[(k, pr, s3)], mat_ov_slice[(m, qr, s4)]] +=\
                                        comb[0] * raw(v_bior(i, j, m, m)[:,:,ov_slice[(m, ql, s2)],ov_slice[(m, qr, s4)]](p, q, 1, 3)
                                                    @ v_dens[mat_slice[i],mat_slice[j],mat_ov_slice[(k, pl, s1)],mat_ov_slice[(k, pr, s3)]](p, q, 0, 2)  # both right
                                                    + v_bior(k, k, i, j)[ov_slice[(k, pl, s1)],ov_slice[(k, pr, s3)],:,:](0, 2, p, q)
                                                    @ v_dens[mat_ov_slice[(m, ql, s2)],mat_ov_slice[(m, qr, s4)],mat_slice[i],mat_slice[j]](1, 3, p, q)  # both left ... both left on the same index is handled on higher level
                                                    - v_bior(i, k, j, m)[:,ov_slice[(k, pl, s1)],:,ov_slice[(m, qr, s4)]](p, 0, q, 3)
                                                    @ v_dens[mat_slice[i],mat_ov_slice[(m, ql, s2)],mat_slice[j],mat_ov_slice[(k, pr, s3)]](p, 1, q, 2)  # one right one left
                                                    - v_bior(k, i, j, m)[ov_slice[(k, pl, s1)],:,:,ov_slice[(m, qr, s4)]](0, p, q, 3)
                                                    @ v_dens[mat_ov_slice[(m, ql, s2)],mat_slice[i],mat_slice[j],mat_ov_slice[(k, pr, s3)]](1, p, q, 2)  # one right one left
                                                    - v_bior(i, k, m, j)[:,ov_slice[(k, pl, s1)],ov_slice[(m, qr, s4)],:](p, 0, 3, q)
                                                    @ v_dens[mat_slice[i],mat_ov_slice[(m, ql, s2)],mat_ov_slice[(k, pr, s3)],mat_slice[j]](p, 1, 2, q)  # one right one left
                                                    - v_bior(k, i, m, j)[ov_slice[(k, pl, s1)],:,ov_slice[(m, qr, s4)],:](0, p, 3, q)
                                                    @ v_dens[mat_ov_slice[(m, ql, s2)],mat_slice[i],mat_ov_slice[(k, pr, s3)],mat_slice[j]](1, p, 2, q))  # one right one left
    """



class grads_and_hessian(object):
    def __init__(self, n_occ, n_virt, n_frozen, d_slices):
        # frozen is assuming the lowest occs as frozen
        self.n_occ = n_occ
        self.d_slices = d_slices
        self.n_virt = n_virt
        self.ten_shape = (sum(n_occ) + sum(n_virt), sum(n_occ) + sum(n_virt))
        self.ten_shape_large = (sum(n_occ) + sum(n_virt), sum(n_occ) + sum(n_virt), sum(n_occ) + sum(n_virt), sum(n_occ) + sum(n_virt))
        self.mat_slice = {0: slice(0, n_occ[0] + n_virt[0]),
                    1: slice(n_occ[0] + n_virt[0], sum(n_occ) + sum(n_virt))}
        n_occ_a = [x // 2 for x in n_occ]
        #n_occ_b = [x // 2 for x in n_occ]
        n_virt_a = [x // 2 for x in n_virt]
        #n_virt_b = [x // 2 for x in n_virt]
        n_frozen_a = [x // 2 for x in n_frozen]
        f = n_occ[0] + n_virt[0]
        # in the following the keys denote (frag_ind, occ or virt, spin (0 = alpha and 1 = beta))
        """
        # this is incorporating frozen orbs
        self.ov_slice = {(0, "i", 0): slice(n_frozen_a[0], n_occ_a[0]),
                         (0, "a", 0): slice(n_occ_a[0], n_occ_a[0] + n_virt_a[0]),
                         (0, "i", 1): slice(n_frozen[0] + n_occ_a[0] + n_virt_a[0], n_occ[0] + n_virt_a[0]),
                         (0, "a", 1): slice(n_occ[0] + n_virt_a[0], n_occ[0] + n_virt[0]),
                         (1, "i", 0): slice(n_frozen_a[1], n_occ_a[1]),
                         (1, "a", 0): slice(n_occ_a[1], n_occ_a[1] + n_virt_a[1]),
                         (1, "i", 1): slice(n_frozen[1] + n_occ_a[1] + n_virt_a[1], n_occ[1] + n_virt_a[1]),
                         (1, "a", 1): slice(n_occ[1] + n_virt_a[1], n_occ[1] + n_virt[1])}
        self.ov_slice_ranges = {(0, "i", 0): (n_frozen_a[0], n_occ_a[0]),
                         (0, "a", 0): (n_occ_a[0], n_occ_a[0] + n_virt_a[0]),
                         (0, "i", 1): (n_frozen[0] + n_occ_a[0] + n_virt_a[0], n_occ[0] + n_virt_a[0]),
                         (0, "a", 1): (n_occ[0] + n_virt_a[0], n_occ[0] + n_virt[0]),
                         (1, "i", 0): (n_frozen_a[1], n_occ_a[1]),
                         (1, "a", 0): (n_occ_a[1], n_occ_a[1] + n_virt_a[1]),
                         (1, "i", 1): (n_frozen[1] + n_occ_a[1] + n_virt_a[1], n_occ[1] + n_virt_a[1]),
                         (1, "a", 1): (n_occ[1] + n_virt_a[1], n_occ[1] + n_virt[1])}
        self.mat_ov_slice = {(0, "i", 0): self.ov_slice[(0, "i", 0)],
                             (0, "a", 0): self.ov_slice[(0, "a", 0)],
                             (0, "i", 1): self.ov_slice[(0, "i", 1)],
                             (0, "a", 1): self.ov_slice[(0, "a", 1)],
                             (1, "i", 0): slice(f + n_frozen_a[1], f + n_occ_a[1]),
                             (1, "a", 0): slice(f + n_occ_a[1], f + n_occ_a[1] + n_virt_a[1]),
                             (1, "i", 1): slice(f + n_frozen[1] + n_occ_a[1] + n_virt_a[1], f + n_occ[1] + n_virt_a[1]),
                             (1, "a", 1): slice(f + n_occ[1] + n_virt_a[1], f + n_occ[1] + n_virt[1])}
        self.mat_ov_slice_ranges = {(0, "i", 0): self.ov_slice_ranges[(0, "i", 0)],
                             (0, "a", 0): self.ov_slice_ranges[(0, "a", 0)],
                             (0, "i", 1): self.ov_slice_ranges[(0, "i", 1)],
                             (0, "a", 1): self.ov_slice_ranges[(0, "a", 1)],
                             (1, "i", 0): (f + n_frozen_a[1], f + n_occ_a[1]),
                             (1, "a", 0): (f + n_occ_a[1], f + n_occ_a[1] + n_virt_a[1]),
                             (1, "i", 1): (f + n_frozen[1] + n_occ_a[1] + n_virt_a[1], f + n_occ[1] + n_virt_a[1]),
                             (1, "a", 1): (f + n_occ[1] + n_virt_a[1], f + n_occ[1] + n_virt[1])}
        """
        # this is not incorporating frozen orbs
        self.ov_slice = {(0, "i", 0): slice(0, n_occ_a[0]),
                         (0, "a", 0): slice(n_occ_a[0], n_occ_a[0] + n_virt_a[0]),
                         (0, "i", 1): slice(n_occ_a[0] + n_virt_a[0], n_occ[0] + n_virt_a[0]),
                         (0, "a", 1): slice(n_occ[0] + n_virt_a[0], n_occ[0] + n_virt[0]),
                         (1, "i", 0): slice(0, n_occ_a[1]),
                         (1, "a", 0): slice(n_occ_a[1], n_occ_a[1] + n_virt_a[1]),
                         (1, "i", 1): slice(n_occ_a[1] + n_virt_a[1], n_occ[1] + n_virt_a[1]),
                         (1, "a", 1): slice(n_occ[1] + n_virt_a[1], n_occ[1] + n_virt[1])}
        self.ov_slice_ranges = {(0, "i", 0): (0, n_occ_a[0]),
                         (0, "a", 0): (n_occ_a[0], n_occ_a[0] + n_virt_a[0]),
                         (0, "i", 1): (n_occ_a[0] + n_virt_a[0], n_occ[0] + n_virt_a[0]),
                         (0, "a", 1): (n_occ[0] + n_virt_a[0], n_occ[0] + n_virt[0]),
                         (1, "i", 0): (0, n_occ_a[1]),
                         (1, "a", 0): (n_occ_a[1], n_occ_a[1] + n_virt_a[1]),
                         (1, "i", 1): (n_occ_a[1] + n_virt_a[1], n_occ[1] + n_virt_a[1]),
                         (1, "a", 1): (n_occ[1] + n_virt_a[1], n_occ[1] + n_virt[1])}
        self.mat_ov_slice = {(0, "i", 0): self.ov_slice[(0, "i", 0)],
                             (0, "a", 0): self.ov_slice[(0, "a", 0)],
                             (0, "i", 1): self.ov_slice[(0, "i", 1)],
                             (0, "a", 1): self.ov_slice[(0, "a", 1)],
                             (1, "i", 0): slice(f, f + n_occ_a[1]),
                             (1, "a", 0): slice(f + n_occ_a[1], f + n_occ_a[1] + n_virt_a[1]),
                             (1, "i", 1): slice(f + n_occ_a[1] + n_virt_a[1], f + n_occ[1] + n_virt_a[1]),
                             (1, "a", 1): slice(f + n_occ[1] + n_virt_a[1], f + n_occ[1] + n_virt[1])}
        self.mat_ov_slice_ranges = {(0, "i", 0): self.ov_slice_ranges[(0, "i", 0)],
                             (0, "a", 0): self.ov_slice_ranges[(0, "a", 0)],
                             (0, "i", 1): self.ov_slice_ranges[(0, "i", 1)],
                             (0, "a", 1): self.ov_slice_ranges[(0, "a", 1)],
                             (1, "i", 0): (f, f + n_occ_a[1]),
                             (1, "a", 0): (f + n_occ_a[1], f + n_occ_a[1] + n_virt_a[1]),
                             (1, "i", 1): (f + n_occ_a[1] + n_virt_a[1], f + n_occ[1] + n_virt_a[1]),
                             (1, "a", 1): (f + n_occ[1] + n_virt_a[1], f + n_occ[1] + n_virt[1])}
        
        """
        # this does not incorporate frozen orbs and is spin blocked differently
        self.ov_slice = {(0, "i"): slice(0, n_occ[0]),
                    (0, "a"): slice(n_occ[0], n_occ[0] + n_virt[0]),
                    (1, "i"): slice(0, n_occ[1]),
                    (1, "a"): slice(n_occ[1], n_occ[1] + n_virt[1])}
        self.ov_slice_ranges = {(0, "i"): (0, n_occ[0]),
                    (0, "a"): (n_occ[0], n_occ[0] + n_virt[0]),
                    (1, "i"): (0, n_occ[1]),
                    (1, "a"): (n_occ[1], n_occ[1] + n_virt[1])}
        self.mat_ov_slice = {(0, "i"): slice(0, n_occ[0]),
                        (0, "a"): slice(n_occ[0], n_occ[0] + n_virt[0]),
                        (1, "i"): slice(n_occ[0] + n_virt[0], sum(n_occ) + n_virt[0]),
                        (1, "a"): slice(sum(n_occ) + n_virt[0], sum(n_occ) + sum(n_virt))}
        self.mat_ov_slice_ranges = {(0, "i"): (0, n_occ[0]),
                        (0, "a"): (n_occ[0], n_occ[0] + n_virt[0]),
                        (1, "i"): (n_occ[0] + n_virt[0], sum(n_occ) + n_virt[0]),
                        (1, "a"): (sum(n_occ) + n_virt[0], sum(n_occ) + sum(n_virt))}
        """
        #print(self.ov_slice)
        #print(self.mat_ov_slice)
        
    def one_p_dens(self, mat, dl_, dr_, dens):
        #mat = np.zeros(tuple(sum(n_occ) + sum(n_virt), sum(n_occ) + sum(n_virt)))
        #for i in range(2):
        #    for j in range(2):
        for key in one_p_dens_catalog:
            Dchgs, field_ops, diagram = one_p_dens_catalog[key]
            for perm in range(2):
                i, j = key
                if perm:
                    dl = tl_tensor(tl.tensor(dl_.T, dtype=tl.float64))
                    dr = tl_tensor(tl.tensor(dr_.T, dtype=tl.float64))
                    field_ops = tuple(reversed(field_ops))
                    Dchgs = tuple(reversed(Dchgs))
                    i = 1 - i
                    j = 1 - j
                else:
                    dl = tl_tensor(tl.tensor(dl_, dtype=tl.float64))
                    dr = tl_tensor(tl.tensor(dr_, dtype=tl.float64))
                if i == j:
                    # do monomer stuff
                    for chg in self.d_slices[i]:
                        #for frag, op in enumerate(field_ops):
                            #if op == "delta":
                            #    delta = tl_tensor(tl.eye(n_occ[frag] + n_virt[frag], dtype=tl.float64))
                            #else:
                        dens0 = dens[i][field_ops[0]][(chg, chg)]
                        #if i == 0:
                        mat[self.mat_slice[i], self.mat_slice[j]] += diagram(dl[self.d_slices[i][chg], :],
                                                                dr[self.d_slices[i][chg], :],
                                                                dens0, perm)
                        #elif i == 1:
                        #    mat[mat_slice[i], mat_slice[j]] += diagram(dl[:, d_slices[j][chg]],
                        #                                        dr[:, d_slices[j][chg]],
                        #                                        dens0, perm)
                        #else:
                        #    raise ValueError("i needs to be 0 or 1")
                    continue
                # do dimer stuff
                for bra_chgi in self.d_slices[i]:
                    for ket_chgi in self.d_slices[i]:
                        if bra_chgi - ket_chgi != Dchgs[0]:
                            continue
                        for bra_chgj in self.d_slices[j]:
                            for ket_chgj in self.d_slices[j]:
                                if bra_chgj - ket_chgj != Dchgs[1]:
                                    continue
                                n_i1 = get_prefactor_exponent([sum(self.n_occ), sum(self.n_occ)], [(bra_chgi, ket_chgi), (bra_chgj, ket_chgj)])
                                densi = dens[i][field_ops[0]][(bra_chgi, ket_chgi)]
                                densj = dens[j][field_ops[1]][(bra_chgj, ket_chgj)]
                                mat[self.mat_slice[i], self.mat_slice[j]] += diagram(dl[self.d_slices[i][bra_chgi], self.d_slices[j][bra_chgj]],
                                                                        dr[self.d_slices[i][ket_chgi], self.d_slices[j][ket_chgj]],
                                                                        densi, densj, n_i1, perm)
        return tl_tensor(tl.tensor(mat, dtype=tl.float64))
    
    def two_p_dens(self, mat, dl_, dr_, dens):
        #mat = np.zeros(tuple(sum(n_occ) + sum(n_virt), sum(n_occ) + sum(n_virt)))
        #for i in range(2):
        #    for j in range(2):
        for key in two_p_dens_catalog:
            Dchgs, field_ops, diagram = two_p_dens_catalog[key]
            for perm in range(2):
                i, j, k, l = key
                if perm:
                    dl = tl_tensor(tl.tensor(dl_.T, dtype=tl.float64))
                    dr = tl_tensor(tl.tensor(dr_.T, dtype=tl.float64))
                    field_ops = tuple(reversed(field_ops))
                    Dchgs = tuple(reversed(Dchgs))
                    i, j, k, l = 1-i, 1-j, 1-k, 1-l
                else:
                    dl = tl_tensor(tl.tensor(dl_, dtype=tl.float64))
                    dr = tl_tensor(tl.tensor(dr_, dtype=tl.float64))
                if len(field_ops) == 1:
                    # do monomer stuff
                    for chg in self.d_slices[i]:
                        dens0 = dens[i][field_ops[0]][(chg, chg)]
                        mat[self.mat_slice[i], self.mat_slice[i], self.mat_slice[i], self.mat_slice[i]] += diagram(dl[self.d_slices[i][chg], :],
                                                                                                                   dr[self.d_slices[i][chg], :],
                                                                                                                   dens0, perm)
                    continue
                # do dimer stuff
                for bra_chg0 in self.d_slices[0 + perm]:
                    for ket_chg0 in self.d_slices[0 + perm]:
                        if bra_chg0 - ket_chg0 != Dchgs[0]:
                            continue
                        for bra_chg1 in self.d_slices[1 - perm]:
                            for ket_chg1 in self.d_slices[1 - perm]:
                                if bra_chg1 - ket_chg1 != Dchgs[1]:
                                    continue
                                n_i1 = get_prefactor_exponent([sum(self.n_occ), sum(self.n_occ)], [(bra_chg0, ket_chg0), (bra_chg1, ket_chg1)])
                                dens0 = dens[i][field_ops[0]][(bra_chg0, ket_chg0)]
                                dens1 = dens[j][field_ops[1]][(bra_chg1, ket_chg1)]
                                mat[self.mat_slice[i], self.mat_slice[j], self.mat_slice[k], self.mat_slice[l]] += diagram(
                                                                        dl[self.d_slices[0 + perm][bra_chg0], self.d_slices[1 - perm][bra_chg1]],
                                                                        dr[self.d_slices[0 + perm][ket_chg0], self.d_slices[1 - perm][ket_chg1]],
                                                                        dens0, dens1, n_i1, perm)
        return tl_tensor(tl.tensor(mat, dtype=tl.float64))
    
    def build_s_inv_2(self, zero, S):  # build ints_symm.S ** (-2)
        for i in range(2):
            for j in range(2):
                zero[self.mat_slice[i], self.mat_slice[j]] = raw(S[i, j])
        s_inv = precise_numpy_inverse(zero)  # this only works if s is actually close to the identity
        s_inv_2_raw = s_inv @ s_inv
        return {(i, j): tl_tensor(tl.tensor(s_inv_2_raw[self.mat_slice[i], self.mat_slice[j]], dtype=tl.float64)) for i in range(2) for j in range(2)}

    def orb_grads(self, dl, dr, dens, ints):
        print("start orb_grad routine")
        ints_symm, ints_bior, nuc_rep = ints
        grad = np.zeros(self.ten_shape)
        print("computing one particle XR density")
        h_dens = self.one_p_dens(np.zeros(self.ten_shape), dl, dr, dens)
        #print("computing s ** (-2)")
        #s_inv_2 = self.build_s_inv_2(np.zeros(self.ten_shape), ints_symm.S)
        print("computing t")
        t_diagram(grad, h_dens, ints_bior, self.ov_slice, self.mat_ov_slice, self.mat_slice)  # update mat in place
        t_diagram2(grad, h_dens, ints_bior, self.ov_slice, self.mat_ov_slice, self.mat_slice)  # update mat in place
        print("computing u")
        u_diagram(grad, h_dens, ints_bior, self.ov_slice, self.mat_ov_slice, self.mat_slice)  # update mat in place
        u_diagram2(grad, h_dens, ints_bior, self.ov_slice, self.mat_ov_slice, self.mat_slice)  # update mat in place
        print("computing two particle XR density")
        v_dens = self.two_p_dens(np.zeros(self.ten_shape_large), dl, dr, dens)
        print("computing v")
        #v_diagram(grad, dl, dr, dens, ints_bior, ints_symm, s_inv_2, self.d_slices, self.n_occ, self.mat_ov_slice, self.ov_slice)  # update mat in place
        v_diagram(grad, v_dens, ints_bior, self.ov_slice, self.mat_ov_slice, self.mat_slice)  # update mat in place
        v_diagram2(grad, v_dens, ints_bior, self.ov_slice, self.mat_ov_slice, self.mat_slice)  # update mat in place
        print("done")
        # enforce antisymmetry, because term can be taken with x_pq, but also with - x_qp not only in the derivate (d/dx), but also in the actual term
        # where x is the transformation matrix for the orbitals ... symmetry will hence be enforced dividing x into two parts and transforming one of them
        return 1 * (grad - grad.T)  #TODO: Original paper also didn't include a "normalization" ... I guess one needs to figure out what works best in the future
        #return 0.01 * grad  # without enforcing symmetry smaller imaginary parts are obtained for dl and dr, but enforcing symmetry seems to also yield sufficiently small values
    
    def orb_hess_diag(self, dl, dr, dens, ints):
        print("start orb_hess routine")
        ints_symm, ints_bior, nuc_rep = ints
        hess_diag = np.zeros((*self.ten_shape, *self.ten_shape))
        print("computing one particle XR density")
        h_dens = self.one_p_dens(np.zeros(self.ten_shape), dl, dr, dens)
        #print("computing s ** (-2)")
        #s_inv_2 = self.build_s_inv_2(np.zeros(self.ten_shape), ints_symm.S)
        print("computing t")
        t_diagram_hess(hess_diag, h_dens, ints_bior, self.ov_slice, self.mat_ov_slice, self.mat_slice)  # update mat in place
        print("computing u")
        u_diagram_hess(hess_diag, h_dens, ints_bior, self.ov_slice, self.mat_ov_slice, self.mat_slice)  # update mat in place
        print("computing two particle XR density")
        v_dens = self.two_p_dens(np.zeros(self.ten_shape_large), dl, dr, dens)
        print("computing v")
        v_diagram_hess(hess_diag, v_dens, ints_bior, self.ov_slice, self.mat_ov_slice, self.mat_slice)
        #hess_diag = 1 * (hess_diag + np.transpose(hess_diag, (2,3,0,1)))  # this is actually a term from the equations, so it needs factor 1
        #v_diagram_hess(hess_diag, dl, dr, dens, ints_bior, ints_symm, s_inv_2, self.d_slices, self.n_occ, self.mat_ov_slice, self.ov_slice)  # update mat in place
        print("computing grad like term")
        grad_term = np.zeros(self.ten_shape)
        #TODO: the following should be loaded from cache
        t_diagram2(grad_term, h_dens, ints_bior, self.ov_slice, self.mat_ov_slice, self.mat_slice)  # update mat in place
        u_diagram2(grad_term, h_dens, ints_bior, self.ov_slice, self.mat_ov_slice, self.mat_slice)  # update mat in place
        v_diagram2(grad_term, v_dens, ints_bior, self.ov_slice, self.mat_ov_slice, self.mat_slice)  # update mat in place
        # not sure, if this term is allowed to be (2,3,0,1) transposed or not...
        for frag in range(2):
            hess_diag[self.mat_slice[frag],self.mat_slice[frag],self.mat_slice[frag],self.mat_slice[frag]] -= 0.5 * np.einsum("kj,il->ijkl", np.identity(self.n_occ[frag] + self.n_virt[frag]), grad_term[self.mat_slice[frag],self.mat_slice[frag]])
        #hess_diag -= 0.5 * np.einsum("kj,il->ijkl", np.identity(self.ten_shape[0]), grad_term)  # minus to correct sign compared to gradient and 0.5 because of later transpose(2,3,0,1)
        print("done")
        # since far bigger tensors than those the size of an ERI need to be taken into account for the calculation
        # of the gradients already (2p densities) there is not really a need to only give the diagonal, but rather
        # to be as precise with little computational effort as possible, so the resulting tensor could also be more
        # than the diagonal. For now the 4d tensor with full blocks 0,0,0,0 and 1,1,1,1 is returned.
        hess_diag = 1 * (hess_diag - np.transpose(hess_diag, (1,0,2,3)) - np.transpose(hess_diag, (0,1,3,2)) + np.transpose(hess_diag, (1,0,3,2)))
        return 1 * (hess_diag + np.transpose(hess_diag, (2,3,0,1)))  # this is actually a term from the equations, so it needs factor 1
        #hess_diag = 0.5 * (hess_diag + np.transpose(hess_diag, (1,0,3,2)))
        #return 0.5 * (hess_diag + np.transpose(hess_diag, (2,3,0,1)))
        #return hess_diag
