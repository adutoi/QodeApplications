#    (C) Copyright 2023 Anthony D. Dutoi
# 
#    This file is part of Qode.
# 
#    Qode is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
# 
#    Qode is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
# 
#    You should have received a copy of the GNU General Public License
#    along with Qode.  If not, see <http://www.gnu.org/licenses/>.
#

import sys
import numpy as np
import tensorly as tl
import Be631g
import diagrammatic_expansion   # defines information structure for housing results of diagram evaluations
import XR_term                  # knows how to use ^this information to pack a matrix for use in XR model
import S_diagrams               # contains definitions of actual diagrams needed for S operator in BO rep
import SH_diagrams
import torch
from orb_projection import transformation_mat, orb_proj_ints, orb_proj_density

torch.set_num_threads(4)  # here we set the number of the CPU cores, even though pytorch is not used yet

#tl.set_backend("pytorch")

#########
# Load data
#########

displacement = sys.argv[1]

Be = Be631g.monomer_data(None)
n_orb = Be.basis.n_spatial_orb  # number of spatial orbitals for one atom
C     = Be.basis.MOcoeffs       # n_orb x n_orb for restricted orbitals
Be.load_states("load=states:16-115-550:thresh=1e-6:4.5:u.pickle/4.5", None, ("all","all","all","all","all"))

S_2 = np.load("atomic_states/integrals/Be2_{}_S.npy".format(displacement))
sig01 = C.T @ S_2[0:n_orb,n_orb:2*n_orb] @ C
overlaps = {}   # fragment blocked, in terms of spin orbitals
overlaps[0,1] = np.zeros((2*n_orb,2*n_orb))
overlaps[1,0] = np.zeros((2*n_orb,2*n_orb))
overlaps[0,0] = None
overlaps[1,1] = None
overlaps[0,1][:n_orb,:n_orb] = sig01
overlaps[0,1][n_orb:,n_orb:] = sig01
overlaps[1,0][:n_orb,:n_orb] = sig01.T
overlaps[1,0][n_orb:,n_orb:] = sig01.T

Be.rho['n_states'] = {+1:4, 0:11, -1:8}     # monkey patch! (do at lower level)
Be.rho['n_elec']   = {+1:3, 0:4,  -1:5}     # monkey patch! (do at lower level)
BeN_rho = [Be.rho, Be.rho]

#########
# build overlap matrices
#########

# Tensor decompositions scale more or less the same as an orbital rotation,
# however, we only have to do the rotation step once, while the tensor
# decomposition requires waaaaaay more steps, so we reduce the dimensions of
# the tensors, by rotating into a projected basis,
# before decomposing them, to save a lot of time.
# This works very well for S, but since h and v can't be truncated in this
# projected basis, this is not applicable to SH

s01 = overlaps[(0, 1)]
s10 = overlaps[(1, 0)]

# tl.tensor read-in requires tensors, not None
overlaps[(0, 0)] = np.zeros_like(overlaps[(0, 1)])
overlaps[(1, 1)] = np.zeros_like(overlaps[(0, 1)])

# get (truncated) orbital rotation matrices
U0, U1, full_U = transformation_mat(s01, s10, thresh=1e-12)
# rotate overlaps into projected basis and make them tl.tensor
overlaps_for_S = {key:orb_proj_ints(U0, U1, key, tl.tensor(overlaps[key])) for key in overlaps}

# rotate densities into projected basis and make them tl.tensor
# here the tl.tensor read in is done in orb_proj_density
BeN_rho[0] = {op_string:{charges:orb_proj_density(U0, BeN_rho[0][op_string][charges])
                         for charges in BeN_rho[0][op_string]} for op_string in BeN_rho[0] if len(op_string) < 6}
BeN_rho[1] = {op_string:{charges:orb_proj_density(U1, BeN_rho[1][op_string][charges])
                         for charges in BeN_rho[1][op_string]} for op_string in BeN_rho[1] if len(op_string) < 6}

# reintroduce monkey patch for densities
for i in range(len(BeN_rho)):
    BeN_rho[i]["n_elec"] = Be.rho["n_elec"]
    BeN_rho[i]["n_states"] = Be.rho["n_states"]

S_blocks = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=overlaps_for_S, diagrams=S_diagrams)

active_diagrams = {}
active_diagrams[0] = ["identity"]
active_diagrams[1] = []
active_diagrams[2] = ["order1_CT1"]#, "order2_CT0", "order2_CT2"] #, "order3_CT1", "order4_CT0"] #, "order4_CT2"]

dimer01 = (0,1)    # In theory, a subsystem of the full system

Stest = {}
Stest[6]  = XR_term.dimer_matrix(S_blocks, active_diagrams, dimer01, [(+1,+1)])
Stest[7]  = XR_term.dimer_matrix(S_blocks, active_diagrams, dimer01, [(0,+1),(+1,0)])
Stest[8]  = XR_term.dimer_matrix(S_blocks, active_diagrams, dimer01, [(0,0),(+1,-1),(-1,+1)])
Stest[9]  = XR_term.dimer_matrix(S_blocks, active_diagrams, dimer01, [(0,-1),(-1,0)])
Stest[10] = XR_term.dimer_matrix(S_blocks, active_diagrams, dimer01, [(-1,-1)])

#########
# build Hamiltonian matrices (SH)
#########

SH_integrals = {}

h_integrals_ao = np.load(f"atomic_states/integrals/Be2_{displacement}_h_AO.npy")
v_integrals_ao = np.load(f"atomic_states/integrals/Be2_{displacement}_V_AO.npy")

U_h = np.zeros_like(h_integrals_ao)
U_h[:n_orb, :n_orb] = C#[:n_orb, :n_orb]
U_h[n_orb:2 * n_orb, n_orb:2 * n_orb] = C#[:n_orb, :n_orb]
U_h[2 * n_orb:3 * n_orb, 2 * n_orb:3 * n_orb] = C#[n_orb:, n_orb:]
U_h[3 * n_orb:4 * n_orb, 3 * n_orb:4 * n_orb] = C#[n_orb:, n_orb:]

h = U_h.T @ h_integrals_ao @ U_h

h = h.reshape((2, 2, n_orb, 2, 2, n_orb))  # spin frag spatial_orbs
h = np.swapaxes(h, 0, 1)
h = np.swapaxes(h, 3, 4)
h = h.reshape((2, 2 * n_orb, 2, 2 * n_orb)) # frag spin_orb

SH_integrals["h"] = {(0, 0): h[0, :, 0, :], (0, 1): h[0, :, 1, :],
                     (1, 0): h[1, :, 0, :], (1, 1): h[1, :, 1, :]}

# This step can be improved by utilizing the block-diagonal structure of U_h (4 diag blocks in restricted and 2 in unrestricted)
v = np.einsum("mi,mjkl->ijkl", U_h, np.einsum("nj,mnkl->mjkl", U_h, np.einsum("ok,mnol->mnkl", U_h, np.einsum("pl,mnop->mnol", U_h, v_integrals_ao))))

# antisymmetrize
v = (1/4) * (v - np.transpose(v, (0, 1, 3, 2)))

v = v.reshape((2, 2, n_orb, 2, 2, n_orb, 2, 2, n_orb, 2, 2, n_orb))  # spin frag spatial_orbs
v = np.swapaxes(v, 0, 1)
v = np.swapaxes(v, 3, 4)
v = np.swapaxes(v, 6, 7)
v = np.swapaxes(v, 9, 10)
v = v.reshape((2, 2 * n_orb, 2, 2 * n_orb, 2, 2 * n_orb, 2, 2 * n_orb))

from itertools import product
SH_integrals["v"] = {(i, j, k, l): v[i, :, j, :, k, :, l, :] for i, j, k, l in product([0, 1], [0, 1], [0, 1], [0, 1])}

# exchange terms are always taken care of by antisymmetrization, even if the fragments are different!
print("norm of v0101(pqrs) + v0110(pqsr)", np.linalg.norm(SH_integrals["v"][0, 1, 0, 1] + np.swapaxes(SH_integrals["v"][0, 1, 1, 0], 2, 3)))

# H1 pure/coupling should include diagonal terms, where h consists only of nuclear attraction integrals
# which are diagonal in their fragments, but exclude the atom number of this specific fragment
"""
H_integrals = {}
# "s" is not required for H1 and H2 terms
H_integrals["h"] = {key:SH_integrals["h"][key] for key in SH_integrals["h"]}#SH_integrals["h"]
H_integrals["v"] = {key:SH_integrals["v"][key] for key in SH_integrals["v"]}#SH_integrals["v"]

H_integrals["h"][(0, 0)] = np.load(f"test-data-4.5/U_1_0_0.npy")
H_integrals["h"][(1, 1)] = np.load(f"test-data-4.5/U_0_1_1.npy")

H_blocks = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=H_integrals, diagrams=SH_diagrams)
SH_blocks = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=SH_integrals, diagrams=SH_diagrams)

active_diagrams_H = {}
active_diagrams_H[2] = ["H1", "H2_pure_2_body"]

active_diagrams_SH = {}
active_diagrams_SH[2] = ["S1H1"]
"""

# correct for diagonals of higher electron orders by building Fock like one-electron integrals

D0 = Be.rho["ca"][0,0][0][0]
D1 = Be.rho["ca"][0,0][0][0]

# exchange terms are always taken care of by antisymmetrization, even if the fragments are different! Therefore give every J term a factor of 2
two_p_mean_field = {(0, 0): 2 * (np.einsum("sr,prqs->pq", D0, SH_integrals["v"][0, 0, 0, 0])
                                 + np.einsum("sr,prqs->pq", D1, SH_integrals["v"][0, 1, 0, 1])),
                    (0, 1): 2 * (np.einsum("sr,prqs->pq", D0, SH_integrals["v"][0, 0, 1, 0])
                                 + np.einsum("sr,prqs->pq", D1, SH_integrals["v"][0, 1, 1, 1])),
                    (1, 0): 2 * (np.einsum("sr,prqs->pq", D0, SH_integrals["v"][1, 0, 0, 0])
                                 + np.einsum("sr,prqs->pq", D1, SH_integrals["v"][1, 1, 0, 1])),
                    (1, 1): 2 * (np.einsum("sr,prqs->pq", D1, SH_integrals["v"][1, 1, 1, 1])
                                 + np.einsum("sr,prqs->pq", D0, SH_integrals["v"][1, 0, 1, 0]))}

fock_ints = {key: SH_integrals["h"][key] + two_p_mean_field[key] for key in SH_integrals["h"]}

SH_integrals_fock = {}
SH_integrals_fock["h"] = fock_ints

SH_integrals["s"] = overlaps
SH_integrals_fock["s"] = overlaps

tl.set_backend("pytorch")
# here we changed the backend, so everything, which shall be handled by anything involving
# a tensorly object, needs to be tl.tensor
# overlaps is not tl.tensor, while densities are already tl.tensor (list of list of tl.tensor),
# see how S is evaluated

# make integrals tl.tensor
"""
SH_integrals = {kind:{subblock:tl.tensor(SH_integrals[kind][subblock]) for subblock in SH_integrals[kind]} for kind in SH_integrals}
SH_integrals_fock = {kind:{subblock:tl.tensor(SH_integrals_fock[kind][subblock]) for subblock in SH_integrals_fock[kind]} for kind in SH_integrals_fock}
"""

U0 = tl.tensor(U0)
U1 = tl.tensor(U1)

# rotate integrals into projected basis and make them tl.tensor
SH_integrals = {kind:{subblock:orb_proj_ints(U0, U1, subblock, tl.tensor(SH_integrals[kind][subblock]))
                      for subblock in SH_integrals[kind]} for kind in SH_integrals}
SH_integrals_fock = {kind:{subblock:orb_proj_ints(U0, U1, subblock, tl.tensor(SH_integrals_fock[kind][subblock]))
                           for subblock in SH_integrals_fock[kind]} for kind in SH_integrals_fock}


#TODO: decompose the ERIs and densities with estimated rank from occ orb number


SH_blocks = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=SH_integrals, diagrams=SH_diagrams)

SH_blocks_fock = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=SH_integrals_fock, diagrams=SH_diagrams)

#active_diagrams_SH = {}
#active_diagrams_SH[2] = ["H1", "H2_pure_2_body", "S1H1"]
H1, H2, S1H1, S1H2 = {}, {}, {}, {}
H1[2] = ["H1"]
H2[2] = ["H2"]#_pure_2_body"]
S1H1[2] = ["S1H1"]
S1H2[2] = ["S1H2"]

SHtest = {}
SHtest2 = {}
import time
start = time.time()
#SHtest[6] = XR_term.dimer_matrix(H_blocks, active_diagrams_H, dimer01, [(+1,+1)])
#SHtest[7] = XR_term.dimer_matrix(H_blocks, active_diagrams_H, dimer01, [(0,+1),(+1,0)])
#SHtest[8] = XR_term.dimer_matrix(H_blocks, active_diagrams_H, dimer01, [(0,0),(+1,-1),(-1,+1)])
#SHtest[9] = XR_term.dimer_matrix(H_blocks, active_diagrams_H, dimer01, [(0,-1),(-1,0)])
#SHtest[10] = XR_term.dimer_matrix(H_blocks, active_diagrams_H, dimer01, [(-1,-1)])

#SHtest[6] = XR_term.dimer_matrix(SH_blocks, active_diagrams_SH, dimer01, [(+1,+1)])
#SHtest[7] = XR_term.dimer_matrix(SH_blocks, active_diagrams_SH, dimer01, [(0,+1),(+1,0)])
#SHtest[8] = XR_term.dimer_matrix(SH_blocks, active_diagrams_SH, dimer01, [(0,0),(+1,-1),(-1,+1)])
#SHtest[9] = XR_term.dimer_matrix(SH_blocks, active_diagrams_SH, dimer01, [(0,-1),(-1,0)])
#SHtest[10] = XR_term.dimer_matrix(SH_blocks, active_diagrams_SH, dimer01, [(-1,-1)])

SHtest[6] = XR_term.dimer_matrix(SH_blocks, H1, dimer01, [(+1,+1)])
SHtest[7] = XR_term.dimer_matrix(SH_blocks, H1, dimer01, [(0,+1),(+1,0)])
SHtest[8] = XR_term.dimer_matrix(SH_blocks, H1, dimer01, [(0,0),(+1,-1),(-1,+1)])
SHtest[9] = XR_term.dimer_matrix(SH_blocks, H1, dimer01, [(0,-1),(-1,0)])
SHtest[10] = XR_term.dimer_matrix(SH_blocks, H1, dimer01, [(-1,-1)])
H1_time = time.time()
#SHtest[6] += XR_term.dimer_matrix(SH_blocks, H2, dimer01, [(+1,+1)])
#SHtest[7] += XR_term.dimer_matrix(SH_blocks, H2, dimer01, [(0,+1),(+1,0)])
#SHtest[8] += XR_term.dimer_matrix(SH_blocks, H2, dimer01, [(0,0),(+1,-1),(-1,+1)])
#SHtest[9] += XR_term.dimer_matrix(SH_blocks, H2, dimer01, [(0,-1),(-1,0)])
#SHtest[10] += XR_term.dimer_matrix(SH_blocks, H2, dimer01, [(-1,-1)])
H2_time = time.time()
SHtest2[6] = XR_term.dimer_matrix(SH_blocks, S1H1, dimer01, [(+1,+1)])
SHtest2[7] = XR_term.dimer_matrix(SH_blocks, S1H1, dimer01, [(0,+1),(+1,0)])
SHtest2[8] = XR_term.dimer_matrix(SH_blocks, S1H1, dimer01, [(0,0),(+1,-1),(-1,+1)])
SHtest2[9] = XR_term.dimer_matrix(SH_blocks, S1H1, dimer01, [(0,-1),(-1,0)])
SHtest2[10] = XR_term.dimer_matrix(SH_blocks, S1H1, dimer01, [(-1,-1)])
"""
SHtest[6] += XR_term.dimer_matrix(SH_blocks_fock, S1H1, dimer01, [(+1,+1)])
SHtest[7] += XR_term.dimer_matrix(SH_blocks_fock, S1H1, dimer01, [(0,+1),(+1,0)])
SHtest[8] += XR_term.dimer_matrix(SH_blocks_fock, S1H1, dimer01, [(0,0),(+1,-1),(-1,+1)])
SHtest[9] += XR_term.dimer_matrix(SH_blocks_fock, S1H1, dimer01, [(0,-1),(-1,0)])
SHtest[10] += XR_term.dimer_matrix(SH_blocks_fock, S1H1, dimer01, [(-1,-1)])
"""
S1H1_time = time.time()
#SHtest[6] += XR_term.dimer_matrix(SH_blocks, S1H2, dimer01, [(+1,+1)])
#SHtest[7] += XR_term.dimer_matrix(SH_blocks, S1H2, dimer01, [(0,+1),(+1,0)])
#SHtest[8] += XR_term.dimer_matrix(SH_blocks, S1H2, dimer01, [(0,0),(+1,-1),(-1,+1)])
#SHtest[9] += XR_term.dimer_matrix(SH_blocks, S1H2, dimer01, [(0,-1),(-1,0)])
#SHtest[10] += XR_term.dimer_matrix(SH_blocks, S1H2, dimer01, [(-1,-1)])
S1H2_time = time.time()
print("timings: H1, H2, S1H1, S1H2", H1_time - start, H2_time - H1_time, S1H1_time - H2_time, S1H2_time - S1H1_time)


# Test against XR code data
"""
# generate the correct integrals
h_ref = {}
v_ref = {}
for i in [0, 1]:
    for j in [0, 1]:
        kin = np.load(f"test-data-4.5/T_{i}_{j}.npy")
        h_ref[(i, j)] = kin
        for k in [0, 1]:  # this sums over the nuclei of U
            nuc = np.load(f"test-data-4.5/U_{k}_{i}_{j}.npy")
            h_ref[(i, j)] += nuc
            for l in [0, 1]:
                v = np.load(f"test-data-4.5/V_{i}_{j}_{k}_{l}.npy")
                v_ref[(i, j, k, l)] = v

# compare the integrals
for key in h_ref:
    print(f"h_me - h_ref for {key}", np.linalg.norm(SH_integrals["h"][key] - h_ref[key]))

for key in v_ref:
    print(f"v_me - v_ref for {key}", np.linalg.norm(SH_integrals["v"][key] - v_ref[key]))
"""

# compare H1 and H2 to XR implementation
"""
H1_0_1_ref_voff = np.load("test-data-4.5/V_off/H2_0_1.npy")
H1_0_ref_voff = np.load("test-data-4.5/V_off/H1_0.npy")
H1_1_ref_voff = np.load("test-data-4.5/V_off/H1_1.npy")
H1_0_1_ref_hoff = np.load("test-data-4.5/h_off/H2_0_1.npy")
H1_0_ref_hoff = np.load("test-data-4.5/h_off/H1_0.npy")
H1_1_ref_hoff = np.load("test-data-4.5/h_off/H1_1.npy")
H1_0_1_ref = np.load("test-data-4.5/H2_0_1.npy")
H1_0_ref = np.load("test-data-4.5/H1_0.npy")
H1_1_ref = np.load("test-data-4.5/H1_1.npy")

H1_0_1_ref_hoff_from_sub = H1_0_1_ref - H1_0_1_ref_voff
H1_0_1_ref_voff_from_sub = H1_0_1_ref - H1_0_1_ref_hoff

idx6  = np.load("atomic_states/states/16-115-550/load=states:16-115-550:thresh=1e-6:4.5:u.pickle/4.5/idx-6.npy")
idx7  = np.load("atomic_states/states/16-115-550/load=states:16-115-550:thresh=1e-6:4.5:u.pickle/4.5/idx-7.npy")
idx8  = np.load("atomic_states/states/16-115-550/load=states:16-115-550:thresh=1e-6:4.5:u.pickle/4.5/idx-8.npy")
idx9  = np.load("atomic_states/states/16-115-550/load=states:16-115-550:thresh=1e-6:4.5:u.pickle/4.5/idx-9.npy")
idx10 = np.load("atomic_states/states/16-115-550/load=states:16-115-550:thresh=1e-6:4.5:u.pickle/4.5/idx-10.npy")

len6  = len(idx6)
len7  = len(idx7)
len8  = len(idx8)
len9  = len(idx9)
len10 = len(idx10)
dim = len6 + len7 + len8 + len9 + len10

reference_S = np.zeros((dim,dim))

for i in range(len6):
    ii = idx6[i]
    for j in range(len6):
        jj = idx6[j]
        reference_S[ii,jj] = SHtest[6][i,j]

for i in range(len7):
    ii = idx7[i]
    for j in range(len7):
        jj = idx7[j]
        reference_S[ii,jj] = SHtest[7][i,j]

for i in range(len8):
    ii = idx8[i]
    for j in range(len8):
        jj = idx8[j]
        reference_S[ii,jj] = SHtest[8][i,j]

for i in range(len9):
    ii = idx9[i]
    for j in range(len9):
        jj = idx9[j]
        reference_S[ii,jj] = SHtest[9][i,j]

for i in range(len10):
    ii = idx10[i]
    for j in range(len10):
        jj = idx10[j]
        reference_S[ii,jj] = SHtest[10][i,j]

H1_me = reference_S
print("H1 me, H1 ref, H2 ref", np.linalg.norm(H1_me), np.linalg.norm(H1_0_1_ref_voff_from_sub), np.linalg.norm(H1_0_1_ref_hoff_from_sub))
print("H2 me - H2 ref diags", np.linalg.norm(np.diag(H1_me) - np.diag(H1_0_1_ref_hoff_from_sub)))
print("H2 me - H2 ref", np.linalg.norm(H1_me - H1_0_1_ref_hoff_from_sub))
print("H1 me - H1 ref", np.linalg.norm(H1_me - H1_0_1_ref_voff))
print("H1 me - H1 ref", np.linalg.norm(H1_me - H1_0_1_ref_voff_from_sub))
print("H me - H1 from sub - H2 from sub", np.linalg.norm(H1_me - H1_0_1_ref_voff_from_sub - H1_0_1_ref_hoff_from_sub))
"""

#########
# evaluate against full brute force reference (XR')
#########

start = 0
for n_elec in [6,7,8,9,10]:
    Sref = np.load("atomic_states/states/16-115-550/load=states:16-115-550:thresh=1e-6:4.5:u.pickle/{}/S-{}.npy".format(displacement,n_elec))
    S = Stest[n_elec] 
    dim, _ = S.shape
    Sref -= np.identity(dim)
    S    -= np.identity(dim)
    Sdiff = S-Sref
    print("{:2d} Frobenius norm of Sref:   ".format(n_elec), np.linalg.norm(Sref))
    print("   Frobenius norm of S:      ",                np.linalg.norm(S))
    print("   Frobenius norm of S-Sref: ",                np.linalg.norm(Sdiff))
    full_H_ref = np.load("atomic_states/states/16-115-550/load=states:16-115-550:thresh=1e-6:4.5:u.pickle/{}/H-{}.npy".format(displacement,n_elec))
    Sref += np.identity(dim)
    SH_ref = Sref @ full_H_ref
    SH_diff = SHtest[n_elec] - SH_ref
    print("   Frobenius norm of SH, SHref: ",                np.linalg.norm(SHtest[n_elec]), np.linalg.norm(SH_ref))
    print("   Frobenius norm of SH-SHref: ",                np.linalg.norm(SH_diff))
    H1_final = SHtest[n_elec]
    S1H1_final = SHtest2[n_elec]
    S1H1_contracted = np.tensordot(S, H1_final, axes=([1], [0]))
    SH_diff = S1H1_contracted - S1H1_final
    print("norms of S1H1_contracted, S1H1, and diff", np.linalg.norm(S1H1_contracted), np.linalg.norm(S1H1_final), np.linalg.norm(SH_diff))
    # the following evaluates how good we can approximate S1H1 - [S1 H1], by only calculating
    # S1H1[i1, i2, j1, j2] if [S1 H1][i1, i2, j1, j2] is significant
    """
    both_large_count = 0
    contracted_large_count = 0
    final_large_count = 0
    final_large_sum_of_squares = 0
    for i in range(len(H1_final)):
        for j in range(len(H1_final)):
            if abs(SH_diff[i, j]) >= 1e-5:
                if abs(S1H1_contracted[i, j]) >= 1e-9 and abs(S1H1_final[i, j]) >= 1e-9:
                    both_large_count += 1
                elif abs(S1H1_contracted[i, j]) >= 1e-9 and abs(S1H1_final[i, j] )<= 1e-9:
                    contracted_large_count += 1
                elif abs(S1H1_contracted[i, j]) <= 1e-9 and abs(S1H1_final[i, j]) >= 1e-9:
                    final_large_count += 1
                    final_large_sum_of_squares += S1H1_final[i, j] ** 2
                else:
                    continue
    print("both large, contracted large, final large, final large norm", both_large_count, contracted_large_count, final_large_count, np.sqrt(final_large_sum_of_squares))
    """
    
