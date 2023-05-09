#    (C) Copyright 2023 Anthony D. Dutoi and Marco Bauer
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

import sys
import numpy as np
import tensorly as tl
#import Be631g
from   Be631g   import monomer_data as Be
import diagrammatic_expansion   # defines information structure for housing results of diagram evaluations
import XR_term                  # knows how to use ^this information to pack a matrix for use in XR model
import S_diagrams               # contains definitions of actual diagrams needed for S operator in BO rep
import SH_diagrams
##ADD import torch
from orb_projection import transformation_mat, orb_proj_ints, orb_proj_density
from   get_ints import get_ints

##ADD torch.set_num_threads(4)  # here we set the number of the CPU cores, even though pytorch is not used yet

#tl.set_backend("pytorch")

#########
# Load data
#########

# Read information about the supersystem from the command line (assume a homogeneous system of identical, evenly spaced fragments in a line)
# and load up some hard-coded info for frozen-core, generalized CI 6-31G Be atoms, most importantly reduced descriptions of the fragment
# many-electron states that are suitable for contracting with the integrals.
#n_frag       = sys.argv[1]    # The number of fragments (presumed in a line for now)
#displacement = sys.argv[2]    # The distance between those fragments
#states       = sys.argv[3]    # Location of the file to use for the fragment statespace (homogeneous system for now)
#if len(sys.argv)==5:  n_states = tuple(sys.argv[4].split("-"))    # How many of the given fragment states to use for each charge state
#else:                 n_states = ("all","all","all","all","all")
n_frag       = 2
displacement = sys.argv[1]
states       = "load=states:16-115-550:thresh=1e-6:4.5:u.pickle/4.5"
n_states     = ("all","all","all","all","all")

#Be = Be631g.monomer_data(None)
#n_orb = Be.basis.n_spatial_orb  # number of spatial orbitals for one atom
#C     = Be.basis.MOcoeffs       # n_orb x n_orb for restricted orbitals
#Be.load_states(states, n_states)

"""
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

BeN_rho = [Be.rho, Be.rho]
"""

# Assemble the supersystem from the displaced fragments
BeN = [Be((0,0,m*float(displacement))) for m in range(int(n_frag))]
integrals, nuc_repulsion = get_ints(BeN)
for m,frag in enumerate(BeN):  frag.load_states(states, n_states)

BeN_rho = [frag.rho for frag in BeN]   # diagrammatic_expansion.blocks should take BeN directly? (pull n_states and n_elec out of rho and put one level higher)


n_orb = BeN[0].basis.n_spatial_orb  # number of spatial orbitals for one atom
C     = BeN[0].basis.MOcoeffs       # n_orb x n_orb for restricted orbitals


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

overlaps = {}
overlaps[(0, 1)] = integrals.S[(0, 1)]
overlaps[(1, 0)] = integrals.S[(1, 0)]

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
BeN_rho_alt = {}
BeN_rho_alt[0] = {op_string:{charges:orb_proj_density(U0, BeN_rho[0][op_string][charges])
                         for charges in BeN_rho[0][op_string]} for op_string in BeN_rho[0] if len(op_string) < 6}
BeN_rho_alt[1] = {op_string:{charges:orb_proj_density(U1, BeN_rho[1][op_string][charges])
                         for charges in BeN_rho[1][op_string]} for op_string in BeN_rho[1] if len(op_string) < 6}

# reintroduce monkey patch for densities
for i in range(len(BeN_rho)):
    BeN_rho_alt[i]["n_elec"] = BeN_rho[i]["n_elec"]
    BeN_rho_alt[i]["n_states"] = BeN_rho[i]["n_states"]

S_blocks = diagrammatic_expansion.blocks(densities=BeN_rho_alt, integrals=overlaps_for_S, diagrams=S_diagrams)

active_diagrams = {}
active_diagrams[0] = ["identity"]
active_diagrams[1] = []
active_diagrams[2] = ["order1_CT1", "order2_CT0", "order2_CT2", "order3_CT1", "order4_CT0", "order4_CT2"]

dimer01 = (0,1)    # In theory, a subsystem of the full system

Stest = {}
Stest[6]  = XR_term.dimer_matrix(S_blocks, active_diagrams, dimer01, [(+1,+1)])
Stest[7]  = XR_term.dimer_matrix(S_blocks, active_diagrams, dimer01, [(0,+1),(+1,0)])
Stest[8]  = XR_term.dimer_matrix(S_blocks, active_diagrams, dimer01, [(0,0),(+1,-1),(-1,+1)])
Stest[9]  = XR_term.dimer_matrix(S_blocks, active_diagrams, dimer01, [(0,-1),(-1,0)])
Stest[10] = XR_term.dimer_matrix(S_blocks, active_diagrams, dimer01, [(-1,-1)])

#

SH_blocks_Tony = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=integrals, diagrams=SH_diagrams)

active_SH_diagrams_Tony = {}
active_SH_diagrams_Tony[0] = []
active_SH_diagrams_Tony[1] = ["order1", "order2"]
active_SH_diagrams_Tony[2] = []

SHtest1_0_Tony = XR_term.monomer_matrix(SH_blocks_Tony, active_SH_diagrams_Tony, 0, [0,+1,-1])
SHtest1_1_Tony = XR_term.monomer_matrix(SH_blocks_Tony, active_SH_diagrams_Tony, 1, [0,+1,-1])
SHtest2_Tony = {}
SHtest2_Tony[6]  = XR_term.dimer_matrix(SH_blocks_Tony, active_SH_diagrams_Tony, dimer01, [(+1,+1)])
SHtest2_Tony[7]  = XR_term.dimer_matrix(SH_blocks_Tony, active_SH_diagrams_Tony, dimer01, [(0,+1),(+1,0)])
SHtest2_Tony[8]  = XR_term.dimer_matrix(SH_blocks_Tony, active_SH_diagrams_Tony, dimer01, [(0,0),(+1,-1),(-1,+1)])
SHtest2_Tony[9]  = XR_term.dimer_matrix(SH_blocks_Tony, active_SH_diagrams_Tony, dimer01, [(0,-1),(-1,0)])
SHtest2_Tony[10] = XR_term.dimer_matrix(SH_blocks_Tony, active_SH_diagrams_Tony, dimer01, [(-1,-1)])

#########
# build Hamiltonian matrices (SH)
#########

class _empty(object):  pass    # Basically just a dictionary

SH_integrals = _empty()

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

SH_integrals.h = {(0, 0): h[0, :, 0, :], (0, 1): h[0, :, 1, :],
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
SH_integrals.v = {(i, j, k, l): v[i, :, j, :, k, :, l, :] for i, j, k, l in product([0, 1], [0, 1], [0, 1], [0, 1])}

# exchange terms are always taken care of by antisymmetrization, even if the fragments are different!
print("norm of v0101(pqrs) + v0110(pqsr)", np.linalg.norm(SH_integrals.v[0, 1, 0, 1] + np.swapaxes(SH_integrals.v[0, 1, 1, 0], 2, 3)))

# correct for diagonals of higher electron orders by building Fock like one-electron integrals

D0 = BeN[0].rho["ca"][0,0][0][0]
D1 = BeN[0].rho["ca"][0,0][0][0]

# exchange terms are always taken care of by antisymmetrization, even if the fragments are different! Therefore give every J term a factor of 2
two_p_mean_field = {(0, 0): 2 * (np.einsum("sr,prqs->pq", D0, SH_integrals.v[0, 0, 0, 0])
                                 + np.einsum("sr,prqs->pq", D1, SH_integrals.v[0, 1, 0, 1])),
                    (0, 1): 2 * (np.einsum("sr,prqs->pq", D0, SH_integrals.v[0, 0, 1, 0])
                                 + np.einsum("sr,prqs->pq", D1, SH_integrals.v[0, 1, 1, 1])),
                    (1, 0): 2 * (np.einsum("sr,prqs->pq", D0, SH_integrals.v[1, 0, 0, 0])
                                 + np.einsum("sr,prqs->pq", D1, SH_integrals.v[1, 1, 0, 1])),
                    (1, 1): 2 * (np.einsum("sr,prqs->pq", D1, SH_integrals.v[1, 1, 1, 1])
                                 + np.einsum("sr,prqs->pq", D0, SH_integrals.v[1, 0, 1, 0]))}

fock_ints = {key: SH_integrals.h[key] + two_p_mean_field[key] for key in SH_integrals.h}

SH_integrals_fock = _empty()
SH_integrals_fock.h = fock_ints

SH_integrals.s = overlaps
SH_integrals_fock.s = overlaps

##ADD tl.set_backend("pytorch")
# here we changed the backend, so everything, which shall be handled by anything involving
# a tensorly object, needs to be tl.tensor
# overlaps is not tl.tensor, while densities are already tl.tensor (list of list of tl.tensor),
# see how S is evaluated

U0 = tl.tensor(U0)
U1 = tl.tensor(U1)

# rotate integrals into projected basis and make them tl.tensor
SH_integrals.s      = {subblock:orb_proj_ints(U0, U1, subblock, tl.tensor(SH_integrals.s[subblock]))      for subblock in SH_integrals.s}
SH_integrals.h      = {subblock:orb_proj_ints(U0, U1, subblock, tl.tensor(SH_integrals.h[subblock]))      for subblock in SH_integrals.h}
SH_integrals.v      = {subblock:orb_proj_ints(U0, U1, subblock, tl.tensor(SH_integrals.v[subblock]))      for subblock in SH_integrals.v}
SH_integrals_fock.s = {subblock:orb_proj_ints(U0, U1, subblock, tl.tensor(SH_integrals_fock.s[subblock])) for subblock in SH_integrals_fock.s}
SH_integrals_fock.h = {subblock:orb_proj_ints(U0, U1, subblock, tl.tensor(SH_integrals_fock.h[subblock])) for subblock in SH_integrals_fock.h}


#TODO: decompose the ERIs and densities with estimated rank from occ orb number


SH_blocks = diagrammatic_expansion.blocks(densities=BeN_rho_alt, integrals=SH_integrals, diagrams=SH_diagrams)

SH_blocks_fock = diagrammatic_expansion.blocks(densities=BeN_rho_alt, integrals=SH_integrals_fock, diagrams=SH_diagrams)

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

body1_ref = np.load("reference/test-data-4.5/H1_0.npy")
print("1-body error 0:", np.linalg.norm(SHtest1_0_Tony - body1_ref))
print("1-body error 1:", np.linalg.norm(SHtest1_1_Tony - body1_ref))
