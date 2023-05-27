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
import time
import numpy
import tensorly
import torch
import qode.util
from   qode.atoms.integrals.fragments import AO_integrals, semiMO_integrals, spin_orb_integrals, Nuc_repulsion
import diagrammatic_expansion   # defines information structure for housing results of diagram evaluations
import XR_term                  # knows how to use ^this information to pack a matrix for use in XR model
import S_diagrams               # contains definitions of actual diagrams needed for S operator in BO rep
import SH_diagrams              # contains definitions of actual diagrams needed for SH operator in BO rep
from   Be631g import monomer_data as Be

# Basically just a dictionary
class _empty(object):  pass

# Helper function passed to integrals engine to get them back as tensorly tensors
def tensorly_wrapper(rule):
    def wrap_it(*indices):
        return tensorly.tensor(rule(*indices), dtype=tensorly.float64)
    return wrap_it

#torch.set_num_threads(4)
#tensorly.set_backend("pytorch")

#########
# Load data
#########

# Information about the Be2 supersystem
n_frag       = 2
displacement = sys.argv[1]
states       = "load=states:16-115-550:thresh=1e-6:4.5:u.pickle/4.5"
n_states     = ("all","all","all","all","all")

# Assemble the supersystem from the displaced fragments
BeN = [Be((0,0,m*float(displacement))) for m in range(int(n_frag))]
for m,frag in enumerate(BeN):  frag.load_states(states, n_states)      # load the density tensors
BeN_rho = [frag.rho for frag in BeN]                                   # deprecate this:  diagrammatic_expansion.blocks should take BeN directly? (n_states and n_elec one level higher)

# Get the fragment-partitioned integrals
fragMO_ints = semiMO_integrals(AO_integrals(BeN), [frag.basis.MOcoeffs for frag in BeN], cache=True)    # get AO integrals and transform to frag MO basis
integrals = spin_orb_integrals(fragMO_ints, rule_wrappers=[tensorly_wrapper], cache=True)               # promote to spin-orbital rep (spin blocked)

# Add the nuclear repulsion matrix to the integrals
integrals.N = Nuc_repulsion(BeN).matrix

# Add the summed kinetic and nuclear-attraction integrals (dimer specific)
integrals.h = {}
integrals.h[0, 0] = integrals.T[0, 0] + integrals.U[0, 0, 0] + integrals.U[1, 0, 0]
integrals.h[0, 1] = integrals.T[0, 1] + integrals.U[0, 0, 1] + integrals.U[1, 0, 1]
integrals.h[1, 0] = integrals.T[1, 0] + integrals.U[0, 1, 0] + integrals.U[1, 1, 0]
integrals.h[1, 1] = integrals.T[1, 1] + integrals.U[0, 1, 1] + integrals.U[1, 1, 1]

# Add Fock-like one-electron integrals (dimer specific)
D0 = BeN[0].rho["ca"][0,0][0][0]
D1 = BeN[0].rho["ca"][0,0][0][0]
two_p_mean_field = {(0, 0): 2 * (  numpy.einsum("sr,prqs->pq", D0, integrals.V[0, 0, 0, 0])
                                 + numpy.einsum("sr,prqs->pq", D1, integrals.V[0, 1, 0, 1])),
                    (0, 1): 2 * (  numpy.einsum("sr,prqs->pq", D0, integrals.V[0, 0, 1, 0])
                                 + numpy.einsum("sr,prqs->pq", D1, integrals.V[0, 1, 1, 1])),
                    (1, 0): 2 * (  numpy.einsum("sr,prqs->pq", D0, integrals.V[1, 0, 0, 0])
                                 + numpy.einsum("sr,prqs->pq", D1, integrals.V[1, 1, 0, 1])),
                    (1, 1): 2 * (  numpy.einsum("sr,prqs->pq", D0, integrals.V[1, 0, 1, 0])
                                 + numpy.einsum("sr,prqs->pq", D1, integrals.V[1, 1, 1, 1]))}
integrals.f = {key: integrals.h[key] + two_p_mean_field[key] for key in integrals.h}

# In theory, a subsystem of the full system
dimer01 = (0,1)

#########
# build matrices
#########

S_blocks  = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=integrals.S, diagrams=S_diagrams)
SH_blocks = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=integrals,   diagrams=SH_diagrams)

active_S_diagrams = {}
active_S_diagrams[0] = ["identity"]
active_S_diagrams[1] = []
active_S_diagrams[2] = ["s01", "s01s10", "s01s01", "s01s01s10", "s01s01s10s10", "s01s01s01s10"]
Stest = {}
Stest[6]  = XR_term.dimer_matrix(S_blocks, active_S_diagrams, dimer01, [(+1,+1)])
Stest[7]  = XR_term.dimer_matrix(S_blocks, active_S_diagrams, dimer01, [(0,+1),(+1,0)])
Stest[8]  = XR_term.dimer_matrix(S_blocks, active_S_diagrams, dimer01, [(0,0),(+1,-1),(-1,+1)])
Stest[9]  = XR_term.dimer_matrix(S_blocks, active_S_diagrams, dimer01, [(0,-1),(-1,0)])
Stest[10] = XR_term.dimer_matrix(S_blocks, active_S_diagrams, dimer01, [(-1,-1)])

active_SH_diagrams = {}
active_SH_diagrams[0] = []
active_SH_diagrams[1] = ["n00", "t00", "u000", "v0000"]
active_SH_diagrams[2] = []
SHtest2 = {}
SHtest1_0 = XR_term.monomer_matrix(SH_blocks, active_SH_diagrams, 0,       [0,+1,-1])
SHtest1_1 = XR_term.monomer_matrix(SH_blocks, active_SH_diagrams, 1,       [0,+1,-1])
SHtest2[6]  = XR_term.dimer_matrix(SH_blocks, active_SH_diagrams, dimer01, [(+1,+1)])
SHtest2[7]  = XR_term.dimer_matrix(SH_blocks, active_SH_diagrams, dimer01, [(0,+1),(+1,0)])
SHtest2[8]  = XR_term.dimer_matrix(SH_blocks, active_SH_diagrams, dimer01, [(0,0),(+1,-1),(-1,+1)])
SHtest2[9]  = XR_term.dimer_matrix(SH_blocks, active_SH_diagrams, dimer01, [(0,-1),(-1,0)])
SHtest2[10] = XR_term.dimer_matrix(SH_blocks, active_SH_diagrams, dimer01, [(-1,-1)])

H1, S1H1 = {}, {}
H1[1]   = ["t00", "u000"]
H1[2]   = ["t01", "u001", "u101", "u100"]
S1H1[2] = ["s10t01", "s10u001", "s10u101", "s01t00", "s01u000", "s01u100", "s10t00", "s10u000", "s10u100", "s01t01", "s01u001", "s01u101"]
SHtest = {}
start = time.time()
SHtest[6]    = XR_term.dimer_matrix(SH_blocks, H1,   dimer01, [(+1,+1)])
SHtest[7]    = XR_term.dimer_matrix(SH_blocks, H1,   dimer01, [(0,+1),(+1,0)])
SHtest[8]    = XR_term.dimer_matrix(SH_blocks, H1,   dimer01, [(0,0),(+1,-1),(-1,+1)])
SHtest[9]    = XR_term.dimer_matrix(SH_blocks, H1,   dimer01, [(0,-1),(-1,0)])
SHtest[10]   = XR_term.dimer_matrix(SH_blocks, H1,   dimer01, [(-1,-1)])
H1_time = time.time()
H2_time = time.time()
SHtest11 = {}
SHtest11[6]  = XR_term.dimer_matrix(SH_blocks, S1H1, dimer01, [(+1,+1)])
SHtest11[7]  = XR_term.dimer_matrix(SH_blocks, S1H1, dimer01, [(0,+1),(+1,0)])
SHtest11[8]  = XR_term.dimer_matrix(SH_blocks, S1H1, dimer01, [(0,0),(+1,-1),(-1,+1)])
SHtest11[9]  = XR_term.dimer_matrix(SH_blocks, S1H1, dimer01, [(0,-1),(-1,0)])
SHtest11[10] = XR_term.dimer_matrix(SH_blocks, S1H1, dimer01, [(-1,-1)])
S1H1_time = time.time()
S1H2_time = time.time()
print("timings: H1, H2, S1H1, S1H2", H1_time - start, H2_time - H1_time, S1H1_time - H2_time, S1H2_time - S1H1_time)

#########
# evaluate against full brute force reference (XR')
#########

start = 0
for n_elec in [6,7,8,9,10]:
    Sref = numpy.load("atomic_states/states/16-115-550/load=states:16-115-550:thresh=1e-6:4.5:u.pickle/{}/S-{}.npy".format(displacement,n_elec))
    S = Stest[n_elec] 
    dim, _ = S.shape
    Sref -= numpy.identity(dim)
    S    -= numpy.identity(dim)
    Sdiff = S-Sref
    print("{:2d} Frobenius norm of Sref:   ".format(n_elec), numpy.linalg.norm(Sref))
    print("   Frobenius norm of S:      ",                   numpy.linalg.norm(S))
    print("   Frobenius norm of S-Sref: ",                   numpy.linalg.norm(Sdiff))
    full_H_ref = numpy.load("atomic_states/states/16-115-550/load=states:16-115-550:thresh=1e-6:4.5:u.pickle/{}/H-{}.npy".format(displacement,n_elec))
    Sref += numpy.identity(dim)
    SH_ref = Sref @ full_H_ref
    SH_diff = SHtest[n_elec] - SH_ref
    print("   Frobenius norm of SH, SHref: ",                numpy.linalg.norm(SHtest[n_elec]), numpy.linalg.norm(SH_ref))
    print("   Frobenius norm of SH-SHref: ",                 numpy.linalg.norm(SH_diff))
    H1_final = SHtest[n_elec]
    S1H1_final = SHtest11[n_elec]
    S1H1_contracted = numpy.tensordot(S, H1_final, axes=([1], [0]))
    SH_diff = S1H1_contracted - S1H1_final
    print("norms of S1H1_contracted, S1H1, and diff", numpy.linalg.norm(S1H1_contracted), numpy.linalg.norm(S1H1_final), numpy.linalg.norm(SH_diff))

body1_ref = numpy.load("reference/test-data-4.5/H1_0.npy")
print("1-body error 0:", numpy.linalg.norm(SHtest1_0 - body1_ref))
print("1-body error 1:", numpy.linalg.norm(SHtest1_1 - body1_ref))
