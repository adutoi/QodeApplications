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
import numpy
import torch
import tensorly
import qode.util
import qode.math
from   qode.atoms.integrals.fragments import AO_integrals, fragMO_integrals, spin_orb_integrals, Nuc_repulsion
import diagrammatic_expansion   # defines information structure for housing results of diagram evaluations
import XR_term                  # knows how to use ^this information to pack a matrix for use in XR model
import S_diagrams               # contains definitions of actual diagrams needed for S operator in BO rep
import SH_diagrams              # contains definitions of actual diagrams needed for SH operator in BO rep
from   Be631g import monomer_data as Be
import excitonic

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

# "Assemble" the supersystem from the displaced fragments
BeN = [Be((0,0,m*float(displacement))) for m in range(int(n_frag))]
for m,frag in enumerate(BeN):  frag.load_states(states, n_states)      # load the density tensors
BeN_rho = [frag.rho for frag in BeN]                                   # deprecate this:  diagrammatic_expansion.blocks should take BeN directly? (n_states and n_elec one level higher)

# Get the fragment-partitioned integrals
fragMO_ints = fragMO_integrals(AO_integrals(BeN), [frag.basis.MOcoeffs for frag in BeN], cache=True)    # get AO integrals and transform to frag MO basis
integrals = spin_orb_integrals(fragMO_ints, rule_wrappers=[tensorly_wrapper], cache=True)               # promote to spin-orbital rep (spin blocked)
integrals.N = Nuc_repulsion(BeN).matrix                                                                 # Add nuclear repulsion matrix to the integrals

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

# The engines that build the terms
S_blocks  = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=integrals.S, diagrams=S_diagrams)
SH_blocks = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=integrals,   diagrams=SH_diagrams)

# for comparison
body1_ref = numpy.load("reference/test-data-4.5/H1_0.npy")    # from old XR code (same as next line, but unaffected by hack)
body2_ref = numpy.load("reference/test-data-4.5/H2_0_1.npy")  # from old XR code (no S) hacked to use symmetric integrals
Sref, Href = {}, {}
for n_elec in [6,7,8,9,10]:  # full S and model-space-BO H from brute-force on dimer (ie, exact target, but one-body and two-body together)
    Sref[n_elec] = numpy.load("atomic_states/states/16-115-550/load=states:16-115-550:thresh=1e-6:4.5:u.pickle/{}/S-{}.npy".format(displacement,n_elec))
    Href[n_elec] = numpy.load("atomic_states/states/16-115-550/load=states:16-115-550:thresh=1e-6:4.5:u.pickle/{}/H-{}.npy".format(displacement,n_elec))

#########
# build and test
#########

# S
if True:

    active_S_diagrams = {
        0:  ["identity"],
        2:  ["s01", "s01s10", "s01s01", "s01s01s10", "s01s01s10s10", "s01s01s01s10"]
    }
    S = {}
    S[6]  = XR_term.dimer_matrix(S_blocks, active_S_diagrams, (0,1), [(+1,+1)])
    S[7]  = XR_term.dimer_matrix(S_blocks, active_S_diagrams, (0,1), [(0,+1),(+1,0)])
    S[8]  = XR_term.dimer_matrix(S_blocks, active_S_diagrams, (0,1), [(0,0),(+1,-1),(-1,+1)])
    S[9]  = XR_term.dimer_matrix(S_blocks, active_S_diagrams, (0,1), [(0,-1),(-1,0)])
    S[10] = XR_term.dimer_matrix(S_blocks, active_S_diagrams, (0,1), [(-1,-1)])

    for n_elec in [6,7,8,9,10]:
        dim, _ = Sref[n_elec].shape
        Id = numpy.identity(dim)
        print("{:2d} Frobenius norm of Sref and S:   ".format(n_elec), numpy.linalg.norm(Sref[n_elec]-Id), numpy.linalg.norm(S[n_elec]-Id))
        print("   Frobenius norm of S-Sref: ", numpy.linalg.norm(S[n_elec]-Sref[n_elec]))

# H1
if True:

    active_H1_diagrams = {1: ["n00", "t00", "u000", "v0000"]}
    H1_0 = XR_term.monomer_matrix(SH_blocks, active_H1_diagrams, 0, [0,+1,-1])
    H1_1 = XR_term.monomer_matrix(SH_blocks, active_H1_diagrams, 1, [0,+1,-1])

    print("1-body error 0:", numpy.linalg.norm(H1_0 - body1_ref))
    print("1-body error 1:", numpy.linalg.norm(H1_1 - body1_ref))
    vals, vecs = qode.util.sort_eigen(numpy.linalg.eig(H1_0))
    print("1-body ground-state energy:", vals[0])

    active_H2_diagrams = {2: ["n01", "t01", "u001", "u101", "u100", "v0101", "v0011", "v0010", "v0100"]}
    H2blocked = XR_term.dimer_matrix(SH_blocks, active_H2_diagrams, (0,1), [(0,0),(0,+1),(0,-1),(+1,0),(+1,+1),(+1,-1),(-1,0),(-1,+1),(-1,-1)])

    # well, this sucks.  reorder the states
    dims0 = [BeN[0].rho['n_states'][chg] for chg in [0,+1,-1]]
    dims1 = [BeN[1].rho['n_states'][chg] for chg in [0,+1,-1]]
    mapping2 = [[None]*sum(dims0) for _ in range(sum(dims1))]
    idx = 0
    beg0 = 0
    for dim0 in dims0:
        beg1 = 0
        for dim1 in dims1:
            for m in range(dim0):
                for n in range(dim1):
                    mapping2[beg0+m][beg1+n] = idx
                    idx += 1
            beg1 += dim1
        beg0 += dim0
    mapping = []
    for m in range(sum(dims0)):
        for n in range(sum(dims1)):
            mapping += [mapping2[m][n]]
    H2 = numpy.zeros(H2blocked.shape)
    for i,i_ in enumerate(mapping):
        for j,j_ in enumerate(mapping):
            H2[i,j] = H2blocked[i_,j_]

    print("norms of H2, H2ref, and diff", numpy.linalg.norm(H2), numpy.linalg.norm(body2_ref), numpy.linalg.norm(H2-body2_ref))
    out, resources = qode.util.output(log=qode.util.textlog(echo=True)), qode.util.parallel.resources(1)
    E, T = excitonic.ccsd(([body1_ref,body1_ref],[[None,body2_ref]]), out, resources)
    out.log("\nTotal Excitonic CCSD Energy (ref)  = ", E)
    E, T = excitonic.ccsd(([H1_0,H1_1],[[None,H2]]), out, resources)
    out.log("\nTotal Excitonic CCSD Energy (test) = ", E)

# Full H, Marco-style
if True:

    active_SH_diagrams = {
        1: [
               #"n00",
               "t00",
               "u000",
               "v0000"
           ],
        2: [
               #"s01n00",
               #"s01n11",
               "s01t00",
               "s10t00",
               "s10u000",
               "s01u000",
               "s01v0000",
               "s10v0000",
               #
               #"n01",
               "u100",
               "v0101",
               #
               "t01",
               "s10t01",
               "s01t01",
               "v0011",
               ##"s01v0011",
               "s10v0011",
               #
               "u001",
               "u101",
               "v0010",
               "v0100",
               #
               #"s01n01",
               "s01u100",
               "s10u100",
               "s01v0101",
               #
               ##"s01u001",
               ##"s01u101",
               "s10u001",
               "s10u101",
               "s01v0100",
               ##"s01v0010",
               "s10v0010"
               ##"s10v0100"
           ]
    }
    SH = {}
    SH[6]  = XR_term.dimer_matrix(SH_blocks, active_SH_diagrams, (0,1), [(+1,+1)])
    SH[7]  = XR_term.dimer_matrix(SH_blocks, active_SH_diagrams, (0,1), [(0,+1),(+1,0)])
    SH[8]  = XR_term.dimer_matrix(SH_blocks, active_SH_diagrams, (0,1), [(0,0),(+1,-1),(-1,+1)])
    SH[9]  = XR_term.dimer_matrix(SH_blocks, active_SH_diagrams, (0,1), [(0,-1),(-1,0)])
    SH[10] = XR_term.dimer_matrix(SH_blocks, active_SH_diagrams, (0,1), [(-1,-1)])

    #active_S1H1_diagrams = {2: ["s10t01", "s10u001", "s10u101", "s01t00", "s01u000", "s01u100", "s10t00", "s10u000", "s10u100", "s01t01", "s01u001", "s01u101"]}

    for n_elec in [6,7,8,9,10]:
        H = qode.math.precise_numpy_inverse(Sref[n_elec]) @ SH[n_elec]
        print("{:2d} Frobenius norm of H, Href: ".format(n_elec), numpy.linalg.norm(H), numpy.linalg.norm(Href[n_elec]))
        print(   "   Frobenius norm of H-Href:  ",         numpy.linalg.norm(H-Href[n_elec]))
        vals, vecs = qode.util.sort_eigen(numpy.linalg.eig(Href[n_elec]))
        print("   2-body ground-state electronic energy (ref):    ", vals[0])
        vals, vecs = qode.util.sort_eigen(numpy.linalg.eig(H))
        print("   2-body ground-state electronic energy (test):   ", vals[0])
        #S1H1_contracted = numpy.tensordot(S1[n_elec], H1[n_elec], axes=([1], [0]))
        #diff = S1H1_contracted - S1H1[n_elec]
        #print("norms of S1H1_contracted, S1H1, and diff", numpy.linalg.norm(S1H1_contracted), numpy.linalg.norm(S1H1[n_elec]), numpy.linalg.norm(diff))
