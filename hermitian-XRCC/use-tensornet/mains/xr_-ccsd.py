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

# Usage:
#     python [-u] <this-file.py> <displacement> <states>
# where <states> can be the name of any one of directories in atomic_states/states/16-115-550

import sys
import pickle
import numpy
#import torch
import tensorly
import qode.util
import qode.math
import excitonic
import diagrammatic_expansion   # defines information structure for housing results of diagram evaluations
import XR_term                  # knows how to use ^this information to pack a matrix for use in XR model
import S_diagrams               # contains definitions of actual diagrams needed for S operator in BO rep
import Sn_diagrams              # contains definitions of actual diagrams needed for SH operator in BO rep
import St_diagrams              # contains definitions of actual diagrams needed for SH operator in BO rep
import Su_diagrams              # contains definitions of actual diagrams needed for SH operator in BO rep
import Sv_diagrams              # contains definitions of actual diagrams needed for SH operator in BO rep
from   get_ints import get_ints

# needed for unpickling?!
class empty(object):  pass     # Basically just a dictionary class

#torch.set_num_threads(4)
#tensorly.set_backend("pytorch")

#########
# Load data
#########

# Information about the Be2 supersystem
n_frag       = 2
displacement = float(sys.argv[1])
states       = "atomic_states/{}.pkl".format(sys.argv[2])

# "Assemble" the supersystem for the displaced fragments and get integrals
BeN = []
for m in range(int(n_frag)):
    Be = pickle.load(open(states,"rb"))
    for elem,coords in Be.atoms:  coords[2] += m * displacement    # displace along z
    BeN += [Be]
symm_ints, bior_ints, nuc_rep = get_ints(BeN)

# The engines that build the terms
BeN_rho = [frag.rho for frag in BeN]   # diagrammatic_expansion.blocks should take BeN directly? (n_states and n_elec one level higher)
S_blocks       = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=symm_ints.S,                     diagrams=S_diagrams)
Sn_blocks      = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=(symm_ints.S, nuc_rep),          diagrams=Sn_diagrams)
St_blocks_symm = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=(symm_ints.S, symm_ints.T),      diagrams=St_diagrams)
Su_blocks_symm = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=(symm_ints.S, symm_ints.U),      diagrams=Su_diagrams)
Sv_blocks_symm = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=(symm_ints.S, symm_ints.V),      diagrams=Sv_diagrams)
St_blocks_bior = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=(symm_ints.S, bior_ints.T),      diagrams=St_diagrams)
Su_blocks_bior = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=(symm_ints.S, bior_ints.U),      diagrams=Su_diagrams)
Sv_blocks_bior = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=(symm_ints.S, bior_ints.V),      diagrams=Sv_diagrams)
Sv_blocks_half = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=(symm_ints.S, bior_ints.V_half), diagrams=Sv_diagrams)

# charges under consideration
monomer_charges = [0, +1, -1]
dimer_charges = {
                 6:  [(+1, +1)],
                 7:  [(0, +1), (+1, 0)],
                 8:  [(0, 0), (+1, -1), (-1, +1)],
                 9:  [(0, -1), (-1, 0)],
                 10: [(-1, -1)]
                }
all_dimer_charges = [(0,0), (0,+1), (0,-1), (+1,0), (+1,+1), (+1,-1), (-1,0), (-1,+1), (-1,-1)]

#########
# Build and test
#########

H1 = []
for m in [0,1]:
    H1_m  = XR_term.monomer_matrix(Sn_blocks, {
                          1: [
                              "n00"
                             ]
                         }, m, monomer_charges)

    H1_m += XR_term.monomer_matrix(St_blocks_bior, {
                          1: [
                              "t00"
                             ]
                         }, m, monomer_charges)
    H1_m += XR_term.monomer_matrix(Su_blocks_bior, {
                          1: [
                              "u000"
                             ]
                         }, m, monomer_charges)

    H1_m += XR_term.monomer_matrix(Sv_blocks_bior, {
                          1: [
                              "v0000"
                             ]
                         }, m, monomer_charges)

    H1 += [H1_m]



S2     = XR_term.dimer_matrix(S_blocks, {
                        0: [
                            "identity"
                           ],
                        2: [
                            "s01"
                            #"s01s10", "s01s01",
                            #"s01s01s10",
                            #"s01s01s10s10", "s01s01s01s10"
                           ]
                       },  (0,1), all_dimer_charges)

S2inv = qode.math.precise_numpy_inverse(S2)



S2H2  = XR_term.dimer_matrix(Sn_blocks, {
                       1: [
                           "n00"
                          ],
                       2: [
                           "n01",
                           "s01n00", "s01n11", "s01n01"
                          ]
                      }, (0,1), all_dimer_charges)

S2H2 += XR_term.dimer_matrix(St_blocks_symm, {
                       1: [
                           "t00"
                          ],
                       2: [
                           "t01",
                           #"s01t00", "s10t00",
                           #"s10t01", "s01t01"
                          ]
                      }, (0,1), all_dimer_charges)
S2H2 += XR_term.dimer_matrix(Su_blocks_symm, {
                       1: [
                           "u000"
                          ],
                       2: [
                           "u100",
                           "u001", "u101",
                           #"s01u000", "s10u000",
                           #"s01u100", "s10u100",
                           #"s10u001", "s10u101", "s01u001", "s01u101"
                          ]
                     }, (0,1), all_dimer_charges)

S2H2 += XR_term.dimer_matrix(St_blocks_bior, {
                       1: [
                           #"t00"
                          ],
                       2: [
                           #"t01",
                           "s01t00", "s10t00",
                           "s10t01", "s01t01"
                          ]
                      }, (0,1), all_dimer_charges)
S2H2 += XR_term.dimer_matrix(Su_blocks_bior, {
                       1: [
                           #"u000"
                          ],
                       2: [
                           #"u100",
                           #"u001", "u101",
                           "s01u000", "s10u000",
                           "s01u100", "s10u100",
                           "s10u001", "s10u101", "s01u001", "s01u101"
                          ]
                      }, (0,1), all_dimer_charges)

S2H2 -= XR_term.dimer_matrix(Sv_blocks_bior, {
                       1: [
                           "v0000"
                          ],
                       2: [
                           "v0101", "v0010", "v0100", "v0011",
                           #"s01v0000", "s10v0000",
                           #"s01v0101", "s10v0010", "s01v0100", "s10v0011", "s01v0010", "s10v0100"#, "s01v0011"
                          ]
                      }, (0,1), all_dimer_charges)

S2H2 += XR_term.dimer_matrix(Sv_blocks_half, {
                       1: [
                           "v0000"
                          ],
                       2: [
                           "v0101", "v0010", "v0100", "v0011",
                           #"s01v0000", "s10v0000",
                           #"s01v0101", "s10v0010", "s01v0100", "s10v0011", "s01v0010", "s10v0100"#, "s01v0011"
                          ]
                      }, (0,1), all_dimer_charges)

S2H2 += XR_term.dimer_matrix(Sv_blocks_bior, {
                       1: [
                           #"v0000"
                          ],
                       2: [
                           #"v0101", "v0010", "v0100", "v0011",
                           "s01v0000", "s10v0000",
                           "s01v0101", "s10v0010", "s01v0100", "s10v0011", "s01v0010", "s10v0100"#, "s01v0011"
                          ]
                      }, (0,1), all_dimer_charges)



H2blocked = S2inv @ S2H2




H2blocked -= XR_term.dimer_matrix(Sn_blocks, {
                       1: [
                           "n00"
                          ]
                      }, (0,1), all_dimer_charges)

H2blocked -= XR_term.dimer_matrix(St_blocks_bior, {
                       1: [
                           "t00"
                          ]
                      }, (0,1), all_dimer_charges)
H2blocked -= XR_term.dimer_matrix(Su_blocks_bior, {
                       1: [
                           "u000"
                          ]
                     }, (0,1), all_dimer_charges)

H2blocked -= XR_term.dimer_matrix(Sv_blocks_bior, {
                       1: [
                           "v0000"
                          ]
                      }, (0,1), all_dimer_charges)



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

out, resources = qode.util.output(log=qode.util.textlog(echo=True)), qode.util.parallel.resources(1)
E, T = excitonic.ccsd((H1,[[None,H2],[None,None]]), out, resources)
out.log("\nTotal Excitonic CCSD Energy (test) = ", E)
