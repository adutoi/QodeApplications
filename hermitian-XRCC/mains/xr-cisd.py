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
from   Be631g   import monomer_data as Be

#torch.set_num_threads(4)
#tensorly.set_backend("pytorch")

#########
# Load data
#########

# Information about the Be2 supersystem
n_frag       = 2
displacement = sys.argv[1]
states       = "{}/4.5".format(sys.argv[2])
n_states     = ("all","all","all","all","all")

# "Assemble" the supersystem from the displaced fragments
BeN = [Be((0,0,m*float(displacement))) for m in range(int(n_frag))]

# Load states and get integrals
for m,frag in enumerate(BeN):  frag.load_states(states, n_states)      # load the density tensors
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

n_elec = 8



S    = XR_term.dimer_matrix(S_blocks, {
                      0: [
                          "identity"
                         ],
                      2: []
                     },  (0,1), dimer_charges[n_elec])

Sinv = qode.math.precise_numpy_inverse(S)



SH   = XR_term.dimer_matrix(Sn_blocks, {
                      1: [
                          "n00"
                         ],
                      2: [
                          "n01"
                         ]
                     }, (0,1), dimer_charges[n_elec])

SH  += XR_term.dimer_matrix(St_blocks_bior, {
                      1: [
                          "t00"
                         ],
                      2: [
                          "t01"
                         ]
                     }, (0,1), dimer_charges[n_elec])
SH  += XR_term.dimer_matrix(Su_blocks_bior, {
                      1: [
                          "u000"
                         ],
                      2: [
                          "u100",
                          "u001", "u101"
                         ]
                     }, (0,1), dimer_charges[n_elec])

SH  += XR_term.dimer_matrix(Sv_blocks_bior, {
                      1: [
                          "v0000"
                         ],
                      2: [
                          "v0101", "v0010", "v0100", "v0011"
                         ]
                     }, (0,1), dimer_charges[n_elec])



vals, vecs = qode.util.sort_eigen(numpy.linalg.eig(Sinv @ SH))
print("2-body ground-state energy:", vals[0])
