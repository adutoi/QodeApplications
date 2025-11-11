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
#     python [-u] <this-file.py> <displacement> <rhos>
# where <rhos> can be the filestem of any one of the .pkl files in atomic_states/rho prepared by Be631g.py.

import time
import sys
import pickle
import numpy
import qode.util
from qode.util import struct, timer
import qode.math
import excitonic
import diagrammatic_expansion   # defines information structure for housing results of diagram evaluations
import XR_term                  # knows how to use ^this information to pack a matrix for use in XR model
import S_diagrams               # contains definitions of actual diagrams needed for S operator in BO rep
from   get_ints import get_ints
from precontract import precontract

#########
# Load data
#########

timings = timer()    # starts the overall clock

# Information about the Be2 supersystem
n_frag       = 2
displacement = sys.argv[1]                                       # need to keep displacement as a string for later
states       = "atomic_states/rho/Be631g-thresh=1e-6:4.5.pkl"    # hard-coded bc very limited testing

# "Assemble" the supersystem for the displaced fragments and get integrals
BeN = []
for m in range(int(n_frag)):
    Be = pickle.load(open(states,"rb"))
    for elem,coords in Be.atoms:  coords[2] += m * float(displacement)    # displace along z
    BeN += [Be]
symm_ints, bior_ints, nuc_rep = get_ints(BeN, project_core=False)

# The engines that build the terms
BeN_rho = [frag.rho for frag in BeN]   # diagrammatic_expansion.blocks should take BeN directly? (n_states and n_elec one level higher)
contract_cache = precontract(BeN_rho, symm_ints.S, timings)
S_blocks       = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=symm_ints.S, diagrams=S_diagrams, contract_cache=contract_cache, timings=timings)

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

for chg in dimer_charges:
    S2 = XR_term.dimer_matrix(S_blocks, {
                                         0: [
                                             "identity"
                                            ],
                                         2: [
                                             "s01",
                                             "s01s10", "s01s01",
                                             "s01s01s10",
                                             "s01s01s10s10", "s01s01s01s10"
                                            ]
                                        },  (0,1), dimer_charges[chg])
    S2ref = numpy.load("atomic_states/states/16-115-550/load=states:16-115-550:thresh=1e-6:4.5:u.pickle/{}/S-{}.npy".format(displacement,chg))

    S2    -= numpy.identity(S2.shape[0])
    S2ref -= numpy.identity(S2.shape[0])
    print(" {:2d} ".format(chg), end="")
    print("Frobenius norm of Sref:    {}".format(numpy.linalg.norm(S2ref)))
    print("    ", end="")
    print("Frobenius norm of S:       {}".format(numpy.linalg.norm(S2)))
    print("    ", end="")
    print("Frobenius norm of S-Sref:  {}".format(numpy.linalg.norm(S2-S2ref)))
