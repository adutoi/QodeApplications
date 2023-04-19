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
import qode.util
import diagrammatic_expansion   # defines information structure for housing results of diagram evaluations
import XR_term                  # knows how to use ^this information to pack a matrix for use in XR model
import S_diagrams               # contains definitions of actual diagrams needed for S operator in BO rep
import SH_diagrams              # contains definitions of actual diagrams needed for SH operator in BO rep
from   get_ints import get_ints
from   Be631g   import monomer_data as Be

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

# Assemble the supersystem from the displaced fragments
BeN = [Be((0,0,m*float(displacement))) for m in range(int(n_frag))]
integrals, nuc_repulsion = get_ints(BeN)
for m,frag in enumerate(BeN):  frag.load_states(states, n_states)

BeN_rho = [frag.rho for frag in BeN]   # diagrammatic_expansion.blocks should take BeN directly? (pull n_states and n_elec out of rho and put one level higher)

#########
# build overlap matrices
#########

dimer01 = (0,1)    # In theory, a subsystem of the full system

###

S_blocks  = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=integrals.S, diagrams=S_diagrams)

active_S_diagrams = {}
active_S_diagrams[0] = ["identity"]
active_S_diagrams[1] = []
active_S_diagrams[2] = ["order1_CT1", "order2_CT0", "order2_CT2", "order3_CT1", "order4_CT0", "order4_CT2"]

Stest = {}
Stest[6]  = XR_term.dimer_matrix(S_blocks, active_S_diagrams, dimer01, [(+1,+1)])
Stest[7]  = XR_term.dimer_matrix(S_blocks, active_S_diagrams, dimer01, [(0,+1),(+1,0)])
Stest[8]  = XR_term.dimer_matrix(S_blocks, active_S_diagrams, dimer01, [(0,0),(+1,-1),(-1,+1)])
Stest[9]  = XR_term.dimer_matrix(S_blocks, active_S_diagrams, dimer01, [(0,-1),(-1,0)])
Stest[10] = XR_term.dimer_matrix(S_blocks, active_S_diagrams, dimer01, [(-1,-1)])

###
"""
SH_blocks = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=integrals,   diagrams=SH_diagrams)

active_SH_diagrams = {}
active_SH_diagrams[0] = []
active_SH_diagrams[1] = ["order1", "order2"]
active_SH_diagrams[2] = []

SHtest1_0 = XR_term.monomer_matrix(SH_blocks, active_SH_diagrams, 0, [0,+1,-1])
SHtest1_1 = XR_term.monomer_matrix(SH_blocks, active_SH_diagrams, 1, [0,+1,-1])
SHtest2 = {}
SHtest2[6]  = XR_term.dimer_matrix(SH_blocks, active_SH_diagrams, dimer01, [(+1,+1)])
SHtest2[7]  = XR_term.dimer_matrix(SH_blocks, active_SH_diagrams, dimer01, [(0,+1),(+1,0)])
SHtest2[8]  = XR_term.dimer_matrix(SH_blocks, active_SH_diagrams, dimer01, [(0,0),(+1,-1),(-1,+1)])
SHtest2[9]  = XR_term.dimer_matrix(SH_blocks, active_SH_diagrams, dimer01, [(0,-1),(-1,0)])
SHtest2[10] = XR_term.dimer_matrix(SH_blocks, active_SH_diagrams, dimer01, [(-1,-1)])
"""
#########
# run tests
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
    print("   Frobenius norm of S:      ",                numpy.linalg.norm(S))
    print("   Frobenius norm of S-Sref: ",                numpy.linalg.norm(Sdiff))
    dumpS = open("viz-S-{}.dat".format(n_elec),"w")
    dumpd = open("viz-d-{}.dat".format(n_elec),"w")
    for i in range(dim):
        I = start + i
        for j in range(dim):
            J = start + j
            dumpS.write(f"{I} {J} {Sref[i][j]}\n")
            dumpd.write(f"{I} {J} {Sdiff[i][j]}\n")
        dumpS.write("\n")
        dumpd.write("\n")
    dumpS.close()
    dumpd.close()
    start = I + 1

###
"""
body1_ref = numpy.load("reference/test-data-4.5/H1_0.npy")
print("1-body error 0:", numpy.linalg.norm(SHtest1_0-body1_ref))
print("1-body error 1:", numpy.linalg.norm(SHtest1_1-body1_ref))
"""
#vals, vecs = qode.util.sort_eigen(numpy.linalg.eig(SHtest2[8]))
#print(vals[0])

