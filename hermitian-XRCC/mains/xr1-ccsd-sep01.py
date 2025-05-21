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
#     python [-u] <this-file.py> <displacement> <rhos1> <rhos2> [no-proj]
# where <rhos> can be the filestem of any one of the .pkl files in atomic_states/ prepared by Be631g.py.

import time
import sys
import pickle
import numpy
#import torch
import tensorly
from tensorly import plugins as tensorly_plugins
import qode.util
from qode.util import struct, timer
import qode.math
import excitonic
import diagrammatic_expansion   # defines information structure for housing results of diagram evaluations
import XR_term                  # knows how to use ^this information to pack a matrix for use in XR model
import S_diagrams               # contains definitions of actual diagrams needed for S operator in BO rep
import St_diagrams              # contains definitions of actual diagrams needed for SH operator in BO rep
import Su_diagrams              # contains definitions of actual diagrams needed for SH operator in BO rep
import Sv_diagrams              # contains definitions of actual diagrams needed for SH operator in BO rep
from   get_ints import get_ints
from precontract import precontract

class empty(object):  pass     # needed for unpickling - remove when all Be-states drivers updated to use struct instead

#torch.set_num_threads(4)
#tensorly.set_backend("pytorch")
tensorly_plugins.use_opt_einsum()
qode.math.tensornet.backend_contract_path(True)

global_timings = timer()
matrix_timings = timer()
integral_timings = timer()
diagram_timings = timer()
precontract_timings = timer()
qode.math.tensornet.initialize_timer()
qode.math.tensornet.tensorly_backend.initialize_timer()

global_timings.start()

#########
# Load data
#########

# Information about the Be2 supersystem
n_frag       = 2
displacement = float(sys.argv[1])
states       = ["rho/{}.pkl".format(sys.argv[2]), "rho/{}.pkl".format(sys.argv[3])]
project_core = True
if len(sys.argv)==5:
    if sys.argv[4]=="no-proj":
        project_core = False

# "Assemble" the supersystem for the displaced fragments and get integrals
BeN = []
print("load states ...")
for m in range(int(n_frag)):
    Be = pickle.load(open(states[m],"rb"))
    for elem,coords in Be.atoms:  coords[2] += m * displacement    # displace along z
    BeN += [Be]
print("get_ints ...")
symm_ints, bior_ints, nuc_rep = get_ints(BeN, project_core, integral_timings)
print("done")

# The engines that build the terms
BeN_rho = [frag.rho for frag in BeN]   # diagrammatic_expansion.blocks should take BeN directly? (n_states and n_elec one level higher)
for BeN_rho_m in BeN_rho:                                # These lines to be removed when synced ...
    BeN_rho_m['n_states_bra'] = BeN_rho_m['n_states']    # ... up with Be states code again (now works with Be-states from main branch).
contract_cache = precontract(BeN_rho, symm_ints.S, precontract_timings)
S_blocks       = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=symm_ints.S,                               diagrams=S_diagrams,  contract_cache=contract_cache, timings=diagram_timings, precon_timings=precontract_timings)
St_blocks_symm = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=struct(S=symm_ints.S, T=symm_ints.T),      diagrams=St_diagrams, contract_cache=contract_cache, timings=diagram_timings, precon_timings=precontract_timings)
Su_blocks_symm = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=struct(S=symm_ints.S, U=symm_ints.U),      diagrams=Su_diagrams, contract_cache=contract_cache, timings=diagram_timings, precon_timings=precontract_timings)
Sv_blocks_symm = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=struct(S=symm_ints.S, V=symm_ints.V),      diagrams=Sv_diagrams, contract_cache=contract_cache, timings=diagram_timings, precon_timings=precontract_timings)
St_blocks_bior = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=struct(S=symm_ints.S, T=bior_ints.T),      diagrams=St_diagrams, contract_cache=contract_cache, timings=diagram_timings, precon_timings=precontract_timings)
Su_blocks_bior = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=struct(S=symm_ints.S, U=bior_ints.U),      diagrams=Su_diagrams, contract_cache=contract_cache, timings=diagram_timings, precon_timings=precontract_timings)
Sv_blocks_bior = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=struct(S=symm_ints.S, V=bior_ints.V),      diagrams=Sv_diagrams, contract_cache=contract_cache, timings=diagram_timings, precon_timings=precontract_timings)
Sv_blocks_half = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=struct(S=symm_ints.S, V=bior_ints.V_half), diagrams=Sv_diagrams, contract_cache=contract_cache, timings=diagram_timings, precon_timings=precontract_timings)
Sv_blocks_diff = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=struct(S=symm_ints.S, V=bior_ints.V_diff), diagrams=Sv_diagrams, contract_cache=contract_cache, timings=diagram_timings, precon_timings=precontract_timings)

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

global_timings.record("setup")
global_timings.start()

#########
# Build and test
#########

print("build H1")

H1 = []
for m in [0,1]:
    H1_m  = XR_term.monomer_matrix(St_blocks_symm, {
                          1: [
                              "t00"
                             ]
                         }, m, monomer_charges, matrix_timings)
    H1_m += XR_term.monomer_matrix(Su_blocks_symm, {
                          1: [
                              "u000"
                             ]
                         }, m, monomer_charges, matrix_timings)

    H1_m += XR_term.monomer_matrix(Sv_blocks_symm, {
                          1: [
                              "v0000"
                             ]
                         }, m, monomer_charges, matrix_timings)
    H1 += [H1_m]



print("build S2inv")

S2     = XR_term.dimer_matrix(S_blocks, {
                        0: [
                            "identity"
                           ],
                        2: [
                            "s01"
                           ]
                       },  (0,1), all_dimer_charges, matrix_timings)

S2inv = qode.math.precise_numpy_inverse(S2)



print("build S2H2 (1e)")

S2H2  = XR_term.dimer_matrix(St_blocks_symm, {
                       1: [
                           "t00"
                          ],
                       2: [
                           "t01"
                          ]
                      }, (0,1), all_dimer_charges, matrix_timings)
S2H2 += XR_term.dimer_matrix(Su_blocks_symm, {
                       1: [
                           "u000"
                          ],
                       2: [
                           "u100",
                           "u001", "u101"
                          ]
                     }, (0,1), all_dimer_charges, matrix_timings)

S2H2 += XR_term.dimer_matrix(St_blocks_bior, {
                       2: [
                           "s01t00", "s01t11",
                           "s01t10", 
                           "s01t01"
                          ]
                      }, (0,1), all_dimer_charges, matrix_timings)
S2H2 += XR_term.dimer_matrix(Su_blocks_bior, {
                       2: [
                           "s01u000", "s01u011",
                           "s01u100", "s01u111",
                           "s01u010", "s01u110", 
                           "s01u001", "s01u101"
                          ]
                      }, (0,1), all_dimer_charges, matrix_timings)

print("build S2H2 (2e)")

S2H2 += XR_term.dimer_matrix(Sv_blocks_diff, {
                       1: [
                           "v0000"
                          ],
                       2: [
                           "v0110", "v0010", "v0100", "v0011"
                          ]
                      }, (0,1), all_dimer_charges, matrix_timings)

S2H2 += XR_term.dimer_matrix(Sv_blocks_bior, {
                       2: [
                           "s01v0000", "s01v1111",
                           "s01v0110", "s01v1110", 
                           "s01v0100", 
                           "s01v1100", "s01v0010", "s01v0111"#, "s01v0011"
                          ]
                      }, (0,1), all_dimer_charges, matrix_timings)



print("build S2H2 (apply S2inv and subtract monomers)")

H2blocked = S2inv @ S2H2

H2blocked -= XR_term.dimer_matrix(St_blocks_symm, {
                       1: [
                           "t00"
                          ]
                      }, (0,1), all_dimer_charges, matrix_timings)
H2blocked -= XR_term.dimer_matrix(Su_blocks_symm, {
                       1: [
                           "u000"
                          ]
                     }, (0,1), all_dimer_charges, matrix_timings)

H2blocked -= XR_term.dimer_matrix(Sv_blocks_symm, {
                       1: [
                           "v0000"
                          ]
                      }, (0,1), all_dimer_charges, matrix_timings)

global_timings.record("build")
global_timings.start()



print("Apply H")

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

out, resources = struct(log=qode.util.textlog(echo=True)), qode.util.parallel.resources(1)
E, T = excitonic.ccsd((H1,[[None,H2],[None,None]]), out, resources)
E += sum(nuc_rep[m1,m2] for m1 in range(n_frag) for m2 in range(m1+1))
out.log("\nTotal Excitonic CCSD Energy (test) = ", E)

global_timings.record("apply")



global_timings.print("GLOBAL")
matrix_timings.print("MATRIX")
integral_timings.print("INTEGRALS")
diagram_timings.print("DIAGRAMS")
precontract_timings.print("PRECONTRACTIONS")
qode.math.tensornet.print_timings("TENSORNET COMPONENTS")
qode.math.tensornet.tensorly_backend.print_timings("TENSORLY BACKEND")
