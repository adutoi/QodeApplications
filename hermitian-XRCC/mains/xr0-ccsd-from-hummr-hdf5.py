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
from diagrams import S_diagrams, ST_diagrams, SU_diagrams, SV_diagrams    # definitions of diagrams needed for S and SH
from   get_ints import get_ints
from precontract import precontract
from diagram_lists import *

from dens_from_hummr import load_densities_json, load_mos_from_hummr

class empty(object):  pass     # needed for unpickling - remove when all Be-states drivers updated to use struct instead

#torch.set_num_threads(4)
#tensorly.set_backend("pytorch")
tensorly_plugins.use_opt_einsum()
qode.math.tensornet.backend_contract_path(False)

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
#states       = ["rho/{}.pkl".format(sys.argv[2]), "rho/{}.pkl".format(sys.argv[3])]
project_core = True
if len(sys.argv)==5:
    if sys.argv[4]=="no-proj":
        project_core = False

# "Assemble" the supersystem for the displaced fragments and get integrals
BeN = []
print("load states ...")
for m in range(int(n_frag)):
    #Be = pickle.load(open(states[m],"rb"))
    Be = empty()
    Be.atoms = [["Be", [0, 0, 0]]]
    Be.core = [0]
    Be.charge = 0
    Be.n_elec_ref = 4
    Be.basis = empty
    Be.basis.n_spatial_orb = 9
    Be.basis.AOcode = "6-31G"
    print("no core and therefore no core projection, as long as core is not provided as frozen core")
    Be.basis.core = []#[0]
    #Be.basis.MOcoeffs = pickle.load(open(f"/home/marco/QodeApplications/tests/ref_data/check_mos_{m}.pkl", "rb"))
    Be.basis.MOcoeffs = load_mos_from_hummr("/home/marco/hummr_tests/Be_mos.C0")
    #print(Be.basis.MOcoeffs)
    #print("hackish reshuffling of basis functions")
    # Mapping the AOs is essential
    #Be.basis.MOcoeffs = Be.basis.MOcoeffs[[0,1,3,4,5,2,6,7,8], :]  # row 2 to 5
    # The map below would yield the same ordering as for psi4, but this
    # would also require adapting the densities, so we simply use the
    # convention from the lible package here.
    #Be.basis.MOcoeffs = Be.basis.MOcoeffs[:, [0,1,2,3,4,8,5,6,7]]  # col 8 to 5
    #Be.rho = load_densities_hdf5(str(sys.argv[2 + m]))
    Be.rho = load_densities_json("/home/marco/hummr_tests/hummr_dens_for_xr.json")
    #for key, val in Be.rho.items():
    #    for key2, val2 in val.items():
    #        print(key, key2, val2.shape)
    Be.rho['n_states'] = {chg_a: chg_dens.shape[m] for (chg_a, chg_b), chg_dens in Be.rho["ca"].items()}
    Be.rho['n_elec'] = {chgs[m]: Be.n_elec_ref - chgs[m] for chgs in Be.rho["ca"]}
    #print(Be.rho["n_states"])
    #print("shapes of MOs and rhos ", Be.basis.MOcoeffs.shape, [(keys, vals.shape) for keys, vals in Be.rho["ca"].items()])
    #print(Be.basis.MOcoeffs)
    if m == 0:
        from qode.math.tensornet import raw
        print(raw(Be.rho["ca"][(0,0)])[0, 0, :9, :9])
        print(raw(Be.rho["ccaa"][(0,0)])[0, 0, :3, :3, :3, :3])
    #print(raw(Be.rho["a"][(1,0)])[0, 0, 1:9])
    #print(raw(Be.rho["a"][(1,0)])[1, 0, 1:9])
    #print(raw(Be.rho["a"][(1,0)])[2, 0, 1:9])
    #print(numpy.linalg.norm(raw(Be.rho["ca"][(0,0)])[0, 0, 1:9, 1:9]))
    #print(numpy.linalg.norm(raw(Be.rho["ca"][(0,0)])[0, 0, 10:18, 10:18]))
    #print(numpy.linalg.norm(raw(Be.rho["ca"][(0,0)])[0, 0, 1:9, 10:18]))
    #print(numpy.linalg.norm(raw(Be.rho["ca"][(0,0)])[0, 0, 10:18, 1:9]))
    #print(numpy.linalg.norm(raw(Be.rho["ca"][(0,0)])[0, 0, [0,9], [0,9]]))
    #print(numpy.linalg.norm(raw(Be.rho["ca"][(0,0)])[0, 0, :, :]))
    #print(numpy.linalg.norm(raw(Be.rho["ccaa"][(0,0)])[0, 0, :, :, :, :]))
    #print(numpy.linalg.norm(raw(Be.rho["ccaa"][(0,0)])[0, 0, 10:18, 1:9, 10:18, 1:9]))
    for elem,coords in Be.atoms:  coords[2] += m * displacement    # displace along z
    BeN += [Be]
print("get_ints ...")
symm_ints, bior_ints, nuc_rep = get_ints(BeN, project_core, integral_timings, spin_ints=False, backend="lible")#"hdf5")
print("done")

#eri_final_hummr_pre = numpy.loadtxt("eri.dat")
#eri_final_hummr = eri_final_hummr_pre.reshape((9, 9, 9, 9))

print("U @ ca ", raw(bior_ints.U[0,0,0]("p", "q") @ BeN[0].rho["ca"][(0,0)][0,0,:,:]("p", "q")))
print("T @ ca ", raw(bior_ints.T[0,0]("p", "q") @ BeN[0].rho["ca"][(0,0)][0,0,:,:]("p", "q")))
#print("V(prrs) @ ca ", numpy.einsum("prrq,pq->", eri_final_hummr, raw(BeN[0].rho["ca"][(0,0)][0,0,:,:])))
#print("norm ca 00 ", numpy.linalg.norm(raw(Be.rho["ca"][(0,0)][0,0,:,:])))
print("V @ ccaa ", raw(bior_ints.V[0,0,0,0]("p", "q", "r", "s") @ BeN[0].rho["ccaa"][(0,0)][0,0,:,:,:,:]("p", "q", "s", "r")))
print("V @ ccaa ", raw(bior_ints.V[0,0,0,0]("p", "q", "r", "s") @ BeN[0].rho["ccaa"][(0,0)][0,0,:,:,:,:]("p", "q", "r", "s")))
#print("V ", numpy.linalg.norm(raw(bior_ints.V[0,0,0,0])))
#print("norm ccaa 00 ", numpy.linalg.norm(raw(Be.rho["ccaa"][(0,0)][0,0,:,:,:,:])))
#print("T ", numpy.linalg.norm(raw(bior_ints.T[0,0])))
#print("U ", numpy.linalg.norm(raw(bior_ints.U[0,0,0])))
#ccaa00 = raw(Be.rho["ccaa"][(0,0)][0,0,:,:,:,:])
#ccaa00 = raw(bior_ints.V[0,0,0,0])
#lten = ccaa00.shape[0]
#for i in range(lten):
#    for j in range(lten):
#        for k in range(lten):
#            for l in range(lten):
#                if abs(ccaa00[i,j,k,l]) > 1e-2:
#                    print([i,j,k,l], ccaa00[i,j,k,l])

print("test two elec ints antisymm")
print("V0101 + V1001 ", numpy.linalg.norm(raw(bior_ints.V[0,1,0,1]) + numpy.swapaxes(raw(bior_ints.V[1,0,0,1]), 2, 3)))
print("V0101 + V0110 ", numpy.linalg.norm(raw(bior_ints.V[0,1,0,1]) + numpy.swapaxes(raw(bior_ints.V[0,1,1,0]), 0, 1)))
print("V0101 + V1010 ", numpy.linalg.norm(raw(bior_ints.V[0,1,0,1]) - numpy.swapaxes(numpy.swapaxes(raw(bior_ints.V[1,0,1,0]), 0, 1), 2, 3)))
print("test two part tRDM antisymm")
print("ccaa_pqrs + ccaa_pqsr ", numpy.linalg.norm(raw(BeN[0].rho["ccaa"][(0,0)][0,0,:,:,:,:]) + numpy.swapaxes(raw(BeN[0].rho["ccaa"][(0,0)][0,0,:,:,:,:]), 2, 3)))
print("ccaa_pqrs + ccaa_qprs ", numpy.linalg.norm(raw(BeN[0].rho["ccaa"][(0,0)][0,0,:,:,:,:]) + numpy.swapaxes(raw(BeN[0].rho["ccaa"][(0,0)][0,0,:,:,:,:]), 0, 1)))
print("ccaa_pqrs + ccaa_qpsr ", numpy.linalg.norm(raw(BeN[0].rho["ccaa"][(0,0)][0,0,:,:,:,:]) - numpy.swapaxes(numpy.swapaxes(raw(BeN[0].rho["ccaa"][(0,0)][0,0,:,:,:,:]), 0, 1), 2, 3)))

#numpy.save(open("ccaa00_hummr.npy", mode="wb"), ccaa00)
#numpy.save(open("V_antisym_hummr.npy", mode="wb"), raw(bior_ints.V[0,0,0,0]))
#numpy.save(open("ca00_hummr.npy", mode="wb"), raw(Be.rho["ca"][(0,0)][0,0,:,:]))

# The engines that build the terms
BeN_rho = [frag.rho for frag in BeN]   # diagrammatic_expansion.blocks should take BeN directly? (n_states and n_elec one level higher)
for BeN_rho_m in BeN_rho:                                # These lines to be removed when synced ...
    BeN_rho_m['n_states_bra'] = BeN_rho_m['n_states']    # ... up with Be states code again (now works with Be-states from main branch).
contract_cache = precontract(BeN_rho, symm_ints.S, precontract_timings)
S_blocks       = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=symm_ints.S,                               diagrams=S_diagrams,  contract_cache=contract_cache, timings=diagram_timings, precon_timings=precontract_timings)
ST_blocks_symm = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=struct(S=symm_ints.S, T=symm_ints.T),      diagrams=ST_diagrams, contract_cache=contract_cache, timings=diagram_timings, precon_timings=precontract_timings)
SU_blocks_symm = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=struct(S=symm_ints.S, U=symm_ints.U),      diagrams=SU_diagrams, contract_cache=contract_cache, timings=diagram_timings, precon_timings=precontract_timings)
SV_blocks_symm = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=struct(S=symm_ints.S, V=symm_ints.V),      diagrams=SV_diagrams, contract_cache=contract_cache, timings=diagram_timings, precon_timings=precontract_timings)
ST_blocks_bior = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=struct(S=symm_ints.S, T=bior_ints.T),      diagrams=ST_diagrams, contract_cache=contract_cache, timings=diagram_timings, precon_timings=precontract_timings)
SU_blocks_bior = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=struct(S=symm_ints.S, U=bior_ints.U),      diagrams=SU_diagrams, contract_cache=contract_cache, timings=diagram_timings, precon_timings=precontract_timings)
SV_blocks_bior = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=struct(S=symm_ints.S, V=bior_ints.V),      diagrams=SV_diagrams, contract_cache=contract_cache, timings=diagram_timings, precon_timings=precontract_timings)
#SV_blocks_half = diagrammatic_expansiegrals=struct(S=symm_ints.S, V=bior_ints.V_half), diagrams=SV_diagrams, contract_cache=contract_cache, timings=diagram_timings, precon_timings=precontract_timings)
#SV_blocks_diff = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=struct(S=symm_ints.S, V=bior_ints.V_diff), diagrams=SV_diagrams, contract_cache=contract_cache, timings=diagram_timings, precon_timings=precontract_timings)

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
    H1 += [  XR_term.monomer_matrix(ST_blocks_symm, {1: ST1[0]}, m, monomer_charges, matrix_timings) \
           + XR_term.monomer_matrix(SU_blocks_symm, {1: SU1[0]}, m, monomer_charges, matrix_timings) \
           + XR_term.monomer_matrix(SV_blocks_symm, {1: SV1[0]}, m, monomer_charges, matrix_timings) ]
#for i, row in enumerate(H1[0]):
#    for j, elem in enumerate(row):
#        if abs(elem) > 1e-2:
#            print((i,j), elem)
#print(numpy.linalg.norm(H1[0]))
#print(H1[0])
#print(H1[1])
print("build H2 (1e)")

H2 =   XR_term.dimer_matrix(ST_blocks_bior, {1: ST1[0], 2: ST2[0]}, (0,1), all_dimer_charges, matrix_timings) \
     + XR_term.dimer_matrix(SU_blocks_bior, {1: SU1[0], 2: SU2[0]}, (0,1), all_dimer_charges, matrix_timings)
#for i, row in enumerate(H2):
#    for j, elem in enumerate(row):
#        if abs(elem) > 1e-2:
#            print((i,j), elem)
#print(numpy.linalg.norm(H2))
print("build H2 (2e)")

H2 +=  XR_term.dimer_matrix(SV_blocks_bior, {1: SV1[0], 2: SV2[0]}, (0,1), all_dimer_charges, matrix_timings)
#print(numpy.linalg.norm(H2))
#print(H2)
print("finish H2 (subtract monomers)")

H2blocked = H2
H2blocked -=  XR_term.dimer_matrix(ST_blocks_symm, {1: ST1[0]}, (0,1), all_dimer_charges, matrix_timings) \
            + XR_term.dimer_matrix(SU_blocks_symm, {1: SU1[0]}, (0,1), all_dimer_charges, matrix_timings) \
            + XR_term.dimer_matrix(SV_blocks_symm, {1: SV1[0]}, (0,1), all_dimer_charges, matrix_timings)
#print(numpy.linalg.norm(H2))
#for i, row in enumerate(H2):
#    for j, elem in enumerate(row):
#        if abs(elem) > 1e-2:
#            print((i,j), elem)
#print(numpy.linalg.norm(H2))
#H2 = numpy.zeros_like(H2)
global_timings.record("build")
global_timings.start()
"""
from qode.util import sort_eigen
import scipy as sp
nn = 2
#sl = [slice(0, nn)]
sl = [slice(0, 2), slice(2, 4), slice(4, 6)]
d_slices = [sl, sl]
#chg0 = chg1 = 0
chgs = [0, 1, -1]
full = numpy.zeros((3 * nn, 3 * nn, 3 * nn, 3 * nn))
for chg0 in chgs:
    for chg1 in chgs:
        full[d_slices[0][chg0], d_slices[1][chg1], d_slices[0][chg0], d_slices[1][chg1]] +=\
            numpy.einsum("ij,kl->ikjl", H1[0][d_slices[0][chg0], d_slices[0][chg0]], numpy.eye(nn)) +\
            numpy.einsum("ij,kl->ikjl", numpy.eye(nn), H1[1][d_slices[1][chg1], d_slices[1][chg1]])
full = full.reshape(9 * nn * nn, 9 * nn * nn)
#print(full)
full_eigvals_raw, full_eigvec_l_unsorted, full_eigvec_r_unsorted = sp.linalg.eig(full, left=True, right=True)
full_eigvals_check, full_eigvec_r = sort_eigen((full_eigvals_raw, full_eigvec_r_unsorted))
print(full_eigvals_check)
"""
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
#E, T = excitonic.ccsd((H1,[[None,H2],[None,None]]), out, resources)
E, T = excitonic.fci((H1,[[None,H2],[None,None]]), out, target_state=slice(0, 20))
E += sum(nuc_rep[m1,m2] for m1 in range(n_frag) for m2 in range(m1+1))
out.log("\nTotal Excitonic Energy = ", E)

print("ref from HCI without frozen core = ", -29.22740841 -0.00004272)  # selected casscf + hci pt correction

global_timings.record("apply")

global_timings.print("GLOBAL")
matrix_timings.print("MATRIX")
integral_timings.print("INTEGRALS")
diagram_timings.print("DIAGRAMS")
precontract_timings.print("PRECONTRACTIONS")
qode.math.tensornet.print_timings("TENSORNET COMPONENTS")
qode.math.tensornet.tensorly_backend.print_timings("TENSORLY BACKEND")
