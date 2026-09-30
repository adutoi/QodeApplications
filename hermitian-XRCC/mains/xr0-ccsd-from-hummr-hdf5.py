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
from dens_provider import DensBackend

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
target_charge = 0
target_multiplicity = 1
displacement = float(sys.argv[1])
#states       = ["rho/{}.pkl".format(sys.argv[2]), "rho/{}.pkl".format(sys.argv[3])]
project_core = True
if len(sys.argv)==5:
    if sys.argv[4]=="no-proj":
        project_core = False

roots = {
            0:  {1: 4},#, 3: 2},
            +1: {2: 4},
            -1: {2: 4, 4: 2},
        }

dens_builder = DensBackend("hummr")

# compute only unique fragments
frags = [["Be", "/home/marco/hummr_tests/scratch/be_631g_fci.inp", roots, 0]]
(mo_coeffs, states) = dens_builder.init_backend(frags)[0]

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


    #Be.basis.MOcoeffs = load_mos_from_hummr("/home/marco/hummr_tests/Be_mos.C0")
    Be.basis.MOcoeffs = mo_coeffs

    #Be.rho = load_densities_json("/home/marco/hummr_tests/hummr_dens_for_xr.json",
    #                             spin_adapted=True)
    Be.rho = dens_builder.build_densities("Be", 0, states, states)

    #Be.rho['n_states'] = {chg_a: chg_dens.shape[m] for (chg_a, chg_b), chg_dens in Be.rho["ca"].items()}
    #Be.rho['n_elec'] = {chgs[m]: Be.n_elec_ref - chgs[m] for chgs in Be.rho["ca"]}
    # TODO: give general build with mult = None for unrestricted
    Be.rho['n_states'] = {chg_a: {mult_a: mult_dens.shape[m] for (mult_a, mult_b, _, _), mult_dens in chg_dens.items() if mult_a == mult_b}
                          for (chg_a, chg_b), chg_dens in Be.rho["ca"].items()}
    Be.rho['n_elec'] = {chgs[m]: Be.n_elec_ref - chgs[m] for chgs in Be.rho["ca"]}

    for elem,coords in Be.atoms:  coords[2] += m * displacement    # displace along z
    BeN += [Be]
print("get_ints ...")
symm_ints, bior_ints, nuc_rep = get_ints(BeN, project_core, integral_timings, spin_ints=False, backend="lible")#"hdf5")
print("done")

print(BeN[0].rho["ca"][(0,0)][(1,1,0,1)][0,0,:,:])
#print(BeN[1].rho["a"][(0,-1)][(1,2,1,0)])
#print(BeN[0].rho["a"][(1,0)])
#print(BeN[0].rho["c"][(0,1)])

#eri_final_hummr_pre = numpy.loadtxt("eri.dat")
#eri_final_hummr = eri_final_hummr_pre.reshape((9, 9, 9, 9))

#print("U @ ca ", raw(bior_ints.U[0,0,0]("p", "q") @ BeN[0].rho["ca"][(0,0)][0,0,:,:]("p", "q")))
#print("T @ ca ", raw(bior_ints.T[0,0]("p", "q") @ BeN[0].rho["ca"][(0,0)][0,0,:,:]("p", "q")))
#print("V(prrs) @ ca ", numpy.einsum("prrq,pq->", eri_final_hummr, raw(BeN[0].rho["ca"][(0,0)][0,0,:,:])))
#print("norm ca 00 ", numpy.linalg.norm(raw(Be.rho["ca"][(0,0)][0,0,:,:])))
#print("V @ ccaa ", raw(bior_ints.V[0,0,0,0]("p", "q", "r", "s") @ BeN[0].rho["ccaa"][(0,0)][0,0,:,:,:,:]("p", "q", "s", "r")))
#print("V @ ccaa ", raw(bior_ints.V[0,0,0,0]("p", "q", "r", "s") @ BeN[0].rho["ccaa"][(0,0)][0,0,:,:,:,:]("p", "q", "r", "s")))
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

#print("test two elec ints antisymm")
#print("V0101 + V1001 ", numpy.linalg.norm(raw(bior_ints.V[0,1,0,1]) + numpy.swapaxes(raw(bior_ints.V[1,0,0,1]), 2, 3)))
#print("V0101 + V0110 ", numpy.linalg.norm(raw(bior_ints.V[0,1,0,1]) + numpy.swapaxes(raw(bior_ints.V[0,1,1,0]), 0, 1)))
#print("V0101 + V1010 ", numpy.linalg.norm(raw(bior_ints.V[0,1,0,1]) - numpy.swapaxes(numpy.swapaxes(raw(bior_ints.V[1,0,1,0]), 0, 1), 2, 3)))
#print("test two part tRDM antisymm")
#print("ccaa_pqrs + ccaa_pqsr ", numpy.linalg.norm(raw(BeN[0].rho["ccaa"][(0,0)][0,0,:,:,:,:]) + numpy.swapaxes(raw(BeN[0].rho["ccaa"][(0,0)][0,0,:,:,:,:]), 2, 3)))
#print("ccaa_pqrs + ccaa_qprs ", numpy.linalg.norm(raw(BeN[0].rho["ccaa"][(0,0)][0,0,:,:,:,:]) + numpy.swapaxes(raw(BeN[0].rho["ccaa"][(0,0)][0,0,:,:,:,:]), 0, 1)))
#print("ccaa_pqrs + ccaa_qpsr ", numpy.linalg.norm(raw(BeN[0].rho["ccaa"][(0,0)][0,0,:,:,:,:]) - numpy.swapaxes(numpy.swapaxes(raw(BeN[0].rho["ccaa"][(0,0)][0,0,:,:,:,:]), 0, 1), 2, 3)))

#numpy.save(open("ccaa00_hummr.npy", mode="wb"), ccaa00)
#numpy.save(open("V_antisym_hummr.npy", mode="wb"), raw(bior_ints.V[0,0,0,0]))
#numpy.save(open("ca00_hummr.npy", mode="wb"), raw(Be.rho["ca"][(0,0)][0,0,:,:]))

# The engines that build the terms
BeN_rho = [frag.rho for frag in BeN]   # diagrammatic_expansion.blocks should take BeN directly? (n_states and n_elec one level higher)
for BeN_rho_m in BeN_rho:                                # These lines to be removed when synced ...
    BeN_rho_m['n_states_bra'] = BeN_rho_m['n_states']    # ... up with Be states code again (now works with Be-states from main branch).
contract_cache = precontract(BeN_rho, symm_ints.S, precontract_timings)
S_blocks       = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=symm_ints.S,                               diagrams=S_diagrams,  contract_cache=contract_cache, timings=diagram_timings, precon_timings=precontract_timings, target_multiplicity=target_multiplicity)
ST_blocks_symm = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=struct(S=symm_ints.S, T=symm_ints.T),      diagrams=ST_diagrams, contract_cache=contract_cache, timings=diagram_timings, precon_timings=precontract_timings, target_multiplicity=target_multiplicity)
SU_blocks_symm = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=struct(S=symm_ints.S, U=symm_ints.U),      diagrams=SU_diagrams, contract_cache=contract_cache, timings=diagram_timings, precon_timings=precontract_timings, target_multiplicity=target_multiplicity)
SV_blocks_symm = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=struct(S=symm_ints.S, V=symm_ints.V),      diagrams=SV_diagrams, contract_cache=contract_cache, timings=diagram_timings, precon_timings=precontract_timings, target_multiplicity=target_multiplicity)
ST_blocks_bior = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=struct(S=symm_ints.S, T=bior_ints.T),      diagrams=ST_diagrams, contract_cache=contract_cache, timings=diagram_timings, precon_timings=precontract_timings, target_multiplicity=target_multiplicity)
SU_blocks_bior = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=struct(S=symm_ints.S, U=bior_ints.U),      diagrams=SU_diagrams, contract_cache=contract_cache, timings=diagram_timings, precon_timings=precontract_timings, target_multiplicity=target_multiplicity)
SV_blocks_bior = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=struct(S=symm_ints.S, V=bior_ints.V),      diagrams=SV_diagrams, contract_cache=contract_cache, timings=diagram_timings, precon_timings=precontract_timings, target_multiplicity=target_multiplicity)
#SV_blocks_half = diagrammatic_expansiegrals=struct(S=symm_ints.S, V=bior_ints.V_half), diagrams=SV_diagrams, contract_cache=contract_cache, timings=diagram_timings, precon_timings=precontract_timings)
#SV_blocks_diff = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=struct(S=symm_ints.S, V=bior_ints.V_diff), diagrams=SV_diagrams, contract_cache=contract_cache, timings=diagram_timings, precon_timings=precontract_timings)

# charges under consideration
monomer_charges = [0]#[0, +1, -1]
# spin-adapted
monomer_sectors = [
    (chg, mult)
    for chg in monomer_charges
    for mult in sorted(BeN[0].rho['n_states'][chg])
]
# unrestricted
#monomer_sectors = [
#    (chg, None)
#    for chg in monomer_charges
#]

all_dimer_charges = []
for i in monomer_charges:
    for j in monomer_charges:
        #if i + j != target_charge:
        #    continue
        all_dimer_charges.append((i,j))

def _multiplicities_couple(mult1, mult2, target_mult):
    """
    True if fragment multiplicities mult1 and mult2 can couple
    to target_mult.
    """

    if mult1 != None and mult2 != None:
        return (
            abs(mult1 - mult2) + 1 <= target_mult <= mult1 + mult2 - 1
            and
            (mult1 + mult2 + target_mult) % 2 == 1
        )
    elif mult1 == None and mult2 == None:
        return True
    else:
        raise NotImplementedError("One fragment is unrestricted and the other one is spin-adapted")

dimer_sectors = []
for chg_pair in all_dimer_charges:
    for mult_a in sorted(BeN[0].rho['n_states'][chg_pair[0]]):
        for mult_b in sorted(BeN[0].rho['n_states'][chg_pair[1]]):
            #if not _multiplicities_couple(mult_a, mult_b, target_multiplicity):
            #    print(f"mults {mult_a} and {mult_b} cannot yield {target_multiplicity} in zeroth order coupling")
            #    continue
            dimer_sectors.append(((chg_pair[0], mult_a), (chg_pair[1], mult_b)))

#dimer_charges = {
#                 6:  [(+1, +1)],
#                 7:  [(0, +1), (+1, 0)],
#                 8:  [(0, 0), (+1, -1), (-1, +1)],
#                 9:  [(0, -1), (-1, 0)],
#                 10: [(-1, -1)]
#                }
#all_dimer_charges = [(0,0), (0,+1), (0,-1), (+1,0), (+1,+1), (+1,-1), (-1,0), (-1,+1), (-1,-1)]

global_timings.record("setup")
global_timings.start()

#########
# Build and test
#########

print("build H1")

H1 = []
for m in [0,1]:
    H1 += [  XR_term.monomer_matrix(ST_blocks_symm, {1: ST1[0]}, m, monomer_sectors, matrix_timings) \
           + XR_term.monomer_matrix(SU_blocks_symm, {1: SU1[0]}, m, monomer_sectors, matrix_timings) \
           + XR_term.monomer_matrix(SV_blocks_symm, {1: SV1[0]}, m, monomer_sectors, matrix_timings) ]
#for i, row in enumerate(H1[0]):
#    for j, elem in enumerate(row):
#        if abs(elem) > 1e-2:
#            print((i,j), elem)
#print(numpy.linalg.norm(H1[0]))
#print(H1[0])
#print(H1[1])
print("build H2 (1e)")

H2 =   XR_term.dimer_matrix(ST_blocks_bior, {1: ST1[0], 2: ST2[0]}, (0,1), dimer_sectors, matrix_timings) \
     + XR_term.dimer_matrix(SU_blocks_bior, {1: SU1[0], 2: SU2[0]}, (0,1), dimer_sectors, matrix_timings)
#for i, row in enumerate(H2):
#    for j, elem in enumerate(row):
#        if abs(elem) > 1e-2:
#            print((i,j), elem)
#print(numpy.linalg.norm(H2))
print("build H2 (2e)")

H2 +=  XR_term.dimer_matrix(SV_blocks_bior, {1: SV1[0], 2: SV2[0]}, (0,1), dimer_sectors, matrix_timings)
#print(numpy.linalg.norm(H2))
#print(H2)
print("finish H2 (subtract monomers)")

H2blocked = H2
H2blocked -=  XR_term.dimer_matrix(ST_blocks_symm, {1: ST1[0]}, (0,1), dimer_sectors, matrix_timings) \
            + XR_term.dimer_matrix(SU_blocks_symm, {1: SU1[0]}, (0,1), dimer_sectors, matrix_timings) \
            + XR_term.dimer_matrix(SV_blocks_symm, {1: SV1[0]}, (0,1), dimer_sectors, matrix_timings)
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
"""
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
"""
# -------------------------------------------------------------------------
# Reorder the dimer Hamiltonian from dimer-sector ordering
#
#     dimer_sectors:
#         ((charge_1, mult_1), (charge_2, mult_2))
#
# into the tensor-product ordering
#
#     fragment-1 basis  x  fragment-2 basis
#
# where each fragment's local basis is ordered as
#
#     charge -> multiplicity -> state.
#
# H2blocked is in the ordering produced by XR_term.dimer_matrix().
# H2 will be in the ordering expected by FCI and by the state optimizer.
# -------------------------------------------------------------------------

rho0 = BeN[0].rho
rho1 = BeN[1].rho

# Local basis states of each fragment in the desired (charge, multiplicity)
# ordering.  Each entry is
#
#     (charge, multiplicity, state)
#
# and the position in the list is the local tensor-product basis index.
#
# In the unrestricted case the multiplicity dictionary has only the
# unrestricted level, so the same construction can be used with mult=None
# if the corresponding n_states dictionaries are structured that way.

frag0_states = []

for chg in monomer_charges:
    for mult in rho0['n_states'][chg]:
        for state in range(rho0['n_states'][chg][mult]):
            frag0_states.append((chg, mult, state))

frag1_states = []

for chg in monomer_charges:
    for mult in rho1['n_states'][chg]:
        for state in range(rho1['n_states'][chg][mult]):
            frag1_states.append((chg, mult, state))


# Map every dimer-sector basis state to its position in H2blocked.
#
# dimer_matrix() uses exactly this ordering:
#
#     for sector in dimer_sectors:
#         all states of fragment 1 in that sector
#         x
#         all states of fragment 2 in that sector
#
# Therefore construct the inverse lookup explicitly.

dimer_index = {}

idx = 0

for (chg1, mult1), (chg2, mult2) in dimer_sectors:

    n0 = rho0['n_states'][chg1][mult1]
    n1 = rho1['n_states'][chg2][mult2]

    for state0 in range(n0):
        for state1 in range(n1):

            dimer_index[
                (chg1, mult1, state0,
                 chg2, mult2, state1)
            ] = idx

            idx += 1


# Construct the permutation from the desired tensor-product ordering
# to the current dimer-sector ordering.
#
# H2new[i,j] = H2blocked[mapping[i], mapping[j]].

mapping = []

for chg0, mult0, state0 in frag0_states:

    for chg1, mult1, state1 in frag1_states:

        key = (
            chg0, mult0, state0,
            chg1, mult1, state1,
        )

        mapping.append(dimer_index[key])


# Apply the same permutation to rows and columns because the
# supersystem Hamiltonian is square.

H2 = H2blocked[numpy.ix_(mapping, mapping)]

print("H1[0] diagonal")
print(numpy.diagonal(H1[0]))
#print("H2 diagonal")
#print(numpy.diagonal(H2))
#print("H2[0,:]")
#print(H2[0, :])
#print("H2[:,0]")
#print(H2[:, 0])

out, resources = struct(log=qode.util.textlog(echo=True)), qode.util.parallel.resources(1)
#E, T = excitonic.ccsd((H1,[[None,H2],[None,None]]), out, resources)
E, T = excitonic.fci((H1,[[None,H2],[None,None]]), out, target_state=0)#target_state=slice(0, 20))
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
