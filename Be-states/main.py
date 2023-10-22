#    (C) Copyright 2023 Anthony D. Dutoi
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
import qode

from qode.many_body.self_consistent_field.fermionic import RHF_RoothanHall_Nonorthogonal, RHF_RoothanHall_Orthonormal
from get_ints import get_ints
from qode.atoms.integrals.fragments import unblock_2, unblock_last2, unblock_4
import psi4_check

from CI_space_traits import CI_space_traits
import field_op_ham
import configurations

class _empty(object):  pass

n_threads = 1
dist = float(sys.argv[1])
if len(sys.argv)==3:  n_threads = int(sys.argv[2])


frag0 = _empty()
frag0.atoms = [("Be",[0,0,0])]
frag0.n_elec_ref = 4	# The number of electrons in the reference state of the monomer ("cation (+1)" and "anion (-1)" and technically interpreted relative to the reference, not zero, as would be the chemical definition)
frag0.basis = _empty()
frag0.basis.AOcode = "6-31G"
frag0.basis.n_spatial_orb = 9
frag0.basis.MOcoeffs = numpy.identity(frag0.basis.n_spatial_orb)    # rest of code assumes spin-restricted orbitals
frag0.basis.core = [0]	# indices of spatial MOs to freeze in CI portions



psi4_check.print_HF_energy(
    "".join("{} {} {} {}\n".format(A,x,y,z) for A,(x,y,z) in frag0.atoms),
    frag0.basis.AOcode
    )

symm_ints, bior_ints, nuc_rep = get_ints([frag0], spin_ints=False)
N, S, T, U, V = nuc_rep[0,0], symm_ints.S[0,0], symm_ints.T[0,0], symm_ints.U[0,0,0], symm_ints.V[0,0,0,0]
E, e, frag0.basis.MOcoeffs = RHF_RoothanHall_Nonorthogonal(frag0.n_elec_ref, (S, T+U, V), thresh=1e-12)
print(E)

symm_ints, bior_ints, nuc_rep = get_ints([frag0], spin_ints=False)
N, S, T, U, V = nuc_rep[0,0], symm_ints.S[0,0], symm_ints.T[0,0], symm_ints.U[0,0,0], symm_ints.V[0,0,0,0]
E, e, _ = RHF_RoothanHall_Orthonormal(frag0.n_elec_ref, (T+U, V), thresh=1e-12)
print(E)



frag1 = _empty()
frag1.atoms = [("Be",[0,0,dist])]
frag1.n_elec_ref = 4
frag1.basis = _empty()
frag1.basis.AOcode = "6-31G"
frag1.basis.n_spatial_orb = 9
frag1.basis.MOcoeffs = frag0.basis.MOcoeffs
frag1.basis.core = [0]

symm_ints, bior_ints, nuc_rep = get_ints([frag0,frag1])

num_elec_atom_dn = frag0.n_elec_ref // 2
num_elec_atom_up = frag0.n_elec_ref - num_elec_atom_dn
num_spatial_atom = frag0.basis.n_spatial_orb



dn_configs_atom = configurations.all_configs(num_spatial_atom, num_elec_atom_dn-len(frag0.basis.core), frozen_occ_orbs=frag0.basis.core)
up_configs_atom = configurations.all_configs(num_spatial_atom, num_elec_atom_up-len(frag0.basis.core), frozen_occ_orbs=frag0.basis.core)
configs_atom    = configurations.tensor_product_configs([up_configs_atom,up_configs_atom], [num_spatial_atom,num_spatial_atom])

N, S, T, U, V = nuc_rep[0,0], symm_ints.S[0,0], symm_ints.T[0,0], symm_ints.U[0,0,0], symm_ints.V[0,0,0,0]
h = T + U

CI_space_atom = qode.math.linear_inner_product_space(CI_space_traits(configs_atom))
H     = CI_space_atom.lin_op(field_op_ham.Hamiltonian(h,V, n_threads=n_threads))
guess = CI_space_atom.member(CI_space_atom.aux.basis_vec("000000011000000011"))

print((guess|H|guess) + N)
(Eval,Evec), = qode.math.lanczos.lowest_eigen(H, [guess], thresh=1e-8)
print("\nE_gs = {}\n".format(Eval+N))



dimer_core = frag0.basis.core + [c+frag0.basis.n_spatial_orb for c in frag1.basis.core]

dn_configs_dimer = configurations.all_configs(2*num_spatial_atom, 2*num_elec_atom_dn-len(dimer_core), frozen_occ_orbs=dimer_core)
up_configs_dimer = configurations.all_configs(2*num_spatial_atom, 2*num_elec_atom_up-len(dimer_core), frozen_occ_orbs=dimer_core)
dn_configs_decomp = configurations.decompose_configs(dn_configs_dimer, [num_spatial_atom, num_spatial_atom])
up_configs_decomp = configurations.decompose_configs(up_configs_dimer, [num_spatial_atom, num_spatial_atom])
combine_configs = configurations.config_combination([num_spatial_atom, num_spatial_atom])
nested = []
for config_dn1,configs_dn0 in dn_configs_decomp:
    for config_up1,configs_up0 in up_configs_decomp:
        config_1  = combine_configs([config_up1, config_dn1])
        configs_0 = configurations.tensor_product_configs([configs_up0, configs_dn0], [num_spatial_atom, num_spatial_atom])
        nested += [(config_1, configs_0)]
configs_dimer = configurations.recompose_configs(nested, [2*num_spatial_atom, 2*num_spatial_atom])

N = nuc_rep[0,0] + nuc_rep[1,1] + nuc_rep[0,1]
T = unblock_2(    bior_ints.T, [frag0,frag1], spin_orbs=True)
U = unblock_last2(bior_ints.U, [frag0,frag1], spin_orbs=True)
V = unblock_4(    bior_ints.V, [frag0,frag1], spin_orbs=True)
h = T + U[0] + U[1]

core = [0,9,18,27]
orbs = list(range(4*9))
for p in orbs:
    for q in orbs:
        if (q in core) and (p!=q):  h[p,q] = 0
        if (p in core) and (p!=q):  h[p,q] = 0
        for r in orbs:
            for s in orbs:
                if (r in core) and (p!=r) and (q!=r):  V[p,q,r,s] = 0
                if (s in core) and (p!=s) and (q!=s):  V[p,q,r,s] = 0
                if (p in core) and (p!=r) and (p!=s):  V[p,q,r,s] = 0
                if (q in core) and (q!=r) and (q!=s):  V[p,q,r,s] = 0

CI_space_dimer = qode.math.linear_inner_product_space(CI_space_traits(configs_dimer))
H     = CI_space_dimer.lin_op(field_op_ham.Hamiltonian(h,V, n_threads=n_threads))
guess = CI_space_dimer.member(CI_space_dimer.aux.basis_vec("000000011000000011000000011000000011"))

print((guess|H|guess) + N)
(Eval,Evec), = qode.math.lanczos.lowest_eigen(H, [guess], thresh=1e-8)
print("\nE_gs = {}\n".format(Eval+N))



















exit()

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

print("starting H1")

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


print("starting S2")

S2     = XR_term.dimer_matrix(S_blocks, {
                        0: [
                            "identity"
                           ]
                       },  (0,1), all_dimer_charges)

S2inv = qode.math.precise_numpy_inverse(S2)



print("starting S2H2")
print("starting N")

S2H2   = XR_term.dimer_matrix(Sn_blocks, {
                        2: [
                            "n01"
                           ]
                       }, (0,1), all_dimer_charges)

print("starting T")
S2H2  += XR_term.dimer_matrix(St_blocks_bior, {
                        2: [
                            "t01"
                           ]
                       }, (0,1), all_dimer_charges)
print("starting U")
S2H2  += XR_term.dimer_matrix(Su_blocks_bior, {
                        2: [
                            "u100",
                            "u001", "u101"
                           ]
                       }, (0,1), all_dimer_charges)

print("starting V")
S2H2  += XR_term.dimer_matrix(Sv_blocks_bior, {
                        2: [
                            "v0101", "v0010", "v0100", "v0011"
                           ]
                       }, (0,1), all_dimer_charges)


print("finished H build")

H2blocked = S2inv @ S2H2

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
