#    (C) Copyright 2018, 2023 Anthony D. Dutoi
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
#     python [-u] monomer_fci-main.py <distance> <nC-nN-nA>
# where nC, nN, and nA are the numbers of cationic, neutral, and anionic states to use, respectively

import sys
import math
import pickle
import numpy

import qode
from qode.util import parallel, output, textlog
import qode.atoms.integrals.spatial_to_spin as spatial_to_spin
import qode.atoms.integrals.external_engines.psi4_ints as integrals
from   qode.many_body.self_consistent_field.fermionic import RHF_RoothanHall_Nonorthogonal

import field_op_ham
import configurations
from CI_space_traits import CI_space_traits
import excitonic
import psi4_check

from qode.util.PyC import Double

def MO_transform(H, V, C):
    H = C.T @ H @ C
    for _ in range(4):  V = numpy.tensordot(V, C, axes=([0],[0]))       # cycle through the tensor axes (this assumes everything is real)
    return H, V

basis_string = "6-31G"

n_threads = 1
dist = float(sys.argv[1])
if len(sys.argv)==3:  n_threads = int(sys.argv[2])



# Normal AO SCF of Be atom
n_elec_1 = 4
Be_1 = """\
Be
"""
S_1, T_1, U_1, V_1, X_1 = integrals.AO_ints(Be_1, basis_string)
H_1 = T_1 + U_1
_, _, C_1 = RHF_RoothanHall_Nonorthogonal(n_elec_1, (S_1,H_1,V_1), thresh=1e-12)
H_1_MO, V_1_MO = MO_transform(H_1, V_1, C_1)

# Set up dimer
n_elec_2 = 8
Be_2 = """\
Be
Be  1  {distance:f}
""".format(distance=dist)
S_2, T_2, U_2, V_2, X_2 = integrals.AO_ints(Be_2, basis_string)
H_2 = T_2 + U_2
Enuc_2 = X_2.mol.nuclear_repulsion_energy()

# Normal AO SCF Be dimer
energy_2, _, C_2 = RHF_RoothanHall_Nonorthogonal(n_elec_2, (S_2,H_2,V_2), thresh=1e-12)
print("As computed here     = ", energy_2 + Enuc_2)
H_2_MO, V_2_MO = MO_transform(H_2, V_2, C_2)

psi4_check.print_HF_energy(Be_2, basis_string)

# Put everything in terms of spin orbitals
H_1_MO = spatial_to_spin.one_electron(H_1_MO, "blocked")
V_1_MO = spatial_to_spin.two_electron(V_1_MO, "blocked") / 2
H_2_MO = spatial_to_spin.one_electron(H_2_MO, "blocked")
V_2_MO = spatial_to_spin.two_electron(V_2_MO, "blocked") / 2
V_1_MO -= V_1_MO.transpose(1,0,2,3)
V_1_MO -= V_1_MO.transpose(0,1,3,2)
V_1_MO /= 4
V_2_MO -= V_2_MO.transpose(1,0,2,3)
V_2_MO -= V_2_MO.transpose(0,1,3,2)
V_2_MO /= 4



num_elec_atom     = 4
num_spat_orb_atom = 9
core_orb_atom     = []
num_elec_atom_dn  = num_elec_atom//2
num_elec_atom_up  = num_elec_atom - num_elec_atom_dn

dn_configs_atom = configurations.all_configs(num_spat_orb_atom, num_elec_atom_dn-len(core_orb_atom), frozen_occ_orbs=core_orb_atom)
up_configs_atom = configurations.all_configs(num_spat_orb_atom, num_elec_atom_up-len(core_orb_atom), frozen_occ_orbs=core_orb_atom)
configs_atom    = configurations.tensor_product_configs([up_configs_atom,up_configs_atom], [num_spat_orb_atom,num_spat_orb_atom])

CI_space_atom = qode.math.linear_inner_product_space(CI_space_traits(configs_atom))
H     = CI_space_atom.lin_op(field_op_ham.Hamiltonian(H_1_MO, V_1_MO, n_threads=n_threads))
guess = CI_space_atom.member(CI_space_atom.aux.basis_vec([0,1,9,10]))

print((guess|H|guess))
(Eval,Evec), = qode.math.lanczos.lowest_eigen(H, [guess], thresh=1e-8)
print("\nmonomer E_gs = {}\n".format(Eval))



num_elec_dimer     = 2 * num_elec_atom
num_spat_orb_dimer = 2 * num_spat_orb_atom
core_orb_dimer     = []
num_elec_dimer_dn  = num_elec_dimer//2
num_elec_dimer_up  = num_elec_dimer - num_elec_dimer_dn

dn_configs_dimer = configurations.all_configs(num_spat_orb_dimer, num_elec_dimer_dn-len(core_orb_dimer), frozen_occ_orbs=core_orb_dimer)
up_configs_dimer = configurations.all_configs(num_spat_orb_dimer, num_elec_dimer_up-len(core_orb_dimer), frozen_occ_orbs=core_orb_dimer)
configs_dimer    = configurations.tensor_product_configs([dn_configs_dimer,up_configs_dimer], [num_spat_orb_dimer,num_spat_orb_dimer])

CI_space_dimer = qode.math.linear_inner_product_space(CI_space_traits(configs_dimer))
H     = CI_space_dimer.lin_op(field_op_ham.Hamiltonian(H_2_MO, V_2_MO, n_threads=n_threads))
guess = CI_space_dimer.member(CI_space_dimer.aux.basis_vec([0,1,2,3,18,19,20,21]))

print((guess|H|guess) + Enuc_2)
(Eval,Evec), = qode.math.lanczos.lowest_eigen(H, [guess], thresh=1e-8)
print("\ndimer E_gs = {}\n".format(Eval + Enuc_2))
