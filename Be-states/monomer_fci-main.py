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
from CI_space_traits import CI_space_traits
import excitonic
import psi4_check

from qode.util.PyC import Double

def MO_transform(H, V, C):
    H = C.T @ H @ C
    for _ in range(4):  V = numpy.tensordot(V, C, axes=([0],[0]))       # cycle through the tensor axes (this assumes everything is real)
    return H, V

basis_string = "6-31G"

dist = sys.argv[1]



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
""".format(distance=float(dist))
S_2, T_2, U_2, V_2, X_2 = integrals.AO_ints(Be_2, basis_string)
H_2 = T_2 + U_2
Enuc_2 = X_2.mol.nuclear_repulsion_energy()

# Normal AO SCF Be dimer
energy_2, _, C_2 = RHF_RoothanHall_Nonorthogonal(n_elec_2, (S_2,H_2,V_2), thresh=1e-12)
print("As computed here     = ", energy_2 + Enuc_2)
H_2_MO, V_2_MO = MO_transform(H_2, V_2, C_2)

psi4_check.print_HF_energy(Be_2, basis_string)

# Put everything in terms of spin orbitals
H_1_MO = spatial_to_spin.one_electron_blocked(H_1_MO)
V_1_MO = spatial_to_spin.two_electron_blocked(V_1_MO) / 2
H_2_MO = spatial_to_spin.one_electron_blocked(H_2_MO)
V_2_MO = spatial_to_spin.two_electron_blocked(V_2_MO) / 2





CI_space = qode.math.linear_inner_product_space(CI_space_traits)



# Given that dict keys are hard-coded, some of these could be hard coded too, but this makes it easier to read.
num_elec_atom         = 4	# For neutral (deviations handled explicitly, locally)
num_core_elec_atom    = 2
num_spat_orb_atom     = 9

num_elec_atom_dn = num_elec_atom//2
num_elec_atom_up = num_elec_atom - num_elec_atom_dn
configs = field_op_ham.fci_configs(num_spat_orb_atom, num_elec_atom_dn, num_elec_atom_up, num_core_elec_atom//2)

# Set up Hamiltonian and promote it and tensor product basis to living in that space
H = field_op_ham.Hamiltonian(H_1_MO, V_1_MO)
guess = numpy.zeros((1,len(configs)), dtype=Double.numpy)
idx = field_op_ham.find_index(0b000000011000000011, configs)
guess[0,idx] = 1

H = CI_space.lin_op(H)
guess = CI_space.member((configs,guess))
print((guess|H|guess))

# Find the dimer ground state (orthonormalize the basis because Lanczos only for Hermitian case, then back to non-ON basis)
print("Ground-state calculation ... ", flush=True)
(Eval,Evec), = qode.math.lanczos.lowest_eigen(H, [guess], thresh=1e-8)
print("... Done.  \n\nE_gs = {}\n".format(Eval))



# Given that dict keys are hard-coded, some of these could be hard coded too, but this makes it easier to read.
num_elec_dimer      = 2 * num_elec_atom
num_core_elec_dimer = 2 * num_core_elec_atom
num_spat_orb_dimer  = 2 * num_spat_orb_atom

num_elec_dimer_dn = num_elec_dimer//2
num_elec_dimer_up = num_elec_dimer - num_elec_dimer_dn
configs = field_op_ham.fci_configs(num_spat_orb_dimer, num_elec_dimer_dn, num_elec_dimer_up, num_core_elec_dimer//2)

# Set up Hamiltonian and promote it and tensor product basis to living in that space
H = field_op_ham.Hamiltonian(H_2_MO, V_2_MO)
guess = numpy.zeros((1,len(configs)), dtype=Double.numpy)
idx = field_op_ham.find_index(0b000000000000001111000000000000001111, configs)
guess[0,idx] = 1

H = CI_space.lin_op(H)
guess = CI_space.member((configs,guess))
print((guess|H|guess)+ Enuc_2)

# Find the dimer ground state (orthonormalize the basis because Lanczos only for Hermitian case, then back to non-ON basis)
print("Ground-state calculation ... ", flush=True)
(Eval,Evec), = qode.math.lanczos.lowest_eigen(H, [guess], thresh=1e-8)
print("... Done.  \n\nE_gs = {}\n".format(Eval+Enuc_2))