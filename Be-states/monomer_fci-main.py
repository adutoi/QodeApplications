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
import pickle
import numpy

from qode.util import parallel, output, textlog
import qode.atoms.integrals.spatial_to_spin as spatial_to_spin
import qode.atoms.integrals.external_engines.psi4_ints as integrals
from   qode.many_body.self_consistent_field.fermionic import RHF_RoothanHall_Nonorthogonal

import LMO_frozen_special_fci    # compute atom FCI states
from fci_space import fci_space_traits
import excitonic
import psi4_check

def MO_transform(H, V, C):
    H = C.T @ H @ C
    for _ in range(4):  V = numpy.tensordot(V, C, axes=([0],[0]))       # cycle through the tensor axes (this assumes everything is real)
    return H, V

basis_string = "6-31G"



dist, state_string = sys.argv[1:]



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

psi4_check.print_dimer_E(X_2.mol, basis_string)


# Put everything in terms of spin orbitals
C_1    = spatial_to_spin.one_electron_blocked(C_1)
H_1_MO = spatial_to_spin.one_electron_blocked(H_1_MO)
V_1_MO = spatial_to_spin.two_electron_blocked(V_1_MO)
#S_2    = spatial_to_spin.one_electron_blocked(S_2)
#H_2    = spatial_to_spin.one_electron_blocked(H_2)
#V_2    = spatial_to_spin.two_electron_blocked(V_2)
#C_2    = spatial_to_spin.one_electron_blocked(C_2)
#H_2_MO = spatial_to_spin.one_electron_blocked(H_2_MO)
#V_2_MO = spatial_to_spin.two_electron_blocked(V_2_MO)








n_1e_states, n_2e_states, n_3e_states = state_string.split("-")
n_states = { 2:int(n_2e_states),  1:int(n_1e_states),  3:int(n_3e_states) }	# keys reference number of valence electrons explicitly (this is a frozen-core Be2 code!)

print("Loading data ...", end="", flush=True)
h_mat       = H_1_MO
V_mat       = V_1_MO
C_scf_atom  = C_1
print("Done.")

frag_dim  = n_states[1] + n_states[2] + n_states[3]
frag_idx = { 2: (0, n_states[2]),  1: (n_states[2], n_states[2]+n_states[1]),  3: (n_states[2]+n_states[1], frag_dim) }
super_dim = frag_dim**2
# Given that dict keys are hard-coded, some of these could be hard coded too, but this makes it easier to read.
num_elec_atom         = 4	# For neutral (deviations handled explicitly, locally)
num_core_elec_atom    = 2
num_valence_elec_atom = num_elec_atom - num_core_elec_atom
num_spin_orbs_atom    = h_mat.shape[0]
num_spat_orbs_atom    = num_spin_orbs_atom // 2

# Compute atomic states
print("Building atomic eigenstates ...", flush=True)
U_1e, U_2e, U_3e, nrg = LMO_frozen_special_fci.compute_fci_vecs(num_spin_orbs_atom, num_core_elec_atom, h_mat, V_mat, n_states[1], n_states[2], n_states[3])	# Atomic FCI calcs
U_1e = U_1e[:,:n_states[1]].copy()	# Slice out the requisite number of 1e- eigenstates (in the basis of atomic orbitals)
U_2e = U_2e[:,:n_states[2]].copy()	# Slice out the requisite number of 2e- eigenstates (in the basis of atomic orbitals)
U_3e = U_3e[:,:n_states[3]].copy()	# Slice out the requisite number of 3e- eigenstates (in the basis of atomic orbitals)

print("monomer FCI done", nrg)
