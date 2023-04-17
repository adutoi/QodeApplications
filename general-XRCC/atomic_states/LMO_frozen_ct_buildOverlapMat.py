#    (C) Copyright 2018, 2019 Yuhong Liu and Anthony Dutoi
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
from multiprocessing import Pool, cpu_count
import numpy
from mol_fci import frozen_core_fci_states
from qode.util.PyC import import_C, BigInt, Double

LMO_frozen_ct_oneAtomBasisToTwoAtomBasisVec = import_C("LMO_frozen_ct_oneAtomBasisToTwoAtomBasisVec", flags="-O2 -fopenmp -lblas -llapack -lm")



def compute_transform_matrix(C_Be_long, 
			                 S_Be2, 
			                 C_Be2, 
			                 num_elec_atom, 
			                 num_spat_orbs_atom, 
			                 num_core_elec_dimer,
			                 U_1e,
			                 U_3e):
	#
	#
	# print("Be   C_HF_LONG   shape =", C_Be_long.shape)
	# print("Be2  S_Be2_halfE shape =", S_Be2.shape)
	# print("Be2  C_Be2_halfE shape =", C_Be2.shape)
	S_Atomic_overlap = C_Be_long.T * S_Be2 * C_Be2
	S_Atomic_overlap = (S_Atomic_overlap.T).copy()  # This ensures the contigous memory blocks for C code.
	print("S Atomic Overlap Matrix shape =", S_Atomic_overlap.shape)
	# print(numpy.diag(S_Atomic_overlap.T * S_Atomic_overlap))
	# print(S_Atomic_overlap.sum(), S_Atomic_overlap.mean(), S_Atomic_overlap.max(), S_Atomic_overlap.min())

	num_elec_dimer      = num_elec_atom       * 2
	num_spin_orbs_atom  = num_spat_orbs_atom  * 2
	num_spat_orbs_dimer = num_spat_orbs_atom  * 2
	num_spin_orbs_dimer = num_spat_orbs_dimer * 2
	num_core_elec_atom  = num_core_elec_dimer // 2
	num_valence_elec_atom = num_elec_atom - num_core_elec_atom

	
	Be_1e_fci_states  = numpy.array(frozen_core_fci_states(num_spin_orbs_atom, num_elec_atom-1, num_core_elec_atom)[1], dtype=BigInt.numpy)
	Be_3e_fci_states  = numpy.array(frozen_core_fci_states(num_spin_orbs_atom, num_elec_atom+1, num_core_elec_atom)[1], dtype=BigInt.numpy)

	Be2_1e_fci_states = numpy.array(frozen_core_fci_states(num_spin_orbs_dimer, num_core_elec_dimer + num_valence_elec_atom - 1, num_core_elec_dimer)[1], dtype=BigInt.numpy)
	Be2_3e_fci_states = numpy.array(frozen_core_fci_states(num_spin_orbs_dimer, num_core_elec_dimer + num_valence_elec_atom + 1, num_core_elec_dimer)[1], dtype=BigInt.numpy)
	
	#
	Be_n_state_1e  = Be_1e_fci_states.shape[0]
	Be_n_state_3e  = Be_3e_fci_states.shape[0]
	Be2_n_state_1e = Be2_1e_fci_states.shape[0]
	Be2_n_state_3e = Be2_3e_fci_states.shape[0]
	print('Be  1 val elec: num of FCI states =', Be_1e_fci_states.shape)
	print('Be  3 val elec: num of FCI states =', Be_3e_fci_states.shape)
	print('Be2 1 val elec: num of FCI states =', Be2_1e_fci_states.shape)
	print('Be2 3 val elec: num of FCI states =', Be2_3e_fci_states.shape)

	atomVec1e = numpy.matrix(numpy.zeros((Be2_n_state_1e, U_1e.shape[1]), dtype=Double.numpy))
	atomVec3e = numpy.matrix(numpy.zeros((Be2_n_state_3e, U_3e.shape[1]), dtype=Double.numpy))

	# 1e part:
	print('1e part')
	LMO_frozen_ct_oneAtomBasisToTwoAtomBasisVec.one_atom_basis_to_two_atom_basis( num_spin_orbs_atom, num_valence_elec_atom-1, Be_n_state_1e, Be2_n_state_1e, Be_1e_fci_states, Be2_1e_fci_states, S_Atomic_overlap, U_1e, U_1e.shape[1], atomVec1e, cpu_count() )

	# 3e part:
	print('3e part')
	LMO_frozen_ct_oneAtomBasisToTwoAtomBasisVec.one_atom_basis_to_two_atom_basis( num_spin_orbs_atom, num_valence_elec_atom+1, Be_n_state_3e, Be2_n_state_3e, Be_3e_fci_states, Be2_3e_fci_states, S_Atomic_overlap, U_3e, U_3e.shape[1], atomVec3e, cpu_count() )

	return atomVec1e, atomVec3e



def getBasisOverlapMatrix( C_scf_atom, C_scf_dimer, S_dimer, num_elec_atom, num_spat_orbs_atom, num_core_elec_dimer, U_1e, U_3e ):
	U_1e  = numpy.matrix(U_1e)
	U_3e  = numpy.matrix(U_3e)
	C_Be  = numpy.matrix( C_scf_atom )
	C_Be2 = numpy.matrix( C_scf_dimer )
	S_Be2 = numpy.matrix( S_dimer)
	num_spin_orbs_atom  = num_spat_orbs_atom * 2
	num_spin_orbs_dimer = num_spin_orbs_atom * 2
	#
	# C_Be_long
	C_Be_long = numpy.matrix(numpy.zeros((num_spin_orbs_dimer,num_spin_orbs_atom), dtype=Double.numpy))
	print("C_Be_long dim =", C_Be_long.shape)
	# Atom A:
	#
	c_ul = C_Be[:num_spat_orbs_atom,:num_spat_orbs_atom].copy()  # ul: Upper Left
	C_Be_long[:num_spat_orbs_atom,:num_spat_orbs_atom] = c_ul
	C_Be_long[num_spat_orbs_atom*2:num_spat_orbs_atom*3, num_spat_orbs_atom:] = c_ul
	atomHighVec1e, atomHighVec3e = compute_transform_matrix(C_Be_long, S_Be2, C_Be2, num_elec_atom, num_spat_orbs_atom, num_core_elec_dimer, U_1e, U_3e)
	# Clean up C_Be_long vector:
	C_Be_long.fill(0)
	# Atom B:
	#
	C_Be_long[num_spat_orbs_atom:num_spat_orbs_atom*2, :num_spat_orbs_atom] = c_ul
	C_Be_long[num_spat_orbs_atom*3:,num_spat_orbs_atom:] = c_ul
	atomLowVec1e, atomLowVec3e = compute_transform_matrix(C_Be_long, S_Be2, C_Be2, num_elec_atom, num_spat_orbs_atom, num_core_elec_dimer, U_1e, U_3e)
	return 	atomHighVec1e, atomHighVec3e, atomLowVec1e, atomLowVec3e




# def compute_det(first_idx, second_idx, num_valence_elec_atom, Be_fci_states, Be2_fci_states, S_Atomic_overlap):
# 	dimension = num_valence_elec_atom
# 	C_det = numpy.zeros((dimension,dimension))
# 	for k in range(dimension):
# 		for l in range(dimension):
# 			# print('Be2 FCI', Be2_fci_states[first_idx,k], 'Be FCI', Be_fci_states[second_idx,l
# 			C_det[k,l] = S_Atomic_overlap[Be2_fci_states[first_idx, k], Be_fci_states[second_idx, l]]
# 	return numpy.linalg.det(C_det)
# 
# # def compute_row(ith, num_valence_elec_atom, Be_n_state, Be_fci_states, Be2_fci_states, S_Atomic_overlap):
# def compute_row(*args):
# 	ith, num_valence_elec_atom, Be_n_state, Be_fci_states, Be2_fci_states, S_Atomic_overlap = args[0]
# 	results_row = numpy.matrix(numpy.zeros((1,Be_n_state)))
# 	for j in range(Be_n_state):
# 		results_row[0,j] = compute_det(ith,j, num_valence_elec_atom, Be_fci_states, Be2_fci_states, S_Atomic_overlap)
# 	return results_row
# 
# raise RuntimeError
# print("Computing Conversion Matrix...")
# myPool0  = Pool(20)
# results0 = myPool0.map(compute_column, [[ i, num_valence_elec_atom-1, Be_n_state_1e, Be_1e_fci_states, Be2_1e_fci_states, S_Atomic_overlap ] for i in range(Be2_n_state_1e)], )
# myPool0.terminate()
# myPool1  = Pool(20)
# results1 = myPool1.map(compute_column, [[ i, num_valence_elec_atom+1, Be_n_state_3e, Be_3e_fci_states, Be2_3e_fci_states, S_Atomic_overlap ] for i in range(Be2_n_state_3e)], )
# myPool1.terminate()
# print("Computation Done. Filling In Numbers...")
# for i in range(Be2_n_state_1e):
# 	for j in range(Be_n_state_1e):
# 		C_Mol_overlap_1e[i,j] = results0[i][0,j]
# print("C_Mol_overlap shape =", C_Mol_overlap_1e.shape)
# for i in range(Be2_n_state_3e):
# 	for j in range(Be_n_state_3e):
# 		C_Mol_overlap_3e[i,j] = results1[i][0,j]
# print("C_Mol_overlap shape =", C_Mol_overlap_3e.shape)
# return numpy.matrix(C_Mol_overlap_1e), numpy.matrix(C_Mol_overlap_3e)

