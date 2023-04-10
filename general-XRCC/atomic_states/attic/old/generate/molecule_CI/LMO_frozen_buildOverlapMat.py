#    (C) Copyright 2018 Anthony D. Dutoi
# 
#    This file is part of Qode.
# 
#    Qode is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
# 
#    Qode is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
# 
#    You should have received a copy of the GNU General Public License
#    along with Qode.  If not, see <http://www.gnu.org/licenses/>.
#
import numpy as np
from mol_fci import frozen_core_fci_states




def compute_det(first_idx, second_idx, num_valence_elec_atom, Be_fci_states, Be2_fci_states, S_Atomic_overlap):
	dimension = num_valence_elec_atom
	C_det = np.zeros((dimension,dimension))
	for k in range(dimension):
		for l in range(dimension):
			# print('Be2 FCI', Be2_fci_states[first_idx,k], 'Be FCI', Be_fci_states[second_idx,l
			C_det[k,l] = S_Atomic_overlap[Be2_fci_states[first_idx, k], Be_fci_states[second_idx, l]]
	return np.linalg.det(C_det)

def compute_column(*args):
	ith, num_valence_elec_atom, Be_n_state, Be_fci_states, Be2_fci_states, S_Atomic_overlap = args[0]
	results_vec = np.matrix(np.zeros((1,Be_n_state)))
	for j in range(Be_n_state):
		results_vec[0,j] = compute_det(ith,j, num_valence_elec_atom, Be_fci_states, Be2_fci_states, S_Atomic_overlap)
	return results_vec




def compute_transform_matrix(C_Be_long, 
			                 S_Be2, 
			                 C_Be2, 
			                 num_elec_atom, 
			                 num_spat_orbs_atom, 
			                 num_core_elec_dimer):
	#
	#
	# print("Be   C_HF_LONG   shape =", C_Be_long.shape)
	# print("Be2  S_Be2_halfE shape =", S_Be2.shape)
	# print("Be2  C_Be2_halfE shape =", C_Be2.shape)
	S_Atomic_overlap = C_Be_long.T * S_Be2 * C_Be2
	S_Atomic_overlap = (S_Atomic_overlap.T).copy()
	print("Atomic Be-Be2 Overlap Matrix shape =", S_Atomic_overlap.shape)

	num_elec_dimer      = num_elec_atom       * 2
	num_spin_orbs_atom  = num_spat_orbs_atom  * 2
	num_spat_orbs_dimer = num_spat_orbs_atom  * 2
	num_spin_orbs_dimer = num_spat_orbs_dimer * 2
	num_core_elec_atom  = num_core_elec_dimer // 2
	num_valence_elec_atom = num_elec_atom - num_core_elec_atom

	
	Be_fci_states = np.array(frozen_core_fci_states(num_spin_orbs_atom, num_elec_atom, num_core_elec_atom)[1])
	Be2_fci_states = np.array(frozen_core_fci_states(num_spin_orbs_dimer, num_core_elec_dimer + num_valence_elec_atom, num_core_elec_dimer)[1])
	Be_n_state  = Be_fci_states.shape[0]
	Be2_n_state = Be2_fci_states.shape[0]
	
	print('Be  full elec: num of FCI states =', Be_n_state)
	print('Be2 half elec: num of FCI states =', Be2_n_state)

	C_Mol_overlap = np.zeros((Be2_n_state, Be_n_state))
	print( 'C_Mol_overlap = %d Bytes = %.3f GB' %(C_Mol_overlap.nbytes, C_Mol_overlap.nbytes/1024**3) )

	# raise RuntimeError
	# print("Computing Conversion Matrix...")
	from multiprocessing import Pool
	myPool  = Pool(25)
	results = myPool.map(compute_column, [[ i, num_valence_elec_atom, Be_n_state, Be_fci_states, Be2_fci_states, S_Atomic_overlap ] for i in range(Be2_n_state)], )
	myPool.terminate()
	
	# print("Computation Done. Filling In Numbers...")
	for i in range(Be2_n_state):
		for j in range(Be_n_state):
			C_Mol_overlap[i,j] = results[i][0,j]
	print("C_Mol_overlap shape =", C_Mol_overlap.shape)

	# np.save( "S_Be_Be2_{}.npy", S_Atomic_overlap)
	# print("OneBody to TwoBody Transformation Matrix Done.")
	return np.matrix(C_Mol_overlap), S_Atomic_overlap





def getBasisOverlapMatrix( C_scf_atom,     C_scf_dimer,  S_dimer, \
	                       num_elec_atom,  num_spat_orbs_atom,    num_core_elec_dimer, U_2e ):
	U_2e = np.matrix(U_2e)
	# C_scf_oneBody: Converged Be SCF vectors (Be_C_HF_spin.npy)
	C_Be    = np.matrix( C_scf_atom )
	# C_scf_twoBody Converged Be2 4e SCF vectors (Be2_4e_100ang_C_HF_spin.npy)
	C_Be2   = np.matrix( C_scf_dimer )
	# S_dimer Overlap Matrix of Be2 w/ half num of electrons (Be2_4e_100ang_S_spin.npy)
	S_Be2   = np.matrix( S_dimer)

	# print("Be  C_HF        shape =", C_Be.shape)
	# print("Be2 C_HF(4e-)   shape =", C_Be2.shape)

	num_spin_orbs_atom  = num_spat_orbs_atom * 2
	num_spin_orbs_dimer = num_spin_orbs_atom * 2
	

	# C_Be_long
	C_Be_long = np.matrix(np.zeros((num_spin_orbs_dimer,num_spin_orbs_atom)))
	print("C_Be_long dim =", C_Be_long.shape)

	# Atom A:
	#
	# print('copy into C_Be_Long[ :',num_spat_orbs_atom,', :',num_spat_orbs_atom,']')
	# print('copy into C_Be_Long[',num_spin_orbs_atom,':',num_spat_orbs_atom*3,',', num_spat_orbs_atom,': ]')
	c_ul = C_Be[:num_spat_orbs_atom,:num_spat_orbs_atom].copy()  # ul: Upper Left
	C_Be_long[:num_spat_orbs_atom,:num_spat_orbs_atom] = c_ul
	C_Be_long[num_spat_orbs_atom*2:num_spat_orbs_atom*3, num_spat_orbs_atom:] = c_ul
	C_OverlapHigh, S_Atomic_overlap_High = compute_transform_matrix(C_Be_long, S_Be2, C_Be2, num_elec_atom, num_spat_orbs_atom, num_core_elec_dimer)
	# np.save("S_Be_Be2_High.npy", S_Atomic_overlap_High)

	# Clean up C_Be_long vector:
	C_Be_long.fill(0)
	
	# Atom B:
	#
	# print('copy into C_Be_Long[ ', num_spat_orbs_atom,':', num_spin_orbs_atom, ', :',num_spat_orbs_atom,']')
	# print('copy into C_Be_Long[',num_spat_orbs_atom*3,':, ',num_spat_orbs_atom, ': ]')
	C_Be_long[num_spat_orbs_atom:num_spat_orbs_atom*2, :num_spat_orbs_atom] = c_ul
	C_Be_long[num_spat_orbs_atom*3:,num_spat_orbs_atom:] = c_ul
	C_OverlapLow, S_Atomic_overlap_Low = compute_transform_matrix(C_Be_long, S_Be2, C_Be2, num_elec_atom, num_spat_orbs_atom, num_core_elec_dimer)
	# np.save("S_Be_Be2_Low.npy", S_Atomic_overlap_Low)

	return C_OverlapHigh * U_2e, C_OverlapLow * U_2e






