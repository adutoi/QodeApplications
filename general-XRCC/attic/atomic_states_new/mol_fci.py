#    (C) Copyright 2018 Yuhong Liu
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
# Generic Code for FCI of any restricted HF results.
#
# 
import sys
import copy
import numpy as np
import time
from multiprocessing import Pool




def add_electron( configs, next_possible ):
	# configs:       for one electron :   [[0],[1],[2],[3], ..., [n]]
	#                for two electrons:   [[0,1],[0,2], ..., [n1_max, n2_max]] 
	#                for three electrons: [[0,1,2],[0,1,3], ..., [n1_max,n2_max,n3_max]]
	#                ... ...
	#                ... ...
	#
	# next_possible: ONLY HOLD ONE ELECTRON ORBITAL NUMBERS. e.g., [[3],[4], ... [n4_max]]
	#
	new_configs = []
	for config in configs:
		for orb in next_possible:
			if orb[0] > config[-1]:
				new_configs += [ copy.deepcopy( config + orb ) ]
	return new_configs


def fci_all_states(num_spin_orbs, num_elec):
	# Generates all possible FCI states (No Spin Restriction)
	#
	#
	possible_orb_list = []
	for n in range(num_elec):
		possible_orbs = [ [i] for i in range(n, num_spin_orbs - num_elec + n + 1 ) ]
		possible_orb_list += [ copy.deepcopy(possible_orbs) ]

	if num_elec > 1:
		fci_states = add_electron( possible_orb_list[0], possible_orb_list[1] )
		for i in range(1, num_elec-1):
			fci_states = add_electron( fci_states, possible_orb_list[i+1] )
	elif num_elec == 1:
		fci_states = possible_orb_list[0]
	else:
		print("num of electrons must >= 1")
		raise ValueError

	return fci_states



def fci_Sz0_states(num_spin_orbs, num_elec):
	num_alpha_orbs = num_spin_orbs // 2
	num_alpha_elec = num_elec // 2
	alpha_states   = fci_all_states(num_alpha_orbs, num_alpha_elec)
	beta_states    = [ [orb+num_alpha_orbs for orb in config] for config in alpha_states ]
	fci_states     = []
	for i in range(len(alpha_states)):
		for j in range(len(beta_states)):
			fci_states += [ copy.deepcopy(alpha_states[i] + beta_states[j]) ]
	return fci_states

def frozen_core_Sz0_fci_states(num_spin_orbs, num_elec, num_core_elec):
	num_spatial_orbs = num_spin_orbs // 2
	alpha_num_core_elec = num_core_elec // 2
	alpha_num_elec      = num_elec // 2
	alpha_core = [ i for i in range(alpha_num_core_elec) ]
	beta_core  = [ i+num_spatial_orbs for i in range(alpha_num_core_elec) ]
	valence_states = fci_all_states(num_spatial_orbs - alpha_num_core_elec, alpha_num_elec - alpha_num_core_elec )
	alpha_valence_states = [[orb+alpha_num_core_elec for orb in config] for config in valence_states]
	beta_valence_states  = [[orb+alpha_num_core_elec+num_spatial_orbs for orb in config] for config in valence_states]
	fci_states = []
	val_states = []
	for i in range(len(alpha_valence_states)):
		for j in range(len(beta_valence_states)):
			val_states += [ copy.deepcopy(alpha_valence_states[i]) + copy.deepcopy(beta_valence_states[j]) ]
			fci_states += [ copy.deepcopy(alpha_core) + copy.deepcopy(alpha_valence_states[i]) + copy.deepcopy(beta_core) + copy.deepcopy(beta_valence_states[j]) ]
	# np.save('data/new_fci_states.npy',fci_states)
	return fci_states, val_states

def frozen_core_fci_states(num_spin_orbs, num_elec, num_core_elec):
	"""\
	Builds a list of all possible configurations of num_elec in num_spin_orbs, assuming that num_core_elec (must be an even number)
	are frozen in the lowest-index alpha and beta orbitals.  The orbital indexing is such that all alphas precede all betas.
	"""
	num_val_elec = num_elec      - num_core_elec
	num_val_orbs = num_spin_orbs - num_core_elec
	#
	alpha_core = [ i                  for i in range(num_core_elec//2) ]	# indices of the alpha core orbitals
	beta_core  = [ i+num_spin_orbs//2 for i in range(num_core_elec//2) ]	# indices of the beta  core orbitals
	core_orbs  = alpha_core + beta_core					# indices of all the core orbitals
	val_orbs   = [ i for i in range(num_spin_orbs) if i not in core_orbs ]	# indices of all the valence orbitals
	#
	possible_orb_list = []		# [ [all the places e1 might end up], [all the places e2 might end up], [all the places e3 might end up], ... ]
	for n in range(num_val_elec):
		possible_orbs = val_orbs[n : n + num_val_orbs-num_val_elec+1] 
		possible_orb_list += [[ [i] for i in possible_orbs ]]
	#
	valence_states = []		# A list of configs, where each config is a list of occupied orbitals (eg, [[0,1,2],[0,1,3], ..., [N-3,N-2,N-1]]
	if num_val_elec < 1:
		raise ValueError("num of electrons < 1")
	else:
		valence_states = possible_orb_list[0]
		for i in range(num_val_elec-1):
			valence_states = add_electron( valence_states, possible_orb_list[i+1] )
	#
	fci_states = [ sorted(core_orbs+state) for state in valence_states ]	# interweave constant core states into list of occupied orbitals
	#
	return fci_states, valence_states

def frozen_core_ct_fci_states(num_ct, num_spin_orbs, num_elec, num_core_elec):
	# up to num_ct of electrons can be transfered.
	# num_ct must be less than num atom valence electrons.
	states, val_states = frozen_core_fci_states(num_spin_orbs, num_elec, num_core_elec)
	num_spatial_orbs   = num_spin_orbs // 2
	num_core_elec_atom = num_core_elec // 2
	num_max_elec = num_ct + num_elec // 2 - num_core_elec_atom
	print('charge transfer =', num_ct)
	ct_states     = []
	ct_val_states = []
	ct = 0
	for item in states:
		idx = item.index(num_spatial_orbs)
		val_a = idx - num_core_elec_atom
		val_b = num_elec - num_core_elec - val_a
		# print(val_states[ct], val_a, val_b)
		if val_a > num_max_elec or val_b > num_max_elec:
			pass
		else:
			ct_states += [item]
			ct_val_states += [val_states[ct]]
		ct += 1
	print("%d / %d states" %(len(ct_states), len(states)) )
	return ct_states, ct_val_states


def ci_states_spin_assert( ci_state1, ci_state2, num_spatial_orbs ):
	# Get the numbers of alpha spins in both states
	state1_a_spin = 0
	for orb_num in ci_state1:
		if orb_num < num_spatial_orbs:
			state1_a_spin += 1
	state2_a_spin = 0
	for orb_num in ci_state2:
		if orb_num < num_spatial_orbs:
			state2_a_spin += 1	
	
	# print("state1 alpha:", state1_a_spin, "state2 alpha:", state2_a_spin)
	if state1_a_spin == state2_a_spin:
		return True
	else:
		return False

def analyze_sign(diff_location1, diff_location2):
	if len(diff_location1) != len(diff_location2):
		raise ValueError
	num_diff_orb = len(diff_location1)
	if num_diff_orb == 2:
		return pow( -1.0, abs(diff_location1[0] - diff_location2[0]) ) * pow( -1.0, abs(diff_location1[1] - diff_location2[1]) )
	elif num_diff_orb == 1:
		return pow( -1.0, abs(diff_location1[0] - diff_location2[0]) )
	elif num_diff_orb == 0:
		return 1.0
	else:
		return 0.0


def different_orb_info(ci_state1, ci_state2):
	# ci_state1 and ci_state2 must be spin-compatible at this point.
	#
	num_diff_orb = 0
	diff_orbs1 = []
	diff_orbs2 = []
	diff_location1 = []
	diff_location2 = []
	# check ci_state1
	for orb_num1 in ci_state1:
		if orb_num1 in ci_state2:
			pass
		else:
			num_diff_orb += 1
			diff_orbs1     += [orb_num1]
			diff_location1 += [ci_state1.index(orb_num1)]
	# check ci_state2
	for orb_num2 in ci_state2:
		if orb_num2 in ci_state1:
			pass
		else:
			diff_orbs2     += [orb_num2]
			diff_location2 += [ci_state2.index(orb_num2)]

	return num_diff_orb, diff_orbs1, diff_orbs2, diff_location1, diff_location2


def get_num_a_b_electrons(ci_state, num_spatial_orbs):
	num_alpha = 0
	num_beta  = 0
	for orb_num in ci_state:
		if 0 <= orb_num < num_spatial_orbs:
			num_alpha += 1
		elif num_spatial_orbs <= orb_num < 2*num_spatial_orbs:
			num_beta  += 1 
		else:
			raise ValueError
	return num_alpha, num_beta



def compute_hamiltonian_value(state1, state2, num_spin_orbs, num_elec, h_mat, V_mat):
	num_diff_orb, diff_orbs1, diff_orbs2, diff_location1, diff_location2 = different_orb_info(state1, state2)
	h_value = 0.0
	#
	#
	if num_diff_orb > 2:
		# More than 2 e- are in different MOs. Value is zero.
		...
	elif num_diff_orb == 2:
		# core Hamiltonian is zero
		# two-e part can be non-zero.
		# print(num_diff_orb, diff_orbs1, diff_orbs2)
		sign = analyze_sign(diff_location1, diff_location2)
		# print(diff_location1, diff_location2, sign)
		p, q = diff_orbs1
		r, s = diff_orbs2
		h_value += V_mat[p*num_spin_orbs + q, r*num_spin_orbs + s] - V_mat[p*num_spin_orbs + q, s*num_spin_orbs + r]
		h_value *= sign

	elif num_diff_orb == 1:
		# core H and two-e part both can be non-zero
		# print(num_diff_orb, diff_orbs1, diff_orbs2)
		sign = analyze_sign(diff_location1, diff_location2)
		# print(diff_location1, diff_loscation2, sign)
		p = diff_orbs1[0]
		r = diff_orbs2[0]
		# One e- part:
		h_value += h_mat[p,r]
		# Two e- part:
		common_orbs = [item for item in state1 if item != p]
		for i in common_orbs:
			h_value += V_mat[p*num_spin_orbs + i, r*num_spin_orbs + i] - V_mat[p*num_spin_orbs + i, i*num_spin_orbs + r]
		h_value *= sign

	else:  # same config:
		# print(num_diff_orb, diff_orbs1, diff_orbs2)
		# num_diff_orb == 0
		# core H is non-zero.
		for i in state1:
			h_value += h_mat[i,i]
		# two-e part can be non-zero
		for i in state1:
			for j in state1:
				h_value += 0.5 * ( V_mat[i*num_spin_orbs + j, i*num_spin_orbs + j] - V_mat[i*num_spin_orbs + j, j*num_spin_orbs + i] )

	return h_value



def compute_row(*args):
	# Pack Inputs Like this: row_idx, num_spatial_orbs, num_elec, h_mat, V_mat):
	row_idx, fci_states, num_spin_orbs, num_elec, h_mat, V_mat = args[0]
	# print("ROW =",row_idx)
	row_fci = []
	for i in range(len(fci_states)):
		row_fci += [ compute_hamiltonian_value(fci_states[row_idx], fci_states[i], num_spin_orbs, num_elec, h_mat, V_mat) ]
	return row_fci



def FCI_matrix( h_mat, V_mat, num_spin_orbs, num_elec, num_core_elec=None, frozen_core=False, force_sz0=False ):
	# 
	print('num_elec =', num_elec)
	print('num_spin_orbs =', num_spin_orbs)
	if frozen_core:
		print('\tBuilding Frozen Core Hamiltonian...')
		print('\tNum Frozen Core Elec =',num_core_elec)
		if force_sz0:
			print("\tGenerating Sz=0 States")
			fci_states = frozen_core_Sz0_fci_states(num_spin_orbs, num_elec, num_core_elec)[0]
		else:
			print('\tSz not set to 0')
			fci_states = frozen_core_fci_states(num_spin_orbs, num_elec, num_core_elec)[0]
	else:
		fci_states = fci_all_states(num_spin_orbs, num_elec)
	#
	#
	H_dim = len(fci_states)
	print("\tFCI Hamiltonian Builder:")
	print('\tnum of FCI states =',H_dim)
	#
	input_pack = [ [i, fci_states, num_spin_orbs, num_elec, h_mat, V_mat] for i in range(H_dim) ]
	myPool = Pool(64)
	fci_H  = myPool.map(compute_row, input_pack)
	myPool.terminate()
	print("Processes terminated.")
	fci_H  = np.array(fci_H)
	print("FCI Hamiltonian Shape =", fci_H.shape)
	return fci_H, fci_states





def main(prefix, h_mat_file, V_mat_file, num_elec, num_core_elec=None, frozen_core=False, force_sz0=False):
	h_mat     =  np.load(h_mat_file)
	V_mat     =  np.load(V_mat_file)
	num_spin_orbs  = h_mat.shape[0]
	print('h shape =', h_mat.shape)
	print('V shape =', V_mat.shape)
	#
	if frozen_core:
		print("FROZEN CORE ACTIVATED. NUM FROZEN ELEC =", num_core_elec)
		H_fci, fci_states = FCI_matrix(h_mat, V_mat, num_spin_orbs, num_elec, num_core_elec, frozen_core, force_sz0)
	else:
		print("Non-Frozen Core Calculation")
		H_fci, fci_states = FCI_matrix(h_mat, V_mat, num_spin_orbs, num_elec)


	print('File Prefix =',prefix)
	np.save('data/'+prefix + '_fci_ham_mat.npy', H_fci)
	print("Saving FCI Configs onto Disk...")
	np.save('data/'+prefix+"_all_states_int64.npy", np.array(fci_states, dtype=np.int64))
	import numpyLanczos
	e = numpyLanczos.lanczos_eig( np.array(H_fci) )
	# sys.exit(0)
	# print("Diagonalizing FCI Matrix...")
	# E,C = np.linalg.eigh(H_fci)
	# print("Lowest EigenValue = %.10f" %(E[0]))
	# print("Lowest 20 EigenValues:")
	# print(E[:20])
	# print("Saving FCI Vectors onto Disk...")
	# np.save('data/'+prefix+"C_FCI.npy", C)
	# np.save('data/'+prefix+"E_FCI.npy", E)
	return H_fci

if __name__ == "__main__":
	#
	#
	#
	prefix      = sys.argv[1]  #'Be_'
	h_mat_file  = sys.argv[2]  #'Be_h_spin.npy'
	V_mat_file  = sys.argv[3]  #'Be_V_spin.npy'
	num_elec      = int(sys.argv[4])
	num_core_elec = int(sys.argv[5])
	frozen_core = True
	force_sz0   = False
	H_fci = main(prefix, h_mat_file, V_mat_file, num_elec, num_core_elec, frozen_core, force_sz0)


