#    (C) Copyright 2018, 2019, 2023 Anthony D. Dutoi and Yuhong Liu
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

import math
import numpy
from   qode.util.PyC import import_C, Double, BigInt
from   compress_tensors import compress

build    = import_C("density_tensors", flags="-O2")
#contract = import_C("H_contractions",  flags="-O2")

#contract.monomer_2e.return_type(float)




def FCIcomboMat(n_elec, n_spin_orbs):
	""" returns a matrix listing the number of configurations of n_elec (rows) in n_spin_orbs (columns) ... only the upper triangle is defined """
	combination_matrix = numpy.zeros((n_elec,n_spin_orbs), dtype=BigInt.numpy)
	for i in range(n_elec):
		for j in range(i, n_spin_orbs):
			combination_matrix[i,j] = math.factorial(j) // ( math.factorial(j-i) * math.factorial(i) )
	return combination_matrix



# Better would be to allocate storage of the correct format the first time and pass it to the C code
def numpy_storage_to_lists(nparray_1d, n_bra, n_ket, n_spin_orbs, free_indices):
	n_ops = sum(len(i) for i in free_indices)
	tensor_size = n_spin_orbs**n_ops
	tensor_shape = [n_spin_orbs]*n_ops
	if tensor_size * n_bra * n_ket != nparray_1d.shape[0]:  raise ValueError
	result = [[None for i in range(n_ket)] for j in range(n_bra)]
	idx = 0
	for i in range(n_bra):
		for j in range(n_ket):
			#print(i,j)
			result[i][j] = compress(nparray_1d[idx: idx+tensor_size].reshape(tensor_shape), free_indices, compression="none")
			idx += tensor_size
	del nparray_1d
	return result



def build_density_tensors(z_lists, n_orbs, n_core, n_threads=1, bra_det=False):
	# Target
	#densities = { 'n_elec':{}, 'n_states':{}, 'aa':{}, 'caaa':{}, 'a':{}, 'caa':{}, 'ccaaa':{}, 'ca':{}, 'ccaa':{}, 'c':{}, 'cca':{}, 'cccaa':{}, 'cc':{}, 'ccca':{} }
	densities = { 'n_elec':{}, 'n_states':{}, 'aa':{}, 'a':{}, 'caa':{}, 'ca':{}, 'ccaa':{}, 'c':{}, 'cca':{}, 'cc':{}}

	# Admin
	n_chg_states = len(z_lists)
	n_spin_orbs  = 2 * n_orbs
	n_core_elec  = 2 * n_core

	# Set up arrays for communicating with C code
	idx             = {}	# save mapping of charge state to enumeration for C index value
	n_elec          = numpy.zeros(n_chg_states, dtype=BigInt.numpy)
	n_configs       = numpy.zeros(n_chg_states, dtype=BigInt.numpy)
	n_states        = numpy.zeros(n_chg_states, dtype=BigInt.numpy)
	z_configs       = []
	z_coeffs        = []
	z_coeffs_bra    = []
	combo_mat_list  = []
	for i,(chg,states) in enumerate(z_lists.items()):
		# Store also for use at higher levels
		densities['n_states'][chg] = states.coeffs.shape[0]
		densities['n_elec'  ][chg] = states.configs.shape[1]
		#
		idx[chg]        = i
		n_elec[i]       = densities['n_elec'  ][chg]
		n_configs[i]    = states.configs.shape[0]
		n_states[i]     = densities['n_states'][chg]
		z_configs      += [numpy.array(     states.configs, dtype=BigInt.numpy)]   	# Makes copy (to enforce data type) ... would be better if original were numpy array, then just:  z_configs += [states.configs]
		z_coeffs       += [numpy.array(list(states.coeffs), dtype=Double.numpy)]	# Makes copy (to forces physical storage to align with logical indexing) ... would be better if original were single numpy array, then just:  z_coeffss += [states.coeffs]
		if bra_det == False:
			z_coeffs_bra += [numpy.array(list(states.coeffs), dtype=Double.numpy)]
		else:
			z_coeffs_bra += [numpy.eye(n_configs[i])]
		combo_mat_list += [FCIcomboMat(n_elec[i]-n_core_elec, n_spin_orbs-n_core_elec)]
	
	if bra_det == False:
		n_states_bra = n_states
	else:
		n_states_bra = n_configs

	# Compute density tensors
	total_size = 0
	for bra_chg in z_lists:
		for ket_chg in z_lists:
			n_bra_states = z_lists[bra_chg].coeffs.shape[0]
			n_ket_states = z_lists[ket_chg].coeffs.shape[0]
			n_tensors    = n_bra_states * n_ket_states
			chg_diff     = bra_chg - ket_chg
			#
			if chg_diff==+2:
				# aa
				print("aa", bra_chg, ket_chg)
				allocation = n_tensors*(n_spin_orbs**2)
				result = numpy.zeros(allocation, dtype=Double.numpy)
				build.aa_tensor(result, idx[bra_chg], idx[ket_chg], n_elec, n_states_bra, n_states, z_coeffs_bra, z_coeffs, n_configs, z_configs, n_orbs, n_core, combo_mat_list, n_threads)
				densities['aa'][bra_chg,ket_chg] = numpy_storage_to_lists(result, n_bra_states, n_ket_states, n_spin_orbs, ((0,),(1,)))
				total_size += allocation
				# caaa
				#print("caaa", bra_chg, ket_chg)
				#allocation = n_tensors*(n_spin_orbs**4)
				#result = numpy.zeros(allocation, dtype=Double.numpy)
				#build.caaa_tensor(result, idx[bra_chg], idx[ket_chg], n_elec, n_states_bra, n_states, z_coeffs_bra, z_coeffs, n_configs, z_configs, n_orbs, n_core, combo_mat_list, n_threads)
				#densities['caaa'][bra_chg,ket_chg] = numpy_storage_to_lists(result, n_bra_states, n_ket_states, n_spin_orbs, ((0,1),(2,3)))
				#total_size += allocation
			if chg_diff==+1:
				# a
				print("a", bra_chg, ket_chg)
				allocation = n_tensors*(n_spin_orbs**1)
				result = numpy.zeros(allocation, dtype=Double.numpy)
				build.a_tensor(result, idx[bra_chg], idx[ket_chg], n_elec, n_states_bra, n_states, z_coeffs_bra, z_coeffs, n_configs, z_configs, n_orbs, n_core, combo_mat_list, n_threads)
				densities['a'][bra_chg,ket_chg] = numpy_storage_to_lists(result, n_bra_states, n_ket_states, n_spin_orbs, ((0,),))
				total_size += allocation
				# caa
				print("caa", bra_chg, ket_chg)
				allocation = n_tensors*(n_spin_orbs**3)
				result = numpy.zeros(allocation, dtype=Double.numpy)
				build.caa_tensor(result, idx[bra_chg], idx[ket_chg], n_elec, n_states_bra, n_states, z_coeffs_bra, z_coeffs, n_configs, z_configs, n_orbs, n_core, combo_mat_list, n_threads)
				densities['caa'][bra_chg,ket_chg] = numpy_storage_to_lists(result, n_bra_states, n_ket_states, n_spin_orbs, ((0,),(1,2)))
				total_size += allocation
				# ccaaa
				print("ccaaa", bra_chg, ket_chg)
				allocation = n_tensors*(n_spin_orbs**5)
				result = numpy.zeros(allocation, dtype=Double.numpy)
				build.ccaaa_tensor(result, idx[bra_chg], idx[ket_chg], n_elec, n_states, z_coeffs, n_configs, z_configs, n_orbs, n_core, combo_mat_list, n_threads)
				densities['ccaaa'][bra_chg,ket_chg] = numpy_storage_to_lists(result, n_bra_states, n_ket_states, n_spin_orbs, ((0,1),(2,3,4)))
				total_size += allocation
			if chg_diff==0:
				# Identity not needed since constants removed from calculation at a higher level (because result is trivial too)
				# ca
				print("ca", bra_chg, ket_chg)
				allocation = n_tensors*(n_spin_orbs**2)
				result = numpy.zeros(allocation, dtype=Double.numpy)
				build.ca_tensor(result, idx[bra_chg], idx[ket_chg], n_elec, n_states_bra, n_states, z_coeffs_bra, z_coeffs, n_configs, z_configs, n_orbs, n_core, combo_mat_list, n_threads)
				densities['ca'][bra_chg,ket_chg] = numpy_storage_to_lists(result, n_bra_states, n_ket_states, n_spin_orbs, ((0,),(1,)))
				total_size += allocation
				# ccaa
				print("ccaa", bra_chg, ket_chg)
				allocation = n_tensors*(n_spin_orbs**4)
				result = numpy.zeros(allocation, dtype=Double.numpy)
				build.ccaa_tensor(result, idx[bra_chg], idx[ket_chg], n_elec, n_states_bra, n_states, z_coeffs_bra, z_coeffs, n_configs, z_configs, n_orbs, n_core, combo_mat_list, n_threads)
				densities['ccaa'][bra_chg,ket_chg] = numpy_storage_to_lists(result, n_bra_states, n_ket_states, n_spin_orbs, ((0,1),(2,3)))
				total_size += allocation
			if chg_diff==-1:
				# c
				print("c", bra_chg, ket_chg)
				allocation = n_tensors*(n_spin_orbs**1)
				result = numpy.zeros(allocation, dtype=Double.numpy)
				build.c_tensor(result, idx[bra_chg], idx[ket_chg], n_elec, n_states_bra, n_states, z_coeffs_bra, z_coeffs, n_configs, z_configs, n_orbs, n_core, combo_mat_list, n_threads)
				densities['c'][bra_chg,ket_chg] = numpy_storage_to_lists(result, n_bra_states, n_ket_states, n_spin_orbs, ((0,),))
				total_size += allocation
				# cca
				print("cca", bra_chg, ket_chg)
				allocation = n_tensors*(n_spin_orbs**3)
				result = numpy.zeros(allocation, dtype=Double.numpy)
				build.cca_tensor(result, idx[bra_chg], idx[ket_chg], n_elec, n_states_bra, n_states, z_coeffs_bra, z_coeffs, n_configs, z_configs, n_orbs, n_core, combo_mat_list, n_threads)
				densities['cca'][bra_chg,ket_chg] = numpy_storage_to_lists(result, n_bra_states, n_ket_states, n_spin_orbs, ((0,1),(2,)))
				total_size += allocation
				# cccaa
				print("cccaa", bra_chg, ket_chg)
				allocation = n_tensors*(n_spin_orbs**5)
				result = numpy.zeros(allocation, dtype=Double.numpy)
				build.cccaa_tensor(result, idx[bra_chg], idx[ket_chg], n_elec, n_states, z_coeffs, n_configs, z_configs, n_orbs, n_core, combo_mat_list, n_threads)
				densities['cccaa'][bra_chg,ket_chg] = numpy_storage_to_lists(result, n_bra_states, n_ket_states, n_spin_orbs, ((0,1,2),(3,4)))
				total_size += allocation
			if chg_diff==-2:
				# cc
				print("cc", bra_chg, ket_chg)
				allocation = n_tensors*(n_spin_orbs**2)
				result = numpy.zeros(allocation, dtype=Double.numpy)
				build.cc_tensor(result, idx[bra_chg], idx[ket_chg], n_elec, n_states_bra, n_states, z_coeffs_bra, z_coeffs, n_configs, z_configs, n_orbs, n_core, combo_mat_list, n_threads)
				densities['cc'][bra_chg,ket_chg] = numpy_storage_to_lists(result, n_bra_states, n_ket_states, n_spin_orbs, ((0,),(1,)))
				total_size += allocation
				# ccca
				#print("ccca", bra_chg, ket_chg)
				#allocation = n_tensors*(n_spin_orbs**4)
				#result = numpy.zeros(allocation, dtype=Double.numpy)
				#build.ccca_tensor(result, idx[bra_chg], idx[ket_chg], n_elec, n_states_bra, n_states, z_coeffs_bra, z_coeffs, n_configs, z_configs, n_orbs, n_core, combo_mat_list, n_threads)
				#densities['ccca'][bra_chg,ket_chg] = numpy_storage_to_lists(result, n_bra_states, n_ket_states, n_spin_orbs, ((0,1),(2,3)))
				#total_size += allocation

	# Return density tensors
	return densities, total_size
