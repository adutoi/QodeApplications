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
import time
import copy
import ctypes as ct
import numpy as np
import mol_fci
import numpyLanczos
from multiprocessing import Pool, cpu_count
import fci_index

class Hv_Matrix(object):
	def __init__(self, num_elec, num_core_elec, num_spin_orbs, h_mat, V_mat):
		self.num_elec      = num_elec
		self.num_core_elec = num_core_elec
		self.num_spin_orbs = num_spin_orbs
		self.h_mat         = h_mat
		self.V_mat         = V_mat
		self.states, self.val_states = mol_fci.frozen_core_fci_states(self.num_spin_orbs, self.num_elec, self.num_core_elec)
		self.states        = np.array(self.states,     dtype=np.int64)
		self.val_states    = np.array(self.val_states, dtype=np.int64)
		self.fidx          = fci_index.fci_index(self.num_elec-self.num_core_elec, self.num_spin_orbs-self.num_core_elec)
		self.comboMat      = self.fidx.getComboMat()
		self.lib_handler   = ct.cdll.LoadLibrary('./LMO_frozen_Hamiltonian_filter.so')
		self.c_num_elec       = ct.c_longlong( self.num_elec )
		self.c_num_core_elec  = ct.c_longlong( self.num_core_elec )
		self.c_num_spin_orbs  = ct.c_longlong( self.num_spin_orbs )
		self.c_num_fci_states = ct.c_longlong( self.states.shape[0] )
		self.c_h_mat          = self.h_mat.ctypes.data_as( ct.POINTER( ct.c_double ) )
		self.c_V_mat          = self.V_mat.ctypes.data_as( ct.POINTER( ct.c_double ) )
		self.c_states         = self.states.ctypes.data_as( ct.POINTER( ct.c_longlong ) )
		self.c_val_states     = self.val_states.ctypes.data_as( ct.POINTER( ct.c_longlong ) )
		self.c_comboMat       = self.comboMat.ctypes.data_as( ct.POINTER( ct.c_longlong ) )
		self.c_nthd           = ct.c_int( cpu_count() )

	def __call__(self,v):
		Hv = np.zeros( v.shape, dtype=np.float64 )
		if len( v.shape ) == 2:
		    c_n_vec = ct.c_longlong( v.shape[1] )
		else:
			c_n_vec = ct.c_longlong( 1 )	    	
		c_Hv = Hv.ctypes.data_as( ct.POINTER( ct.c_double ) )
		c_v  = v.ctypes.data_as(  ct.POINTER( ct.c_double ) )
		self.lib_handler.compute_HPsi( self.c_num_elec, self.c_num_core_elec, self.c_num_spin_orbs, self.c_num_fci_states, \
		                               c_v, self.c_states, self.c_val_states, self.c_h_mat, self.c_V_mat, c_Hv,  \
		                               self.c_comboMat, c_n_vec, self.c_nthd )
		return Hv


def get_core_elec(num_spin_orbs, num_core_elec):
	if num_core_elec > num_spin_orbs:
		raise RuntimeError
	num_spat_orbs = num_spin_orbs // 2
	return [i for i in range(num_core_elec//2)], [i+num_spat_orbs for i in range(num_core_elec//2)]

def C_Hamiltionan(fci_states, num_spin_orbs, num_elec, h_mat ,V_mat):
	# void generic_fci(long long num_fci_states, long long* fci_states, long long num_spin_orbs, long long num_elec, double* h_mat, double* V_mat, double* FCI_H)
	c_handler = ct.cdll.LoadLibrary('./generic_hamiltonian.so')
	print('generic_hamiltonian.so loaded...')
	c_num_elec       = ct.c_longlong( num_elec )
	c_num_spin_orbs  = ct.c_longlong( num_spin_orbs )
	c_num_fci_states = ct.c_longlong( len(fci_states) )
	c_h_mat          = h_mat.ctypes.data_as( ct.POINTER( ct.c_double ) )
	c_V_mat          = V_mat.ctypes.data_as( ct.POINTER( ct.c_double ) )

	np_fci_states   = np.array( fci_states, dtype=np.int64 )
	c_states        = np_fci_states.ctypes.data_as( ct.POINTER( ct.c_longlong ) )

	fci_H = np.zeros( (len(fci_states), len(fci_states)), dtype=np.float64)
	# print(fci_H.shape)
	c_fci_H = fci_H.ctypes.data_as( ct.POINTER( ct.c_double ) )
	c_handler.generic_fci(c_num_fci_states, c_states, c_num_spin_orbs, c_num_elec, c_h_mat, c_V_mat, c_fci_H)
	# print(fci_H)
	# print(fci_states)
	return fci_H

def fci_solution(fci_states, num_spin_orbs, num_elec, h_mat, V_mat, dump=False):
	new_H = C_Hamiltionan(fci_states, num_spin_orbs, num_elec, h_mat, V_mat)
	#if dump:  np.save("atomicH.npy", new_H)
	return np.linalg.eigh(new_H)

def fci_Hv_solution(num_elec, num_core_elec, num_spin_orbs, h_mat, V_mat, num_eig):
	mat = Hv_Matrix(num_elec, num_core_elec, num_spin_orbs, h_mat, V_mat)
	E,U = numpyLanczos.Hv_eigh(mat, num_eig)
	return E,U


def compute_fci_vecs(num_spin_orbs_atom, num_core_elec_atom, h_mat, V_mat, num_1e_states=16, num_2e_states=30, num_3e_states=30 ):
	num_spat_orbs_atom = num_spin_orbs_atom // 2
	num_val_orbs_atom  = num_spin_orbs_atom - num_core_elec_atom
	#
	val_orbs = [ [i+num_core_elec_atom//2] for i in range(num_val_orbs_atom//2)] + [[i+num_core_elec_atom//2+num_spat_orbs_atom] for i in range(num_val_orbs_atom//2)]
	core_elec = get_core_elec(num_spin_orbs_atom, num_core_elec_atom)[0] + get_core_elec(num_spin_orbs_atom, num_core_elec_atom)[1]
	#
	#
	# 1)  One Elec WaveFunction: (DIRTY CODE FOR Be)
	print('1e FCI')
	one_elec_states = [ sorted( core_elec + item )  for item in val_orbs ]
	np.save("configs_1e.npy",one_elec_states)
	E0,U0 = fci_solution(one_elec_states, num_spin_orbs_atom, 3, h_mat, V_mat)
	#np.save('Be1+_C_FCI.npy', U0)
	#np.save('Be1+_C_FCI_energies.npy', E0)

	# 2) Two Elec WaveFunction: Just Be atom
	print('2e FCI')
	two_elec_states, two_elec_val_states = mol_fci.frozen_core_fci_states(num_spin_orbs_atom, 4, num_core_elec_atom)
	np.save("configs_2e.npy",two_elec_states)
	E1,U1 = fci_solution(two_elec_states, num_spin_orbs_atom, 4, h_mat, V_mat, dump=True)
	#np.save('Be_C_FCI.npy', U1)
	#np.save('Be_C_FCI_energies.npy', E1)

	# 3) Three Elec Wavefunction:
	print('3e FCI')
	three_elec_states, three_elec_val_states = mol_fci.frozen_core_fci_states(num_spin_orbs_atom, 5, num_core_elec_atom)
	np.save("configs_3e.npy",three_elec_states)
	if len(three_elec_states)<1000:
		E2,U2 = fci_solution(three_elec_states, num_spin_orbs_atom, 5, h_mat, V_mat)		# for small matrices (6-31G) use this, otherwise block*dim>mat_dim
	else:
		E2,U2 = fci_Hv_solution(5, 2, num_spin_orbs_atom, h_mat, V_mat, num_3e_states)
	#np.save('Be1-_C_FCI.npy', U2)
	#np.save('Be1-_C_FCI_energies.npy', E2)

	E0 = E0.tolist()
	E1 = E1.tolist()
	E2 = E2.tolist()
	E = np.array(E1[:num_2e_states] + E0[:num_1e_states] + E2[:num_3e_states])
	#np.save("all_Be_FCI_energies.npy",E)

	return U0, U1, U2, E






if __name__ == "__main__":
	num_spin_orbs_atom = 110
	num_spat_orbs_atom = 55
	num_core_elec_atom = 2
	num_val_orbs_atom  = num_spin_orbs_atom - num_core_elec_atom
	h_mat = np.load('Be_h_spin.npy')
	V_mat = np.load('Be_V_spin.npy')
	# Zero Elec WaveFunction ???
	#
	#

	# 1)  One Elec WaveFunction: (DIRTY CODE FOR Be)
	#
	#
	val_orbs = [ [i+num_core_elec_atom//2] for i in range(num_val_orbs_atom//2)] + [[i+num_core_elec_atom//2+num_spat_orbs_atom] for i in range(num_val_orbs_atom//2)]
	# print(val_orbs)
	core_elec = get_core_elec(num_spin_orbs_atom, num_core_elec_atom)[0] + get_core_elec(num_spin_orbs_atom, num_core_elec_atom)[1]
	# print('core elec :', core_elec)
	one_elec_states = [ sorted( core_elec + item )  for item in val_orbs ]
	# for line in one_elec_states:
	# 	print(line)
	print('-'*80)
	E,U0 = fci_solution(one_elec_states, num_spin_orbs_atom, 3, h_mat, V_mat)
	# E,U0 = fci_Hv_solution(3, 2, num_spin_orbs_atom, h_mat, V_mat, 16)
	#np.save('Be1+_C_FCI.npy', U0)

	# 2) Two Elec WaveFunction: Just Be atom
	#
	#
	# print('-'*80)
	two_elec_states, two_elec_val_states = mol_fci.frozen_core_fci_states(num_spin_orbs_atom, 4, num_core_elec_atom)
	E,U1 = fci_solution(two_elec_states, num_spin_orbs_atom, 4, h_mat, V_mat)
	# E,U1 = fci_Hv_solution(4, 2, num_spin_orbs_atom, h_mat, V_mat, 30)
	#np.save('Be_C_FCI.npy', U1)


	# 3) Three Elec Wavefunction:
	#
	#
	print('-'*80)
	three_elec_states, three_elec_val_states = mol_fci.frozen_core_fci_states(num_spin_orbs_atom, 5, num_core_elec_atom)
	# E,U2 = fci_solution(three_elec_states, num_spin_orbs_atom, 5, h_mat, V_mat)
	E,U2 = fci_Hv_solution(5, 2, num_spin_orbs_atom, h_mat, V_mat, 30)
	#np.save('test_vecs.npy',U2)
	#np.save('Be1-_C_FCI.npy', U2)


	# Four Elec Wavefunction:
	#
	#
	#four_elec_states, four_elec_val_states = mol_fci.frozen_core_fci_states(num_spin_orbs_atom, 6, num_core_elec_atom)
	#fci_H,E,U4 = fci_solution(four_elec_states, num_spin_orbs_atom, 6, h_mat, V_mat)
	#np.save('Be_2-_C_FCI.npy', U4)







