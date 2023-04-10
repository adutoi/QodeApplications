#    (C) Copyright 2018, 2019 Yuhong Liu and Anthony D. Dutoi
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
import numpy
import mol_fci
import numpyLanczos
from multiprocessing import Pool, cpu_count
import fci_index
from qode.util.PyC import import_C, BigInt, Double

generic_hamiltonian           = import_C("generic_hamiltonian", flags="-O2 -fopenmp -lm")
LMO_frozen_Hamiltonian_filter = import_C("LMO_frozen_Hamiltonian_filter", flags="-O2 -fopenmp -lm")





class Hv_Matrix(object):
	def __init__(self, num_elec, num_core_elec, num_spin_orbs, h_mat, V_mat):
		self.num_elec         = num_elec
		self.num_core_elec    = num_core_elec
		self.num_spin_orbs    = num_spin_orbs
		self.h_mat            = h_mat
		self.V_mat            = V_mat
		self.states, self.val_states = mol_fci.frozen_core_fci_states(self.num_spin_orbs, self.num_elec, self.num_core_elec)
		self.states           = numpy.array(self.states,     dtype=BigInt.numpy)
		self.val_states       = numpy.array(self.val_states, dtype=BigInt.numpy)
		self.fidx             = fci_index.fci_index(self.num_elec-self.num_core_elec, self.num_spin_orbs-self.num_core_elec)
		self.comboMat         = self.fidx.getComboMat()
		self.c_num_fci_states = self.states.shape[0]
		self.c_nthd           = cpu_count()
	def __call__(self,v):
		Hv = numpy.zeros( v.shape, dtype=Double.numpy )
		if len( v.shape ) == 2:
		    n_vec = v.shape[1]
		else:
		    n_vec = 1	    	
		LMO_frozen_Hamiltonian_filter.compute_HPsi( self.num_elec, self.num_core_elec, self.num_spin_orbs, self.num_fci_states, \
		                                            v, self.states, self.val_states, self.h_mat, self.V_mat, Hv, self.comboMat, n_vec, self.nthd )
		return Hv


def get_core_elec(num_spin_orbs, num_core_elec):
	if num_core_elec > num_spin_orbs:
		raise RuntimeError
	num_spat_orbs = num_spin_orbs // 2
	return [i for i in range(num_core_elec//2)], [i+num_spat_orbs for i in range(num_core_elec//2)]

def C_Hamiltionan(fci_states, num_spin_orbs, num_elec, h_mat ,V_mat):
	print('generic_hamiltonian.c loaded...')
	num_fci_states = len(fci_states)
	np_fci_states  = numpy.array( fci_states, dtype=BigInt.numpy )
	fci_H = numpy.zeros( (len(fci_states), len(fci_states)), dtype=Double.numpy)
	generic_hamiltonian.generic_fci( num_fci_states, np_fci_states, num_spin_orbs, num_elec, h_mat, V_mat, fci_H )
	return fci_H

def fci_solution(fci_states, num_spin_orbs, num_elec, h_mat, V_mat, dump=False):
	new_H = C_Hamiltionan(fci_states, num_spin_orbs, num_elec, h_mat, V_mat)
	#if dump:  numpy.save("integrals/atomicH.npy", new_H)
	return numpy.linalg.eigh(new_H)

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
	numpy.save("integrals/configs_1e.npy",one_elec_states)
	E0,U0 = fci_solution(one_elec_states, num_spin_orbs_atom, 3, h_mat, V_mat)
	#numpy.save('integrals/Be1+_C_FCI.npy', U0)
	#numpy.save('integrals/Be1+_C_FCI_energies.npy', E0)

	# 2) Two Elec WaveFunction: Just Be atom
	print('2e FCI')
	two_elec_states, two_elec_val_states = mol_fci.frozen_core_fci_states(num_spin_orbs_atom, 4, num_core_elec_atom)
	numpy.save("integrals/configs_2e.npy",two_elec_states)
	E1,U1 = fci_solution(two_elec_states, num_spin_orbs_atom, 4, h_mat, V_mat, dump=True)
	#numpy.save('integrals/Be_C_FCI.npy', U1)
	#numpy.save('integrals/Be_C_FCI_energies.npy', E1)

	# 3) Three Elec Wavefunction:
	print('3e FCI')
	three_elec_states, three_elec_val_states = mol_fci.frozen_core_fci_states(num_spin_orbs_atom, 5, num_core_elec_atom)
	numpy.save("integrals/configs_3e.npy",three_elec_states)
	if len(three_elec_states)<1000:
		E2,U2 = fci_solution(three_elec_states, num_spin_orbs_atom, 5, h_mat, V_mat)		# for small matrices (6-31G) use this, otherwise block*dim>mat_dim
	else:
		E2,U2 = fci_Hv_solution(5, 2, num_spin_orbs_atom, h_mat, V_mat, num_3e_states)
	#numpy.save('integrals/Be1-_C_FCI.npy', U2)
	#numpy.save('integrals/Be1-_C_FCI_energies.npy', E2)

	E0 = E0.tolist()
	E1 = E1.tolist()
	E2 = E2.tolist()
	E = numpy.array(E1[:num_2e_states] + E0[:num_1e_states] + E2[:num_3e_states])
	#numpy.save("integrals/all_Be_FCI_energies.npy",E)

	return U0, U1, U2, E






if __name__ == "__main__":
	num_spin_orbs_atom = 110
	num_spat_orbs_atom = 55
	num_core_elec_atom = 2
	num_val_orbs_atom  = num_spin_orbs_atom - num_core_elec_atom
	h_mat = numpy.load('integrals/Be_h_spin.npy')
	V_mat = numpy.load('integrals/Be_V_spin.npy')
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
	#numpy.save('integrals/Be1+_C_FCI.npy', U0)

	# 2) Two Elec WaveFunction: Just Be atom
	#
	#
	# print('-'*80)
	two_elec_states, two_elec_val_states = mol_fci.frozen_core_fci_states(num_spin_orbs_atom, 4, num_core_elec_atom)
	E,U1 = fci_solution(two_elec_states, num_spin_orbs_atom, 4, h_mat, V_mat)
	# E,U1 = fci_Hv_solution(4, 2, num_spin_orbs_atom, h_mat, V_mat, 30)
	#numpy.save('integrals/Be_C_FCI.npy', U1)


	# 3) Three Elec Wavefunction:
	#
	#
	print('-'*80)
	three_elec_states, three_elec_val_states = mol_fci.frozen_core_fci_states(num_spin_orbs_atom, 5, num_core_elec_atom)
	# E,U2 = fci_solution(three_elec_states, num_spin_orbs_atom, 5, h_mat, V_mat)
	E,U2 = fci_Hv_solution(5, 2, num_spin_orbs_atom, h_mat, V_mat, 30)
	#numpy.save('integrals/test_vecs.npy',U2)
	#numpy.save('integrals/Be1-_C_FCI.npy', U2)


	# Four Elec Wavefunction:
	#
	#
	#four_elec_states, four_elec_val_states = mol_fci.frozen_core_fci_states(num_spin_orbs_atom, 6, num_core_elec_atom)
	#fci_H,E,U4 = fci_solution(four_elec_states, num_spin_orbs_atom, 6, h_mat, V_mat)
	#numpy.save('integrals/Be_2-_C_FCI.npy', U4)







