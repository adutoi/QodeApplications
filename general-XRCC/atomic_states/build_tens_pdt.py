#    (C) Copyright 2018 Anthony D. Dutoi and Yuhong Liu
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
from math import factorial as fct
import numpy
import mol_fci					# generate list of all frozen-core dimer FCI configurations
import LMO_frozen_buildTwoBodyVec_wrapper	# take tensor product of atomic states in dimer basis, freezing dimer core



def state(eA, iA, eB, iB, atomAvec, atomBvec, num_spin_orbs_dimer, num_core_elec_dimer, num_valence_elec_atom):

	num_valence_elec = eA + eB
	num_elec         = num_core_elec_dimer + num_valence_elec
	block_dims = (   ( fct(num_spin_orbs_dimer-num_core_elec_dimer) // fct(num_spin_orbs_dimer-num_elec) ) // fct(num_valence_elec),   1   )
	nominal_block = numpy.zeros(block_dims)

	FCI_1e_Configs = numpy.array( mol_fci.frozen_core_fci_states(num_spin_orbs_dimer, num_core_elec_dimer + num_valence_elec_atom - 1, num_core_elec_dimer )[1], dtype=numpy.int64)
	FCI_2e_Configs = numpy.array( mol_fci.frozen_core_fci_states(num_spin_orbs_dimer, num_core_elec_dimer + num_valence_elec_atom,     num_core_elec_dimer )[1], dtype=numpy.int64)
	FCI_3e_Configs = numpy.array( mol_fci.frozen_core_fci_states(num_spin_orbs_dimer, num_core_elec_dimer + num_valence_elec_atom + 1, num_core_elec_dimer )[1], dtype=numpy.int64)
	FCI_Configs = { 2: FCI_2e_Configs, 1: FCI_1e_Configs, 3: FCI_3e_Configs }

	vec = LMO_frozen_buildTwoBodyVec_wrapper.build( iA, iB, eA, eB, atomAvec[eA], atomBvec[eB],
	                                                FCI_Configs[eA], FCI_Configs[eB], num_spin_orbs_dimer, num_core_elec_dimer )
	nominal_block[:,0] = vec[:]

	return ( num_elec, nominal_block, 0, block_dims )



def basis(n_states, atomAvec, atomBvec, num_spin_orbs_dimer, num_core_elec_dimer, num_valence_elec_atom, _neutral=False):

	FCI_1e_Configs = numpy.array( mol_fci.frozen_core_fci_states(num_spin_orbs_dimer, num_core_elec_dimer + num_valence_elec_atom - 1, num_core_elec_dimer )[1], dtype=numpy.int64)
	FCI_2e_Configs = numpy.array( mol_fci.frozen_core_fci_states(num_spin_orbs_dimer, num_core_elec_dimer + num_valence_elec_atom,     num_core_elec_dimer )[1], dtype=numpy.int64)
	FCI_3e_Configs = numpy.array( mol_fci.frozen_core_fci_states(num_spin_orbs_dimer, num_core_elec_dimer + num_valence_elec_atom + 1, num_core_elec_dimer )[1], dtype=numpy.int64)
	FCI_Configs = { 2: FCI_2e_Configs, 1: FCI_1e_Configs, 3: FCI_3e_Configs }

	tot_elec_nums = []
	for eA in (2,1,3):
		for i in range(n_states[eA]):
			for eB in (2,1,3):
				for j in range(n_states[eB]):
					tot_elec_nums += [ num_core_elec_dimer + eA + eB ]

	counter       = { 6:0,  7:0,  8:0,  9:0,  10:0  }
	tns_pdt_idx   = { 6:[], 7:[], 8:[], 9:[], 10:[] }
	n_blocked_idx = []
	for i,n in enumerate(tot_elec_nums):
		tns_pdt_idx[n] += [i]
		n_blocked_idx  += [(n,counter[n])]
		counter[n]     += 1

	e_block_dims = {}
	e_block_dims[ 6] = (   ( fct(num_spin_orbs_dimer-4) // fct(num_spin_orbs_dimer- 6) ) // fct(2),   len(tns_pdt_idx[ 6])   )
	e_block_dims[ 7] = (   ( fct(num_spin_orbs_dimer-4) // fct(num_spin_orbs_dimer- 7) ) // fct(3),   len(tns_pdt_idx[ 7])   )
	e_block_dims[ 8] = (   ( fct(num_spin_orbs_dimer-4) // fct(num_spin_orbs_dimer- 8) ) // fct(4),   len(tns_pdt_idx[ 8])   )
	e_block_dims[ 9] = (   ( fct(num_spin_orbs_dimer-4) // fct(num_spin_orbs_dimer- 9) ) // fct(5),   len(tns_pdt_idx[ 9])   )
	e_block_dims[10] = (   ( fct(num_spin_orbs_dimer-4) // fct(num_spin_orbs_dimer-10) ) // fct(6),   len(tns_pdt_idx[10])   )

	e_block_basis = {}
	e_block_basis[ 8] = numpy.zeros(e_block_dims[ 8])
	if _neutral:
		e_block_basis[ 6] = None
		e_block_basis[ 7] = None
		e_block_basis[ 9] = None
		e_block_basis[10] = None
	else:
		e_block_basis[ 6] = numpy.zeros(e_block_dims[ 6])
		e_block_basis[ 7] = numpy.zeros(e_block_dims[ 7])
		e_block_basis[ 9] = numpy.zeros(e_block_dims[ 9])
		e_block_basis[10] = numpy.zeros(e_block_dims[10])

	count = 0
	tens_pdt_basis = []
	for eA in (2,1,3):
		for i in range(n_states[eA]):
			for eB in (2,1,3):
				for j in range(n_states[eB]):
					n, idx = n_blocked_idx[count]
					if n!=num_core_elec_dimer+eA+eB:  raise LogicError()
					count += 1
					if n==8 or not _neutral:
						print("{}e-{}e state: {} {}".format(eA, eB, i, j))
						vec = LMO_frozen_buildTwoBodyVec_wrapper.build( i, j, eA, eB, atomAvec[eA], atomBvec[eB],
						                                                FCI_Configs[eA], FCI_Configs[eB], num_spin_orbs_dimer, num_core_elec_dimer )
						e_block_basis[n][:,idx] = vec[:]
					tens_pdt_basis += [( n, e_block_basis[n], idx, e_block_dims[n] )]

	return tens_pdt_basis, tns_pdt_idx, n_blocked_idx


def basis_neutral(n_states, atomAvec, atomBvec, num_spin_orbs_dimer, num_core_elec_dimer, num_valence_elec_atom):
	full_basis, tns_pdt_idx, n_blocked_idx = basis(n_states, atomAvec, atomBvec, num_spin_orbs_dimer, num_core_elec_dimer, num_valence_elec_atom, _neutral=True)
	neut_basis = [full_basis[i] for i in tns_pdt_idx[8]]
	return neut_basis, tns_pdt_idx[8]


