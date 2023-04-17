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
import numpy
from math      import factorial
from fci_index import fci_index
from qode.util.PyC import import_C, Double

LMO_frozen_buildTwoBodyVec = import_C("LMO_frozen_buildTwoBodyVec", flags="-O2 -fopenmp")



#  This feeds info into C function that turns atomic Low/High into Dimer vector
#  Orbital enumeration follows the pattern:
#         alpha ===> { -core- | --- valence --- }{ -core- | --- valence --- } <=== beta



def build( i, j, num_valence_elec_A, num_valence_elec_B, atomA_states, atomB_states, atomA_configs, atomB_configs, num_spin_orbs, num_core_elec ):
	# i:                   index of state from atomA_states to use
	# j:                   index of state from atomB_states to use
	# num_valence_elec_A:  number of valence electrons on atom A
	# num_valence_elec_B:  number of valence electrons on atom B
	# atomA_states:        requisite number of eigenstates (of appropriate electron number) of atom A, given in the basis of dimer orbitals
	# atomB_states:        requisite number of eigenstates (of appropriate electron number) of atom B, given in the basis of dimer orbitals
	# atomA_configs:       list of all dimer valence configurations for number of electrons on A, where each config is a list of orbital indices (excluding core indices, but core are indexed)
	# atomB_configs:       list of all dimer valence configurations for number of electrons on B, where each config is a list of orbital indices (excluding core indices, but core are indexed)
	# num_spin_orbs:       total number of orbitals for the dimer
	# num_core_elec:       number of frozen core electrons in the dimer
	#
	num_valence_elec  = num_valence_elec_A + num_valence_elec_B
	num_valence_orbs  = num_spin_orbs - num_core_elec
	num_dimer_configs = ( factorial(num_valence_orbs) // factorial(num_valence_orbs - num_valence_elec) ) // factorial(num_valence_elec)
	atomA_state       = atomA_states[:,i].copy()						# load the state of atom A
	atomB_state       = atomB_states[:,j].copy()						# load the state of atom B
	dimer_state       = numpy.zeros(num_dimer_configs, dtype=Double.numpy)			# allocation for the tensor product state
	comboMat          = fci_index( num_valence_elec, num_valence_orbs ).getComboMat()	# preload combinitorics for efficiency
	#
	LMO_frozen_buildTwoBodyVec.Transform_ABvecs_Into_TwoBody_Basis(dimer_state,
	                                                               num_spin_orbs, 
	                                                               num_core_elec,
	                                                               num_valence_elec_A,
	                                                               num_valence_elec_B,
	                                                               atomA_state,
	                                                               atomB_state,
	                                                               atomA_configs,
	                                                               atomB_configs,
	                                                               atomA_configs.shape[0],
	                                                               atomB_configs.shape[0], 
	                                                               comboMat)
	return dimer_state
