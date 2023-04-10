#    (C) Copyright 2018, 2019 Yuhong Liu and Anthony Dutoi
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
import sys
import time
import numpy
from multiprocessing import cpu_count
import mol_fci
from fci_index import fci_index
from qode.util.PyC import import_C, BigInt, Double

LMO_frozen_Hamiltonian = import_C("LMO_frozen_Hamiltonian", flags="-O2 -lm")



class _action(object):
	def __init__(self, H, num_elec):
		self.configs, self.valence_configs = mol_fci.frozen_core_fci_states(H.num_spin_orbs, num_elec, H.num_core_elec)
		self.configs, self.valence_configs = numpy.array(self.configs, dtype=BigInt.numpy), numpy.array(self.valence_configs, dtype=BigInt.numpy)
		self.comboMat = fci_index( num_elec-H.num_core_elec, H.num_spin_orbs-H.num_core_elec ).getComboMat()
		self.H               = H
		self.num_elec        = num_elec
		self.num_fci_configs = self.configs.shape[0]
	def __call__(self, Psi, i, num, dims):
		(dim,I) = dims
		HPsi = numpy.zeros((dim,num), dtype=Double.numpy)
		H = self.H
		LMO_frozen_Hamiltonian.compute_HPsi(self.num_elec,
		                                    H.num_core_elec,
		                                    H.num_spin_orbs,
		                                    self.num_fci_configs,
		                                    Psi,
		                                    self.configs,
		                                    self.valence_configs,
		                                    H.h,
		                                    H.V,
		                                    HPsi,
		                                    self.comboMat,
		                                    i,
		                                    I,
		                                    num,
		                                    cpu_count()   )
		return HPsi

class Hamiltonian(object):
	def __init__(self, h, V, num_core_elec, num_elec=None):		# num_elec is a list of all electron numbers desired for this Hamiltonian
		self.h             = h
		self.V             = V
		self.num_core_elec = num_core_elec
		self.num_spin_orbs = h.shape[0]
		self.H_action = {}
		for n in num_elec:  self.H_action[n] = _action(H=self, num_elec=n)
	def __call__(self, Psi):
		n, block, ii, (dim,I) = Psi
		multivec = isinstance(ii,tuple)
		if not multivec:  ii = (ii,1)	# means act on single vector, assumes I is int
		#
		i,num = ii
		block = self.H_action[n](block, i, num, (dim,I))
		ii = 0,num
		#
		if not multivec:  ii = 0
		return (n, block, ii, (dim,num))
