#    (C) Copyright 2018, 2019, 2023 Yuhong Liu and Anthony Dutoi
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
from multiprocessing import cpu_count
from qode.util.PyC import import_C, BigInt, Double

field_op = import_C("field_op", flags="-O2 -lm")
field_op.find_index.return_type(int)



def find_index(config, configs):
	return field_op.find_index(config, configs, len(configs))

def _all_nelec_configs(num_orb, num_elec, _outer_config=0):
	configs = []
	for p in range(num_elec-1, num_orb):
		config = _outer_config + 2**p
		if num_elec==1:
			configs += [config]
		else:
			configs += _all_nelec_configs(p, num_elec-1, _outer_config=config)
	return configs    # not suitable for end users because not a numpy BigInt array (needed for recursive use)

def fci_configs(num_spat_orb, num_elec_dn, num_elec_up, num_core_orb):
	active_orb  = num_spat_orb - num_core_orb
	active_dn   = num_elec_dn  - num_core_orb
	active_up   = num_elec_up  - num_core_orb
	configs_dn  = _all_nelec_configs(active_orb, active_dn)
	configs_up  = _all_nelec_configs(active_orb, active_up)
	core_shift  = 2**num_core_orb
	core_config = core_shift - 1
	spin_shift  = 2**num_spat_orb
	for i in range(len(configs_dn)):
		configs_dn[i] *= core_shift
		configs_dn[i] += core_config
		configs_dn[i] *= spin_shift
	for i in range(len(configs_up)):
		configs_up[i] *= core_shift
		configs_up[i] += core_config
	configs = []
	for config_dn in configs_dn:
		for config_up in configs_up:
			configs += [config_dn + config_up]
	return numpy.array(configs, dtype=BigInt.numpy)



class _action(object):
	def __init__(self, H, configs):
		self.H = H
		self.configs = configs
		self.thresh = 1e-10
	def __call__(self, Psi, vec_0, num_vecs):
		HPsi = numpy.zeros((num_vecs,len(self.configs)), dtype=Double.numpy)
		field_op.opPsi_1e(self.H.h,                # tensor of matrix elements (integrals)
		                  Psi,                     # block of row vectors: input vectors to act on
		                  HPsi,                    # block of row vectors: incremented by output
		                  self.configs,            # bitwise occupation strings stored as integers ... so, max 64 orbs for FCI ;-) [no checking here!]
		                  self.H.num_spin_orbs,    # edge dimension of the integrals tensor.  cannot be bigger than the number of bits in a BigInt (64)
		                  vec_0,                   # index of first vector in block to act upon
		                  num_vecs,                # how many vectors we are acting on simultaneously
		                  len(self.configs),       # how many configurations are there (call signature is ok as long as PyInt not longer than BigInt)
		                  self.thresh,             # threshold for ignoring integrals and coefficients (avoiding expensive index search)
		                  cpu_count())             # number of OMP threads to spread the work over
		field_op.opPsi_2e(self.H.V,                # tensor of matrix elements (integrals)
		                  Psi,                     # block of row vectors: input vectors to act on
		                  HPsi,                    # block of row vectors: incremented by output
		                  self.configs,            # bitwise occupation strings stored as integers ... so, max 64 orbs for FCI ;-) [no checking here!]
		                  self.H.num_spin_orbs,    # edge dimension of the integrals tensor.  cannot be bigger than the number of bits in a BigInt (64)
		                  vec_0,                   # index of first vector in block to act upon
		                  num_vecs,                # how many vectors we are acting on simultaneously
		                  len(self.configs),       # how many configurations are there (call signature is ok as long as PyInt not longer than BigInt)
		                  self.thresh,             # threshold for ignoring integrals and coefficients (avoiding expensive index search)
		                  cpu_count())             # number of OMP threads to spread the work over
		return HPsi



class Hamiltonian(object):
	def __init__(self, h, V, configs):
		self.h = h
		self.V = V
		self.num_spin_orbs = h.shape[0]
		self.H_action = _action(H=self, configs=configs)
	def __call__(self, Psi):
		n_unused, block, ii, block_dims_unused = Psi
		try:
			i, num = ii
			jj = 0, num
		except:
			i, num = ii, 1
			jj = 0
		Hblock = self.H_action(block, i, num)
		return (n_unused, Hblock, jj, block_dims_unused)
