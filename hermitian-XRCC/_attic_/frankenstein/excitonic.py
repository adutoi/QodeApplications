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
import numpy
import qode
from   qode.util import sort_eigen
from   qode.many_body import CCSD
from   qode.many_body.hierarchical_fluctuations.excitonic import SD_operator, H_012_operator, inv_diagE, BakerCampbellHausdorff, space_traits
from   hamiltonian import braket_loops



def ccsd(H, out, resources, diis_start=0):
	monomer_Hamiltonians, dimer_Couplings = H
	states_per_frag = [h.shape[0] for h in monomer_Hamiltonians]

	out.log("Digesting H ...")
	E0, H = H_012_operator(states_per_frag, monomer_Hamiltonians, dimer_Couplings)

	out.log("Setting up fragment-energy preconditioner ...")
	invH0 = inv_diagE([numpy.diag(F) for F in monomer_Hamiltonians], states_per_frag)

	out.log("Initializing T ...")
	T = SD_operator(states_per_frag)		# Initialized to zero

	BCH = BakerCampbellHausdorff(states_per_frag, resources)
	out.log("Baker-Campbell-Hausdorff module initialized.")

	out.log("Performing coupled-cluster iterations ...")
	t_ccsd_start = time.time()
	dE, T = CCSD(H, T, invH0, BCH, space_traits, out.log.sublog(), thresh=1e-10, diis_start=diis_start)
	E = E0 + dE
	t_ccsd_end = time.time()

	out.log("CCSD Time {}".format(t_ccsd_end - t_ccsd_start))
	out.log("CCSD Energy =", E, 'Hartree')
	return E, T



def fci(H, out):
	monomer_Hamiltonians, dimer_Hamiltonians = H
	N_frag = len(monomer_Hamiltonians)
	states_per_frag = monomer_Hamiltonians[0].shape[0]
	for h in monomer_Hamiltonians:
		if h.shape[0]!=states_per_frag:  raise AssertionError	# Only for homogeneous systems right now

	out.log("Building ...")
	Hmat = numpy.zeros((states_per_frag**N_frag, states_per_frag**N_frag))
	braket_loops(Hmat, N_frag, states_per_frag, H)

	out.log("Diagonalizing ...")
	dim = Hmat.shape[0]
	if True:	# Because Lanczos does not yet handle non-hermitian matrices
		vals, vecs = sort_eigen(numpy.linalg.eig(Hmat))
		E = vals[0]
	else:
		traits = qode.math.numpy_space.real_traits(dim)
		S = qode.math.linear_inner_product_space(traits)
		H = S.lin_op(Hmat)
		v = S.member(qode.util.basis_vector(0,dim))
		(Eval,Evec), = qode.math.lanczos.lowest_eigen(H, [v], thresh=1e-8)
		print("... Done.  \n\nE_gs = {}\n".format(Eval))
		E = Eval

	out.log("FCI Energy  =", E)
	return E
