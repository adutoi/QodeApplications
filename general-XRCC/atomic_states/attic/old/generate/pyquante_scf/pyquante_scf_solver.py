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
# Run under python2.7
# All the Info Saved are in SPATIAL ORBITALS and are NOT OthorNormal NOR CONVERGED.
#

import numpy as np
from PyQuante.Molecule import Molecule
from PyQuante.PyQuante2 import SCF         
from PyQuante.cints import ijkl2intindex as intindex
from PyQuante.Ints  import getT, getV

def scf_solver(molecule, basis='6-31G'):
	print molecule
	print "\tCharge = {} Multiplicity = {}\n".format(molecule.charge, molecule.multiplicity)
	print "Running HF/" + basis + '\n'
	solver = SCF(molecule, method="HF",basis=basis)
	solver.iterate()
	print "SCF Energy = {}  a.u.\n".format(solver.energy)
	return solver


def get_elec_rep_mat(v_pyquante, nbfs):
	V_mat = np.zeros((nbfs**2,nbfs**2), dtype=np.float64)
	# Conversion from PyQuante to my v_mat indexings...
	for i in range(nbfs):
		for j in range(nbfs):
			ij = i*(i+1)/2+j
			for k in range(nbfs):
				for l in range(nbfs):
					kl = k*(k+1)/2+l
					V_mat[i*nbfs+k, j*nbfs+l] = v_pyquante[intindex(i,j,k,l)] 
	return V_mat


def save_scf_info(solver):
	print "\nSaving E, C, S, h, V, T and N Info in PRIMITITIVE SPATIAL BASIS for {}".format(solver.molecule.name)
	bfs   = solver.basis_set.get()        # spatital basis functions
	nbfs  = len(bfs)                      # num of spatial basis functions
	atoms = solver.molecule.atoms
	# 1.Overlap;  2.core Hamiltonian;   3.Elec Repulsion;
	# 4. Kinetic; 5.Nuclear Attraction; 6.Converged HF vector 
	S_mat = np.matrix(solver.S)
	h_mat = np.matrix(solver.h) 
	V_mat = np.matrix(get_elec_rep_mat(solver.ERI, nbfs))
	T_mat = np.matrix(getT(bfs))
	N_mat = np.matrix(getV(bfs,atoms))
	C_mat = np.matrix(solver.solver.orbs.copy())

	prefix = solver.molecule.name
	# print "Saving Prefix = " + prefix
	np.save(prefix + "_MolOrb_Evals.npy", solver.solver.orbe)
	np.save(prefix + "_C_HF.npy", C_mat)
	np.save(prefix + "_S.npy", S_mat)
	np.save(prefix + "_h.npy", h_mat)
	np.save(prefix + "_V.npy", V_mat)
	np.save(prefix + "_T.npy", T_mat)
	np.save(prefix + "_N.npy", N_mat)
	np.save(prefix + "_Nucl_Attr_E.npy", np.array([solver.Enuke]))
	print "Info Saved...\n"



if __name__ == "__main__":
	import sys
	molecule = eval( open(sys.argv[1],'r').read() )
	if len(sys.argv) == 3:
		solver   = scf_solver(molecule, sys.argv[2])
	else:
		solver   = scf_solver(molecule)
	save_scf_info(solver)
	
	#
	#
	#
	# molecule = Molecule('H2O',
	#                [(8,  ( 0.00000000,     0.00000000,     0.04851804)),
	#                 (1,  ( 0.75300223,     0.00000000,    -0.51923377)),
	#                 (1,  (-0.75300223,     0.00000000,    -0.51923377))],
	#                 units='Angstrom')
	# name = 'Be2_4e'
	# molecule = Molecule( name,
	#                      [(4,  ( 0.00000000,     0.00000000,    0.00000000)),
	#                       (4,  ( 0.00000000,     0.00000000,    1.00000000))],
	#                      units='Angstrom',
	#                      charge=4)
	# name = 'Be'
	# molecule = Molecule( name,
	#                      [(4,  ( 0.00000000,     0.00000000,    0.00000000))],
	#                       units='Angstrom', charge=0, multiplicity=1 )
	# name = 'h2'
	# molecule = Molecule( name,
	#                      [(1,  ( 0.00000000,     0.00000000,    0.00000000)),
	#                       (1,  ( 0.00000000,     0.00000000,    1.00000000))],
	#                       units='Angstrom')


	


