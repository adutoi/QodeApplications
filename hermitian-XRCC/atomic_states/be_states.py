#    (C) Copyright 2018 Anthony D. Dutoi
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

import os
import sys
from math import sqrt
import pickle
import numpy
import qode
from qode.util import parallel, output, textlog
import LMO_frozen_special_fci						# compute atom FCI states
import LMO_frozen_buildOverlapMat, LMO_frozen_ct_buildOverlapMat	# put atomic states in dimer basis
import LMO_frozen_Hamiltonian_wrapper					# act hamiltonian on dimer state
from fci_space import fci_space_traits
import build_tens_pdt
from monomer_rotations import monomer_rotations
import excitonic



dist, state_string, compression = sys.argv[1:] 

compression    = compression.split("=")
compression[1] = "=".join(compression[1:])
n_1e_states, n_2e_states, n_3e_states = state_string.split("-")
n_states = { 2:int(n_2e_states),  1:int(n_1e_states),  3:int(n_3e_states) }	# keys reference number of valence electrons explicitly (this is a frozen-core Be2 code!)

print("Loading data ...", end="", flush=True)
h_mat       = numpy.load("integrals/Be_h.npy")
V_mat       = numpy.load("integrals/Be_V.npy")
C_scf_atom  = numpy.load("integrals/Be_C.npy")
C_scf_dimer = numpy.load("integrals/Be2_{}_C.npy".format(dist))
S_dimer     = numpy.load("integrals/Be2_{}_S.npy".format(dist))
print("Done.")

frag_dim  = n_states[1] + n_states[2] + n_states[3]
frag_idx = { 2: (0, n_states[2]),  1: (n_states[2], n_states[2]+n_states[1]),  3: (n_states[2]+n_states[1], frag_dim) }
super_dim = frag_dim**2
# Given that dict keys are hard-coded, some of these could be hard coded too, but this makes it easier to read.
num_elec_atom         = 4	# For neutral (deviations handled explicitly, locally)
num_core_elec_atom    = 2
num_valence_elec_atom = num_elec_atom - num_core_elec_atom
num_spin_orbs_atom    = h_mat.shape[0]
num_spat_orbs_atom    = num_spin_orbs_atom // 2
num_elec_dimer        = 2 * num_elec_atom
num_core_elec_dimer   = 2 * num_core_elec_atom
num_spin_orbs_dimer   = 2 * num_spin_orbs_atom

# Compute atomic states in the basis of dimer orbitals (A/Low refers to each of the two atoms)
print("Building atomic eigenstates in dimer orbital basis ...", flush=True)
U_1e, U_2e, U_3e, nrg = LMO_frozen_special_fci.compute_fci_vecs(num_spin_orbs_atom, num_core_elec_atom, h_mat, V_mat, n_states[1], n_states[2], n_states[3])	# Atomic FCI calcs
U_1e = U_1e[:,:n_states[1]].copy()	# Slice out the requisite number of 1e- eigenstates (in the basis of atomic orbitals)
U_2e = U_2e[:,:n_states[2]].copy()	# Slice out the requisite number of 2e- eigenstates (in the basis of atomic orbitals)
U_3e = U_3e[:,:n_states[3]].copy()	# Slice out the requisite number of 3e- eigenstates (in the basis of atomic orbitals)
atomAvec2e, atomBvec2e                         = LMO_frozen_buildOverlapMat.getBasisOverlapMatrix(    C_scf_atom, C_scf_dimer, S_dimer, 
                                                                                                      num_elec_atom, num_spat_orbs_atom, num_core_elec_dimer, U_2e )
atomAvec1e, atomAvec3e, atomBvec1e, atomBvec3e = LMO_frozen_ct_buildOverlapMat.getBasisOverlapMatrix( C_scf_atom, C_scf_dimer, S_dimer,
                                                                                                      num_elec_atom, num_spat_orbs_atom, num_core_elec_dimer, U_1e, U_3e )
atomAvec = { 2: atomAvec2e,  1: atomAvec1e,  3: atomAvec3e  }
atomBvec = { 2: atomBvec2e,  1: atomBvec1e,  3: atomBvec3e  }
print("... Done.")

# Set up Hamiltonian and promote it and tensor product basis to living in that space
print("Setting up Hamiltonian ... ", end="", flush=True)
h_mat = numpy.load("integrals/Be2_{}_h.npy".format(dist))
V_mat = numpy.load("integrals/Be2_{}_V.npy".format(dist))
H = LMO_frozen_Hamiltonian_wrapper.Hamiltonian(h_mat, V_mat, num_core_elec_dimer, num_elec=[6,7,8,9,10])
fci_space = qode.math.linear_inner_product_space(fci_space_traits)
H = fci_space.lin_op(H)
print("Done.")


# Obtain a small number of states for each monomer
if   compression[0]=="load":

	print("Loading pre-computed monomer transformations ... ", end="", flush=True)
	u   = pickle.load(open(compression[1].replace(":","/"),"rb"))
	states_per_frag = len(u[2]) + len(u[1]) + len(u[3])
	path = "states/" + state_string + "/load="+compression[1].replace("/",":")
	print("Done.")

else:

	# Find the dimer ground state (orthonormalize the basis because Lanczos only for Hermitian case, then back to non-ON basis)
	print("Ground-state calculation ... ", flush=True)
	guess = fci_space.member(build_tens_pdt.state(2,0, 2,0, atomAvec, atomBvec, num_spin_orbs_dimer, num_core_elec_dimer, num_valence_elec_atom))
	(Eval,Evec), = qode.math.lanczos.lowest_eigen(H, [guess], thresh=1e-8)
	print("... Done.  \n\nE_gs = {}\n".format(Eval))

	print("Building tensor-product dimer states ...", flush=True)
	neut_basis, tns_pdt_idx = build_tens_pdt.basis_neutral(n_states, atomAvec, atomBvec, num_spin_orbs_dimer, num_core_elec_dimer, num_valence_elec_atom)
	neut_basis = qode.math.vector_set(fci_space, [fci_space.member(v) for v in neut_basis])
	print("... Done.")

	# Expand projected vector to the full dimer-product length to make dens-mat calc easier
	Evec = numpy.array(neut_basis.project(Evec))
	Evec /= numpy.linalg.norm(Evec)
	Evec_long = numpy.zeros(super_dim)
	for c,i in zip(Evec, tns_pdt_idx):  Evec_long[i] = c
	Evec = neut_basis.deproject(Evec)
	Evec /= sqrt(Evec|Evec)
	print("\nEnergy after projection = {}\n".format(Evec|H|Evec))

	# Build reduced density matrices for fragments A and B for the dimer ground state
	print("Diagonalizing atomic reduced density matrices ... ", end="", flush=True)
	u, d, tag = monomer_rotations(Evec_long, compression, frag_idx)
	states_per_frag = len(u[2]) + len(u[1]) + len(u[3])
	path = "states/" + state_string + "/" + tag + "/"
	print("Done.")

	# Set path for output and dump transformation
	print("Saving monomer transformations ... ", end="", flush=True)
	pickle.dump(u, open(path+"/"+dist+"/u.pickle","wb"))
	pickle.dump(d, open(path+"/"+dist+"/d.pickle","wb"))
	print("Done.")

open(path+"/"+dist+"/nstates={}".format(states_per_frag),"w").close()

# Build the Hamiltonian transformed into the basis of natural monomer states
new_n_states = { 2: len(u[2]),  1: len(u[1]),  3: len(u[3])  }
new_super_dim = states_per_frag**2

print("Transforming monomer states ... ", end="", flush=True)
u[2] = numpy.array(u[2]).T
u[1] = numpy.array(u[1]).T
u[3] = numpy.array(u[3]).T
### This is the connection point to the new code.  Spits out monomer eigenstates in full configuration basis.
numpy.save(path+"/"+dist+"/U_1e.npy",U_1e)
numpy.save(path+"/"+dist+"/U_2e.npy",U_2e)
numpy.save(path+"/"+dist+"/U_3e.npy",U_3e)
Z_1e = U_1e.dot(u[1])
Z_2e = U_2e.dot(u[2])
Z_3e = U_3e.dot(u[3])
numpy.save(path+"/"+dist+"/Z_1e.npy",Z_1e)
numpy.save(path+"/"+dist+"/Z_2e.npy",Z_2e)
numpy.save(path+"/"+dist+"/Z_3e.npy",Z_3e)
###
atomAvec2e = atomAvec2e.dot(u[2])	# overwrites old names!
atomBvec2e = atomBvec2e.dot(u[2])
atomAvec1e = atomAvec1e.dot(u[1])
atomBvec1e = atomBvec1e.dot(u[1])
atomAvec3e = atomAvec3e.dot(u[3])
atomBvec3e = atomBvec3e.dot(u[3])
atomAvec   = { 2: atomAvec2e,  1: atomAvec1e,  3: atomAvec3e  }
atomBvec   = { 2: atomBvec2e,  1: atomBvec1e,  3: atomBvec3e  }
print("Done.")

print("Building tensor-product dimer states ... ", flush=True)
tens_pdt_basis, tns_pdt_idx, n_blocked_idx = build_tens_pdt.basis(new_n_states, atomAvec, atomBvec, num_spin_orbs_dimer, num_core_elec_dimer, num_valence_elec_atom)  # Overwrites old name
tens_pdt_basis = [fci_space.member(v) for v in tens_pdt_basis]
print("... Done.")

print("Building the compressed Hamiltonian ... ", end="", flush=True)
h_n = {}
for n in (6,7,8,9,10):
	n_basis   = qode.math.vector_set(fci_space, [tens_pdt_basis[i] for i in tns_pdt_idx[n]] )
	### For testing Marco's code
	numpy.save(path+"/"+dist+"/idx-{}.npy".format(n), tns_pdt_idx[n])
	numpy.save(path+"/"+dist+"/S-{}.npy".format(n), n_basis.overlaps())
	###
	n_basis_H = n_basis.project(H)
	#h_n[n]    = fci_space.dot_vec_blocks(n_basis_H,n_basis)		... why is this not working!!!
	h_n[n]    = numpy.array([[ (bra_H|ket) for ket in n_basis ] for bra_H in n_basis_H])
	### For testing Marco's code
	numpy.save(path+"/"+dist+"/H-{}.npy".format(n), h_n[n])
	###
h = numpy.zeros((new_super_dim,new_super_dim))
for I,(m,i) in enumerate(n_blocked_idx):
	for J,(n,j) in enumerate(n_blocked_idx):
		if m==n:  h[I,J] = h_n[n][i,j]
print("Done.")

### This is here to help debug the new code.
numpy.save(path+"/"+dist+"/h.npy", h)
###

# Extract the lowest eigenvalue from the compressed Hamiltonian for comparison
print("Diagonalizing ... ", end="", flush=True)
Evals,Evecs = numpy.linalg.eig(h)
Evals,Evecs = zip(*sorted(zip( Evals, Evecs.T.tolist() ), key=lambda p: p[0]))
print("Done.  \n\nE_gs  = {}  [compressed H]\n".format(Evals[0]))

# Dump compressed Hamiltonian separated into monomer and dimer parts
vecs = []
for n in (2,1,3):
	beg, end = frag_idx[n]
	tmp = numpy.zeros((frag_dim,new_n_states[n]))
	tmp[beg:end, : ] = u[n]
	vecs += tmp.T.tolist()
u = numpy.array(vecs)

f  = u.dot(numpy.diag(nrg).dot(u.T))
h0 = numpy.zeros((states_per_frag*states_per_frag, states_per_frag*states_per_frag))
for i in range(states_per_frag):
	for j in range(states_per_frag):
		Mij = i*states_per_frag + j
		for k in range(states_per_frag):
			for l in range(states_per_frag):
				Nkl = k*states_per_frag + l
				if i==k:  h0[Mij][Nkl] += f[j][l]
				if j==l:  h0[Mij][Nkl] += f[i][k]
numpy.save(path+"/"+dist+"/f.npy", f)
V = h - h0
numpy.save(path+"/"+dist+"/V.npy", V)

# Check our work
out, resources  = output(log=textlog(echo=True)), parallel.resources(1)
out.log("EXCITONIC CCSD CALCULATION")
excitonic.ccsd(([f,f],[[None,V],[None,None]]), out, resources)

f_mat = numpy.load(path+"/"+dist+"/f.npy")     # Could call fci.main() here, but until everything is rock solid, I like independence of this simple dimer-only H-build
V_mat = numpy.load(path+"/"+dist+"/V.npy")
out.log("Building FCI matrix ...")
for i in range(states_per_frag):
	for j in range(states_per_frag):
		Mij = i*states_per_frag + j
		for k in range(states_per_frag):
			for l in range(states_per_frag):
				Nkl = k*states_per_frag + l
				if i==k:  V_mat[Mij][Nkl] += f_mat[j][l]
				if j==l:  V_mat[Mij][Nkl] += f_mat[i][k]
out.log("Diagonalizing ...")
vals, vecs = numpy.linalg.eig(V_mat)
vals, vecs = zip(*sorted(zip( vals, vecs.T.tolist() ), key=lambda p: p[0]))
out.log("FCI Energy  =", vals[0])

out.log.write(path+"/"+dist+"/log.txt")
