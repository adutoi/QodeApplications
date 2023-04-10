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

# Use within a Psi4 conda environment for integrals

# This computes the frozen-core FCI energy using both the MO and biorthogonal semi-MO basis (so both the defintion of core and resolution of the identity are slightly different)
# runs with just: python hard_way.py


import numpy
from qode.util import sort_eigen
from qode.math import precise_numpy_inverse
import qode.atoms.integrals.spatial_to_spin as spatial_to_spin
from qode.atoms.integrals.fragments import AO_integrals, semiMO_integrals, block_2, unblock_2, bra_transformed, spin_orb_integrals, unblock_last2, unblock_4
from qode.fermion_field import state
from qode.fermion_field.occ_strings import all_occ_strings
from qode.fermion_field.state import create, annihilate, op_string
from qode.many_body.self_consistent_field.fermionic import RHF_RoothanHall_Nonorthogonal
from BeCustom import monomer_data as Be

class empty(object):  pass



def inv_rt(M):
	vals, vecs = numpy.linalg.eigh(M)
	vals = numpy.diag([1/numpy.sqrt(v) for v in vals])
	return vecs @ vals @ vecs.T

def get_ints(fragments):
	basis = fragments[0].basis.AOcode
	AO_ints     = AO_integrals(basis, fragments)
	SemiMO_ints = semiMO_integrals(AO_ints, [frag.basis.MOcoeffs for frag in fragments], cache=True)     # Cache because multiple calls to each block during biorthogonalization
	return SemiMO_ints, AO_ints

def MO_transform(h, V, C):
	h = C.T @ h @ C
	for _ in range(4):  V = numpy.tensordot(V, C, axes=([0],[0]))       # cycle through the tensor axes (this assumes everything is real)
	return h, V

def H_build(occ_basis, h, V):
	n_orb = h.shape[0]
	#if V.shape[0]!=n_orb:         raise AssertionError
	#if len(occ_basis[0])!=n_orb:  raise AssertionError

	dim = len(occ_basis)
	Hmat = numpy.zeros((dim,dim))

	c = [create(i)     for i in range(n_orb)]
	a = [annihilate(i) for i in range(n_orb)]

	for p in range(n_orb):
		for q in range(n_orb):
			#print(p,q)
			ca = op_string( c[p], a[q] )
			for j,ket in enumerate(occ_basis):
				Hket = ca | ket
				if Hket.is_not_null():
					for i,bra in enumerate(occ_basis):
						Hmat[i,j] += h[p][q] * (bra|Hket)
	for p in range(n_orb):
		for q in range(n_orb):
			for r in range(n_orb):
				for s in range(n_orb):
					#print(p,q,r,s)
					ccaa = op_string( c[p], c[q], a[r], a[s] )
					for j,ket in enumerate(occ_basis):
						Hket = ccaa | ket
						if Hket.is_not_null():
							for i,bra in enumerate(occ_basis):
								Hmat[i,j] += V[p][q][s][r] * (bra|Hket)
	return Hmat

def scf(n_elec, S, h, V):
	E, e, C = RHF_RoothanHall_Nonorthogonal(n_elec, (S,h,V), thresh=1e-12)
	return MO_transform(h, V, C)

def spin_ints(h, V):
	h = spatial_to_spin.one_electron_blocked(h)
	V = spatial_to_spin.two_electron_blocked(V)
	V = (1/4.) * (V - numpy.swapaxes(V,2,3))
	return h, V

def integrals(fragments, n_elec):
	SemiMOints, AOints = get_ints(fragments)

	sMO_S = unblock_2(    fragments, SemiMOints.S)
	sMO_h = unblock_2(    fragments, SemiMOints.T)
	sMO_U = unblock_last2(fragments, SemiMOints.U)
	sMO_V = unblock_4(    fragments, SemiMOints.V)
	for _,U in sMO_U.items():  sMO_h += U
	sMO_Sinv   = precise_numpy_inverse(sMO_S)

	bsMO_h = sMO_Sinv @ sMO_h
	bsMO_V = numpy.tensordot(sMO_Sinv,  sMO_V, axes=([1],[1]))
	bsMO_V = numpy.tensordot(sMO_Sinv, bsMO_V, axes=([1],[1]))
	bsMO = empty()
	bsMO.h, bsMO.V = spin_ints(bsMO_h, bsMO_V)

	AO_S = unblock_2(    fragments, AOints.S)
	AO_h = unblock_2(    fragments, AOints.T)
	AO_U = unblock_last2(fragments, AOints.U)
	AO_V = unblock_4(    fragments, AOints.V)
	for _,U in AO_U.items():  AO_h += U
	MO = empty()
	MO.h, MO.V = spin_ints(*scf(n_elec, AO_S, AO_h, AO_V))

	return bsMO, MO

def FCI_occ_strings(n_elec, n_orb, n_core):
	n_active = n_orb - n_core
	core = [True]*(n_core//2)
	occ_strings = all_occ_strings(n_active, n_elec-n_core)
	new_occ_strings = []
	for occ_string in occ_strings:
		alphas = occ_string[:(n_active//2)]
		betas  = occ_string[(n_active//2):]
		new_occ_strings += [core + alphas + core + betas]
	return new_occ_strings

def MO_FCI_basis(n_elec, n_orb, n_core):
	return [state.configuration(occ_string) for occ_string in FCI_occ_strings(n_elec, n_orb, n_core)]



def semiMO_FCI_basis(n_elec_frag, Dn_elec_frag, n_orb_frag, n_core_frag, n_frag):
	occ_stringsM = {}
	for D in Dn_elec_frag:
		occ_stringsM[D] = FCI_occ_strings(n_elec_frag+D, n_orb_frag, n_core_frag)
	occ_strings = { 0:[[]] }
	for M in range(n_frag):
		new_occ_strings = {}
		for D in occ_strings:
			for Dm in occ_stringsM:
				Dtot = D + Dm
				if Dtot not in new_occ_strings:  new_occ_strings[Dtot] = []
				for occ_string in occ_strings[D]:
					for occ_stringM in occ_stringsM[Dm]:
						new_occ_strings[Dtot] += [occ_string+occ_stringM]
		occ_strings = new_occ_strings
	return [state.configuration(occ_string) for occ_string in occ_strings[0]]	# overall deviation in elec number should be zero



n_frag       = 2
displacement = 9.0
n_core_frag  = 1	# number of core spatial orbs per fragment

BeN = [Be((0,0,m*displacement)) for m in range(n_frag)]

n_core_frag *= 2
n_orb_frag   = 2 * BeN[0].basis.n_spatial_orb
n_elec_frag  =     BeN[0].n_elec
n_core       = n_frag * n_core_frag
n_orb        = n_frag * n_orb_frag
n_elec       = n_frag * n_elec_frag

bsMOints, MOints = integrals(BeN, n_elec)

max_cation = n_elec_frag - n_core_frag
max_anion  = n_orb_frag  - n_elec_frag
Dn_elec_frag = range(-max_cation, +(max_anion+1))
basis = semiMO_FCI_basis(n_elec_frag, Dn_elec_frag, n_orb_frag, n_core_frag, n_frag)
H = H_build(basis, bsMOints.h, bsMOints.V)
vals, vecs = sort_eigen(numpy.linalg.eig(H))
print("biorthogonal semi-MO frozen-core FCI energy = ", vals[0])

basis = MO_FCI_basis(n_elec, n_orb, n_core)
H = H_build(basis, MOints.h, MOints.V)
vals, vecs = sort_eigen(numpy.linalg.eigh(H))
print("                  MO frozen-core FCI energy = ", vals[0])
