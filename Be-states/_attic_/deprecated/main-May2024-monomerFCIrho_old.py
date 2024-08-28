#    (C) Copyright 2023 Anthony D. Dutoi
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

# Usage: python main-May2024-monomerFCIrho.py 4.5
# "4.5" is the bond distance (for example)

import sys
import numpy
import qode
from qode.many_body.self_consistent_field.fermionic import RHF_RoothanHall_Nonorthogonal, RHF_RoothanHall_Orthonormal
from get_ints import get_ints
from qode.atoms.integrals.fragments import unblock_2, unblock_last2, unblock_4
import psi4_check
from CI_space_traits import CI_space_traits
import field_op_ham
import configurations
import qode.util
from qode.util.PyC import Double
import densities_old
import pickle
class empty(object):  pass



basis_label = "6-31G"
n_spatial_orb = 9
n_threads = 1
dist = float(sys.argv[1])
if len(sys.argv)==3:  n_threads = int(sys.argv[2])



frag0 = empty()
frag0.atoms = [("Be",[0,0,0])]
frag0.n_elec_ref = 4	# The number of electrons in the reference state of the monomer ("cation (+1)" and "anion (-1)" and technically interpreted relative to the reference, not zero, as would be the chemical definition)
frag0.basis = empty()
frag0.basis.AOcode = basis_label
frag0.basis.n_spatial_orb = n_spatial_orb
frag0.basis.MOcoeffs = numpy.identity(frag0.basis.n_spatial_orb)    # rest of code assumes spin-restricted orbitals
frag0.basis.core = [0]	# indices of spatial MOs to freeze in CI portions



psi4_check.print_HF_energy(
    "".join("{} {} {} {}\n".format(A,x,y,z) for A,(x,y,z) in frag0.atoms),
    frag0.basis.AOcode
    )

symm_ints, bior_ints, nuc_rep = get_ints([frag0], spin_ints=False)
N, S, T, U, V = nuc_rep[0,0], symm_ints.S[0,0], symm_ints.T[0,0], symm_ints.U[0,0,0], symm_ints.V[0,0,0,0]
E, e, frag0.basis.MOcoeffs = RHF_RoothanHall_Nonorthogonal(frag0.n_elec_ref, (S, T+U, V), thresh=1e-12)
print(E)

symm_ints, bior_ints, nuc_rep = get_ints([frag0], spin_ints=False)
N, S, T, U, V = nuc_rep[0,0], symm_ints.S[0,0], symm_ints.T[0,0], symm_ints.U[0,0,0], symm_ints.V[0,0,0,0]
E, e, _ = RHF_RoothanHall_Orthonormal(frag0.n_elec_ref, (T+U, V), thresh=1e-12)
print(E)



symm_ints, bior_ints, nuc_rep = get_ints([frag0], spin_ints=True)
N, S, T, U, V = nuc_rep[0,0], symm_ints.S[0,0], symm_ints.T[0,0], symm_ints.U[0,0,0], symm_ints.V[0,0,0,0]
h = T + U

n_spatial_orb = frag0.basis.n_spatial_orb
spatial_core  = frag0.basis.core
n_spin_orb    = 2 * n_spatial_orb
spin_core     = spatial_core + [p+n_spatial_orb for p in spatial_core]

states = {}
for charge, n_subset in [(+1, 5), (0, 5), (-1, 5)]:

    n_elec        = frag0.n_elec_ref - charge
    n_active_elec = n_elec - len(spin_core)
    configs = configurations.all_configs(n_spin_orb, n_active_elec, frozen_occ_orbs=spin_core)
    n_config = len(configs)

    CI_space_atom = qode.math.linear_inner_product_space(CI_space_traits(configs))
    H = CI_space_atom.lin_op(field_op_ham.Hamiltonian(h,V, n_threads=n_threads))
    CI_basis = [CI_space_atom.member(v) for v in CI_space_atom.aux.complete_basis()]

    Hmat = numpy.zeros((n_config,n_config), dtype=Double.numpy, order="C")
    for j,w in enumerate(CI_basis):
        Hmat[j,j] = N
        Hw = H|w
        for i,v in enumerate(CI_basis):
            Hmat[i,j] += v|Hw

    evals, evecs = qode.util.sort_eigen(numpy.linalg.eigh(Hmat))
    print()
    print(evals)
    print(evals[0])

    states[charge] = empty()
    states[charge].configs = configs
    states[charge].coeffs = [evecs[:,i] for i in range(n_subset)]



rho = densities_old.build_tensors(states, n_spatial_orb, spatial_core, n_threads)

pickle.dump(rho, open("Be631g.pkl", "wb"))
