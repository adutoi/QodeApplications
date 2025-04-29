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
from get_ints_Be import get_ints
from qode.atoms.integrals.fragments import unblock_2, unblock_last2, unblock_4
import psi4_check
from CI_space_traits import CI_space_traits
import field_op_ham
import configurations
import qode.util
from qode.util.PyC import Double
import densities
import pickle

def get_fci_states(dist, n_state_list=[(+1, 4), (0, 11), (-1, 8)]):
    class empty(object):  pass


    #basis_label = "cc-pvtz"  # according to alavi paper aug-cc-pvtz should be slightly better
    basis_label = "6-31G"
    #basis_label = "cc-pvdz"
    #n_spatial_orb = 14
    n_spatial_orb = 9  #30  # 9 for 6-31g and 46 for aug-cc-pvtz
    n_threads = 4
    #dist = float(sys.argv[1])
    #if len(sys.argv)==3:  n_threads = int(sys.argv[2])



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
    n_spin_orb    = 2 * n_spatial_orb
    spin_core     = frag0.basis.core + [p+n_spatial_orb for p in frag0.basis.core]

    states = {}
    for charge, n_subset in n_state_list:
        n_elec        = frag0.n_elec_ref - charge
        n_active_elec = n_elec - len(spin_core)
        configs = configurations.all_configs(n_spin_orb, n_active_elec, frozen_occ_orbs=spin_core)
        n_config = len(configs)

        CI_space_atom = qode.math.linear_inner_product_space(CI_space_traits(configs))
        H = CI_space_atom.lin_op(field_op_ham.Hamiltonian(h,V, n_threads=n_threads))
        CI_basis = [CI_space_atom.member(v) for v in CI_space_atom.aux.complete_basis()]

        Hmat = numpy.zeros((n_config,n_config), dtype=Double.numpy)
        for j,w in enumerate(CI_basis):
            Hmat[j,j] = N
            Hw = H|w
            for i,v in enumerate(CI_basis):
                Hmat[i,j] += v|Hw

        #evals, evecs = qode.util.sort_eigen(numpy.linalg.eigh(Hmat))
        # TODO: the following setup is Be specific. Generalize reference and cationic part like anionic part to make it at least general for singlet monomers
        if charge <= 0:
            guess = []
            if charge == 0:
                if n_subset % 2 != 1:
                    raise NotImplementedError("pick an uneven number of neutral guess vectors")
                ref = [0, n_spatial_orb+0]
                guess.append(CI_space_atom.member(CI_space_atom.aux.basis_vec(ref + [1, n_spatial_orb+1])))  # gs determinant
                for ex in range((n_subset - 1) // 2):
                    ex += 2
                    guess.append(CI_space_atom.member(CI_space_atom.aux.basis_vec(ref + [ex, n_spatial_orb+1])))  # alpha excitation
                    guess.append(CI_space_atom.member(CI_space_atom.aux.basis_vec(ref + [n_spatial_orb + ex, 1])))  # beta excitation
            if charge == -1:
                if n_subset % 2 != 0:
                    raise NotImplementedError("pick an even number of anionic guess vectors")
                ref = [0, 1, n_spatial_orb+0, n_spatial_orb+1]
                for ex in range(n_subset // 2):
                    ex += 2
                    guess.append(CI_space_atom.member(CI_space_atom.aux.basis_vec(ref + [ex])))  # alpha excitation
                    guess.append(CI_space_atom.member(CI_space_atom.aux.basis_vec(ref + [n_spatial_orb + ex])))  # beta excitation
            eigpairs = numpy.array(qode.math.lanczos.lowest_eigen_one_by_one(H, guess, num=n_subset, thresh=1e-8))
            evals, evecs = eigpairs[:, 0], eigpairs[:, 1]
            print()
            #print(evals)
            print(evals[0])
            #print(evecs)

            states[charge] = empty()
            states[charge].configs = configs
            states[charge].coeffs = [i.v for i in evecs]
        else:  # the cation matrix cannot be diagonalized with lanczos for Be with frozen core
            evals, evecs = qode.util.sort_eigen(numpy.linalg.eigh(Hmat))
            print(evals[0])
            states[charge] = empty()
            states[charge].configs = configs
            states[charge].coeffs = [evecs[:,i] for i in range(n_subset)]

    #rho = densities.build_tensors(states, n_spin_orb, n_elec)
    return states, n_spin_orb, n_elec, n_threads, frag0
