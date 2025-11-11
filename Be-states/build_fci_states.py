#    (C) Copyright 2023, 2025 Anthony D. Dutoi and Marco Bauer
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
from qode.many_body.fermion_field import CI_space_traits, field_op_ham, configurations
import qode.util
from qode.util import struct
from qode.util.PyC import Double
import densities
import pickle

def get_fci_states(dist, n_state_list=[(+1, 4), (0, 11), (-1, 8)], backend="psi4", monomer_charges=[[0, +1, -1], [0, +1, -1]]):


    #basis_label = "cc-pvtz"  # according to alavi paper aug-cc-pvtz should be slightly better
    basis_label = "6-31G"
    #basis_label = "cc-pvdz"
    #n_spatial_orb = 14
    n_spatial_orb = 9  #30  # 9 for 6-31g and 46 for aug-cc-pvtz
    n_threads = 4
    #dist = float(sys.argv[1])
    #if len(sys.argv)==3:  n_threads = int(sys.argv[2])



    frag0 = struct()
    frag0.atoms = [("Be",[0,0,0])]
    frag0.n_elec_ref = 4	# The number of electrons in the reference state of the monomer ("cation (+1)" and "anion (-1)" and technically interpreted relative to the reference, not zero, as would be the chemical definition)
    frag0.basis = struct()
    frag0.basis.AOcode = basis_label
    frag0.basis.n_spatial_orb = n_spatial_orb
    frag0.basis.MOcoeffs = numpy.identity(frag0.basis.n_spatial_orb)    # rest of code assumes spin-restricted orbitals
    frag0.basis.core = [0]	# indices of spatial MOs to freeze in CI portions

    if "psi4" in backend and not "vlx" in backend:
        import psi4_check
        psi4_check.print_HF_energy(
            "".join("{} {} {} {}\n".format(A,x,y,z) for A,(x,y,z) in frag0.atoms),
            frag0.basis.AOcode
            )
    elif "vlx" in backend and not "psi4" in backend:
        import vlx_check
        mol, basis, scf_driver, hf_result = vlx_check.print_HF_energy(
            "".join("{} {} {} {}\n".format(A,x,y,z) for A,(x,y,z) in frag0.atoms),
            frag0.basis.AOcode
            )
    else:
        raise NotImplementedError(f"SCF backend option {backend} not accepted")


    if ("mtp" in backend and "vlx" in backend) and not "in_house" in backend:
        import veloxchem as vlx
        import multipsi as mtp
        #frag0.basis.MOcoeffs = hf_result['C_alpha']

        states = {}
        for charge, n_states in n_state_list:
            states[charge] = struct()
            states[charge].coeffs = []
            states[charge].configs = []
            # charges under consideration
            # this also includes the spin-flips
            a_b_charges = []
            for chg0 in monomer_charges:
                for chg1 in monomer_charges:
                    a_b_charges.append((chg0,chg1))

            mtp_res = [[], [], []]
            beg = 0
            for comp in a_b_charges:
                print(charge, comp)
                if sum(comp) != charge:
                    continue
                if charge == 0 and comp[0] == 0:
                    # depending on choice of active space, MOs get reordered in multipsi
                    frag0.basis.MOcoeffs = space.get_ordered_mo_coefs()
                # multipsi calculation setup
                space=mtp.OrbSpace(mol,scf_driver.mol_orbs, charge=comp)
                space.spin_restricted = False  # this is also necessary for neutral restricted references
                space.fci(n_frozen=len(frag0.basis.core))
                CIdrv=mtp.CIDriver()
                # TODO: this only works well for restricted references and only one additional charge
                if 0 not in comp or charge != 0:
                    n_states_chg = int(n_states // 1.5)
                else:
                    n_states_chg = n_states
                ci_results = CIdrv.compute(mol,basis,space, n_states=n_states_chg)

                # transforming determinant basis to be compatible with in house fci density builder
                n_inact = len(space._get_inactive_mos()[0])
                n_tot = len(frag0.basis.MOcoeffs)
                inact_dets_alpha = 0
                inact_dets_beta = 0
                if n_inact > 0:
                    inact_dets_alpha = sum(2**numpy.array([i for i in range(n_inact)]))
                    inact_dets_beta = sum(2**numpy.array([i for i in range(n_tot, n_inact + n_tot)]))
                configs = [int(sum(2**(i.alpha_vector + n_inact)) + inact_dets_alpha
                           + sum(2**(i.beta_vector + n_inact + n_tot)) + inact_dets_beta)
                           for i in ci_results["ci_expansion"].determinant_list()]
                states[charge].configs += configs
                #states[charge].configs = configs + states[charge].configs
                sl = [slice(beg, beg + len(configs))]*n_states_chg
                beg += len(configs)
                mtp_res[0] += ci_results["energies"].tolist()
                mtp_res[1] += sl
                mtp_res[2] += [ci_results["ci_vectors"].to_numpy(i) for i in range(n_states_chg)]  # bit of a waste, but conversion is not that slow

            # sort for lowest energies (one could also think about using all of the states, as they are already build)
            #print(mtp_res[0], ci_results["energies"].tolist())
            sorted_vecs = [(en, sl, vec) for en, sl, vec in sorted(zip(mtp_res[0], mtp_res[1], mtp_res[2]), key=lambda pair: pair[0])]

            # build coeffs in full config space
            for n_s in range(n_states):
                coef = numpy.zeros(len(states[charge].configs))
                en, sl, vec = sorted_vecs[n_s]
                print(en, sl, vec.shape)
                coef[sl] = vec
                states[charge].coeffs.append(coef)

            # in house density builder requires configs in ascending order...
            order = [i for i in range(len(states[charge].configs))]
            new_ord = numpy.array([numpy.array([confs, inds]) for confs, inds in sorted(zip(states[charge].configs, order))])
            order = new_ord[:, 1]
            states[charge].configs = new_ord[:, 0].tolist()
            states[charge].coeffs = numpy.array(states[charge].coeffs)
            states[charge].coeffs[:, order]
            states[charge].coeffs = [i for i in states[charge].coeffs]


        for charge in [0, 1, -1]:
            print(len(states[charge].coeffs))
            print(len(states[charge].configs))
        
        """
        from state_screening import conf_decoder
        for num, coeff in enumerate(states[0].coeffs):
            print("state number ", num)
            for i, elem in enumerate(coeff):
                if elem > 1e-4:
                    print(elem, conf_decoder(states[0].configs[i], frag0.basis.n_spatial_orb))
        raise ValueError("stop here")
        """
                 
        return states, 2 * frag0.basis.n_spatial_orb, frag0.n_elec_ref, n_threads, frag0
    elif "in_house" in backend and not "mtp" in backend:
        symm_ints, bior_ints, nuc_rep = get_ints([frag0], spin_ints=False, backend=backend)
        N, S, T, U, V = nuc_rep[0,0], symm_ints.S[0,0], symm_ints.T[0,0], symm_ints.U[0,0,0], symm_ints.V[0,0,0,0]
        E, e, frag0.basis.MOcoeffs = RHF_RoothanHall_Nonorthogonal(frag0.n_elec_ref, (S, T+U, V), thresh=1e-12)
        print(E)
        if "vlx" in backend:
            frag0.basis.MOcoeffs = hf_result['C_alpha']

        symm_ints, bior_ints, nuc_rep = get_ints([frag0], spin_ints=False, backend=backend)
        N, S, T, U, V = nuc_rep[0,0], symm_ints.S[0,0], symm_ints.T[0,0], symm_ints.U[0,0,0], symm_ints.V[0,0,0,0]
        E, e, _ = RHF_RoothanHall_Orthonormal(frag0.n_elec_ref, (T+U, V), thresh=1e-12)
        print(E)

        symm_ints, bior_ints, nuc_rep = get_ints([frag0], spin_ints=True, backend=backend)
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
            if charge <= -2:#0:
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
                print(evals)
                #print(evecs)

                states[charge] = struct()
                states[charge].configs = configs
                states[charge].coeffs = [i.v for i in evecs]
            else:  # the cation matrix cannot be diagonalized with lanczos for Be with frozen core
                evals, evecs = qode.util.sort_eigen(numpy.linalg.eigh(Hmat))
                print(evals[:n_subset])
                states[charge] = struct()
                states[charge].configs = configs
                states[charge].coeffs = [evecs[:,i] for i in range(n_subset)]

        #rho = densities.build_tensors(states, n_spin_orb, n_elec)
        #return states, n_spin_orb, n_elec, n_threads, frag0
        return states, n_spin_orb, frag0.n_elec_ref, n_threads, frag0
    else:
        raise NotImplementedError(f"FCI backend option {backend} not accepted")
