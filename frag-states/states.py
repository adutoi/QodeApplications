#!/usr/bin/env python3
#    (C) Copyright 2023, 2024, 2025 Anthony D. Dutoi
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
import qode
from qode.util import struct
from qode.util.PyC import Double
from qode.many_body.self_consistent_field.fermionic import RHF_RoothanHall_Nonorthogonal, RHF_RoothanHall_Orthonormal
from qode.atoms.integrals.fragments import unblock_2, unblock_last2, unblock_4
from qode.fermion_field import CI_space_traits, field_op_ham, configurations
from get_ints import get_ints



def get_optimal(frags, statesthresh, n_threads=1):

    for frag in frags:
        frag.basis.MOcoeffs = numpy.identity(frag.basis.n_spatial_orb)    # rest of code assumes spin-restricted orbitals
        symm_ints, bior_ints, nuc_rep = get_ints([frag], spin_ints=False)
        N, S, T, U, V = nuc_rep[0,0], symm_ints.S[0,0], symm_ints.T[0,0], symm_ints.U[0,0,0], symm_ints.V[0,0,0,0]
        E, e, frag.basis.MOcoeffs = RHF_RoothanHall_Nonorthogonal(frag.n_elec_ref, (S, T+U, V), thresh=1e-12)
        print(E)

    # insist identical for this code
    frags[1].basis.MOcoeffs = frags[0].basis.MOcoeffs

    symm_ints, bior_ints, nuc_rep = get_ints(frags)

    num_elec_atom_dn = frags[0].n_elec_ref // 2
    num_elec_atom_up = frags[0].n_elec_ref - num_elec_atom_dn
    num_spatial_atom = frags[0].basis.n_spatial_orb



    dn_configs_atom = configurations.all_configs(num_spatial_atom, num_elec_atom_dn-len(frags[0].basis.core), frozen_occ_orbs=frags[0].basis.core)
    up_configs_atom = configurations.all_configs(num_spatial_atom, num_elec_atom_up-len(frags[0].basis.core), frozen_occ_orbs=frags[0].basis.core)
    configs_atom    = configurations.tensor_product_configs([dn_configs_atom,up_configs_atom], [num_spatial_atom,num_spatial_atom])

    N, S, T, U, V = nuc_rep[0,0], symm_ints.S[0,0], symm_ints.T[0,0], symm_ints.U[0,0,0], symm_ints.V[0,0,0,0]
    h = T + U

    CI_space_atom = qode.math.linear_inner_product_space(CI_space_traits(configs_atom))
    H     = CI_space_atom.lin_op(field_op_ham.Hamiltonian(h,V, n_elec=frags[0].n_elec_ref, n_threads=n_threads))
    guess = CI_space_atom.member(CI_space_atom.aux.basis_vec([0, 1, num_spatial_atom+0, num_spatial_atom+1]))

    print((guess|H|guess) + N)
    (Eval,Evec), = qode.math.lanczos.lowest_eigen(H, [guess], thresh=1e-8)
    print("\nE_gs = {}\n".format(Eval+N))



    dimer_core = frags[0].basis.core + [c+frags[0].basis.n_spatial_orb for c in frags[1].basis.core]

    dn_configs_dimer = configurations.all_configs(2*num_spatial_atom, 2*num_elec_atom_dn-len(dimer_core), frozen_occ_orbs=dimer_core)
    up_configs_dimer = configurations.all_configs(2*num_spatial_atom, 2*num_elec_atom_up-len(dimer_core), frozen_occ_orbs=dimer_core)
    dn_configs_decomp = configurations.decompose_configs(dn_configs_dimer, [num_spatial_atom, num_spatial_atom])
    up_configs_decomp = configurations.decompose_configs(up_configs_dimer, [num_spatial_atom, num_spatial_atom])
    combine_configs = configurations.config_combination([num_spatial_atom, num_spatial_atom])
    all_configs_1 = list()
    all_configs_0 = set()
    nested = []
    for config_dn1,configs_dn0 in dn_configs_decomp:
        for config_up1,configs_up0 in up_configs_decomp:
            config_1  = combine_configs([config_up1, config_dn1])
            configs_0 = configurations.tensor_product_configs([configs_up0, configs_dn0], [num_spatial_atom, num_spatial_atom])
            nested += [(config_1, configs_0)]
            all_configs_1 += [config_1]
            all_configs_0 |= set(configs_0)
    all_configs_0 = list(all_configs_0)
    configs_dimer = configurations.recompose_configs(nested, [2*num_spatial_atom, 2*num_spatial_atom])

    num_elec_dimer = frags[0].n_elec_ref + frags[1].n_elec_ref
    sorted_configs_1 = [[] for n in range(num_elec_dimer+1)]
    sorted_configs_0 = [[] for n in range(num_elec_dimer+1)]
    for config_1 in all_configs_1:
        n = config_1.bit_count()
        sorted_configs_1[n] += [config_1]
    sorted_configs_1 = [sorted(sorted_configs_1_n) for sorted_configs_1_n in sorted_configs_1]
    for config_0 in all_configs_0:
        n = config_0.bit_count()
        sorted_configs_0[n] += [config_0]
    sorted_configs_0 = [sorted(sorted_configs_0_n) for sorted_configs_0_n in sorted_configs_0]

    frag1_to_dimer = [[[] for _ in range(len(sorted_configs_1_n))] for sorted_configs_1_n in sorted_configs_1]
    frag0_to_dimer = [[[] for _ in range(len(sorted_configs_0_n))] for sorted_configs_0_n in sorted_configs_0]
    dimer_to_frags = []
    P = 0
    for config_1,configs_0 in nested:
        n1 = config_1.bit_count()
        i1 = sorted_configs_1[n1].index(config_1)
        for config_0 in configs_0:
            n0 = config_0.bit_count()
            i0 = sorted_configs_0[n0].index(config_0)
            dimer_to_frags += [((n1,i1),(n0,i0))]
            frag1_to_dimer[n1][i1] += [P]
            frag0_to_dimer[n0][i0] += [P]
            P += 1

    N = nuc_rep[0,0] + nuc_rep[1,1] + nuc_rep[0,1]
    T = unblock_2(    bior_ints.T, frags, spin_orbs=True)
    U = unblock_last2(bior_ints.U, frags, spin_orbs=True)
    V = unblock_4(    bior_ints.V, frags, spin_orbs=True)
    h = T + U[0] + U[1]

    core = dimer_core + [c+2*frags[0].basis.n_spatial_orb for c in dimer_core]
    orbs = list(range(4*num_spatial_atom))    # 4 because dimer spin orbs
    for p in orbs:
        for q in orbs:
            if (q in core) and (p!=q):  h[p,q] = 0
            if (p in core) and (p!=q):  h[p,q] = 0
            for r in orbs:
                for s in orbs:
                    if (r in core) and (p!=r) and (q!=r):  V[p,q,r,s] = 0
                    if (s in core) and (p!=s) and (q!=s):  V[p,q,r,s] = 0
                    if (p in core) and (p!=r) and (p!=s):  V[p,q,r,s] = 0
                    if (q in core) and (q!=r) and (q!=s):  V[p,q,r,s] = 0

    CI_space_dimer = qode.math.linear_inner_product_space(CI_space_traits(configs_dimer))
    #H     = CI_space_dimer.lin_op(field_op_ham.Hamiltonian(h,V, n_elec=num_elec_dimer, n_threads=n_threads))    # slower than not using wisdom on many cores (probably bus traffic load)
    H     = CI_space_dimer.lin_op(field_op_ham.Hamiltonian(h,V, n_threads=n_threads))
    guess = CI_space_dimer.member(CI_space_dimer.aux.basis_vec([0, 1, num_spatial_atom+0, num_spatial_atom+1, 2*num_spatial_atom+0, 2*num_spatial_atom+1, 3*num_spatial_atom+0, 3*num_spatial_atom+1]))

    print((guess|H|guess) + N)
    (Eval,Evec), = qode.math.lanczos.lowest_eigen(H, [guess], thresh=1e-8)
    print("\nE_gs = {}\n".format(Eval+N))



    dim_1 = [len(frag1_to_dimer_n) for frag1_to_dimer_n in frag1_to_dimer]
    dim_0 = [len(frag0_to_dimer_n) for frag0_to_dimer_n in frag0_to_dimer]
    rho_1 = [(numpy.zeros((dim_1_n,dim_1_n)) if dim_1_n>0 else None) for dim_1_n in dim_1]
    rho_0 = [(numpy.zeros((dim_0_n,dim_0_n)) if dim_0_n>0 else None) for dim_0_n in dim_0]
    for n in range(num_elec_dimer+1):
        if rho_0[n] is not None:
            for R_list in frag1_to_dimer[num_elec_dimer-n]:
                for P in R_list:
                    (n_i1,i1),(n_i0,i0) = dimer_to_frags[P]
                    for Q in R_list:
                        (n_j1,j1),(n_j0,j0) = dimer_to_frags[Q]
                        rho_0[n][i0,j0] += Evec.v[P] * Evec.v[Q]    # should have n_i1==n_j1==num_elec_dimer-n and  i1==j1  and n_i0==n_j0==n
        if rho_1[n] is not None:
            for R_list in frag0_to_dimer[num_elec_dimer-n]:
                for P in R_list:
                    (n_i1,i1),(n_i0,i0) = dimer_to_frags[P]
                    for Q in R_list:
                        (n_j1,j1),(n_j0,j0) = dimer_to_frags[Q]
                        rho_1[n][i1,j1] += Evec.v[P] * Evec.v[Q]    # should have n_i0==n_j0==num_elec_dimer-n and  i0==j0  and n_i1==n_j1==n

    evals_evecs = {}
    all_evals = []
    for n in range(num_elec_dimer+1):
        if rho_0[n] is not None:
            rho = (rho_1[n] + rho_0[n]) / 2    # relies on fragments being the same
            evals, evecs = qode.util.sort_eigen(numpy.linalg.eigh(rho), order="descending")
            evals_evecs[n] = (evals, evecs)
            all_evals += list(evals)

    if statesthresh.nstates is None:
        thresh = statesthresh.thresh
        if thresh is None:
            raise RuntimeError("must define at least one criterion for keeping monomer states")
    elif statesthresh.thresh is None:
        all_evals = list(reversed(sorted(all_evals)))
        nstates = statesthresh.nstates
        thresh = (all_evals[nstates-1] + all_evals[nstates]) / 2
    else:
        raise RuntimeError("cannot define both thresh and nstates as criteria for keeping monomer states")

    states = {}
    for n, (evals, evecs) in evals_evecs.items():
        chg = frags[0].n_elec_ref - n
        n_config_n = len(evals)
        print("n_config_n", n_config_n)
        for i,e in enumerate(evals):
            if e>thresh:
                if chg not in states:
                    states[chg] = struct(
                        configs = sorted_configs_0[n],  # relies on fragments being the same
                        coeffs  = []
                    )
                tmp = numpy.zeros(n_config_n, dtype=Double.numpy, order="C")
                tmp[:] = evecs[:,i]
                states[chg].coeffs += [tmp]

    for chg,states_chg in states.items():
        num_states = len(states_chg.coeffs)
        if num_states>0:
            print("{}: {} x {}".format(chg, num_states, states_chg.coeffs[0].shape))
            #for config in states_chg.configs:
            #    print("  {:018b}".format(config))

    frags[0].states = states

    ref_chg, ref_idx = 0, 0
    frags[0].state_indices = [(ref_chg,ref_idx)]                # List of all charge and state indices, reference state needs to be first, but otherwise irrelevant order
    for i in range(len(frags[0].states[ref_chg].coeffs)):
        if   i!=ref_idx:  frags[0].state_indices += [(ref_chg,i)]
    for chg in frags[0].states:
        if chg!=ref_chg:  frags[0].state_indices += [(chg,i) for i in range(len(frags[0].states[chg].coeffs))]

    return "nth"
