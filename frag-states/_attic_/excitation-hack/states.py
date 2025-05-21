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
from qode.util import struct, indented, no_print
from qode.util.PyC import Double
from qode.many_body.self_consistent_field.fermionic import RHF_RoothanHall_Nonorthogonal, RHF_RoothanHall_Orthonormal
from qode.atoms.integrals.fragments import unblock_2, unblock_last2, unblock_4
from qode.fermion_field import CI_space_traits, field_op_ham, configurations
from get_ints import get_ints
from trim_states import trim_states


def _fragment_FCI(frag, printout, n_threads):

    n_spatial = frag.basis.n_spatial_orb
    core_orbs = frag.basis.core
    n_elec_dn, n_elec_up, occupied_dn, occupied_up = frag.HartreeFock("n_elec_dn n_elec_up occupied_dn occupied_up")

    occupied = occupied_dn + [n_spatial+i for i in occupied_up]

    N, S, T, U, V = frag.integrals
    h = T + U

    dn_configs = configurations.all_configs(n_spatial, n_elec_dn-len(core_orbs), frozen_occ_orbs=core_orbs)
    up_configs = configurations.all_configs(n_spatial, n_elec_up-len(core_orbs), frozen_occ_orbs=core_orbs)
    configs    = configurations.tensor_product_configs([dn_configs,up_configs], [n_spatial,n_spatial])

    CI_space = qode.math.linear_inner_product_space(CI_space_traits(configs))
    H     = CI_space.lin_op(field_op_ham.Hamiltonian(h,V, n_elec=frag.n_elec_ref, n_threads=n_threads))
    guess = CI_space.member(CI_space.aux.basis_vec(occupied))

    printout("Energy of guess =", (guess|H|guess) + N)
    (Eval,Evec), = qode.math.lanczos.lowest_eigen(H, [guess], thresh=1e-8, printout=indented(printout))
    printout("Ground-state energy = ", Eval+N)

    #archive.ground = struct(energy= Eval+N, state=Evec)
    return

def _dimer_Sz_configs(frag0, frag1):
    # the ultimate ordering of orbitals is (from highest to lowest, as found in a bit string) is:
    # frag1-up frag1-dn frag0-up frag0-dn

    n_spatial_0 = frag0.basis.n_spatial_orb
    n_spatial_1 = frag1.basis.n_spatial_orb
    n_spatial   = n_spatial_0 + n_spatial_1
    core_orbs_0 = frag0.basis.core
    core_orbs_1 = frag1.basis.core
    core_orbs   = core_orbs_0 + [n_spatial_0+c for c in core_orbs_1]
    n_elec_dn_0, n_elec_up_0 = frag0.HartreeFock("n_elec_dn n_elec_up")
    n_elec_dn_1, n_elec_up_1 = frag1.HartreeFock("n_elec_dn n_elec_up")
    n_elec_dn   = n_elec_dn_0 + n_elec_dn_1 
    n_elec_up   = n_elec_up_0 + n_elec_up_1 
    n_elec      = n_elec_dn + n_elec_up

    dn_configs = configurations.all_configs(n_spatial, n_elec_dn-len(core_orbs), frozen_occ_orbs=core_orbs)
    up_configs = configurations.all_configs(n_spatial, n_elec_up-len(core_orbs), frozen_occ_orbs=core_orbs)

    dn_configs_decomp = configurations.decompose_configs(dn_configs, [n_spatial_0, n_spatial_1])
    up_configs_decomp = configurations.decompose_configs(up_configs, [n_spatial_0, n_spatial_1])
    combine_configs = configurations.config_combination([n_spatial_1, n_spatial_1])
    all_configs_1 = list()
    all_configs_0 = set()
    nested = []
    for dn_config_1,dn_configs_0 in dn_configs_decomp:
        for up_config_1,up_configs_0 in up_configs_decomp:
            config_1  = combine_configs([up_config_1, dn_config_1])
            configs_0 = configurations.tensor_product_configs([up_configs_0, dn_configs_0], [n_spatial_0, n_spatial_0])
            nested += [(config_1, configs_0)]
            all_configs_1 += [config_1]
            all_configs_0 |= set(configs_0)
    all_configs_0 = list(all_configs_0)
    configs = configurations.recompose_configs(nested, [2*n_spatial_0, 2*n_spatial_1])

    sorted_configs_1 = [[] for n in range(n_elec+1)]
    sorted_configs_0 = [[] for n in range(n_elec+1)]
    for config_1 in all_configs_1:
        n = config_1.bit_count()
        sorted_configs_1[n] += [config_1]
    for config_0 in all_configs_0:
        n = config_0.bit_count()
        sorted_configs_0[n] += [config_0]
    sorted_configs_1 = [sorted(sorted_configs_1_n) for sorted_configs_1_n in sorted_configs_1]
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

    occupied_dn_0, occupied_up_0 = frag0.HartreeFock("occupied_dn occupied_up")
    occupied_dn_1, occupied_up_1 = frag1.HartreeFock("occupied_dn occupied_up")
    occupied_0 = occupied_dn_0 + [n_spatial_0+i for i in occupied_up_0]
    occupied_1 = occupied_dn_1 + [n_spatial_1+i for i in occupied_up_1]
    occupied = occupied_0 + [2*n_spatial_0+i for i in occupied_1]

    core_orbs_0 = core_orbs_0 + [n_spatial_0+c for c in core_orbs_0]
    core_orbs_1 = core_orbs_1 + [n_spatial_1+c for c in core_orbs_1]
    core_orbs = core_orbs_0 + [2*n_spatial_0+c for c in core_orbs_1]

    return n_elec, 2*n_spatial, core_orbs, occupied, configs, sorted_configs_0, sorted_configs_1, frag0_to_dimer, frag1_to_dimer, dimer_to_frags


def get_optimal(frags, statesthresh, printout=print, n_threads=1):

    printout("Fragment FCI")
    symm_ints, bior_ints, nuc_rep = get_ints(frags, printout=indented(no_print))
    for m,frag in enumerate(frags):
        frag.integrals = nuc_rep[m,m], symm_ints.S[m,m], symm_ints.T[m,m], symm_ints.U[m,m,m], symm_ints.V[m,m,m,m]
    printout("MO integrals initialized")

    for frag in frags:
        _fragment_FCI(frag, indented(printout), n_threads)
    printout("FCI completed")

    # the only reason for encapsulating this one-off outside the present name space is to make really clear
    # which of the variables were just intermediates and which are carried forward
    n_elec, n_spin, core, occupied, configs, configs_0, configs_1, frag0_to_dimer, frag1_to_dimer, dimer_to_frags = _dimer_Sz_configs(*frags)

    N = nuc_rep[0,0] + nuc_rep[1,1] + nuc_rep[0,1]
    T = unblock_2(    bior_ints.T, frags, spin_orbs=True)
    U = unblock_last2(bior_ints.U, frags, spin_orbs=True)
    V = unblock_4(    bior_ints.V, frags, spin_orbs=True)
    h = T + U[0] + U[1]
    orbs = list(range(n_spin))
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

    occupied[-1] += 1
    print(occupied)
    CI_space_dimer = qode.math.linear_inner_product_space(CI_space_traits(configs))
    H     = CI_space_dimer.lin_op(field_op_ham.Hamiltonian(h,V, n_threads=n_threads))
    guess = CI_space_dimer.member(CI_space_dimer.aux.basis_vec(occupied))

    printout((guess|H|guess) + N)
    #(Eval,Evec), = qode.math.lanczos.lowest_eigen(H, [guess], thresh=1e-8)
    n_dim_states = 4
    results = qode.math.lanczos.lowest_eigen_one_by_one(H, [guess]*n_dim_states, thresh=1e-8)
    for _1,bra in results:
        for _2,ket in results:
            printout((bra|ket), end="  ")
        printout()
    #printout(results[0][1]|results[0][1])
    #printout(results[1][1]|results[1][1])
    #printout(results[0][1]|results[1][1])
    Eval,Evec = results[n_dim_states-1]
    printout("\nE_gs = {}\n".format(Eval+N))

    dim_1 = [len(frag1_to_dimer_n) for frag1_to_dimer_n in frag1_to_dimer]
    dim_0 = [len(frag0_to_dimer_n) for frag0_to_dimer_n in frag0_to_dimer]
    rho_1 = [(numpy.zeros((dim_1_n,dim_1_n)) if dim_1_n>0 else None) for dim_1_n in dim_1]
    rho_0 = [(numpy.zeros((dim_0_n,dim_0_n)) if dim_0_n>0 else None) for dim_0_n in dim_0]
    for n in range(n_elec+1):
        if rho_0[n] is not None:
            for R_list in frag1_to_dimer[n_elec-n]:
                for P in R_list:
                    (n_i1,i1),(n_i0,i0) = dimer_to_frags[P]
                    for Q in R_list:
                        (n_j1,j1),(n_j0,j0) = dimer_to_frags[Q]
                        rho_0[n][i0,j0] += Evec.v[P] * Evec.v[Q]    # should have n_i1==n_j1==n_elec-n and  i1==j1  and n_i0==n_j0==n
        if rho_1[n] is not None:
            for R_list in frag0_to_dimer[n_elec-n]:
                for P in R_list:
                    (n_i1,i1),(n_i0,i0) = dimer_to_frags[P]
                    for Q in R_list:
                        (n_j1,j1),(n_j0,j0) = dimer_to_frags[Q]
                        rho_1[n][i1,j1] += Evec.v[P] * Evec.v[Q]    # should have n_i0==n_j0==n_elec-n and  i0==j0  and n_i1==n_j1==n

    rho = {}
    for n in range(n_elec+1):
        if rho_0[n] is not None:
            rho[n] = (rho_1[n] + rho_0[n]) / 2    # relies on fragments being the same

    frags[0].states, frags[0].state_indices = trim_states(rho, statesthresh, frags[0].n_elec_ref, configs_0, printout=indented(printout))
    #frags[1].states, frags[1].state_indices = trim_states(rho, statesthresh, frags[1].n_elec_ref, configs_1, printout=indented(printout))

    return "nth"
