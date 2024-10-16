#    (C) Copyright 2024 Marco Bauer
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

from   get_ints import get_ints
from   get_xr_result import get_xr_states, get_xr_H
from qode.math.tensornet import raw, tl_tensor
#import qode.util
from qode.util import timer, sort_eigen
from state_gradients import state_gradients, get_slices
from state_screening import state_screening, orthogonalize

#import torch
import numpy as np
import tensorly as tl
import pickle
#import scipy as sp

#torch.set_num_threads(4)
#tl.set_backend("pytorch")
#tl.set_backend("numpy")

from   build_fci_states import get_fci_states
import densities


def optimize_states(displacement, max_iter, xr_order, conv_thresh=1e-6, dens_filter_thresh=1e-9):
    ######################################################
    # Initialize integrals and densities
    ######################################################

    n_frag       = 2
    displacement = displacement
    project_core = True
    monomer_charges = [[0, +1, -1], [0, +1, -1]]
    density_options = ["compress=SVD,cc-aa"]
    frozen_orbs = [0, 9]
    n_orbs = 9
    n_occ = [2, 2]  # currently only alpha beta separation, but generalize to frag level not done yet!!!

    # "Assemble" the supersystem for the displaced fragments and get integrals
    BeN = []
    dens = []
    dens_builder_stuff = []
    state_coeffs_og = []
    for m in range(int(n_frag)):
        state_obj, dens_var_1, dens_var_2, n_threads, Be = get_fci_states(displacement, n_state_list=[(1, 4), (0, 9), (-1, 6)])
        for elem,coords in Be.atoms:  coords[2] += m * displacement    # displace along z
        BeN.append(Be)
        #dens.append(densities.build_tensors(state_obj, dens_var_1, dens_var_2, options=density_options, n_threads=n_threads))
        dens_builder_stuff.append([state_obj, dens_var_1, dens_var_2, density_options])
        #state_coeffs_og.append({chg: state_obj[chg].coeffs for chg in state_obj})
    #print(type(raw(dens[0]["a"][(+1,0)])), raw(dens[0]["a"][(+1,0)]).shape)

    int_timer = timer()
    ints = get_ints(BeN, project_core, int_timer)

    state_screening(dens_builder_stuff, ints, monomer_charges, n_orbs, frozen_orbs, n_occ, n_threads=n_threads)  # updates coeffs in dens_builder_stuff in place

    for m in range(int(n_frag)):
        state_coeffs_og.append({chg: dens_builder_stuff[m][0][chg].coeffs for chg in state_obj})
        dens.append(densities.build_tensors(*dens_builder_stuff[m][:-1], options=density_options, n_threads=n_threads))


    #for i in range(raw(dens[0]["ca"][(0,0)]).shape[0]):
    #    print(np.diag(raw(dens[0]["ca"][(0,0)][i, i, :, :])))
    #raise ValueError("stop here")

    state_coeffs_optimized = state_coeffs_og

    
    def iteration(frag_ind, dens_builder_stuff, dens, state_coeffs, dens_eigval_thresh=dens_filter_thresh):
        # In the following the variables are named as if frag_ind = 0, but it also works with frag_ind = 1
        print(f"opt frag {frag_ind}")
        """
        gs_energy_b, gradient_states_b = state_gradients(1, ints, dens_builder_stuff, dens, monomer_charges, n_threads=n_threads, xr_order=xr_order)
        #gs_energy_a, gradient_states_a = state_gradients(0, ints, dens_builder_stuff, dens, monomer_charges, n_threads=n_threads, xr_order=xr_order)

        for chg in monomer_charges[0]:
        #    dens_builder_stuff[0][0][chg].coeffs = [i.copy() for i in state_coeffs[0][chg]]
            dens_builder_stuff[1][0][chg].coeffs = [i.copy() for i in state_coeffs[1][chg]]
            dens_builder_stuff[0][0][chg].coeffs = [i.copy() for i in state_coeffs[0][chg]]
        #dens[0] = densities.build_tensors(*dens_builder_stuff[0], n_threads=n_threads)
        dens[1] = densities.build_tensors(*dens_builder_stuff[1][:-1], options=dens_builder_stuff[1][-1], n_threads=n_threads)  # do this because dens[0] was changed in state_gradients function
        dens[0] = densities.build_tensors(*dens_builder_stuff[0][:-1], options=dens_builder_stuff[0][-1], n_threads=n_threads)  # do this because dens[1] was also changed in state_gradients function
        """
        #gs_energy_b, gradient_states_b = state_gradients(1, ints, dens_builder_stuff, dens, monomer_charges, n_threads=n_threads, xr_order=xr_order)
        gs_energy_a, gradient_states_a = state_gradients(frag_ind, ints, dens_builder_stuff, dens, monomer_charges, n_threads=n_threads, xr_order=xr_order)
        #coeffs_grads = [state_coeffs, gradient_states]
        #pickle.dump(coeffs_grads, open("coeffs_grads.pkl", mode="wb"))
        
        ################################
        # Application of the derivatives
        ################################

        a_coeffs = {chg: np.array(tens.copy()) for chg, tens in state_coeffs[frag_ind].items()}
        #b_coeffs = {chg: np.array(tens.copy()) for chg, tens in state_coeffs[1].items()}
        grad_coeffs_a = {chg: np.array([vec / np.linalg.norm(vec) for vec in val]) for chg, val in gradient_states_a.items()}
        #grad_coeffs_b = {chg: np.array([vec / np.linalg.norm(vec) for vec in val]) for chg, val in gradient_states_b.items()}


        # For Hamiltonian evaluation in extended state basis, i.e. states + gradients, an orthonormal set for each fragment
        # is required, but this choice is not unique. One could e.g. normalize, orthogonalize and then again normalize,
        # or orthogonalize and normalize without previous normalization.
        tot_a_coeffs = {chg: orthogonalize(np.vstack((a_coeffs[chg], grad_coeffs_a[chg]))) for chg in monomer_charges[frag_ind]}
        #tot_b_coeffs = {chg: orthogonalize(np.vstack((b_coeffs[chg], grad_coeffs_b[chg]))) for chg in monomer_charges[1]}

        n_states = [sum(len(state_coeffs[i][chg]) for chg in monomer_charges[i]) for i in range(2)]
        n_confs = [sum(len(state_coeffs[i][chg][0]) for chg in monomer_charges[i]) for i in range(2)] #sum(conf_dict[0].values())

        state_dict = [{chg: len(state_coeffs[i][chg]) for chg in monomer_charges[i]} for i in range(2)]
        conf_dict = [{chg: len(state_coeffs[i][chg][0]) for chg in monomer_charges[i]} for i in range(2)]

        d_slices = [get_slices(state_dict[i], monomer_charges[i]) for i in range(2)]
        c_slices = [get_slices(conf_dict[i], monomer_charges[i]) for i in range(2)]

        #d_slices_double = [get_slices(state_dict[i], monomer_charges[i], type="double") for i in range(2)]
        #c_slices_double = [get_slices(conf_dict[i], monomer_charges[i], type="double") for i in range(2)]

        d_slices_first = [get_slices(state_dict[i], monomer_charges[i], type="first") for i in range(2)]
        #c_slices_first = [get_slices(conf_dict[i], monomer_charges[i], type="first") for i in range(2)]

        d_slices_latter = [get_slices(state_dict[i], monomer_charges[i], type="latter") for i in range(2)]
        #c_slices_latter = [get_slices(conf_dict[i], monomer_charges[i], type="latter") for i in range(2)]

        mat_sl_first, mat_sl_latter = {}, {}
        mat_sl_first[frag_ind] = d_slices_first[frag_ind]
        mat_sl_latter[frag_ind] = d_slices_latter[frag_ind]
        mat_sl_first[1 - frag_ind] = d_slices[1 - frag_ind]
        mat_sl_latter[1 - frag_ind] = d_slices[1 - frag_ind]
            

        #for chg in monomer_charges[0]:
        #    dens_builder_stuff[0][0][chg].coeffs = [i.copy() for i in state_coeffs[0][chg]]
        #    dens_builder_stuff[1][0][chg].coeffs = [i.copy() for i in state_coeffs[1][chg]]
        #dens[0] = densities.build_tensors(*dens_builder_stuff[0], n_threads=n_threads)
        #dens[1] = densities.build_tensors(*dens_builder_stuff[1][:-1], options=dens_builder_stuff[1][-1], n_threads=n_threads)  # do this because dens[1] was also changed in state_gradients function
        #gs_en, gs_vec = get_xr_states(ints, dens, 0)
        #print("old energy", gs_energy)
        #print("is this still the old gs_energy?", gs_en)

        # state gradient provider changes both densities, so the (new) "normal" densities need to be recovered/generated here
        for chg in monomer_charges[frag_ind]:
            dens_builder_stuff[frag_ind][0][chg].coeffs = [i.copy() for i in tot_a_coeffs[chg]]
        dens[frag_ind] = densities.build_tensors(*dens_builder_stuff[frag_ind][:-1], options=dens_builder_stuff[frag_ind][-1], n_threads=n_threads)

        for chg in monomer_charges[1 - frag_ind]:
            dens_builder_stuff[1 - frag_ind][0][chg].coeffs = [i.copy() for i in state_coeffs[1 - frag_ind][chg]]
            #dens_builder_stuff[1][0][chg].coeffs = [i.copy() for i in tot_b_coeffs[chg]]
        dens[1 - frag_ind] = densities.build_tensors(*dens_builder_stuff[1 - frag_ind][:-1], options=dens_builder_stuff[1 - frag_ind][-1], n_threads=n_threads)

        H1, H2 = get_xr_H(ints, dens, xr_order, monomer_charges)
        
        """
        # the following code snippet builds the Hamiltonian in the space of the states and derivatives of both fragments
        # sadly this seems to lead to a complex ground state vector
        def add_to_full(mat, b):  # update in place ... b is (0, 0) or (0, 1), etc.
            if sum(b) == 1:
                diag = False
            else:
                diag = True
            d_sl = {0: d_slices_first, 1: d_slices_latter}
            for chg0 in monomer_charges[0]:
                for chg1 in monomer_charges[1]:
                    #(0,0)
                    mat[d_slices_first[0][chg0], d_sl[b[0]][1][chg1], d_slices_first[0][chg0], d_sl[b[1]][1][chg1]] +=\
                        np.einsum("ij,kl->ikjl", np.eye(state_dict[0][chg0]), H1[1][d_sl[b[0]][1][chg1], d_sl[b[1]][1][chg1]])
                    if diag:
                        mat[d_slices_first[0][chg0], d_sl[b[0]][1][chg1], d_slices_first[0][chg0], d_sl[b[1]][1][chg1]] +=\
                            np.einsum("ij,kl->ikjl", H1[0][d_slices_first[0][chg0], d_slices_first[0][chg0]], np.eye(state_dict[1][chg1]))
                    #(0,1)
                    if diag:
                        mat[d_slices_first[0][chg0], d_sl[b[0]][1][chg1], d_slices_latter[0][chg0], d_sl[b[1]][1][chg1]] +=\
                            np.einsum("ij,kl->ikjl", H1[0][d_slices_first[0][chg0], d_slices_latter[0][chg0]], np.eye(state_dict[1][chg1]))
                    #(1,0)
                    if diag:
                        mat[d_slices_latter[0][chg0], d_sl[b[0]][1][chg1], d_slices_first[0][chg0], d_sl[b[1]][1][chg1]] +=\
                            np.einsum("ij,kl->ikjl", H1[0][d_slices_latter[0][chg0], d_slices_first[0][chg0]], np.eye(state_dict[1][chg1]))
                    #(1,1)
                    mat[d_slices_latter[0][chg0], d_sl[b[0]][1][chg1], d_slices_latter[0][chg0], d_sl[b[1]][1][chg1]] +=\
                        np.einsum("ij,kl->ikjl", np.eye(state_dict[0][chg0]), H1[1][d_sl[b[0]][1][chg1], d_sl[b[1]][1][chg1]])
                    if diag:
                        mat[d_slices_latter[0][chg0], d_sl[b[0]][1][chg1], d_slices_latter[0][chg0], d_sl[b[1]][1][chg1]] +=\
                            np.einsum("ij,kl->ikjl", H1[0][d_slices_latter[0][chg0], d_slices_latter[0][chg0]], np.eye(state_dict[1][chg1]))

        full = H2.reshape(2 * n, 2 * n, 2 * n, 2 * n)
        add_to_full(full, (0, 0))
        add_to_full(full, (0, 1))
        add_to_full(full, (1, 0))
        add_to_full(full, (1, 1))

        full = full.reshape(4 * n**2, 4 * n**2)
        """
        
        full = H2.reshape((2 - frag_ind) * n_states[0], (1 + frag_ind) * n_states[1],
                          (2 - frag_ind) * n_states[0], (1 + frag_ind) * n_states[1])
        
        for chg0 in monomer_charges[0]:
            for chg1 in monomer_charges[1]:
                #(0,0)
                full[mat_sl_first[0][chg0], mat_sl_first[1][chg1], mat_sl_first[0][chg0], mat_sl_first[1][chg1]] +=\
                    np.einsum("ij,kl->ikjl", H1[0][mat_sl_first[0][chg0], mat_sl_first[0][chg0]], np.eye(state_dict[1][chg1])) +\
                    np.einsum("ij,kl->ikjl", np.eye(state_dict[0][chg0]), H1[1][mat_sl_first[1][chg1], mat_sl_first[1][chg1]])
                #(0,1)
                if frag_ind == 0:
                    term = np.einsum("ij,kl->ikjl", H1[0][mat_sl_first[0][chg0], mat_sl_latter[0][chg0]], np.eye(state_dict[1][chg1]))
                else:
                    term = np.einsum("ij,kl->ikjl", np.eye(state_dict[0][chg0]), H1[1][mat_sl_first[1][chg1], mat_sl_latter[1][chg1]])
                full[mat_sl_first[0][chg0], mat_sl_first[1][chg1], mat_sl_latter[0][chg0], mat_sl_latter[1][chg1]] += term
                #(1,0)
                if frag_ind == 0:
                    term = np.einsum("ij,kl->ikjl", H1[0][mat_sl_latter[0][chg0], mat_sl_first[0][chg0]], np.eye(state_dict[1][chg1]))
                else:
                    term = np.einsum("ij,kl->ikjl", np.eye(state_dict[0][chg0]), H1[1][mat_sl_latter[1][chg1], mat_sl_first[1][chg1]])
                full[mat_sl_latter[0][chg0], mat_sl_latter[1][chg1], mat_sl_first[0][chg0], mat_sl_first[1][chg1]] += term
                #(1,1)
                full[mat_sl_latter[0][chg0], mat_sl_latter[1][chg1], mat_sl_latter[0][chg0], mat_sl_latter[1][chg1]] +=\
                    np.einsum("ij,kl->ikjl", H1[0][mat_sl_latter[0][chg0], mat_sl_latter[0][chg0]], np.eye(state_dict[1][chg1])) +\
                    np.einsum("ij,kl->ikjl", np.eye(state_dict[0][chg0]), H1[1][mat_sl_latter[1][chg1], mat_sl_latter[1][chg1]])

        full = full.reshape(2 * n_states[0] * n_states[1],
                            2 * n_states[0] * n_states[1])

        #np.save("H_with_both_grads.npy", full)
        #full += np.identity(len(full)) * 1e-16
        full_eigvals, full_eigvec = sort_eigen(np.linalg.eig(full))
        #if full_eigvals[0] != np.min(full_eigvals):
        #    print("for eig lowest eigval is not initial eigval, even after sorting")
        print(np.min(full_eigvals))
        #print(full_eigvec[0])
        print("relative imag contribution of lowest eigvec of full H with grads with eig", np.linalg.norm(np.imag(full_eigvec[:, 0])) / np.linalg.norm(full_eigvec[:, 0]))
        #print("imaginary part of full matrix", np.linalg.norm(np.imag(full)))  <-- zero
        #print("norm(full - full.T) / norm(full)", np.linalg.norm(full - full.T) / np.linalg.norm(full))

        #svdl, svd_eig, svdr = np.linalg.svd(full)
        #if svd_eig[0] != np.min(svd_eig):
        #    print("for svd lowest eigval is not initial eigval")
        #print(np.min(svd_eig))
        #print("this is how far off svdl @ svdr - id is", np.linalg.norm(svdl @ svdr - np.identity(len(svdl))))
        #print("relative imag contribution of lowest eigvec of full H with grads with svd", np.linalg.norm(np.imag(svdl[0])) / np.linalg.norm(svdl[0]))
        #print("imag contributions of svdl and svdr", np.linalg.norm(np.imag(svdl)), np.linalg.norm(np.imag(svdr)))

        # now determine which elements of the eigvec to keep
        #full_gs_vec = full_eigvec[0].reshape(2 * n, 2 * n)
        full_gs_vec = np.real(full_eigvec[:, 0].reshape((2 - frag_ind) * n_states[0], (1 + frag_ind) * n_states[1]))
        #for i, row in enumerate(full_gs_vec):
        #    for j, val in enumerate(row):
        #        if np.imag(val) >= 1e-14:
        #            print((i,j), val)
        if frag_ind == 0:
            # conjugation is only necessary if full_gs_vec is complex, which it is hopefully not,
            # because the imaginary part will explode in the next eigendecomp, even though it is small here
            dens_mat = np.einsum("ij,kj->ik", full_gs_vec, full_gs_vec)#np.conj(full_gs_vec))  # contract over frag_b part
        else:
            dens_mat = np.einsum("ij,ik->jk", full_gs_vec, full_gs_vec)#np.conj(full_gs_vec))  # contract over frag_a part
        
        #dens_mat += np.identity(len(dens_mat)) * 1e-16
        dens_eigvals, dens_eigvecs = np.linalg.eigh(dens_mat)
        #dens_eigvals_b, dens_eigvecs_b = np.linalg.eigh(dens_mat_b)
        print(dens_eigvals)
        #print(dens_eigvals_b)
        # choosing more/less states here at higher iterations, depending on the diagonal values might be a good way to slightly enlarge/compress the state space automatically
        print("imaginary contribution of dens_eigvecs", np.linalg.norm(np.imag(dens_eigvecs)))
        print("relative imaginary contribution of dens_eigvecs", np.linalg.norm(np.imag(dens_eigvecs)) / np.linalg.norm(dens_eigvecs))
        new_large_vecs = np.real(dens_eigvecs)
        #new_large_vecs_b = np.real(dens_eigvecs_b)#[n:])
        
        #new_large_vecs = dens_mat_a#np.real(dens_eigvecs[n:])
        #print("dropped imag part of eigvecs for new coeffs is", np.linalg.norm(np.imag(dens_eigvecs[n:])))
        
        # the following threshold is very delicate, because if its
        # too large -> truncation errors
        # too small -> numerical inconsistencies through terms to small to resolve even
        # with double precision (at least I think so...where else should the numerical instability come from?)
        # for the first iteration of the Be2 6-31g example something around 1e-9 seems to be the sweet spot
        keepers = []
        for i, vec in enumerate(new_large_vecs.T):
            if dens_eigvals[i] >= dens_eigval_thresh:
                keepers.append(vec)
        print(f"{len(keepers)} states are kept for frag {frag_ind}")
        keepers = np.array(keepers)

        #keepers_a = [i for i in new_large_vecs_a.T]  # if you want to keep all states

        #keepers_b = []
        #for i, vec in enumerate(new_large_vecs_b.T):
        #    if dens_eigvals_a[i] >= thresh:
        #        keepers_b.append(vec)
        #print(f"{len(keepers_b)} states are kept for frag 1")
        
        #large_vec_map = {}
        #for frag in range(2):
        
        large_vec_map = np.zeros((2 * n_states[frag_ind], n_confs[frag_ind]))
        for chg in monomer_charges[frag_ind]:
            large_vec_map[d_slices_first[frag_ind][chg], c_slices[frag_ind][chg]] = state_coeffs[frag_ind][chg]
            large_vec_map[d_slices_latter[frag_ind][chg], c_slices[frag_ind][chg]] = state_coeffs[frag_ind][chg]

        new_vecs = np.einsum("ij,jp->ip", keepers, large_vec_map)
        print(new_vecs.shape)
        #new_vecs_b = np.einsum("ij,jp->ip", np.array(keepers_b), large_vec_map[0])
        #print(new_vecs_b.shape)
        
        chg_sorted_keepers = {chg: [] for chg in monomer_charges[frag_ind]}  #{0: [], 1: [], -1: []}
        #chg_sorted_keepers_b = {chg: [] for chg in monomer_charges[1]}
        #print("c_slices[0]", c_slices[0])
        for state in new_vecs:
            state /= np.linalg.norm(state)
            norms = {chg: np.linalg.norm(state[c_slices[frag_ind][chg]]) for chg in monomer_charges[frag_ind]}
            #print(norms)
            if max(norms.values()) < 0.99:
                ValueError(f"mixed state encountered (different charges are mixed for a state on frag {frag_ind}), see {norms}")
            chg_sorted_keepers[max(norms, key=norms.get)].append(state[c_slices[frag_ind][max(norms, key=norms.get)]])

        for chg, vecs in chg_sorted_keepers.items():
            print(f"for charge {chg} {len(vecs)} states are kept")
            chg_sorted_keepers[chg] = [i for i in orthogonalize(np.array(vecs))]


        #for state in new_vecs_b:
        #    state /= np.linalg.norm(state)
        #    norms = {chg: np.linalg.norm(state[c_slices[1][chg]]) for chg in monomer_charges[1]}
        #    #print(norms)
        #    if max(norms.values()) < 0.99:
        #        ValueError(f"mixed state encountered (different charges are mixed for a state on frag 0), see {norms}")
        #    chg_sorted_keepers_b[max(norms, key=norms.get)].append(state[c_slices[1][max(norms, key=norms.get)]])

        #for chg, vecs in chg_sorted_keepers_b.items():
        #    chg_sorted_keepers_b[chg] = [i for i in orthogonalize(np.array(vecs))]

        #new_vecs_a = np.einsum("ij,ip->jp", new_large_vecs_a, large_vec_map[0])
        #new_vecs_b = np.einsum("ij,jp->ip", new_large_vecs_b, large_vec_map[1])
        #print(new_vecs_a.shape)
        #print(np.dot(new_vecs_a[-1], new_vecs_a[-2]), np.dot(new_vecs_a[-1], new_vecs_a[-1]))
        #print(np.dot(chg_sorted_keepers[0][-1], chg_sorted_keepers[0][-2]), np.dot(chg_sorted_keepers[0][-1], chg_sorted_keepers[0][-1]))
        # orthogonalization only necessary if some imaginary part was dropped
        #new_vecs_a = orthogonalize(new_vecs_a)
        #new_vecs_b = orthogonalize(new_vecs_b)
        new_coeffs = chg_sorted_keepers
        """
        # the following is just a safety measure
        #for chg in monomer_charges[frag_ind]:
        #    if len(new_coeffs[chg]) == 0:
        #        print(f"{chg} part of new vectors 0 is empty and therefore filled up with previous coeffs")
        #        new_coeffs[chg] = state_coeffs[frag_ind][chg]
        #    print(f"for charge {chg} {len(new_coeffs[chg])} states are used on frag {frag_ind}")

        #new_coeffs_b = chg_sorted_keepers_b
        #for chg in monomer_charges[1]:
        #    if len(new_coeffs_b[chg]) == 0:
        #        print(f"{chg} part of new vectors 1 is empty and therefore filled up with previous coeffs")
        #        new_coeffs_b[chg] = state_coeffs[0][chg]
        #    print(f"for charge {chg} {len(new_coeffs_b[chg])} states are used on frag 1")

        #new_coeffs_a = {chg: new_vecs_a[d_slices[0][chg], c_slices[0][chg]] for chg in monomer_charges[0]}
        #new_coeffs_b = {chg: new_vecs_b[d_slices[1][chg], c_slices[1][chg]] for chg in monomer_charges[1]}
        #save_obj = [keepers, new_coeffs, state_coeffs, gradient_states_a]
        #pickle.dump(save_obj, open("step_size_stuff.pkl", mode="wb"))
        
        # get step sizes from some heuristic based on the procedure from above
        step_sizes_unsorted = np.zeros(n_states[frag_ind])  # use this if you want to keep the states
        #step_sizes_unsorted = [None for _ in range(n_states[frag_ind])]  # and this if you want to throw them out
        #og_contr = np.concatenate(tuple([keepers[:, d_slices_first[frag_ind][chg]] for chg in monomer_charges[frag_ind]]), axis=1)**2
        #grad_contr = np.concatenate(tuple([keepers[:, d_slices_latter[frag_ind][chg]] for chg in monomer_charges[frag_ind]]), axis=1)**2
        og_contr = np.concatenate(tuple([new_large_vecs.T[:, d_slices_first[frag_ind][chg]] for chg in monomer_charges[frag_ind]]), axis=1)**2
        grad_contr = np.concatenate(tuple([new_large_vecs.T[:, d_slices_latter[frag_ind][chg]] for chg in monomer_charges[frag_ind]]), axis=1)**2
        for i, vec in enumerate(og_contr):
            print("max**2 in og", np.argmax(vec), np.max(vec))
            if np.max(vec) < 0.5:
                print(f"this state does not contain a dominant contribution (max**2 is {np.max(vec)})")
                print("for now this state is therefore thrown away")
                continue
            print("max**2 in grad", grad_contr[i][np.argmax(vec)], np.max(grad_contr[i]))
            print("corresponding norm of grad", np.linalg.norm(grad_contr[i]))
            print("coeff =", np.sqrt(grad_contr[i][np.argmax(vec)] / np.max(vec)), "\n")
            step_sizes_unsorted[np.argmax(vec)] = np.sqrt(grad_contr[i][np.argmax(vec)] / np.max(vec))
            #print("max coeff would be", np.sqrt(np.linalg.norm(grad_contr[i]) / np.max(vec)), "\n")
        step_sizes = {chg: step_sizes_unsorted[d_slices[frag_ind][chg]] for chg in monomer_charges[frag_ind]}

        new_coeffs = {}
        # Apply all the gradients to the subset of states and diagonalize with obtained step size
        # frag A
        for chg in monomer_charges[frag_ind]:
            #tmp = np.array(state_coeffs[frag_ind][chg])
            #tmp -= 0.2 * gradient_states_a[chg]
            tmp = []
            for i, step_size in enumerate(step_sizes[chg]):
                if step_size == None:
                    continue
                tmp.append(state_coeffs[frag_ind][chg][i] - step_size * grad_coeffs_a[chg][i])
            new_coeffs[chg] = [i for i in orthogonalize(np.array(tmp))]
            #dens_builder_stuff[0][0][chg].coeffs = [i for i in orthogonalize(tmp)]
            #dens_builder_stuff[1][0][chg].coeffs = state_coeffs[1][chg].copy()
        """
        """
        # frag B
        #for chg in monomer_charges[1]:
        #    tmp = np.array(state_coeffs[1][chg])
        #    tmp[-1] -= 0.2 * gradient_states[chg][-1]
        #    dens_builder_stuff[1][0][chg].coeffs = [i for i in orthogonalize(tmp)]
        #    dens_builder_stuff[0][0][chg].coeffs = state_coeffs[0][chg]
        """
        for chg in monomer_charges[frag_ind]:
            state_coeffs[frag_ind][chg] = new_coeffs[chg]
        return state_coeffs, dens_builder_stuff, dens, gs_energy_a, full_eigvals[0]
    
    def reduce_screened_state_space(dens_builder_stuff, dens, state_coeffs, dens_eigval_thresh=dens_filter_thresh):
        n_states = [sum(len(state_coeffs[i][chg]) for chg in monomer_charges[i]) for i in range(2)]
        n_confs = [sum(len(state_coeffs[i][chg][0]) for chg in monomer_charges[i]) for i in range(2)]

        state_dict = [{chg: len(state_coeffs[i][chg]) for chg in monomer_charges[i]} for i in range(2)]
        conf_dict = [{chg: len(state_coeffs[i][chg][0]) for chg in monomer_charges[i]} for i in range(2)]

        d_slices = [get_slices(state_dict[i], monomer_charges[i]) for i in range(2)]
        c_slices = [get_slices(conf_dict[i], monomer_charges[i]) for i in range(2)]

        H1, H2 = get_xr_H(ints, dens, xr_order, monomer_charges)
        
        full = H2.reshape(n_states[0], n_states[1],
                          n_states[0], n_states[1])
        
        for chg0 in monomer_charges[0]:
            for chg1 in monomer_charges[1]:
                full[d_slices[0][chg0], d_slices[1][chg1], d_slices[0][chg0], d_slices[1][chg1]] +=\
                    np.einsum("ij,kl->ikjl", H1[0][d_slices[0][chg0], d_slices[0][chg0]], np.eye(state_dict[1][chg1])) +\
                    np.einsum("ij,kl->ikjl", np.eye(state_dict[0][chg0]), H1[1][d_slices[1][chg1], d_slices[1][chg1]])

        full = full.reshape(n_states[0] * n_states[1],
                            n_states[0] * n_states[1])

        full_eigvals, full_eigvec = sort_eigen(np.linalg.eig(full))
        print(np.min(full_eigvals))
        if np.linalg.norm(np.imag(full_eigvec[:, 0])) / np.linalg.norm(full_eigvec[:, 0]) > 1e-10:
            raise ValueError("imaginary part of eigenvector is not negligible")

        full_gs_vec = np.real(full_eigvec[:, 0].reshape((n_states[0], n_states[1])))

        dens_mat_a = np.einsum("ij,kj->ik", full_gs_vec, full_gs_vec)#np.conj(full_gs_vec))  # contract over frag_b part
        dens_mat_b = np.einsum("ij,ik->jk", full_gs_vec, full_gs_vec)#np.conj(full_gs_vec))  # contract over frag_a part
        
        dens_eigvals_a, dens_eigvecs_a = np.linalg.eigh(dens_mat_a)
        dens_eigvals_b, dens_eigvecs_b = np.linalg.eigh(dens_mat_b)
        print(dens_eigvals_a)
        print(dens_eigvals_b)
        dens_eigvals = [dens_eigvals_a, dens_eigvals_b]
        new_large_vecs = [np.real(dens_eigvecs_a), np.real(dens_eigvecs_b)]
        
        # the following threshold is very delicate, because if its
        # too large -> truncation errors
        # too small -> numerical inconsistencies through terms to small to resolve even
        # with double precision (at least I think so...where else should the numerical instability come from?)
        # for the first iteration of the Be2 6-31g example something around 1e-9 seems to be the sweet spot
        keepers = [[], []]
        for frag in range(2):
            for i, vec in enumerate(new_large_vecs[frag].T):
                if dens_eigvals[frag][i] >= dens_eigval_thresh:
                    keepers[frag].append(vec)
            print(f"{len(keepers[frag])} states are kept for frag {frag}")

        large_vec_map = []
        for frag in range(2):
            large_vec_map_ = np.zeros((n_states[frag], n_confs[frag]))
            for chg in monomer_charges[frag]:
                large_vec_map_[d_slices[frag][chg], c_slices[frag][chg]] = state_coeffs[frag][chg]
            large_vec_map.append(large_vec_map_)

        new_vecs = [np.einsum("ij,jp->ip", np.array(keepers[frag]), large_vec_map[frag]) for frag in range(2)]
        
        chg_sorted_keepers = [{chg: [] for chg in monomer_charges[frag]} for frag in range(2)]
        for frag in range(2):
            for state in new_vecs[frag]:
                state /= np.linalg.norm(state)
                norms = {chg: np.linalg.norm(state[c_slices[frag][chg]]) for chg in monomer_charges[frag]}
                if max(norms.values()) < 0.99:
                    ValueError(f"mixed state encountered (different charges are mixed for a state on frag {frag}), see {norms}")
                chg_sorted_keepers[frag][max(norms, key=norms.get)].append(state[c_slices[frag][max(norms, key=norms.get)]])

        for frag in range(2):
            print(f"fragment {frag}")
            for chg, vecs in chg_sorted_keepers[frag].items():
                print(f"for charge {chg} {len(vecs)} states are kept")
                chg_sorted_keepers[frag][chg] = [i for i in orthogonalize(np.array(vecs))]

        for frag in range(2):
            for chg in monomer_charges[frag]:
                state_coeffs[frag][chg] = chg_sorted_keepers[frag][chg]

        #raise ValueError("stop here")

        for frag in range(2):
            for chg in monomer_charges[frag]:
                dens_builder_stuff[frag][0][chg].coeffs = [i for i in state_coeffs[frag][chg]]
            dens[frag] = densities.build_tensors(*dens_builder_stuff[frag][:-1], options=dens_builder_stuff[frag][-1], n_threads=n_threads)
        return state_coeffs, dens_builder_stuff, dens, full_eigvals[0]

    def postprocessing(en, en_extended, en_history, en_with_grads_history, converged):
        en_history.append(en)
        en_with_grads_history.append(en_extended)
        #printer(en_history, en_with_grads_history)
        print(f"History of XR[{xr_order}] energies:", en_history)
        print(f"History of XR[{xr_order}] energies with gradients in the Hamiltonian build"
              "(order is grads on frag 0 then on 1 then on 0 again and so on):", en_with_grads_history)

        if abs(en_history[-1] - en_history[-2]) <= conv_thresh and en_history[-1] - en_with_grads_history[-1] <= conv_thresh:
            if en_history[-1] - en_with_grads_history[-1] < 0.:
                RuntimeWarning("gs energy of larger Hamiltonian (with gradients included) is not lower than the gs energy of the smaller Hamiltonian in this iteration")
            else:
                print("Converged!!!")
                converged = True
        return converged
    

    ######################################################
    # iterative procedure
    ######################################################
    iter = 0
    converged = False

    state_coeffs_optimized, dens_builder_stuff, dens, gs_energy = reduce_screened_state_space(dens_builder_stuff, dens, state_coeffs_optimized)

    en_history, en_with_grads_history = [gs_energy], [0]

    while iter < max_iter:
        iter += 1

        state_coeffs_optimized, dens_builder_stuff, dens, gs_energy_a, gs_en_a_with_grads = iteration(0, dens_builder_stuff, dens, state_coeffs_optimized)
        converged = postprocessing(gs_energy_a, gs_en_a_with_grads, en_history, en_with_grads_history, converged)

        if converged:
            break

        for frag in range(2):  # this should only be necessary for frag 0
            for chg in monomer_charges[frag]:
                dens_builder_stuff[frag][0][chg].coeffs = [i for i in state_coeffs_optimized[frag][chg]]
        dens[0] = densities.build_tensors(*dens_builder_stuff[0][:-1], options=dens_builder_stuff[0][-1], n_threads=n_threads)
        
        #if iter == max_iter:
        #    print("gs energy old", gs_energy_a)
        #    print("gs energy for H with states and gradients", np.real(np.min(full_eigvals)))
        #    print("gs energy new", new_gs_en)
        #    break
        #else:
        #    print(f"from {previous_energy} to {gs_energy_a}")

        state_coeffs_optimized, dens_builder_stuff, dens, gs_energy_b, gs_en_b_with_grads = iteration(1, dens_builder_stuff, dens, state_coeffs_optimized)
        converged = postprocessing(gs_energy_b, gs_en_b_with_grads, en_history, en_with_grads_history, converged)

        if converged:
            break

        for frag in range(2):  # this should only be necessary for frag 1
            for chg in monomer_charges[frag]:
                dens_builder_stuff[frag][0][chg].coeffs = [i for i in state_coeffs_optimized[frag][chg]]
        dens[1] = densities.build_tensors(*dens_builder_stuff[1][:-1], options=dens_builder_stuff[1][-1], n_threads=n_threads)





optimize_states(4.5, 1, 0)













