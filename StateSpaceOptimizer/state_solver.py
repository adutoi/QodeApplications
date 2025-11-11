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

#from   get_ints import get_ints
from   get_xr_result import get_xr_H#, get_xr_states
from qode.math.tensornet import backend_contract_path
#import qode.util
from qode.util import timer, sort_eigen
from state_gradients import state_gradients, get_slices#, get_adapted_overlaps
from state_screening import state_screening, orthogonalize, conf_decoder, is_singlet

#import torch
import numpy as np
import tensorly as tl
#import pickle
import scipy as sp

#torch.set_num_threads(4)
#tl.set_backend("pytorch")
#tl.set_backend("numpy")

#from   build_fci_states import get_fci_states
import densities

tl.plugins.use_opt_einsum()
backend_contract_path(True)

#class empty(object):  pass

def optimize_states(max_iter, xr_order, dens_builder_stuff, ints, n_occ, n_orbs, frozen_orbs, additional_states,
                    conv_thresh=1e-6, dens_filter_thresh=1e-7, begin_from_state_prep=False, grad_level="herm", target_state=0,
                    monomer_charges=[[0, +1, -1], [0, +1, -1]], density_options=["compress=SVD,cc-aa"], n_threads=1):
    """
    max_iter: defines the max_iter for the state solver using an actual gradient
    begin_from_state_prep: determines whether previously to the gradient based optimization a guess shall be provided
                           originating from the same state screening but then optimizing without the gradient (faster but less reliable)
    the current recommendation is to either only use the solver with a gradient with begin_from_state_prep=False or max_iter=0 with begin_from_state_prep=True.
    As already mentioned the first one is more accurate, but the latter one is quite a bit faster
    """
    ######################################################
    # Initialize integrals and density preliminaries
    ######################################################

    if max_iter > 0 and begin_from_state_prep:
        raise ValueError("it is recommended to either use the gradient based solver or the state preparer, not both")
    
    state_coeffs_og = [{chg: dens_builder_stuff[m][0][chg].coeffs for chg in monomer_charges[m]} for m in range(2)]
    
    def reduce_screened_state_space(dens_builder_stuff, dens, state_coeffs, dens_eigval_thresh=dens_filter_thresh, target_state=target_state):
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

        #full_eigvals_raw, full_eigvec_l_unsorted, full_eigvec_r_unsorted = sp.linalg.eig(full, left=True, right=True)
        #full_eigvals_check, full_eigvec_l = sort_eigen((full_eigvals_raw, full_eigvec_l_unsorted))
        # If only the right eigenvector is used later also only ask for it, as this will lead to a significant performace gain
        full_eigvals_raw, full_eigvec_r_unsorted = sp.linalg.eig(full)
        full_eigvals, full_eigvec_r = sort_eigen((full_eigvals_raw, full_eigvec_r_unsorted))
        #if any(full_eigvals - full_eigvals_check):  # this check should be unnecessary and might need to be refined for fully degenerate states
        #    raise ValueError("sorting went wrong with the sorting")
        print(full_eigvals[target_state])
        #if np.linalg.norm(np.imag(full_eigvec_l[:, target_state])) > 1e-10:
        #    raise ValueError("imaginary part of eigenvector is not negligible")
        if np.linalg.norm(np.imag(full_eigvec_r[:, target_state])) > 1e-10:
            raise ValueError("imaginary part of eigenvector is not negligible")

        if type(target_state) == int:
            target_state = [target_state]

        # it would be correct here to also use the left eigvec, but it was found that only the right eigvec
        # yields similar or even better results with much more compact state spaces.
        # TODO: an open question remains how the performance differs for stronger interacting fragments
        full_eigvecs = [full_eigvec_r]#, full_eigvec_l]
        target_vecs = [np.real(full_eigvecs[lr][:, i].reshape((n_states[0], n_states[1]))) for i in target_state for lr in range(len(full_eigvecs))]
        #def get_large_elems(mat, eps=1e-3):
        #    ret = {}
        #    for i, vec in enumerate(mat):
        #        for j, elem in enumerate(vec):
        #            if abs(elem) > eps:
        #                ret[(i, j)] = elem
        #    return ret
        #print(get_large_elems(target_vecs[0]))
        #print(get_large_elems(target_vecs[1]))

        dens_mats = [[np.einsum("ij,kj->ik", vec, vec) for vec in target_vecs],  # contract over frag_b part
                     [np.einsum("ij,ik->jk", vec, vec) for vec in target_vecs]]  # contract over frag_a part

        dens_eigpairs = [[sort_eigen(sp.linalg.eigh(mat), order="descending") for mat in dens_mats[frag]] for frag in range(2)]
        # the following is tricky...
        # building everything here and then orthogonalize will only take the first eigenvector into account,
        # since they build the full space and therefore one needs to either sort it differently or truncate before orthogonalizing
        new_large_vecs_full = [np.real(np.concatenate(tuple([pair[1] for pair in dens_eigpairs[frag]]), axis=1)) for frag in range(2)]
        new_large_vals_full = [np.concatenate([pair[0] for pair in dens_eigpairs[frag]]) for frag in range(2)]

        for frag in range(2):
            to_del = []
            for ind, eigval in enumerate(new_large_vals_full[frag]):
                if eigval < dens_eigval_thresh:
                    to_del.append(ind)
            new_large_vals_full[frag] = np.delete(new_large_vals_full[frag], to_del, 0)
            new_large_vecs_full[frag] = np.delete(new_large_vecs_full[frag], to_del, 1)

        new_large_vecs = [[], []]
        dens_eigvals = [[], []]

        for frag in range(2):
            # the sorting did not work for the single example I've tried, but might do so for other examples
            #new_large_vals_full[frag], new_large_vecs_full[frag] = sort_eigen((new_large_vals_full[frag], new_large_vecs_full[frag]), order="descending")
            for ind, orth_vec in enumerate(orthogonalize(new_large_vecs_full[frag].T, eps=dens_filter_thresh, filter_list=new_large_vals_full[frag])):
                # The left eigvecs are very close to the right eigvecs depending on the system, but this way they are filtered out in a consistent manner
                # and it might also be helpful when looking at multiple states.
                new_large_vecs[frag].append(orth_vec)
                dens_eigvals[frag].append(new_large_vals_full[frag][ind])
        print(np.array(dens_eigvals, dtype=object))

        large_vec_map = []
        for frag in range(2):
            large_vec_map_ = np.zeros((n_states[frag], n_confs[frag]))
            for chg in monomer_charges[frag]:
                large_vec_map_[d_slices[frag][chg], c_slices[frag][chg]] = state_coeffs[frag][chg]
            large_vec_map.append(large_vec_map_)

        new_vecs = [np.einsum("ij,jp->ip", np.array(new_large_vecs[frag]), large_vec_map[frag]) for frag in range(2)]
        
        mixed_states = {}
        chg_sorted_keepers = [{chg: [] for chg in monomer_charges[frag]} for frag in range(2)]
        for frag in range(2):
            for state in new_vecs[frag]:
                state /= np.linalg.norm(state)
                norms = {chg: np.linalg.norm(state[c_slices[frag][chg]]) for chg in monomer_charges[frag]}
                print(norms, max(norms.values()))
                #if max(norms.values()) < 1 - 1e-6:
                #    raise ValueError(f"mixed state encountered (different charges are mixed for a state on frag {frag}), see {norms}")
                if max(norms.values()) > 1 - 1e-6:  # maybe rather go for 1e-8?
                    chg_sorted_keepers[frag][max(norms, key=norms.get)].append(state[c_slices[frag][max(norms, key=norms.get)]])
                else:
                    match = False
                    for key in mixed_states.keys():
                        if abs(max(norms.values()) - max(key)) < 1e-8:
                            # expect them to only turn up pairwise...if this is not the case, something more sophisticated needs to implemented here
                            # also expect the chg order in the key to be the same as in the current norms dict
                            for i, chg in enumerate(norms.keys()):
                                if norms[chg] < 1e-6:
                                    continue
                                if abs(key[i]**2 + norms[chg]**2 - 1) > 5e-8:
                                    raise ValueError("the charge contributions were found to be neither negligible nor adding up pairwise "
                                                     f"{i} {key[i]} {abs(key[i]**2 + norms[chg]**2 - 1)}")
                                chg_state = state[c_slices[frag][chg]] + mixed_states[key][c_slices[frag][chg]]
                                chg_sorted_keepers[frag][chg].append(chg_state / np.linalg.norm(chg_state))
                            del mixed_states[key]
                            match = True
                            break
                    if not match:
                        mixed_states[tuple(norms.values())] = state
        if mixed_states:
            print(mixed_states.keys())
            print("care, the above combs of states couldn't be resolved and will be neglected")
            #raise NotImplementedError("the current step requires are more refined charge filtering than is currently implemented")

        for frag in range(2):
            print(f"fragment {frag}")
            for chg, vecs in chg_sorted_keepers[frag].items():
                chg_sorted_keepers[frag][chg] = [i for i in orthogonalize(np.array(vecs))]
                print(f"for charge {chg} {len(chg_sorted_keepers[frag][chg])} states are kept")

        #for frag in range(2):
        #    for chg in monomer_charges[frag]:
        #        if len(chg_sorted_keepers[frag][chg])== 0:
        #            print(f"to prevent algorithm from breaking, because for frag {frag} with charge {chg} no states are kept, we keep the previous states for these charges here")
        #            chg_sorted_keepers[frag][chg] = dens_builder_stuff[frag][0][chg].coeffs

        for chg in monomer_charges[0]:
            for i, vec in enumerate(chg_sorted_keepers[0][chg]):
                big_inds = {ind: elem for ind, elem in enumerate(vec) if abs(elem) > 1e-1}
                print(chg, i, {tuple(conf_decoder(dens_builder_stuff[frag][0][chg].configs[j], n_orbs)): val for j, val in big_inds.items()})

        for frag in range(2):
            for chg in monomer_charges[frag]:
                dens_builder_stuff[frag][0][chg].coeffs = [i for i in chg_sorted_keepers[frag][chg]]

        return chg_sorted_keepers, dens_builder_stuff, full_eigvals[target_state]
    
    def alternate_enlarge_and_opt(frag_ind, dens_builder_stuff, dens, state_coeffs, dens_eigval_thresh=dens_filter_thresh, dets={}, target_state=target_state, grad_level=grad_level):
        print(f"opt frag {frag_ind}")

        if type(target_state) != int:
            if len(target_state) > 1:
                # this has not been done, because the fragment state spaces would get too big
                # for the XR Hamiltonian evaluation, before they can be reduced again via filtering
                raise NotImplementedError("The gradient based optimizer is currently only implemented for single states")

        #############################################
        # Obtaining the derivatives for frag frag_ind
        #############################################

        gs_energy_a, gradient_states_a, dl_prev, dr_prev = state_gradients(frag_ind, ints, dens_builder_stuff, dens, monomer_charges, n_threads=n_threads, xr_order=xr_order, dets=dets, grad_level=grad_level, target_state=target_state)
        if dets:
            # decompress gradients again into full configuration space
            gradient_states_a = {chg: np.einsum("iq,qp->ip", grad, dets[chg]) for chg, grad in gradient_states_a.items()}
        
        ########################################################
        # Application of the derivatives as enlarged basis for H
        ########################################################

        a_coeffs = {chg: np.array(tens.copy()) for chg, tens in state_coeffs[frag_ind].items()}
        grad_coeffs_a_raw = {chg: np.array(val) for chg, val in gradient_states_a.items()}

        # For Hamiltonian evaluation in extended state basis, i.e. states + gradients, an orthonormal set for each fragment
        # is required, but this choice is not unique. One could e.g. normalize, orthogonalize and then again normalize,
        # or orthogonalize and normalize without previous normalization.
        tot_a_coeffs = {chg: np.array([i for i in orthogonalize(np.vstack((a_coeffs[chg], grad_coeffs_a_raw[chg])))])
                        for chg in monomer_charges[frag_ind]}
        
        state_dict = [{chg: len(state_coeffs[i][chg]) for chg in monomer_charges[i]} for i in range(2)]
        conf_dict = [{chg: len(state_coeffs[i][chg][0]) for chg in monomer_charges[i]} for i in range(2)]

        grad_dict = {chg: len(tot_a_coeffs[chg]) - state_dict[frag_ind][chg] for chg in monomer_charges[frag_ind]}
        grad_coeffs = {chg: [tot_a_coeffs[chg][-i] for i in range(grad_dict[chg], 0, -1)] for chg in monomer_charges[frag_ind]}

        n_states = [sum(state_dict[i][chg] for chg in monomer_charges[i]) for i in range(2)]
        n_states[frag_ind] += sum(grad_dict[chg] for chg in monomer_charges[frag_ind])
        n_confs = [sum(conf_dict[i][chg] for chg in monomer_charges[i]) for i in range(2)]

        d_slices = [get_slices(state_dict[i], monomer_charges[i]) for i in range(2)]
        c_slices = [get_slices(conf_dict[i], monomer_charges[i]) for i in range(2)]

        d_slices_first = get_slices(state_dict[frag_ind], monomer_charges[frag_ind], append=grad_dict, type="first") # for i in range(2)]
        d_slices_latter = get_slices(state_dict[frag_ind], monomer_charges[frag_ind], append=grad_dict, type="latter") # for i in range(2)]

        mat_sl_first, mat_sl_latter = {}, {}
        mat_sl_first[frag_ind] = d_slices_first
        mat_sl_latter[frag_ind] = d_slices_latter
        mat_sl_first[1 - frag_ind] = d_slices[1 - frag_ind]
        mat_sl_latter[1 - frag_ind] = d_slices[1 - frag_ind]

        # state gradient provider changes both densities, so the (new) "normal" densities need to be recovered/generated here
        # TODO: here either recompute the densities for the fragment without gradients, which saves memory, or load these densities
        # again, which is of course faster but more memory intensive ... probably better take the second option
        for chg in monomer_charges[frag_ind]:
            dens_builder_stuff[frag_ind][0][chg].coeffs = [i.copy() for i in tot_a_coeffs[chg]]
        dens[frag_ind] = densities.build_tensors(*dens_builder_stuff[frag_ind][:-1], options=dens_builder_stuff[frag_ind][-1], n_threads=n_threads)
    
        for chg in monomer_charges[1 - frag_ind]:  # here no new dens eval is needed, just contract with the inverse of d, to negate the alternation from the gradient determination
            dens_builder_stuff[1 - frag_ind][0][chg].coeffs = [i.copy() for i in state_coeffs[1 - frag_ind][chg]]
        dens[1 - frag_ind] = densities.build_tensors(*dens_builder_stuff[1 - frag_ind][:-1], options=dens_builder_stuff[1 - frag_ind][-1], n_threads=n_threads)

        H1, H2 = get_xr_H(ints, dens, xr_order, monomer_charges)

        full = H2.reshape(n_states[0], n_states[1],
                          n_states[0], n_states[1])
        
        grad_dict_map = {frag_ind: grad_dict, 1 - frag_ind: state_dict[1 - frag_ind]}

        for chg0 in monomer_charges[0]:
            for chg1 in monomer_charges[1]:
                #(state,state)
                full[mat_sl_first[0][chg0], mat_sl_first[1][chg1], mat_sl_first[0][chg0], mat_sl_first[1][chg1]] +=\
                    np.einsum("ij,kl->ikjl", H1[0][mat_sl_first[0][chg0], mat_sl_first[0][chg0]], np.eye(state_dict[1][chg1])) +\
                    np.einsum("ij,kl->ikjl", np.eye(state_dict[0][chg0]), H1[1][mat_sl_first[1][chg1], mat_sl_first[1][chg1]])
                #(state,grad)
                if frag_ind == 0:
                    term = np.einsum("ij,kl->ikjl", H1[0][mat_sl_first[0][chg0], mat_sl_latter[0][chg0]], np.eye(state_dict[1][chg1]))
                else:
                    term = np.einsum("ij,kl->ikjl", np.eye(state_dict[0][chg0]), H1[1][mat_sl_first[1][chg1], mat_sl_latter[1][chg1]])
                full[mat_sl_first[0][chg0], mat_sl_first[1][chg1], mat_sl_latter[0][chg0], mat_sl_latter[1][chg1]] += term
                #(grad,state)
                if frag_ind == 0:
                    term = np.einsum("ij,kl->ikjl", H1[0][mat_sl_latter[0][chg0], mat_sl_first[0][chg0]], np.eye(state_dict[1][chg1]))
                else:
                    term = np.einsum("ij,kl->ikjl", np.eye(state_dict[0][chg0]), H1[1][mat_sl_latter[1][chg1], mat_sl_first[1][chg1]])
                full[mat_sl_latter[0][chg0], mat_sl_latter[1][chg1], mat_sl_first[0][chg0], mat_sl_first[1][chg1]] += term
                #(grad,grad)
                full[mat_sl_latter[0][chg0], mat_sl_latter[1][chg1], mat_sl_latter[0][chg0], mat_sl_latter[1][chg1]] +=\
                    np.einsum("ij,kl->ikjl", H1[0][mat_sl_latter[0][chg0], mat_sl_latter[0][chg0]], np.eye(grad_dict_map[1][chg1])) +\
                    np.einsum("ij,kl->ikjl", np.eye(grad_dict_map[0][chg0]), H1[1][mat_sl_latter[1][chg1], mat_sl_latter[1][chg1]])

        full = full.reshape(n_states[0] * n_states[1],
                            n_states[0] * n_states[1])
        
        #################################################
        # Solve for state of interest (here ground state)
        #################################################
        
        full_eigvals_raw, full_eigvec_l_unsorted, full_eigvec_r_unsorted = sp.linalg.eig(full, left=True, right=True)
        full_eigvals, full_eigvec_l = sort_eigen((full_eigvals_raw, full_eigvec_l_unsorted))
        full_eigvals_check, full_eigvec_r = sort_eigen((full_eigvals_raw, full_eigvec_r_unsorted))
        if any(full_eigvals - full_eigvals_check):  # this check should be unnecessary and might need to be refined for fully degenerate states
            raise ValueError("something went wrong with the sorting")
        print(np.min(full_eigvals))
        print("imag contribution of lowest eigvec of full H with grads with eig", np.linalg.norm(np.imag(full_eigvec_r[:, target_state])))#, np.linalg.norm(np.imag(full_eigvec_l[:, 0])))

        #############################
        # Filter which states to keep
        #############################

        if type(target_state) == int:
            target_state = [target_state]

        # it would be correct here to also use the left eigvec, but it was found that only the right eigvec
        # yields similar or even better results with much more compact state spaces.
        # TODO: an open question remains how the performance differs for stronger interacting fragments
        full_eigvecs = [full_eigvec_r]#, full_eigvec_l]
        target_vecs = [np.real(full_eigvecs[lr][:, i].reshape((n_states[0], n_states[1]))) for i in target_state for lr in range(len(full_eigvecs))]
        #def get_large_elems(mat, eps=1e-3):
        #    ret = {}
        #    for i, vec in enumerate(mat):
        #        for j, elem in enumerate(vec):
        #            if abs(elem) > eps:
        #                ret[(i, j)] = elem
        #    return ret
        #print(get_large_elems(target_vecs[0]))
        #print(get_large_elems(target_vecs[1]))

        dens_mats = [[np.einsum("ij,kj->ik", vec, vec) for vec in target_vecs],  # contract over frag_b part
                     [np.einsum("ij,ik->jk", vec, vec) for vec in target_vecs]]  # contract over frag_a part

        dens_eigpairs = [[sort_eigen(sp.linalg.eigh(mat), order="descending") for mat in dens_mats[frag]] for frag in range(2)]
        # the following is tricky...
        # building everything here and then orthogonalize will only take the first eigenvector into account,
        # since they build the full space and therefore one needs to either sort it differently or truncate before orthogonalizing
        new_large_vecs_full = [np.real(np.concatenate(tuple([pair[1] for pair in dens_eigpairs[frag]]), axis=1)) for frag in range(2)]
        new_large_vals_full = [np.concatenate([pair[0] for pair in dens_eigpairs[frag]]) for frag in range(2)]

        for frag in range(2):
            to_del = []
            for ind, eigval in enumerate(new_large_vals_full[frag]):
                if eigval < dens_eigval_thresh:
                    to_del.append(ind)
            new_large_vals_full[frag] = np.delete(new_large_vals_full[frag], to_del, 0)
            new_large_vecs_full[frag] = np.delete(new_large_vecs_full[frag], to_del, 1)

        new_large_vecs = [[], []]
        dens_eigvals = [[], []]

        for frag in range(2):
            # the filtering only changes the order for multiple states
            #new_large_vals_full[frag], new_large_vecs_full[frag] = sort_eigen((new_large_vals_full[frag], new_large_vecs_full[frag]), order="descending")
            for ind, orth_vec in enumerate(orthogonalize(new_large_vecs_full[frag].T, eps=dens_filter_thresh, filter_list=new_large_vals_full[frag])):#normalize=False)):#[len(dens_eigvals[frag]):, :]):
                # The left eigvecs are very close to the right eigvecs depending on the system, but this way they are filtered out in a consistent manner
                # and it might also be helpful when looking at multiple states.
                # If the threshold is lowered, beware that the orthogonalizer also has a threshold for setting parallel vectors to zero.
                new_large_vecs[frag].append(orth_vec)
                dens_eigvals[frag].append(new_large_vals_full[frag][ind])
        print(np.array(dens_eigvals, dtype=object))

        # the following threshold is very delicate
        keepers = [[], []]
        for frag in range(2):
            for i, vec in enumerate(new_large_vecs[frag]):#.T):
                if dens_eigvals[frag][i] >= dens_eigval_thresh:
                    if np.linalg.norm(np.imag(vec)) > 1e-7:
                        raise ValueError(f"imaginary contribution of relevant density eigvec is {np.linalg.norm(np.imag(vec))}, which is too large")
                    keepers[frag].append(np.real(vec))
            print(f"{len(keepers[frag])} states are kept for frag {frag}")

        large_vec_map = [0, 0]

        large_vec_map[frag_ind] = np.zeros((n_states[frag_ind], n_confs[frag_ind]))
        for chg in monomer_charges[frag_ind]:
            large_vec_map[frag_ind][d_slices_first[chg], c_slices[frag_ind][chg]] = state_coeffs[frag_ind][chg]
            if grad_dict[chg] >= 1:
                large_vec_map[frag_ind][d_slices_latter[chg], c_slices[frag_ind][chg]] = grad_coeffs[chg]

        large_vec_map[1 - frag_ind] = np.zeros((n_states[1 - frag_ind], n_confs[1 - frag_ind]))
        for chg in monomer_charges[1 - frag_ind]:
            large_vec_map[1 - frag_ind][d_slices[1 - frag_ind][chg], c_slices[1 - frag_ind][chg]] = state_coeffs[1 - frag_ind][chg]

        new_vecs = [np.einsum("ij,jp->ip", np.array(keepers[frag]), large_vec_map[frag]) for frag in range(2)]
        
        mixed_states = {}
        chg_sorted_keepers = [{chg: [] for chg in monomer_charges[frag]} for frag in range(2)]
        for frag in range(2):
            for state in new_vecs[frag]:
                state /= np.linalg.norm(state)
                norms = {chg: np.linalg.norm(state[c_slices[frag][chg]]) for chg in monomer_charges[frag]}
                print(norms, max(norms.values()))
                #if max(norms.values()) < 1 - 1e-6:
                #    raise ValueError(f"mixed state encountered (different charges are mixed for a state on frag {frag}), see {norms}")
                if max(norms.values()) > 1 - 1e-6:  # maxbe rather go for 1e-8
                    chg_sorted_keepers[frag][max(norms, key=norms.get)].append(state[c_slices[frag][max(norms, key=norms.get)]])
                else:
                    match = False
                    for key in mixed_states.keys():
                        if abs(max(norms.values()) - max(key)) < 1e-8:
                            # expect them to only turn up pairwise...if this is not the case, something more sophisticated needs to implemented here
                            # also expect the chg order in the key to be the same as in the current norms dict
                            for i, chg in enumerate(norms.keys()):
                                if norms[chg] < 1e-6:
                                    continue
                                if abs(key[i]**2 + norms[chg]**2 - 1) > 5e-8:
                                    raise ValueError("the charge contributions were found to be neither negligible nor adding up pairwise")
                                chg_state = state[c_slices[frag][chg]] + mixed_states[key][c_slices[frag][chg]]
                                chg_sorted_keepers[frag][chg].append(chg_state / np.linalg.norm(chg_state))
                            del mixed_states[key]
                            match = True
                            break
                    if not match:
                        mixed_states[tuple(norms.values())] = state
        if mixed_states:
            print(mixed_states.keys())
            print("care, the above combs of states couldn't be resolved and will be neglected")
            #raise NotImplementedError("the current step requires are more refined charge filtering than is currently implemented")

        for frag in range(2):
            print(f"fragment {frag}")
            for chg, vecs in chg_sorted_keepers[frag].items():
                final_keepers = []
                for vec in orthogonalize(np.array(vecs)):
                    final_keepers.append(vec)
                chg_sorted_keepers[frag][chg] = final_keepers
                print(f"for charge {chg} {len(chg_sorted_keepers[frag][chg])} states are kept")

        #for frag in range(2):
        #    for chg in monomer_charges[frag]:
        #        if len(chg_sorted_keepers[frag][chg])== 0:
        #            print(f"to prevent algorithm from breaking, because for frag {frag} with charge {chg} no states are kept, we keep the previous states for these charges here")
        #            chg_sorted_keepers[frag][chg] = dens_builder_stuff[frag][0][chg].coeffs

        for frag in range(2):
            for chg in monomer_charges[frag]:
                dens_builder_stuff[frag][0][chg].coeffs = [i for i in chg_sorted_keepers[frag][chg]]

        for frag in range(2):
            for chg in monomer_charges[frag]:
                for i, vec in enumerate(dens_builder_stuff[frag][0][chg].coeffs):
                    big_inds = {ind: elem for ind, elem in enumerate(vec) if abs(elem) > 1e-1}
                    print(frag, chg, i, {tuple(conf_decoder(dens_builder_stuff[frag][0][chg].configs[j], n_orbs)): val for j, val in big_inds.items()})

        return chg_sorted_keepers, dens_builder_stuff, gs_energy_a, full_eigvals[target_state]

    def postprocessing(en, en_extended, en_history, en_with_grads_history, converged):
        en_history.append(en)
        en_with_grads_history.append(en_extended)
        print(f"History of XR[{xr_order}] energies:", en_history[1:])
        print(f"History of XR[{xr_order}] energies with gradients in the Hamiltonian build"
              "(order is grads on frag 0 then on 1 then on 0 again and so on):", en_with_grads_history[1:])

        if abs(en_history[-1] - en_history[-2]) <= conv_thresh and en_history[-1] - en_with_grads_history[-1] <= conv_thresh:
            if en_history[-1] - en_with_grads_history[-1] < 0.:
                RuntimeWarning("gs energy of larger Hamiltonian (with gradients included) is not lower than the gs energy of the smaller Hamiltonian in this iteration")
            else:
                print("Converged!!!")
                converged = True
        return converged

    def enlarge_state_space(frag, state_coeffs_optimized, dens_builder_stuff, dens):
        for chg in monomer_charges[frag]:
            print(f"for fragment {frag} with charge {chg} "
                    f"{len(additional_states[frag][chg]) - state_tracker[frag][chg]}"
                    " states still need to be included")
            if len(additional_states[frag][chg]) == state_tracker[frag][chg]:
                screening_done[frag][abs(min(monomer_charges[frag])) + chg] = True
                continue
            # the following thresholds are not set in stone
            # since the xr evaluation scales as the fourth order in the number of states
            # we dont want to overdo it here. 20 is still quite acceptable, but maybe not
            # enough, so a relative increase is provided as well, resulting in an expansion
            # by 1/3 leading to an increase in CPU time of roughly a factor of 3 for the XR evaluation.
            # This also caps the amount of densities, which have to be computed at once, which
            # also saves time, as it roughly scales to the second order in the number of states.
            max_states = max(len(dens_builder_stuff[frag][0][chg].coeffs) * 4 // 3, 20)  # maybe increase to 25 or 30
            max_add = min(max_states - len(dens_builder_stuff[frag][0][chg].coeffs), len(additional_states[frag][chg]) - state_tracker[frag][chg])
            # using the following only makes sense, if sorting in state screening is active, but that was found to be ineffective
            #conf_ind = np.argmax(additional_states[frag][chg][state_tracker[frag][chg] + max_add - 1])
            #conf_ind_pre = np.argmax(additional_states[frag][chg][state_tracker[frag][chg] + max_add - 2])
            #singlet, pair = is_singlet(conf_decoder(dens_builder_stuff[frag][0][chg].configs[conf_ind], n_orbs), n_orbs)
            #if pair != dens_builder_stuff[frag][0][chg].configs[conf_ind_pre] and max_add < len(additional_states[frag][chg]) - state_tracker[frag][chg]:
            #    max_add += 1
            dens_builder_stuff[frag][0][chg].coeffs += additional_states[frag][chg][state_tracker[frag][chg]: state_tracker[frag][chg] + max_add]
            dens_builder_stuff[frag][0][chg].coeffs = [i for i in orthogonalize(np.array(dens_builder_stuff[frag][0][chg].coeffs))]
            state_coeffs_optimized[frag][chg] = dens_builder_stuff[frag][0][chg].coeffs.copy()
            state_tracker[frag][chg] += max_add
        dens[frag] = densities.build_tensors(*dens_builder_stuff[frag][:-1], options=density_options, n_threads=n_threads)
        #print(raw(dens[0]["ca"][(0,0)])[0, 0, :, :])
        #raise ValueError("stop here")
        return state_coeffs_optimized, dens_builder_stuff, dens
    

    state_tracker = {frag: {chg: 0 for chg in monomer_charges[frag]} for frag in range(2)}
    screening_done = np.array([[False for chg in monomer_charges[frag]] for frag in range(2)])
    state_coeffs_optimized = state_coeffs_og
    if begin_from_state_prep:
        safety_iter = 0
    else:
        safety_iter = 100
    screening_energies = []

    # The following code tries to prepare optimized guess states for a solver, by enlarging the state space
    # on both fragments at the same time with states from the screening and then compresses them again.
    # This is referred to as gradient free optimization variant.

    # expanding the state space partially to then reduce and expand again saves a lot of CPU time and memory,
    # but on the other hand some contributions might be lost, since the relevant other state(s) required for
    # a large contribution with an integral might not appear in the same current state space...
    # This could be (partially) circumvented in two ways, which can also be applied both
    # 1. do multiple forward and backward cycles (like e.g. in DMRG)
    # 2. apply some prefiltering making sure pairs of most probably interacting determinants/states are included
    # TODO: Introduce some procedure that "preoptimizes" the determinant space to linear combinations
    #       and use those to further reduce the amount of determinants that need to be added here
    dens = [[], []]

    ####################################
    # Gradient free optimization variant
    ####################################

    while safety_iter < 50 and begin_from_state_prep:
        safety_iter += 1
        for frag in range(2):
            state_coeffs_optimized, dens_builder_stuff, dens = enlarge_state_space(frag, state_coeffs_optimized, dens_builder_stuff, dens)
        if all(screening_done.flatten()):
            break

        state_coeffs_optimized, dens_builder_stuff, gs_energy = reduce_screened_state_space(dens_builder_stuff, dens, state_coeffs_optimized)
        screening_energies.append(gs_energy)
        print("energy development during stepwise incorporation of screened states", screening_energies)


    #####################################
    # Gradient based optimization variant
    #####################################
    if max_iter > 0:
        print("starting iterative state solver now")
        converged = False

        dens[0] = densities.build_tensors(*dens_builder_stuff[0][:-1], options=dens_builder_stuff[0][-1], n_threads=n_threads)
    
    # The following variant tries to optimize one fragment by building its gradients, after previously
    # enlarging the state space of the other fragment with states obtained from the screening. Both are
    # then compressed again.
    en_history, en_with_grads_history = [0], [0]
    iter = 0
    while iter < max_iter:
        iter += 1
        # opt frag 0 and previously enlarge 1
        state_coeffs_optimized, dens_builder_stuff, dens = enlarge_state_space(1, state_coeffs_optimized, dens_builder_stuff, dens)
        state_coeffs_optimized, dens_builder_stuff, gs_energy_a, gs_en_a_with_grads = alternate_enlarge_and_opt(0, dens_builder_stuff, dens, state_coeffs_optimized, dets=additional_states[0], grad_level=grad_level)
        converged = postprocessing(gs_energy_a, gs_en_a_with_grads, en_history, en_with_grads_history, converged)

        if all(screening_done.flatten()):
            break

        dens[1] = densities.build_tensors(*dens_builder_stuff[1][:-1], options=dens_builder_stuff[1][-1], n_threads=n_threads)
        
        # opt frag 1 and previously enlarge 0
        state_coeffs_optimized, dens_builder_stuff, dens = enlarge_state_space(0, state_coeffs_optimized, dens_builder_stuff, dens)
        state_coeffs_optimized, dens_builder_stuff, gs_energy_a, gs_en_a_with_grads = alternate_enlarge_and_opt(1, dens_builder_stuff, dens, state_coeffs_optimized, dets=additional_states[1], grad_level=grad_level)
        converged = postprocessing(gs_energy_a, gs_en_a_with_grads, en_history, en_with_grads_history, converged)

        dens[0] = densities.build_tensors(*dens_builder_stuff[0][:-1], options=dens_builder_stuff[0][-1], n_threads=n_threads)
        
        if all(screening_done.flatten()):
            break

    if len(screening_energies) == 0:
        ret_energies = en_history
    else:
        ret_energies =  screening_energies
    #else:
    #    return [screening_energies, en_history]
    
    #return screening_energies, BeN, ints, dens, dens_builder_stuff  # this is for the orbital solver
    return ret_energies, dens_builder_stuff
