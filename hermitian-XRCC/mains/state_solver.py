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
from qode.math.tensornet import raw, tl_tensor, backend_contract_path
#import qode.util
from qode.util import timer, sort_eigen
from state_gradients import state_gradients, get_slices, get_adapted_overlaps
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

tl.plugins.use_opt_einsum()
backend_contract_path(True)

#class empty(object):  pass

def optimize_states(displacement, max_iter, xr_order, conv_thresh=1e-6, dens_filter_thresh=1e-7, begin_from_state_prep=True):
    """
    max_iter: defines the max_iter for the state solver using an actual gradient
    begin_from_state_prep: determines whether previously to the gradient based optimization a guess shall be provided
                           originating from the same state screening but then optimizing without the gradient (faster but less reliable)
    the current recommendation is to either only use the solver with a gradient with begin_from_state_prep=False or max_iter=0 with begin_from_state_prep=True.
    As already mentioned the first one is more reliable, but the latter one is quite a bit faster
    """
    ######################################################
    # Initialize integrals and density preliminaries
    ######################################################

    n_frag       = 2
    displacement = displacement
    project_core = True
    monomer_charges = [[0, +1, -1], [0, +1, -1]]
    density_options = ["compress=SVD,cc-aa"]
    frozen_orbs = [0, 9]
    n_orbs = 9
    n_occ = [2, 2]  # currently only alpha beta separation, but generalize to frag level not done yet!!!

    #ref_states = pickle.load(open("ref_states.pkl", mode="rb"))
    #ref_mos = pickle.load(open("ref_mos.pkl", mode="rb"))

    # "Assemble" the supersystem for the displaced fragments and get integrals
    BeN = []
    dens = []
    dens_builder_stuff = []
    state_coeffs_og = []
    #pre_opt_states = pickle.load(open("pre_opt_coeffs.pkl", mode="rb"))
    for m in range(int(n_frag)):
        state_obj, dens_var_1, dens_var_2, n_threads, Be = get_fci_states(displacement, n_state_list=[(1, 2), (0, 10), (-1, 10)])
        #Be.basis.MOcoeffs = ref_mos.copy()
        pickle.dump(Be.basis.MOcoeffs, open(f"pre_opt_mos_{m}.pkl", mode="wb"))

        #Be.basis.MOcoeffs = pickle.load(open(f"pre_opt_mos_{m}.pkl", mode="rb"))
        #for chg in monomer_charges[m]:
        #    state_obj[chg].coeffs = pre_opt_states[m][chg].copy()

            #state_obj[chg].coeffs = ref_states[chg].coeffs.copy()
        #    state_obj[chg].configs = ref_states[chg].configs.copy()
            #state_obj[chg].coeffs = [i[:448] for i in state_obj[chg].coeffs]
            #if m == 0:
            #    state_obj[chg].coeffs = ref_states[chg].coeffs.copy()
        #    state_obj[chg].coeffs = ref_states[chg].coeffs[:1]
        for elem,coords in Be.atoms:  coords[2] += m * displacement    # displace along z
        BeN.append(Be)
        #dens.append(densities.build_tensors(state_obj, dens_var_1, dens_var_2, options=density_options, n_threads=n_threads))
        dens_builder_stuff.append([state_obj, dens_var_1, dens_var_2, density_options])
        #state_coeffs_og.append({chg: state_obj[chg].coeffs for chg in state_obj})
        state_coeffs_og.append({chg: state_obj[chg].coeffs for chg in state_obj})
        #print(Be.basis.MOcoeffs)
        #raise ValueError("stop here")
    #print(type(raw(dens[0]["a"][(+1,0)])), raw(dens[0]["a"][(+1,0)]).shape)

    int_timer = timer()
    ints = get_ints(BeN, project_core, int_timer)
    
    def iteration(frag_ind, dens_builder_stuff, dens, state_coeffs, dens_eigval_thresh=dens_filter_thresh, dets={}):
        # In the following the variables are named as if frag_ind = 0, but it also works with frag_ind = 1
        print(f"opt frag {frag_ind}")

        def conf_decoder(conf):
            ret = []
            for bit in range(n_orbs * 2, -1, -1):
                if conf - 2**bit < 0:
                    continue
                conf -= 2**bit
                ret.append(bit)
            return sorted(ret)
        """
        #additional_random = {}
        for chg in monomer_charges[1 - frag_ind]:
            additional_random = []
            big_inds = {}
            for vec in state_coeffs[1 - frag_ind][chg]:
                for ind, elem in enumerate(vec):
                    if elem < 1e-1:
                        continue
                    big_inds[ind] = elem #{ind: elem for ind, elem in enumerate(vec) if abs(elem) > 1e-2}
                #print(chg, i, {tuple(conf_decoder(dens_builder_stuff[frag][0][chg].configs[j])): val for j, val in big_inds.items()})
            n_extra = 2#min(20 - len(state_coeffs[1  - frag_ind][chg]), 2 * len(state_coeffs[1  - frag_ind][chg]))
            new_vec_structure = np.zeros_like(state_coeffs[1 - frag_ind][chg][0])
            for ind in big_inds.keys():
                new_vec_structure[ind] = 1.
            for _ in range(n_extra):
                additional_random.append(np.random.rand(len(new_vec_structure)) * new_vec_structure.copy())
            state_coeffs[1 - frag_ind][chg] = [i for i in orthogonalize(np.array(state_coeffs[1 - frag_ind][chg] + additional_random))]
            dens_builder_stuff[1 - frag_ind][0][chg].coeffs = state_coeffs[1 - frag_ind][chg]
        dens[1 - frag_ind] = densities.build_tensors(*dens_builder_stuff[1 - frag_ind][:-1], options=density_options, n_threads=n_threads)
        """
        
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
        gs_energy_a, gradient_states_a, d = state_gradients(frag_ind, ints, dens_builder_stuff, dens, monomer_charges, n_threads=n_threads, xr_order=xr_order, dets=dets)
        if dets:
            # decompress gradients again into full configuration space
            gradient_states_a = {chg: np.einsum("iq,qp->ip", grad, dets[chg]) for chg, grad in gradient_states_a.items()}
        #coeffs_grads = [state_coeffs, gradient_states]
        #pickle.dump(coeffs_grads, open("coeffs_grads.pkl", mode="wb"))
        
        ################################
        # Application of the derivatives
        ################################

        a_coeffs = {chg: np.array(tens.copy()) for chg, tens in state_coeffs[frag_ind].items()}
        #b_coeffs = {chg: np.array(tens.copy()) for chg, tens in state_coeffs[1].items()}
        grad_coeffs_a = {chg: np.array([vec / np.linalg.norm(vec) for vec in val]) for chg, val in gradient_states_a.items()}
        #grad_coeffs_b = {chg: np.array([vec / np.linalg.norm(vec) for vec in val]) for chg, val in gradient_states_b.items()}

        for chg in monomer_charges[frag_ind]:
            for i, vec in enumerate(grad_coeffs_a[chg]):
                big_inds = {ind: elem for ind, elem in enumerate(vec) if abs(elem) > 1e-1}
                #print(i, [ind for ind, elem in enumerate(vec) if abs(elem) > 1e-1])
                print(chg, i, {tuple(conf_decoder(dens_builder_stuff[frag_ind][0][chg].configs[j])): val for j, val in big_inds.items()})

        # For Hamiltonian evaluation in extended state basis, i.e. states + gradients, an orthonormal set for each fragment
        # is required, but this choice is not unique. One could e.g. normalize, orthogonalize and then again normalize,
        # or orthogonalize and normalize without previous normalization.
        tot_a_coeffs = {chg: np.array([i for i in orthogonalize(np.vstack((a_coeffs[chg], grad_coeffs_a[chg]))) if np.linalg.norm(i) > 0.99])
                        for chg in monomer_charges[frag_ind]}
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

        #if frag_ind == 1:
        #    frag_map = {0: 0, 1: 1}
        #elif frag_ind == 0:
        #    frag_map = {0: 1, 1: 0}
    
        for chg in monomer_charges[1 - frag_ind]:
            dens_builder_stuff[1 - frag_ind][0][chg].coeffs = [i.copy() for i in state_coeffs[1 - frag_ind][chg]]# + [i / np.linalg.norm(i) for i in np.einsum("ip,ki->kp", state_coeffs[1 - frag_ind][chg], get_adapted_overlaps(frag_map, d, d_slices)[chg])]
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

        #full = H2.reshape(2 * n, 2 * n, 2 * n, 2 * n)
        full = H2.reshape(2 * n_states[0], 2 * n_states[1],
                          2 * n_states[0], 2 * n_states[1])
        add_to_full(full, (0, 0))
        add_to_full(full, (0, 1))
        add_to_full(full, (1, 0))
        add_to_full(full, (1, 1))

        #full = full.reshape(4 * n**2, 4 * n**2)
        full = full.reshape(4 * n_states[0] * n_states[1],
                            4 * n_states[0] * n_states[1])
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
        #full_gs_vec = full_eigvec[0].reshape(2 * n_states[0], 2 * n_states[1])
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
        dens_eigvals, dens_eigvecs = sort_eigen(np.linalg.eigh(dens_mat), order="descending")
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
            #print(norms)
            if max(norms.values()) < 0.99:
                ValueError(f"mixed state encountered (different charges are mixed for a state on frag {frag_ind}), see {norms}")
            chg_sorted_keepers[max(norms, key=norms.get)].append(state[c_slices[frag_ind][max(norms, key=norms.get)]])

        for chg, vecs in chg_sorted_keepers.items():
            print(f"for charge {chg} {len(vecs)} states are kept")
            chg_sorted_keepers[chg] = [i for i in orthogonalize(np.array(vecs)) if np.linalg.norm(i) > 0.99]


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
        
        dens_eigvals_a, dens_eigvecs_a = sort_eigen(np.linalg.eigh(dens_mat_a), order="descending")
        dens_eigvals_b, dens_eigvecs_b = sort_eigen(np.linalg.eigh(dens_mat_b), order="descending")
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
                #print(norms)
                if max(norms.values()) < 0.99:
                    ValueError(f"mixed state encountered (different charges are mixed for a state on frag {frag}), see {norms}")
                chg_sorted_keepers[frag][max(norms, key=norms.get)].append(state[c_slices[frag][max(norms, key=norms.get)]])

        for frag in range(2):
            print(f"fragment {frag}")
            for chg, vecs in chg_sorted_keepers[frag].items():
                print(f"for charge {chg} {len(vecs)} states are kept")
                chg_sorted_keepers[frag][chg] = [i for i in orthogonalize(np.array(vecs)) if np.linalg.norm(i) > 0.99]

        for frag in range(2):
            for chg in monomer_charges[frag]:
                state_coeffs[frag][chg] = chg_sorted_keepers[frag][chg]

        def conf_decoder(conf):
            ret = []
            for bit in range(n_orbs * 2, -1, -1):
                if conf - 2**bit < 0:
                    continue
                conf -= 2**bit
                ret.append(bit)
            return sorted(ret)

        for chg in monomer_charges[0]:
            for i, vec in enumerate(chg_sorted_keepers[0][chg]):
                big_inds = {ind: elem for ind, elem in enumerate(vec) if abs(elem) > 1e-1}
                #print(i, [ind for ind, elem in enumerate(vec) if abs(elem) > 1e-1])
                print(chg, i, {tuple(conf_decoder(dens_builder_stuff[frag][0][chg].configs[j])): val for j, val in big_inds.items()})

        #raise ValueError("stop here")

        for frag in range(2):
            for chg in monomer_charges[frag]:
                dens_builder_stuff[frag][0][chg].coeffs = [i for i in state_coeffs[frag][chg]]
            #dens[frag] = densities.build_tensors(*dens_builder_stuff[frag][:-1], options=dens_builder_stuff[frag][-1], n_threads=n_threads)
        return state_coeffs, dens_builder_stuff, full_eigvals[0]
    
    def alternate_enlarge_and_opt(frag_ind, dens_builder_stuff, dens, state_coeffs, dens_eigval_thresh=dens_filter_thresh, dets={}):
        print(f"opt frag {frag_ind}")

        #############################################
        # Obtaining the derivatives for frag frag_ind
        #############################################

        gs_energy_a, gradient_states_a, d = state_gradients(frag_ind, ints, dens_builder_stuff, dens, monomer_charges, n_threads=n_threads, xr_order=xr_order, dets=dets)
        if dets:
            # decompress gradients again into full configuration space
            gradient_states_a = {chg: np.einsum("iq,qp->ip", grad, dets[chg]) for chg, grad in gradient_states_a.items()}
        
        ########################################################
        # Application of the derivatives as enlarged basis for H
        ########################################################

        a_coeffs = {chg: np.array(tens.copy()) for chg, tens in state_coeffs[frag_ind].items()}
        #grad_coeffs_a = {chg: np.array([vec / np.linalg.norm(vec) for vec in val]) for chg, val in gradient_states_a.items()}
        grad_coeffs_a_raw = {chg: np.array(val) for chg, val in gradient_states_a.items()}

        # For Hamiltonian evaluation in extended state basis, i.e. states + gradients, an orthonormal set for each fragment
        # is required, but this choice is not unique. One could e.g. normalize, orthogonalize and then again normalize,
        # or orthogonalize and normalize without previous normalization.
        tot_a_coeffs = {chg: np.array([i for i in orthogonalize(np.vstack((a_coeffs[chg], grad_coeffs_a_raw[chg]))) if np.linalg.norm(i) > 0.99])
                        for chg in monomer_charges[frag_ind]}
        
        state_dict = [{chg: len(state_coeffs[i][chg]) for chg in monomer_charges[i]} for i in range(2)]
        conf_dict = [{chg: len(state_coeffs[i][chg][0]) for chg in monomer_charges[i]} for i in range(2)]

        grad_dict = {chg: len(tot_a_coeffs[chg]) - state_dict[frag_ind][chg] for chg in monomer_charges[frag_ind]}
        #grad_coeffs = {chg: list(reversed([list(reversed(tot_a_coeffs[chg]))[i] for i in range(grad_dict[chg])]))# range(grad_dict[chg], 0, -1)]))
        #               for chg in monomer_charges[frag_ind] if grad_dict[chg] >= 1}
        grad_coeffs = {chg: [tot_a_coeffs[chg][-i] for i in range(grad_dict[chg], 0, -1)] for chg in monomer_charges[frag_ind]}
        #print(grad_dict)
        #print({i: len(val) for i, val in grad_coeffs.items()})

        n_states = [sum(state_dict[i][chg] for chg in monomer_charges[i]) for i in range(2)]
        n_states[frag_ind] += sum(grad_dict[chg] for chg in monomer_charges[frag_ind])
        n_confs = [sum(conf_dict[i][chg] for chg in monomer_charges[i]) for i in range(2)] #sum(conf_dict[0].values())

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
        for chg in monomer_charges[frag_ind]:
            dens_builder_stuff[frag_ind][0][chg].coeffs = [i.copy() for i in tot_a_coeffs[chg]]
        dens[frag_ind] = densities.build_tensors(*dens_builder_stuff[frag_ind][:-1], options=dens_builder_stuff[frag_ind][-1], n_threads=n_threads)
    
        for chg in monomer_charges[1 - frag_ind]:  # here no new dens eval is needed, just contract with the inverse of d, to negate the alternation from the gradient determination
            dens_builder_stuff[1 - frag_ind][0][chg].coeffs = [i.copy() for i in state_coeffs[1 - frag_ind][chg]]
        dens[1 - frag_ind] = densities.build_tensors(*dens_builder_stuff[1 - frag_ind][:-1], options=dens_builder_stuff[1 - frag_ind][-1], n_threads=n_threads)

        H1, H2 = get_xr_H(ints, dens, xr_order, monomer_charges)
        
        #full = H2.reshape((2 - frag_ind) * n_states[0], (1 + frag_ind) * n_states[1],
        #                  (2 - frag_ind) * n_states[0], (1 + frag_ind) * n_states[1])
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
        
        full_eigvals, full_eigvec = sort_eigen(np.linalg.eig(full))
        print(np.min(full_eigvals))
        print("relative imag contribution of lowest eigvec of full H with grads with eig", np.linalg.norm(np.imag(full_eigvec[:, 0])) / np.linalg.norm(full_eigvec[:, 0]))

        # now determine which elements of the eigvec to keep
        #full_gs_vec = np.real(full_eigvec[:, 0].reshape((2 - frag_ind) * n_states[0], (1 + frag_ind) * n_states[1]))
        full_gs_vec = np.real(full_eigvec[:, 0].reshape(n_states[0], n_states[1]))

        #############################
        # Filter which states to keep
        #############################

        dens_mat_a = np.einsum("ij,kj->ik", full_gs_vec, full_gs_vec)#np.conj(full_gs_vec))  # contract over frag_b part
        dens_mat_b = np.einsum("ij,ik->jk", full_gs_vec, full_gs_vec)#np.conj(full_gs_vec))  # contract over frag_a part
        
        # the following descending ordering is very important for the later gram-schmidt orthogonalization
        dens_eigvals_a, dens_eigvecs_a = sort_eigen(np.linalg.eigh(dens_mat_a), order="descending")
        dens_eigvals_b, dens_eigvecs_b = sort_eigen(np.linalg.eigh(dens_mat_b), order="descending")
        print(dens_eigvals_a)
        print(dens_eigvals_b)
        dens_eigvals = [dens_eigvals_a, dens_eigvals_b]
        new_large_vecs = [np.real(dens_eigvecs_a), np.real(dens_eigvecs_b)]
        
        # the following threshold is very delicate, because if its
        # too large -> truncation errors
        # too small -> numerical inconsistencies through terms to small to resolve even
        # with double precision (at least I think so...where else should the numerical instability come from?)
        # for this algorithm applied to the Be2 6-31g example something around 3e-9 seems to be the sweet spot
        keepers = [[], []]
        for frag in range(2):
            for i, vec in enumerate(new_large_vecs[frag].T):
                if dens_eigvals[frag][i] >= dens_eigval_thresh:
                    keepers[frag].append(vec)
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

        def conf_decoder(conf):
            ret = []
            for bit in range(n_orbs * 2, -1, -1):
                if conf - 2**bit < 0:
                    continue
                conf -= 2**bit
                ret.append(bit)
            return sorted(ret)
        
        chg_sorted_keepers = [{chg: [] for chg in monomer_charges[frag]} for frag in range(2)]
        for frag in range(2):
            for state in new_vecs[frag]:
                state /= np.linalg.norm(state)
                norms = {chg: np.linalg.norm(state[c_slices[frag][chg]]) for chg in monomer_charges[frag]}
                #print(norms)
                if max(norms.values()) < 0.99:
                    ValueError(f"mixed state encountered (different charges are mixed for a state on frag {frag}), see {norms}")
                # safety measure, but not sure, if it helps
                #big_inds = {ind: abs(elem) for ind, elem in enumerate(state) if abs(elem) > 1e-1}
                #if max(norms, key=norms.get) == 0:
                #    sub = 0
                #elif max(norms, key=norms.get) == 1:
                #    sub = 120
                #else:
                #    sub = 120 + 16
                #print({tuple(conf_decoder(dens_builder_stuff[frag][0][max(norms, key=norms.get)].configs[j - sub])): val for j, val in big_inds.items()})
                #if not big_inds:  #sum(big_inds.values()) < 0.7:
                #    print("optimized state without any larger contributions encountered. These are thrown out for now...")
                #    continue
                chg_sorted_keepers[frag][max(norms, key=norms.get)].append(state[c_slices[frag][max(norms, key=norms.get)]])

        for frag in range(2):
            print(f"fragment {frag}")
            for chg, vecs in chg_sorted_keepers[frag].items():
                print(f"for charge {chg} {len(vecs)} states are kept")
                #chg_sorted_keepers[frag][chg] = [i for i in orthogonalize(np.array(vecs))]
                final_keepers = []
                for vec in orthogonalize(np.array(vecs)):
                    #big_inds = {ind: elem for ind, elem in enumerate(vec) if abs(elem) > 1e-1}
                    #print(frag, chg, i, {tuple(conf_decoder(dens_builder_stuff[frag][0][chg].configs[j])): val for j, val in big_inds.items()})
                    #if not big_inds:
                    #    continue
                    if np.linalg.norm(vec) < 0.99:
                        continue
                    final_keepers.append(vec)
                chg_sorted_keepers[frag][chg] = final_keepers

        for frag in range(2):
            for chg in monomer_charges[frag]:
                dens_builder_stuff[frag][0][chg].coeffs = [i for i in chg_sorted_keepers[frag][chg]]

        for frag in range(2):
            for chg in monomer_charges[frag]:
                for i, vec in enumerate(dens_builder_stuff[frag][0][chg].coeffs):
                    big_inds = {ind: elem for ind, elem in enumerate(vec) if abs(elem) > 1e-1}
                    #print(i, [ind for ind, elem in enumerate(vec) if abs(elem) > 1e-1])
                    print(frag, chg, i, {tuple(conf_decoder(dens_builder_stuff[frag][0][chg].configs[j])): val for j, val in big_inds.items()})

        #raise ValueError("stop here")
        return chg_sorted_keepers, dens_builder_stuff, gs_energy_a, full_eigvals[0]

    def postprocessing(en, en_extended, en_history, en_with_grads_history, converged):
        en_history.append(en)
        en_with_grads_history.append(en_extended)
        #printer(en_history, en_with_grads_history)
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
    
    ######################################################
    # screen for relevant states
    ######################################################
    print("starting screening now")

    additional_states, confs_and_inds = state_screening(dens_builder_stuff, ints, monomer_charges, n_orbs, frozen_orbs, n_occ, n_threads=n_threads)
                                                        #single_thresh=1/3, double_thresh=1/2, triple_thresh=1/1.5)
    """
    remaining_dets = [{}, {}]
    for frag in range(2):
        for chg in monomer_charges[frag]:
            remaining_dets[frag][chg] = []
            for det_ind in range(len(dens_builder_stuff[frag][0][chg].configs)):
                if det_ind not in confs_and_inds[frag][chg].values():
                    vec = np.zeros(len(dens_builder_stuff[frag][0][chg].configs))
                    vec[det_ind] = 1.
                    remaining_dets[frag][chg].append(vec)
                continue
    """
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

    # expanding the state space partially to then reduce and expand again saves a lot of CPU time and memory,
    # but on the other hand some contributions might be lost, since the relevant other state(s) required for
    # a large contribution with an integral might not appear in the same current state space...
    # This could be (partially) circumvented in two ways, which can also be applied both
    # 1. use linear combinations of the basis functions with similar weights for each of the relevant determinants
    # 2. start by only expanding the neutral space in bigger steps, which probably contains most of the
    # correlation with itself and then expand the other charges, which can then be applied in larger chunks
    while safety_iter < 20:#not all(screening_done.flatten()):
        #if safety_iter >= 10:
        #    break
        safety_iter += 1
        for frag in range(2):
            for chg in monomer_charges[frag]:
                print(f"for fragment {frag} with charge {chg} "
                      f"{len(additional_states[frag][chg]) - state_tracker[frag][chg]}"
                      " states still need to be included")
                if len(additional_states[frag][chg]) == state_tracker[frag][chg]:
                    screening_done[frag][abs(min(monomer_charges[frag])) + chg] = True
                    continue
                #n_states = len(dens_builder_stuff[frag][0][chg])
                # the following thresholds are not set in stone
                # since the xr evaluation scales as the fourth order in the number of states
                # we dont want to overdo it here. 20 is still quite acceptable, but maybe not
                # enough, so a relative increase is provided as well, resulting in an expansion
                # by 1/3 leading to an increase in CPU time of roughly a factor of 3 for the XR evaluation.
                # This also caps the amount of densities, which have to be computed at once, which
                # also saves time, as it roughly scales to the second order in the number of states.
                max_states = max(len(dens_builder_stuff[frag][0][chg].coeffs) * 4 // 3, 20)  # maybe increase to 25 or 30
                #print(max_states)
                max_add = min(max_states - len(dens_builder_stuff[frag][0][chg].coeffs), len(additional_states[frag][chg]) - state_tracker[frag][chg])
                #print(state_tracker[frag][chg], state_tracker[frag][chg] + max_add)
                dens_builder_stuff[frag][0][chg].coeffs += additional_states[frag][chg][state_tracker[frag][chg]: state_tracker[frag][chg] + max_add]
                dens_builder_stuff[frag][0][chg].coeffs = [i for i in orthogonalize(np.array(dens_builder_stuff[frag][0][chg].coeffs)) if np.linalg.norm(i) > 0.99]
                state_coeffs_optimized[frag][chg] = dens_builder_stuff[frag][0][chg].coeffs.copy()
                state_tracker[frag][chg] += max_add
            if safety_iter == 1:
                dens.append(densities.build_tensors(*dens_builder_stuff[frag][:-1], options=density_options, n_threads=n_threads))
            else:
                dens[frag] = densities.build_tensors(*dens_builder_stuff[frag][:-1], options=density_options, n_threads=n_threads)

        if all(screening_done.flatten()):
            break

        state_coeffs_optimized, dens_builder_stuff, gs_energy = reduce_screened_state_space(dens_builder_stuff, dens, state_coeffs_optimized)
        screening_energies.append(gs_energy)
        print("energy development during stepwise incorporation of screened states", screening_energies)

    def enlarge_state_space(frag, state_coeffs_optimized, dens_builder_stuff, dens):
        for chg in monomer_charges[frag]:
            print(f"for fragment {frag} with charge {chg} "
                    f"{len(additional_states[frag][chg]) - state_tracker[frag][chg]}"
                    " states still need to be included")
            if len(additional_states[frag][chg]) == state_tracker[frag][chg]:
                screening_done[frag][abs(min(monomer_charges[frag])) + chg] = True
                continue
            #n_states = len(dens_builder_stuff[frag][0][chg])
            # the following thresholds are not set in stone
            # since the xr evaluation scales as the fourth order in the number of states
            # we dont want to overdo it here. 20 is still quite acceptable, but maybe not
            # enough, so a relative increase is provided as well, resulting in an expansion
            # by 1/3 leading to an increase in CPU time of roughly a factor of 3 for the XR evaluation.
            # This also caps the amount of densities, which have to be computed at once, which
            # also saves time, as it roughly scales to the second order in the number of states.
            max_states = max(len(dens_builder_stuff[frag][0][chg].coeffs) * 4 // 3, 20)  # maybe increase to 25 or 30
            #print(max_states)
            max_add = min(max_states - len(dens_builder_stuff[frag][0][chg].coeffs), len(additional_states[frag][chg]) - state_tracker[frag][chg])
            #print(state_tracker[frag][chg], state_tracker[frag][chg] + max_add)
            dens_builder_stuff[frag][0][chg].coeffs += additional_states[frag][chg][state_tracker[frag][chg]: state_tracker[frag][chg] + max_add]
            dens_builder_stuff[frag][0][chg].coeffs = [i for i in orthogonalize(np.array(dens_builder_stuff[frag][0][chg].coeffs)) if np.linalg.norm(i) > 0.99]
            state_coeffs_optimized[frag][chg] = dens_builder_stuff[frag][0][chg].coeffs.copy()
            state_tracker[frag][chg] += max_add
        dens[frag] = densities.build_tensors(*dens_builder_stuff[frag][:-1], options=density_options, n_threads=n_threads)
        return state_coeffs_optimized, dens_builder_stuff, dens

    #for i in range(raw(dens[0]["ca"][(0,0)]).shape[0]):
    #    print(np.diag(raw(dens[0]["ca"][(0,0)][i, i, :, :])))
    #raise ValueError("stop here")
    """
    #pickle.dump(BeN, open("BeN_with_MOs.pkl", mode="wb"))
    #pickle.dump(dens_builder_stuff, open("state_coeffs_and_configs.pkl", mode="wb"))
    mo_coeffs = []
    configs = []
    state_coeffs = []
    for frag in range(2):
        mo_coeffs.append(BeN[frag].basis.MOcoeffs)
        configs.append([])
        state_coeffs.append([])
        for chg in monomer_charges[frag]:
            configs[frag].append(dens_builder_stuff[frag][0][chg].configs)
            state_coeffs[frag].append(dens_builder_stuff[frag][0][chg].coeffs)
    np.save("mo_coeffs.npy", np.array(mo_coeffs))
    np.save("configs.npy", np.array(configs))
    np.save("state_coeffs.npy", np.array(state_coeffs))
    """


    ######################################################
    # iterative procedure
    ######################################################
    if max_iter > 0:
        print("starting iterative state solver now")
        converged = False

        pickle.dump(state_coeffs_optimized, open("pre_opt_coeffs.pkl", mode="wb"))
        for frag in range(2):
            dens.append(densities.build_tensors(*dens_builder_stuff[frag][:-1], options=density_options, n_threads=n_threads))

        # not required as current densities are already computed in last iteration of while loop
        #for frag in range(2):
        #    dens[frag] = densities.build_tensors(*dens_builder_stuff[frag][:-1], options=density_options, n_threads=n_threads)

        en_history, en_with_grads_history = [0], [0]

        #for chg in monomer_charges[1]:
        #    state_coeffs_optimized[1][chg] = ref_states[chg].coeffs.copy()
        #    dens_builder_stuff[1][0][chg].coeffs = ref_states[chg].coeffs.copy()
        #dens[1] = densities.build_tensors(*dens_builder_stuff[1][:-1], options=dens_builder_stuff[1][-1], n_threads=n_threads)
    
    # The following variant tries to optimize one fragment by building its gradients, while simultaneously
    # enlarging the state space of the other fragment with states obtained from the screening. Both are
    # then compressed again.
    iter = 0
    while iter < max_iter:
        iter += 1
        # opt frag 0 and previously enlarge 1
        state_coeffs_optimized, dens_builder_stuff, dens = enlarge_state_space(1, state_coeffs_optimized, dens_builder_stuff, dens)
        state_coeffs_optimized, dens_builder_stuff, gs_energy_a, gs_en_a_with_grads = alternate_enlarge_and_opt(0, dens_builder_stuff, dens, state_coeffs_optimized, dets=additional_states[0])
        converged = postprocessing(gs_energy_a, gs_en_a_with_grads, en_history, en_with_grads_history, converged)

        if all(screening_done.flatten()):
            break

        for frag in range(2):
            dens[frag] = densities.build_tensors(*dens_builder_stuff[frag][:-1], options=dens_builder_stuff[frag][-1], n_threads=n_threads)
        
        # opt frag 1 and previously enlarge 0
        state_coeffs_optimized, dens_builder_stuff, dens = enlarge_state_space(0, state_coeffs_optimized, dens_builder_stuff, dens)
        state_coeffs_optimized, dens_builder_stuff, gs_energy_a, gs_en_a_with_grads = alternate_enlarge_and_opt(1, dens_builder_stuff, dens, state_coeffs_optimized, dets=additional_states[1])
        converged = postprocessing(gs_energy_a, gs_en_a_with_grads, en_history, en_with_grads_history, converged)

        for frag in range(2):
            dens[frag] = densities.build_tensors(*dens_builder_stuff[frag][:-1], options=dens_builder_stuff[frag][-1], n_threads=n_threads)
        
        if all(screening_done.flatten()):
            break
    """
    # The following variant tries to optimize one fragment, while keeping the other fragment as it is,
    # while starting from the optimized guess states obtained above.
    while iter < max_iter:
        iter += 1
        
        #state_coeffs_optimized, dens_builder_stuff, dens, gs_energy_a, gs_en_a_with_grads = iteration(0, dens_builder_stuff, dens, state_coeffs_optimized, dets=remaining_dets[0])# additional_states[0])
        state_coeffs_optimized, dens_builder_stuff, dens, gs_energy_a, gs_en_a_with_grads = iteration(0, dens_builder_stuff, dens, state_coeffs_optimized)#, dets=additional_states[0])
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
        
        #state_coeffs_optimized, dens_builder_stuff, dens, gs_energy_b, gs_en_b_with_grads = iteration(1, dens_builder_stuff, dens, state_coeffs_optimized, dets=remaining_dets[1])# additional_states[1])
        state_coeffs_optimized, dens_builder_stuff, dens, gs_energy_b, gs_en_b_with_grads = iteration(1, dens_builder_stuff, dens, state_coeffs_optimized)#, dets=additional_states[1])
        converged = postprocessing(gs_energy_b, gs_en_b_with_grads, en_history, en_with_grads_history, converged)

        if converged:
            break

        for frag in range(2):  # this should only be necessary for frag 1
            for chg in monomer_charges[frag]:
                dens_builder_stuff[frag][0][chg].coeffs = [i for i in state_coeffs_optimized[frag][chg]]
        dens[1] = densities.build_tensors(*dens_builder_stuff[1][:-1], options=dens_builder_stuff[1][-1], n_threads=n_threads)
    """

    #return en_history #screening_energies
    #return screening_energies
    return screening_energies, BeN, ints, dens, dens_builder_stuff


#scan = []
#for i in range(12):
#    scan.append(optimize_states(3.9 + i / 10, 0, 0))

#scan = [optimize_states(4.5, 20, 0, dens_filter_thresh=1e-6), optimize_states(4.5, 20, 0, dens_filter_thresh=1e-7),
#        optimize_states(4.5, 20, 0, dens_filter_thresh=1e-8), optimize_states(4.5, 20, 0, dens_filter_thresh=1e-9)]
        #optimize_states(4.5, 0, 0, dens_filter_thresh=1e-10)]

#print(scan)

#print(optimize_states(4.5, 0, 0))#, dens_filter_thresh=3e-9))












