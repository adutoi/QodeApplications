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
from qode.math.tensornet import backend_contract_path#, raw, tl_tensor
#import qode.util
from qode.util import timer, sort_eigen, struct
import qode.util
from state_gradients import state_gradients, get_slices#, get_adapted_overlaps
from state_screening import state_screening, orthogonalize, conf_decoder, is_singlet
from excitonic import fci

#import torch
import numpy as np
import tensorly as tl
#import pickle
import scipy as sp

#torch.set_num_threads(4)
#tl.set_backend("pytorch")
#tl.set_backend("numpy")

#from   build_fci_states import get_fci_states
#import densities

tl.plugins.use_opt_einsum()
backend_contract_path(True)

#class empty(object):  pass

def optimize_states(max_iter, xr_order, dens_builder_stuff, ints, n_occ, n_orbs, frozen_orbs, additional_states,
                    monomer_charges, monomer_multiplicities,
                    conv_thresh=1e-6, dens_filter_thresh=1e-7, begin_from_state_prep=False, grad_level="herm", target_state=0,
                    density_options=["compress=SVD,cc-aa"], n_threads=1):
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

    ############################################################
    # State-space dictionaries
    ############################################################

    def get_state_dict(dens_builder_stuff):
        return [
            {
                chg: {
                    mult: len(
                        dens_builder_stuff[frag][0][chg][mult].coeffs
                    )
                    for mult in monomer_multiplicities[frag][chg]
                }
                for chg in monomer_charges[frag]
            }
            for frag in range(2)
        ]
    
    state_coeffs_og = get_state_dict(dens_builder_stuff)

    ############################################################
    # Build the local sector dictionaries and projectors
    ############################################################

    def get_combined_sector_dict(frag, frag_ind, state_dict, grad_dict):
        """
        Number of local basis states in each charge/multiplicity
        sector.

        For the gradient fragment, the local basis consists of
        ordinary states followed by gradient states in each sector.
        """

        ret = {}

        for chg in monomer_charges[frag]:
            ret[chg] = {}

            for mult in monomer_multiplicities[frag][chg]:

                n_state = state_dict[frag][chg][mult]

                if frag == frag_ind:
                    n_grad = grad_dict[chg][mult]
                else:
                    n_grad = 0

                ret[chg][mult] = n_state + n_grad

        return ret

    def get_sector_slices(sector_dict):
        ret = {}
        offset = 0

        for chg, mult_dict in sector_dict.items():
            ret[chg] = {}

            for mult, n in mult_dict.items():
                ret[chg][mult] = slice(
                    offset,
                    offset + n,
                )
                offset += n

        return ret

    ############################################################
    # Filter the fragment density matrices
    ############################################################

    def filter_fragment_density(
            rho,
            frag,
            orth_eps,
            sector_slices,
        ):
        """
        Filter one fragment density.

        The local basis is ordered as

            [charge][multiplicity][state]

        and, for the gradient fragment, the state dimension in
        each sector already includes the gradient states.

        Returns state-space vectors, still expressed in the
        original local state/gradient basis.
        """

        filtered = {
            chg: {
                mult: []
                for mult in monomer_multiplicities[frag][chg]
            }
            for chg in monomer_charges[frag]
        }

        for chg in monomer_charges[frag]:

            for mult in monomer_multiplicities[frag][chg]:

                sl = sector_slices[frag][chg][mult]

                rho_sector = rho[
                    sl,
                    sl,
                ]

                rho_sector = 0.5 * (
                    rho_sector + rho_sector.T
                )

                eigvals, eigvecs = sp.linalg.eigh(
                    rho_sector
                )

                keep = eigvals > dens_filter_thresh

                eigvecs = eigvecs[:, keep]

                if eigvecs.shape[1] == 0:
                    continue

                eigvecs = orthogonalize(
                    eigvecs.T,
                    eps=orth_eps,
                )

                filtered[chg][mult] = [
                    vec
                    for vec in eigvecs
                ]

        return filtered

    ############################################################
    # Map filtered local states back to the CSF basis
    ############################################################

    def get_local_coefficient_matrix(frag, chg, mult, frag_ind, dens_builder_stuff, gradient_states):
        """
        Coefficient matrix for the complete temporary local basis.

        Rows are ordered exactly as the corresponding FCI basis:
            ordinary states
            followed by gradient states

        Returns:
            n_local_states × n_csfs
        """

        C_state = np.asarray(
            dens_builder_stuff[frag][0][chg][mult].coeffs
        )

        if frag != frag_ind:
            return C_state

        C_grad = np.asarray(
            gradient_states[chg][mult]
        )

        return np.concatenate(
            (C_state, C_grad),
            axis=0,
        )

    ###########
    # Filter and map fragment dens mats
    ###########
    
    def filter_and_map_dens_mats(dens_mats, sector_slices):
        new_state_coeffs = [
            {
                chg: {
                    mult: []
                    for mult in monomer_multiplicities[frag][chg]
                }
                for chg in monomer_charges[frag]
            }
            for frag in range(2)
        ]


        for frag in range(2):

            filtered = filter_fragment_density(
                dens_mats[frag],
                frag,
                dens_filter_thresh,
                sector_slices,
            )

            for chg in monomer_charges[frag]:

                for mult in monomer_multiplicities[frag][chg]:

                    state_vecs = filtered[chg][mult]

                    if len(state_vecs) == 0:
                        continue

                    state_vecs = np.asarray(
                        state_vecs
                    )

                    C = get_local_coefficient_matrix(
                        frag,
                        chg,
                        mult,
                    )

                    print(f"for frag {frag}, chg {chg}, mult {mult} {state_vecs.shape[1]} states are kept")

                    #
                    # New states in the CSF basis.
                    #
                    new_C = np.einsum(
                        "pi,pq->iq",
                        state_vecs,
                        C,
                    )

                    #
                    # Remove numerical linear dependencies.
                    #
                    new_C = orthogonalize(
                        new_C,
                        eps=dens_filter_thresh,
                    )

                    new_state_coeffs[frag][chg][mult] = [
                        vec
                        for vec in new_C
                    ]

        return new_state_coeffs

    
    
    def reduce_screened_state_space(dens_builder_stuff, dens, state_coeffs, target_state=target_state):
        state_dict = get_state_dict(dens_builder_stuff)

        n_states = [
            sum(
                dim
                for mult_dict in state_dict[frag].values()
                for dim in mult_dict.values()
            ) for frag in range(2)
        ]

        sector_slices = get_sector_slices(state_dict)

        H1, H2 = get_xr_H(ints, dens, xr_order, monomer_charges)

        out = struct(log=qode.util.textlog(echo=True))
        E, full_eigvec_l, full_eigvec_r = fci(
            (
                [H1[0], H1[1]],
                [[None, H2],
                [None, None]],
            ),
            out,
            target_state=target_state,
            get_left_and_right = True,
        )

        if np.linalg.norm(np.imag(full_eigvec_r[:, target_state])) > 1e-10:
            raise ValueError("imaginary part of eigenvector is not negligible")

        #############################
        # Filter which states to keep
        #############################

        if type(target_state) == int:
            target_state = [target_state]

        # it would be correct here to also use the left eigvec, but it was found that only the right eigvec
        # yields similar or even better results with much more compact state spaces.
        # TODO: an open question remains how the performance differs for stronger interacting fragments
        full_eigvecs = [full_eigvec_r]#, full_eigvec_l]
        target_vecs = [np.real(full_eigvecs[lr][:, i].reshape((n_states[0], n_states[1]))) for i in range(len(target_state))
                        for lr in range(len(full_eigvecs))]

        dens_mats = [[np.einsum("ij,kj->ik", vec, vec) for vec in target_vecs],  # contract over frag_b part
                     [np.einsum("ij,ik->jk", vec, vec) for vec in target_vecs]]  # contract over frag_a part

        new_state_coeffs = filter_and_map_dens_mats(dens_mats, sector_slices)

        for frag in range(2):
            for chg in monomer_charges[frag]:
                for mult in monomer_multiplicities[frag][chg]:
                    dens_builder_stuff[frag][0][chg][mult].coeffs = [i for i in new_state_coeffs[frag][chg][mult]]

        #for frag in range(2):
        #    for chg in monomer_charges[frag]:
        #        for i, vec in enumerate(dens_builder_stuff[frag][0][chg].coeffs):
        #            big_inds = {ind: elem for ind, elem in enumerate(vec) if abs(elem) > 1e-1}
        #            print(frag, chg, i, {tuple(conf_decoder(dens_builder_stuff[frag][0][chg].configs[j], n_orbs)): val for j, val in big_inds.items()})

        return new_state_coeffs, dens_builder_stuff, E
    
    def alternate_enlarge_and_opt(
            frag_ind,
            dens_builder_stuff,
            dens,
            state_coeffs,
            backend,
            ints,
            monomer_charges,
            target_state=target_state,
            grad_level=grad_level,
            dets=None):

        print(f"opt frag {frag_ind}")

        if not isinstance(target_state, int):
            if len(target_state) > 1:
                raise NotImplementedError(
                    "Gradient based optimization is currently only implemented "
                    "for a single target state"
                )

        ###########################################################
        # Obtain gradients
        ###########################################################

        gs_energy_a, gradient_states_a, dl_prev, dr_prev = state_gradients(
            frag_ind,
            ints,
            dens_builder_stuff,
            dens,
            monomer_charges,
            n_threads=n_threads,
            xr_order=xr_order,
            grad_level=grad_level,
            target_state=target_state,
        )

        #gradient_states_a = {
        #    chg: np.asarray(grad, dtype=float)
        #    for chg, grad in gradient_states_a.items()
        #}

        ###########################################################
        # Enlarge fragment state space with gradients
        ###########################################################

        #a_coeffs = {
        #    chg: np.asarray(state_coeffs[frag_ind][chg], dtype=float).copy()
        #    for chg in monomer_charges[frag_ind]
        #}

        tot_a_coeffs = {
            chg: {
                mult: np.asarray(
                    orthogonalize(
                        np.vstack((
                            state_coeffs[frag_ind][chg][mult],
                            gradient_states_a[chg][mult],
                        ))
                    )
                )
                for mult in monomer_multiplicities[frag_ind][chg]
            }
            for chg in monomer_charges[frag_ind]
        }

        state_dict = get_state_dict(dens_builder_stuff)


        ############################################################
        # Gradient-space dictionaries
        ############################################################

        grad_dict = {
            chg: {
                mult: len(gradient_states_a[chg][mult])
                for mult in monomer_multiplicities[frag_ind][chg]
            }
            for chg in monomer_charges[frag_ind]
        }

        n_states = []

        for frag in range(2):

            n_state = sum(
                dim
                for mult_dict in state_dict[frag].values()
                for dim in mult_dict.values()
            )

            if frag == frag_ind:
                n_grad = sum(
                    dim
                    for mult_dict in grad_dict.values()
                    for dim in mult_dict.values()
                )
                n_states.append(n_state + n_grad)
            else:
                n_states.append(n_state)


        combined_sector_dict = [
            get_combined_sector_dict(frag)
            for frag in range(2)
        ]
        

        sector_slices = [
            get_sector_slices(combined_sector_dict[frag])
            for frag in range(2)
        ]

        

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

        out = struct(log=qode.util.textlog(echo=True))
        E, full_eigvec_l, full_eigvec_r = fci(
            (
                [H1[0], H1[1]],
                [[None, H2],
                [None, None]],
            ),
            out,
            target_state=target_state,
            get_left_and_right = True,
        )

        if np.linalg.norm(np.imag(full_eigvec_r[:, target_state])) > 1e-10:
            raise ValueError("imaginary part of eigenvector is not negligible")

        #############################
        # Filter which states to keep
        #############################

        if type(target_state) == int:
            target_state = [target_state]

        # it would be correct here to also use the left eigvec, but it was found that only the right eigvec
        # yields similar or even better results with much more compact state spaces.
        # TODO: an open question remains how the performance differs for stronger interacting fragments
        full_eigvecs = [full_eigvec_r]#, full_eigvec_l]
        target_vecs = [np.real(full_eigvecs[lr][:, i].reshape((n_states[0], n_states[1]))) for i in range(len(target_state))
                       for lr in range(len(full_eigvecs))]

        dens_mats = [[np.einsum("ij,kj->ik", vec, vec) for vec in target_vecs],  # contract over frag_b part
                     [np.einsum("ij,ik->jk", vec, vec) for vec in target_vecs]]  # contract over frag_a part

        new_state_coeffs = filter_and_map_dens_mats(dens_mats, sector_slices)

        for frag in range(2):
            for chg in monomer_charges[frag]:
                for mult in monomer_multiplicities[frag][chg]:
                    dens_builder_stuff[frag][0][chg][mult].coeffs = [i for i in new_state_coeffs[frag][chg][mult]]

        #for frag in range(2):
        #    for chg in monomer_charges[frag]:
        #        for i, vec in enumerate(dens_builder_stuff[frag][0][chg].coeffs):
        #            big_inds = {ind: elem for ind, elem in enumerate(vec) if abs(elem) > 1e-1}
        #            print(frag, chg, i, {tuple(conf_decoder(dens_builder_stuff[frag][0][chg].configs[j], n_orbs)): val for j, val in big_inds.items()})

        return new_state_coeffs, dens_builder_stuff, gs_energy_a, E

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
            for mult in monomer_multiplicities[frag][chg]:
                print(f"for fragment {frag} with charge {chg} and mult {mult} "
                        f"{len(additional_states[frag][chg][mult]) - state_tracker[frag][chg][mult]}"
                        " states still need to be included")
                if len(additional_states[frag][chg][mult]) == state_tracker[frag][chg][mult]:
                    screening_done[frag][chg][mult] = True
                    continue
                # the following thresholds are not set in stone
                # since the xr evaluation scales as the fourth order in the number of states
                # we dont want to overdo it here. 20 per charge is still quite acceptable, but maybe not
                # enough, so a relative increase is provided as well, resulting in an expansion
                # by 1/3 leading to an increase in CPU time of roughly a factor of 3 for the XR evaluation.
                # This also caps the amount of densities, which have to be computed at once, which
                # also saves time, as it roughly scales to the second order in the number of states.
                max_states = max(len(dens_builder_stuff[frag][0][chg][mult].coeffs) * 4 // 3, 15)  # TODO: play with the threshold a little
                max_add = min(max_states - len(dens_builder_stuff[frag][0][chg][mult].coeffs), len(additional_states[frag][chg][mult]) - state_tracker[frag][chg][mult])
                # using the following only makes sense, if sorting in state screening is active, but that was found to be ineffective
                #conf_ind = np.argmax(additional_states[frag][chg][state_tracker[frag][chg] + max_add - 1])
                #conf_ind_pre = np.argmax(additional_states[frag][chg][state_tracker[frag][chg] + max_add - 2])
                #singlet, pair = is_singlet(conf_decoder(dens_builder_stuff[frag][0][chg].configs[conf_ind], n_orbs), n_orbs)
                #if pair != dens_builder_stuff[frag][0][chg].configs[conf_ind_pre] and max_add < len(additional_states[frag][chg]) - state_tracker[frag][chg]:
                #    max_add += 1
                dens_builder_stuff[frag][0][chg][mult].coeffs += additional_states[frag][chg][mult][state_tracker[frag][chg][mult]: state_tracker[frag][chg][mult] + max_add]
                dens_builder_stuff[frag][0][chg][mult].coeffs = [i for i in orthogonalize(np.array(dens_builder_stuff[frag][0][chg][mult].coeffs))]
                state_coeffs_optimized[frag][chg][mult] = dens_builder_stuff[frag][0][chg][mult].coeffs.copy()
                state_tracker[frag][chg][mult] += max_add
        dens[frag] = densities.build_tensors(*dens_builder_stuff[frag][:-1], options=density_options, n_threads=n_threads)
        #from qode.math.tensornet import raw
        #print(raw(dens[0]["ca"][(0,0)])[0, 0, 1:9, 1:9])
        #print(raw(dens[0]["a"][(1,0)])[0, 0, :])
        #print(raw(dens[0]["a"][(1,0)])[1, 0, :])
        #print(raw(dens[0]["a"][(1,0)])[2, 0, :])
        #print(np.linalg.norm(raw(dens[0]["ca"][(0,0)])[0, 0, :, :]))
        #caa_00 = raw(dens[0]["caa"][(1,0)])[0, 0, :, :, :]
        #print(np.linalg.norm(ccaa_00))
        #print(np.linalg.norm(ccaa_00[[0,9], [0,9], [0,9], [0,9]]))
        #print(np.linalg.norm(ccaa_00[1:9, 1:9, 1:9, 1:9]))
        #print(np.linalg.norm(ccaa_00[10:18, 1:9, 10:18, 1:9]))
        #print(np.linalg.norm(ccaa_00[1:9, 10:18, 10:18, 1:9]))
        #sh = caa_00.shape
        #for i in range(sh[0]):
        #    for j in range(sh[1]):
        #        for k in range(sh[2]):
        #            val = caa_00[i,j,k]
        #            if abs(val) > 0.1:
        #                print(i,j,k,val)
        #print(raw(dens[0]["ccaa"][(0,0)])[0, 0, 0:2, 0:2, 0:2, 0:2])
        #tmp_ccaa = raw(dens[0]["ccaa"][(0,0)])[0, 0]
        #idx = [0, 9]
        #print(tmp_ccaa[np.ix_(idx, idx, idx, idx)])
        #raise ValueError("stop here")
        return state_coeffs_optimized, dens_builder_stuff, dens
    

    state_tracker = {frag: 
                        {chg: 
                            {mult: 0 for mult in monomer_multiplicities[frag][chg]}
                        for chg in monomer_charges[frag]}
                    for frag in range(2)}
    screening_done = np.array([
                                [
                                    [False for mult in monomer_multiplicities[frag][chg]]
                                for chg in monomer_charges[frag]]
                            for frag in range(2)])
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
