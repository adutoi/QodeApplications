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

import numpy as np
import densities
from get_xr_result import get_xr_states, get_xr_H
from qode.util import sort_eigen
import tensorly as tl
from qode.math.tensornet import tl_tensor
import scipy as sp

def get_adapted_overlaps(frag_map, dl, dr, d_slices):
    # TODO: The following only works, if both fragments have the same charges, which are symmetrically sampled around zero
    if sum(d_slices[0].keys()) != 0 or sum(d_slices[1].keys()) != 0:
        raise ValueError("charges are not symmtrically sampled around zero, as currently required by this function")
    overlaps = {}
    for chg in d_slices[frag_map[1]].keys():
        overlaps[chg * (-1)] = np.einsum("kl,il->ki", dl[d_slices[frag_map[0]][chg * (-1)], d_slices[frag_map[1]][chg]],
                                                      dr[d_slices[frag_map[0]][chg * (-1)], d_slices[frag_map[1]][chg]])
    return overlaps

def contract_dens_with_d(dens, dl, dr, frag_map, d_slices, state_dict):  # updates dens in place
    #beware, that the following only works, if both fragments have the same charges, which are symmetrically sampled around zero
    if sum(d_slices[0].keys()) != 0 or sum(d_slices[1].keys()) != 0:
        raise ValueError("charges are not symmtrically sampled around zero, as currently required by this function")
    for op_string in dens:
        if "n_" in op_string:
            continue
        for bra_chg,ket_chg in dens[op_string]:
            dens_inds = [num for num in range(2, len(dens[op_string][(bra_chg,ket_chg)].shape))]
            d_left  = tl_tensor(tl.tensor(dl[d_slices[frag_map[0]][bra_chg * (-1)], d_slices[frag_map[1]][bra_chg]], dtype=tl.float64))
            d_right = tl_tensor(tl.tensor(dr[d_slices[frag_map[0]][ket_chg * (-1)], d_slices[frag_map[1]][ket_chg]], dtype=tl.float64))
            dens[op_string][(bra_chg,ket_chg)] = d_left(0,"i") @ dens[op_string][(bra_chg,ket_chg)]("i","j",*dens_inds) @ d_right(1,"j")
    for chg in d_slices[frag_map[0]]:
        dens["n_states"][chg] = state_dict[frag_map[0]][chg * (-1)]

def get_slices(dict, chgs, append=False, type="standard"):
    if type != "standard" and not append:
        raise ValueError("all types except standard require a dictionary provided by the append keyword argument")
    dummy_ind = 0
    ret = {}
    for chg in chgs:
        if type == "standard":
            ret[chg] = slice(dummy_ind, dummy_ind+dict[chg])
            dummy_ind += dict[chg]
        #elif type == "double":
        #    ret[chg] = slice(dummy_ind, dummy_ind + 2 * dict[chg])
        #    dummy_ind += 2 * dict[chg]
        elif type == "first":
            ret[chg] = slice(dummy_ind, dummy_ind + dict[chg])
            dummy_ind += append[chg] + dict[chg]
        elif type == "latter":
            ret[chg] = slice(dummy_ind + dict[chg], dummy_ind + append[chg] + dict[chg])
            dummy_ind += append[chg] + dict[chg]
        else:
            raise ValueError(f"type {type} is unknown")
    return ret

def get_gs(current_state_dict, d_sl, H1_new, H2_new, monomer_charges, target_state, grad_level):
        #new_gs_en, new_gs_vec = get_xr_states(ints, dens, 0)  # TODO: This is not possible, until solvers are fixed
        #H1_new, H2_new = get_xr_H(ints, dens, xr_order)
        n_states = [sum(current_state_dict[i][chg] for chg in monomer_charges[i]) for i in range(2)]
        H2_new = H2_new.reshape(n_states[0], n_states[1],
                                n_states[0], n_states[1])
        for chg0 in monomer_charges[0]:
            for chg1 in monomer_charges[1]:
                H2_new[d_sl[0][chg0], d_sl[1][chg1], d_sl[0][chg0], d_sl[1][chg1]] +=\
                    np.einsum("ij,kl->ikjl", H1_new[0][d_sl[0][chg0], d_sl[0][chg0]], np.eye(current_state_dict[1][chg1])) +\
                    np.einsum("ij,kl->ikjl", np.eye(current_state_dict[0][chg0]), H1_new[1][d_sl[1][chg1], d_sl[1][chg1]])
        H2_new = H2_new.reshape(n_states[0] * n_states[1],
                                n_states[0] * n_states[1])
        full_eigvals_raw, full_eigvec_l_unsorted, full_eigvec_r_unsorted = sp.linalg.eig(H2_new, left=True, right=True)
        full_eigvals, full_eigvec_l = sort_eigen((full_eigvals_raw, full_eigvec_l_unsorted))
        full_eigvals_check, full_eigvec_r = sort_eigen((full_eigvals_raw, full_eigvec_r_unsorted))
        if any(full_eigvals - full_eigvals_check):  # this check should be unnecessary and might need to be refined for fully degenerate states
            raise ValueError("something went wrong with the sorting")
        #return full_eigvals[0], H2_new, full_eigvec_l[:, 0].T / np.linalg.norm(full_eigvec_l[:, 0]), full_eigvec_r[:, 0].T
        if "herm" in grad_level:
            print("currently state gradients are build with only dr and not dl")
            return full_eigvals[target_state], H2_new, full_eigvec_r[:, target_state].T, full_eigvec_r[:, target_state].T
        else:
            print("currently state gradients are build with dl and dr")
            return full_eigvals[target_state], H2_new, full_eigvec_l[:, target_state].T / np.linalg.norm(full_eigvec_l[:, target_state]), full_eigvec_r[:, target_state].T

def state_gradients(frag_ind, ints, dens_builder_stuff, dens, monomer_charges, n_threads=1, xr_order=0, dets={}, grad_level="herm", target_state=0):
    if xr_order != 0:
        raise NotImplementedError("gradients for higher orders in S than 0 are not implemented yet")

    if frag_ind == 0:
        frag_map = {0: 0, 1: 1}
    elif frag_ind == 1:
        frag_map = {0: 1, 1: 0}
    else:
        raise IndexError("for dimer interactions only frag_ind 0 and 1 are acceptable")

    state_dict = [{chg: len(dens_builder_stuff[i][0][chg].coeffs) for chg in monomer_charges[i]} for i in range(2)]
    if dets:
        conf_dict = [{chg: len(dets[chg]) for chg in monomer_charges[i]} for i in range(2)]
    else:
        conf_dict = [{chg: len(dens_builder_stuff[i][0][chg].coeffs[0]) for chg in monomer_charges[i]} for i in range(2)]

    d_slices = [get_slices(state_dict[i], monomer_charges[i]) for i in range(2)]
    c_slices = [get_slices(conf_dict[i], monomer_charges[i]) for i in range(2)] 

    state_coeffs = [{chg: np.array(states.coeffs) for chg, states in dens_builder_stuff[frag][0].items()} for frag in range(len(dens_builder_stuff))]

    #gs_energy, gs_state = get_xr_states(ints, dens, xr_order)
    H1_for_d, H2_for_d = get_xr_H(ints, dens, xr_order, monomer_charges)
    gs_energy, H_no_dets, dl, dr = get_gs(state_dict, d_slices, H1_for_d, H2_for_d, monomer_charges, target_state, grad_level)
    E = np.real(gs_energy)
    #dl, dr = gs_state
    print("dropping imaginary part of non-diagonalized d with norm", np.linalg.norm(np.imag(dl)), np.linalg.norm(np.imag(dr)))
    dl = np.real(dl).reshape(sum(state_dict[0].values()), sum(state_dict[1].values()))
    dr = np.real(dr).reshape(sum(state_dict[0].values()), sum(state_dict[1].values()))
    if frag_ind == 1:
        dl, dr = dl.T, dr.T

    # normalize d
    # careful, this doesn't yield <Psi_D|Psi_D> = 1, because dl @ dr.T != 1
    # also dr should already be normalized and dl should be normalized from get_gs()
    dl = dl / np.linalg.norm(dl)
    dr = dr / np.linalg.norm(dr)
    #dl_extra = dl_extra / np.linalg.norm(dl_extra)
    
    # build new densities (slater det densities for fragment under optimization and contract_with_d densities for the other one)
    # decomposing the following densities sometimes yields errors, so dont decompose them for now
    print(f"contract densities on fragment {frag_map[1]} with d")
    # alternatively save to new object (less CPU time, but more RAM, since densities dont need to be rebuild)
    contract_dens_with_d(dens[frag_map[1]], dl, dr, frag_map, d_slices, state_dict)
    print(f"build densities between slater determinant and state on fragment {frag_map[0]}")
    # maybe its a good idea to contract d with the ket states here, because it doesnt affect how one has to deal with the
    # determinant to state densities, but maybe eases up the building of them, because it might put a lot more elements
    # of the state below the threshold indicating whether the transition density for this element should even be built or not.
    dens[frag_map[0]] = densities.build_tensors(*dens_builder_stuff[frag_map[0]][:-1], options=dens_builder_stuff[frag_map[0]][-1] + ["bra_det"], n_threads=n_threads, dets=dets)

    # build gradient
    H1, H2 = get_xr_H(ints, dens, xr_order, monomer_charges, bra_det=True)

    print(H1[0].shape, H1[1].shape, H2.shape)
    H2 = H2.reshape((H1[0].shape[0], H1[1].shape[0]))#, H1[0].shape[1], H1[1].shape[1]))

    # the following is the contribution <psi psi | H - E | phi psi>
    if "full" in grad_level:
        # TODO: this is a waste...one can also obtain the corresponding grad terms from biorthogonalizing the kets of the integrals
        # instead of the bras, which is much faster than building the new densities
        dens[frag_map[0]] = densities.build_tensors(*dens_builder_stuff[frag_map[0]][:-1], options=dens_builder_stuff[frag_map[0]][-1] + ["ket_det"], n_threads=n_threads, dets=dets)
        H1_new, H2_new = get_xr_H(ints, dens, xr_order, monomer_charges, ket_det=True)
        print(H1_new[0].shape, H1_new[1].shape, H2_new.shape)
        H2_new = H2_new.reshape((H1_new[0].shape[1], H1_new[1].shape[1]))
    
    # H1 of frag A can be used as is and H1 of frag B needs to be contracted with the state coeffs of frag A. Note, that this is independent of the XR order 
    gradient_states = {}
    new_overlaps = get_adapted_overlaps(frag_map, dl, dr, d_slices)

    # intermediates for S^{-1} terms left
    if grad_level == "full":
        H_dr = np.einsum("ik,k->i", H_no_dets, dr.reshape((H_no_dets.shape[1])))
        dl_partial_H_dr = np.einsum("il,ml->im", dl, H_dr.reshape(dl.shape))

    # intermediates for S^{-1} terms right
    if grad_level == "full":
        H_dl = np.einsum("ik,i->k", H_no_dets, dl.reshape((H_no_dets.shape[0])))
        dl_H_dr_partial = np.einsum("il,ml->im", H_dl.reshape(dl.shape), dr)
    
    for chg in monomer_charges[frag_map[0]]:
        if dets:
            # compress state to smaller configuration space
            c0 = np.einsum("ip,qp->iq", state_coeffs[frag_map[0]][chg], np.array(dets[chg])) 
        else:
            c0 = state_coeffs[frag_map[0]][chg]
        gradient_states[chg] = np.einsum("pi,ki->kp", H1[frag_map[0]][c_slices[frag_map[0]][chg], d_slices[frag_map[0]][chg]], new_overlaps[chg])  # frag A monomer H term
        gradient_states[chg] -= np.einsum("ip,ki->kp", c0, new_overlaps[chg]) * E  # E term
        if frag_ind == 0:
            gradient_states[chg] += H2[c_slices[frag_map[0]][chg], d_slices[frag_map[0]][chg]].T  # dimer H term
        else:
            gradient_states[chg] += H2[d_slices[frag_map[0]][chg], c_slices[frag_map[0]][chg]]  # dimer H term
        # this line also relies on equal charges on fragments A and B
        gradient_states[chg] += np.einsum("ip,ki->kp", c0, H1[frag_map[1]][d_slices[frag_map[0]][chg], d_slices[frag_map[0]][chg]])  # frag B monomer H term
        #if not "full" in grad_level:
        #    gradient_states[chg] *= 2

        # TODO: check for stronger interacting systems whether this should be the one term for the "herm" gradient option
        # the following is the contribution <psi psi | H - E | phi psi>
        if "full" in grad_level:
            gradient_states[chg] += np.einsum("ip,ki->kp", H1_new[frag_map[0]][d_slices[frag_map[0]][chg], c_slices[frag_map[0]][chg]], new_overlaps[chg])  # frag A monomer H term
            gradient_states[chg] -= np.einsum("ip,ik->kp", c0, new_overlaps[chg]) * E  # E term
            if frag_ind == 0:
                gradient_states[chg] += H2_new[c_slices[frag_map[0]][chg], d_slices[frag_map[0]][chg]].T  # dimer H term
            else:
                gradient_states[chg] += H2_new[d_slices[frag_map[0]][chg], c_slices[frag_map[0]][chg]]  # dimer H term
            # this line also relies on equal charges on fragments A and B
            gradient_states[chg] += np.einsum("ip,ki->kp", c0, H1_new[frag_map[1]][d_slices[frag_map[0]][chg], d_slices[frag_map[0]][chg]])  # frag B monomer H term
            
        # the following are the two terms from the derivative with respect to S^{-1} from the left
        if grad_level == "full":
            gradient_states[chg] -= np.einsum("mp,im->ip", c0, dl_partial_H_dr[d_slices[frag_map[0]][chg], d_slices[frag_map[0]][chg]])
            gradient_states[chg] -= np.einsum("kp,ki->ip", c0, dl_partial_H_dr[d_slices[frag_map[0]][chg], d_slices[frag_map[0]][chg]])

        # the following are the two terms from the derivative with respect to S^{-1} from the right
        if grad_level == "full":
            gradient_states[chg] -= np.einsum("im,mp->ip", dl_H_dr_partial[d_slices[frag_map[0]][chg], d_slices[frag_map[0]][chg]], c0)
            gradient_states[chg] -= np.einsum("ki,kp->ip", dl_H_dr_partial[d_slices[frag_map[0]][chg], d_slices[frag_map[0]][chg]], c0)
        
    return gs_energy, gradient_states, dl, dr

