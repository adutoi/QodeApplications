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

"""
def contract_mon_with_d(frag_map, dens_builder_stuff, d, state_coeffs, d_slices):  # updates states.coeffs in the dens_builder_stuff object...original coeffs can be found in state_coeffs
    #beware, that the following only works, if both fragments have the same charges, which are symmetrically sampled around zero
    if sum(d_slices[0].keys()) != 0 or sum(d_slices[1].keys()) != 0:
        raise ValueError("charges are not symmtrically sampled around zero...this can be fixed though, by changing this function")
    for chg in d_slices[frag_map[1]].keys():
        adapted_coeffs = np.tensordot(d[d_slices[frag_map[0]][chg * (-1)], d_slices[frag_map[1]][chg]],
                                      state_coeffs[frag_map[1]][chg], axes=([1], [0]))
        dens_builder_stuff[frag_map[1]][0][chg].coeffs = [i for i in adapted_coeffs]  # dont change charge in dens builder here
"""

def get_adapted_overlaps(frag_map, d, d_slices):
    #beware, that the following only works, if both fragments have the same charges, which are symmetrically sampled around zero
    if sum(d_slices[0].keys()) != 0 or sum(d_slices[1].keys()) != 0:
        raise ValueError("charges are not symmtrically sampled around zero...this can be fixed though, by changing this function")
    overlaps = {}
    for chg in d_slices[frag_map[1]].keys():
        #adapted_coeffs = np.tensordot(d[d_slices[frag_map[0]][chg * (-1)], d_slices[frag_map[1]][chg]],
        #                              state_coeffs[frag_map[1]][chg], axes=([1], [0]))
        #overlaps[chg * (-1)] = np.array([[np.dot(i,j) for j in adapted_coeffs] for i in adapted_coeffs])
        overlaps[chg * (-1)] = np.einsum("kl,il->ki", d[d_slices[frag_map[0]][chg * (-1)], d_slices[frag_map[1]][chg]],
                                                      d[d_slices[frag_map[0]][chg * (-1)], d_slices[frag_map[1]][chg]])
    return overlaps

def contract_dens_with_d(dens, d, frag_map, d_slices, state_dict):  # updates dens in place
    #beware, that the following only works, if both fragments have the same charges, which are symmetrically sampled around zero
    if sum(d_slices[0].keys()) != 0 or sum(d_slices[1].keys()) != 0:
        raise ValueError("charges are not symmtrically sampled around zero...this can be fixed though, by changing this function")
    for op_string in dens:
        if "n_" in op_string:
            continue
        for bra_chg,ket_chg in dens[op_string]:
            dens_inds = [num for num in range(2, len(dens[op_string][(bra_chg,ket_chg)].shape))]
            d_left  = tl_tensor(tl.tensor(d[d_slices[frag_map[0]][bra_chg * (-1)], d_slices[frag_map[1]][bra_chg]], dtype=tl.float64))
            d_right = tl_tensor(tl.tensor(d[d_slices[frag_map[0]][ket_chg * (-1)], d_slices[frag_map[1]][ket_chg]], dtype=tl.float64))
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

def get_gs(current_state_dict, d_sl, H1_new, H2_new, monomer_charges):
        #new_gs_en, new_gs_vec = get_xr_states(ints, dens, 0)  # this is not possible, until solvers are fixed
        #H1_new, H2_new = get_xr_H(ints, dens, xr_order)
        n_states = [sum(current_state_dict[i][chg] for chg in monomer_charges[i]) for i in range(2)]
        H2_new = H2_new.reshape(n_states[0], n_states[1],
                                n_states[0], n_states[1])
        #current_state_dict = [{chg: len(current_state_coeffs[i][chg]) for chg in monomer_charges[i]} for i in range(2)]
        #d_sl = [get_slices(current_state_dict[i], monomer_charges[i]) for i in range(2)]
        #print(d_sl, n_states, H2_new.shape, H1_new[0].shape, H1_new[1].shape)
        for chg0 in monomer_charges[0]:
            for chg1 in monomer_charges[1]:
                H2_new[d_sl[0][chg0], d_sl[1][chg1], d_sl[0][chg0], d_sl[1][chg1]] +=\
                    np.einsum("ij,kl->ikjl", H1_new[0][d_sl[0][chg0], d_sl[0][chg0]], np.eye(current_state_dict[1][chg1])) +\
                    np.einsum("ij,kl->ikjl", np.eye(current_state_dict[0][chg0]), H1_new[1][d_sl[1][chg1], d_sl[1][chg1]])
        H2_new = H2_new.reshape(n_states[0] * n_states[1],
                                n_states[0] * n_states[1])
        new_ens, new_states = sort_eigen(np.linalg.eig(H2_new))
        return new_ens[0], new_states[:, 0].T

def state_gradients(frag_ind, ints, dens_builder_stuff, dens, monomer_charges, n_threads=1, xr_order=0, dets={}):
    if xr_order != 0:
        raise NotImplementedError("gradients for higher orders in S than 0 are not implemented yet")

    if frag_ind == 0:
        frag_map = {0: 0, 1: 1}
    elif frag_ind == 1:
        frag_map = {0: 1, 1: 0}
    else:
        raise IndexError("for dimer interactions only frag_ind 0 and 1 are acceptable")

    state_dict = [{chg: len(dens_builder_stuff[i][0][chg].coeffs) for chg in monomer_charges[i]} for i in range(2)]
    #state_coeffs = [{chg: dens_builder_stuff[frag_map[i]][0][chg].coeffs for chg in monomer_charges[i]} for i in range(2)]
    if dets:
        conf_dict = [{chg: len(dets[chg]) for chg in monomer_charges[i]} for i in range(2)]
    else:
        conf_dict = [{chg: len(dens_builder_stuff[i][0][chg].coeffs[0]) for chg in monomer_charges[i]} for i in range(2)]

    d_slices = [get_slices(state_dict[i], monomer_charges[i]) for i in range(2)]
    c_slices = [get_slices(conf_dict[i], monomer_charges[i]) for i in range(2)] 

    state_coeffs = [{chg: np.array(states.coeffs) for chg, states in dens_builder_stuff[frag][0].items()} for frag in range(len(dens_builder_stuff))]

    #gs_energy, gs_state = get_xr_states(ints, dens, xr_order)
    H1_for_d, H2_for_d = get_xr_H(ints, dens, xr_order, monomer_charges)
    gs_energy, gs_state = get_gs(state_dict, d_slices, H1_for_d, H2_for_d, monomer_charges)
    E = np.real(gs_energy)
    d = gs_state
    print("dropping imaginary part of non-diagonalized d with norm", np.linalg.norm(np.imag(d)))
    d = np.real(d).reshape(sum(state_dict[0].values()), sum(state_dict[1].values()))
    if frag_ind == 1:
        d = d.T

    # normalize d
    # careful, this only works with S = 1, because then <Psi_D|Psi_D> = d * d * <Psi_A|Psi_A> <Psi_B|Psi_B> = d * d * 1 * 1
    # with XR' Psi_D needs to be normalized differently
    d = d / np.linalg.norm(d)
    
    #print(d)
    #print(d_slices)
    #for chg in monomer_charges:
    #    print(chg, chg * (-1), np.linalg.norm(d[d_slices[chg], d_slices[chg * (-1)]]))
    #    print(d[d_slices[chg], d_slices[chg * (-1)]])
    #norm_of_blocks = np.sqrt(sum([np.linalg.norm(d[d_slices[frag_map[0]][chg], d_slices[frag_map[1]][chg * (-1)]])**2 for chg in monomer_charges]))
    #print("norm of blocks", norm_of_blocks)
    #print("total norm - norm of blocks", np.linalg.norm(d) - norm_of_blocks)
    #print("relative norm of d, which is not contained in the 0 0, 1 -1, and -1 1 blocks", (np.linalg.norm(d) - norm_of_blocks) / np.linalg.norm(d))

    #contract_mon_with_d(frag_map, dens_builder_stuff, d, state_coeffs, d_slices)  # reevaluating the densities is not necessary...better contract d with densities on frag B...therefore recycle dens_transform function

    # build new densities (slater det densities for fragment under optimization and contract_with_d densities for the other one)
    # decomposing the following densities sometimes yields errors, so dont decompose them for now
    #print(f"build densities from states contracted with d on fragment {frag_map[1]}")
    #dens[frag_map[1]] = densities.build_tensors(*dens_builder_stuff[frag_map[1]][:-1], options=dens_builder_stuff[frag_map[1]][-1], n_threads=n_threads)
    #dens[frag_map[1]] = densities.build_tensors(*dens_builder_stuff[frag_map[1]][:-1], n_threads=n_threads)
    print(f"contract densities on fragment {frag_map[1]} with d")
    #dens[frag_map[1]] = contract_dens_with_d(dens[frag_map[1]], d, frag_map, d_slices, state_dict)  # alternatively save to new object (less CPU time, but more RAM, since densities dont need to be rebuild)
    contract_dens_with_d(dens[frag_map[1]], d, frag_map, d_slices, state_dict)
    #contract_mon_with_d(frag_map, dens_builder_stuff, d, state_coeffs, d_slices)
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
    """
    dens[frag_map[0]] = densities.build_tensors(*dens_builder_stuff[frag_map[0]][:-1], options=dens_builder_stuff[frag_map[0]][-1] + ["ket_det"], n_threads=n_threads)
    H1_new, H2_new = get_xr_H(ints, dens, xr_order, monomer_charges, ket_det=True)
    print(H1_new[0].shape, H1_new[1].shape, H2_new.shape)
    H2_new = H2_new.reshape((H1_new[0].shape[1], H1_new[1].shape[1]))
    """
    # H1 of frag A can be used as is and H1 of frag B needs to be contracted with the state coeffs of frag A. Note, that this is independent of the XR order 
    gradient_states = {}
    new_overlaps = get_adapted_overlaps(frag_map, d, d_slices)
    #new_overlaps = {}
    
    for chg in monomer_charges[frag_map[0]]:
        if dets:
            # compress state to smaller configuration space
            c0 = np.einsum("ip,qp->iq", state_coeffs[frag_map[0]][chg], np.array(dets[chg])) 
            #new_overlaps[chg] = np.einsum("ip,qp->iq", new_overlaps_pre[chg], np.array(dets[chg]))
        else:
            c0 = state_coeffs[frag_map[0]][chg]
            #new_overlaps[chg] = new_overlaps_pre[chg]
        #gradient_states[chg] = H1[frag_map[0]][c_slices[frag_map[0]][chg], d_slices[frag_map[0]][chg]].T  # frag A monomer H term
        gradient_states[chg] = np.einsum("pi,ki->kp", H1[frag_map[0]][c_slices[frag_map[0]][chg], d_slices[frag_map[0]][chg]], new_overlaps[chg])  # frag A monomer H term
        #print("step1", [np.linalg.norm(i) for i in gradient_states[chg]])
        #gradient_states[chg] -= E * c0  # E term
        #print(gradient_states[chg].shape, c0.shape, new_overlaps[chg].shape)
        gradient_states[chg] -= np.einsum("ip,ki->kp", c0, new_overlaps[chg]) * E
        #print("step2", [np.linalg.norm(i) for i in gradient_states[chg]])
        if frag_ind == 0:
            #gradient_states[chg] += np.einsum("pkii->kp", H2[c_slices[frag_map[0]][chg], d_slices[frag_map[0]][chg], :, :])  # dimer H term
            gradient_states[chg] += H2[c_slices[frag_map[0]][chg], d_slices[frag_map[0]][chg]].T  # dimer H term
        else:
            #gradient_states[chg] += np.einsum("kpii->kp", H2[d_slices[frag_map[0]][chg], c_slices[frag_map[0]][chg], :, :])  # dimer H term
            gradient_states[chg] += H2[d_slices[frag_map[0]][chg], c_slices[frag_map[0]][chg]]  # dimer H term
        #print("step3", [np.linalg.norm(i) for i in gradient_states[chg]])
        # this line also relies on equal charges on fragments A and B
        gradient_states[chg] += np.einsum("ip,ki->kp", c0, H1[frag_map[1]][d_slices[frag_map[0]][chg], d_slices[frag_map[0]][chg]])  # frag B monomer H term
        #print("step4", [np.linalg.norm(i) for i in gradient_states[chg]])
        #gradient_states[chg] *= 2
        # the following is the contribution <psi psi | H - E | phi psi>
        """
        gradient_states[chg] += np.einsum("ip,ki->kp", H1_new[frag_map[0]][d_slices[frag_map[0]][chg], c_slices[frag_map[0]][chg]], new_overlaps[chg])  # frag A monomer H term
        #print("step1", [np.linalg.norm(i) for i in gradient_states[chg]])
        #gradient_states[chg] -= E * c0  # E term
        gradient_states[chg] -= np.einsum("ip,ki->kp", c0, new_overlaps[chg]) * E
        #print("step2", [np.linalg.norm(i) for i in gradient_states[chg]])
        if frag_ind == 0:
            #gradient_states[chg] += np.einsum("pkii->kp", H2[c_slices[frag_map[0]][chg], d_slices[frag_map[0]][chg], :, :])  # dimer H term
            gradient_states[chg] += H2_new[c_slices[frag_map[0]][chg], d_slices[frag_map[0]][chg]].T  # dimer H term
        else:
            #gradient_states[chg] += np.einsum("kpii->kp", H2[d_slices[frag_map[0]][chg], c_slices[frag_map[0]][chg], :, :])  # dimer H term
            gradient_states[chg] += H2_new[d_slices[frag_map[0]][chg], c_slices[frag_map[0]][chg]]  # dimer H term
        #print("step3", [np.linalg.norm(i) for i in gradient_states[chg]])
        # this line also relies on equal charges on fragments A and B
        gradient_states[chg] += np.einsum("ip,ki->kp", c0, H1_new[frag_map[1]][d_slices[frag_map[0]][chg], d_slices[frag_map[0]][chg]])  # frag B monomer H term
        """
    return gs_energy, gradient_states, d

