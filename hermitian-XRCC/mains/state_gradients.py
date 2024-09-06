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

def contract_mon_with_d(frag_map, dens_builder_stuff, d, state_coeffs, d_slices):  # updates states.coeffs in the dens_builder_stuff object...original coeffs can be found in state_coeffs
    #beware, that the following only works, if both fragments have the same charges, which are symmetrically sampled around zero
    if sum(d_slices[0].keys()) != 0 or sum(d_slices[0].keys()) != 0:
        raise ValueError("charges are not symmtrically sampled around zero...this can be fixed though, by changing this function")
    for chg in d_slices[frag_map[1]].keys():
        adapted_coeffs = np.tensordot(d[d_slices[frag_map[0]][chg * (-1)], d_slices[frag_map[1]][chg]],
                                      state_coeffs[frag_map[1]][chg], axes=([1], [0]))
        dens_builder_stuff[frag_map[1]][0][chg * (-1)].coeffs = [i for i in adapted_coeffs]

def get_slices(dict, chgs, type="standard"):
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
            dummy_ind += 2 * dict[chg]
        elif type == "latter":
            ret[chg] = slice(dummy_ind + dict[chg], dummy_ind + 2 * dict[chg])
            dummy_ind += 2 * dict[chg]
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
        print(d_sl, n_states, H2_new.shape, H1_new[0].shape, H1_new[1].shape)
        for chg0 in monomer_charges[0]:
            for chg1 in monomer_charges[1]:
                H2_new[d_sl[0][chg0], d_sl[1][chg1], d_sl[0][chg0], d_sl[1][chg1]] +=\
                    np.einsum("ij,kl->ikjl", H1_new[0][d_sl[0][chg0], d_sl[0][chg0]], np.eye(current_state_dict[1][chg1])) +\
                    np.einsum("ij,kl->ikjl", np.eye(current_state_dict[0][chg0]), H1_new[1][d_sl[1][chg1], d_sl[1][chg1]])
        H2_new = H2_new.reshape(n_states[0] * n_states[1],
                                n_states[0] * n_states[1])
        new_ens, new_states = sort_eigen(np.linalg.eig(H2_new))
        return new_ens[0], new_states[:, 0].T


# the following code snippet can be used to contract d with the densities, to circumvent building them again
# it is however not tested yet
"""
def dens_contract_with_d(density, frag_ind):
    ret = {}
    for dens_key in density:
        #print(dens_key)
        #if dens_key not in ("ca", "ccaa"):
        #    continue
        #print(density[dens_key].keys())
        ret[dens_key] = {}
        for chg in density[dens_key]:
            #if chg != (0,0):
            #    continue
            #print(dens_key, chg, type(density[dens_key][chg][0][0]))
            if type(density[dens_key][chg]) == int:
                ret[dens_key][chg] = density[dens_key][chg]
                continue  # skip n_elec and n_states keys

            ret[dens_key][(chg[0] * (-1), chg[1] * (-1))] = {}

            # this has to be circumvented, because it requires building the full density tensors explicitly
            tmp = np.empty((state_dict[chg[0]], state_dict[chg[1]], *density[dens_key][chg][(0, 0)].shape))
            for i in range(state_dict[chg[0]]):
                for j in range(state_dict[chg[1]]):
                    tmp[i][j] = raw(density[dens_key][chg][(i, j)])
            #print("tmp", tmp)
            #print(dens_key, chg)
            #print(tmp.shape)
            #if frag_ind == 0:
            #print(chg)
            #print(type(d[d_slices[chg[0]], d_slices[chg[1]]]), d[d_slices[chg[0]], d_slices[chg[1]]].shape)
            #print(type(tmp), tmp.shape)
            #print(d[d_slices[chg[0]], d_slices[chg[1]]].shape, tmp.shape)
            # in the following the only non-zero contributions are 0 0, +1 -1, and -1 +1, which can be mapped with a factor of -1
            if frag_ind == 0:
                tmp = np.tensordot(d[d_slices[chg[0]], d_slices[chg[0] * (-1)]], tmp, axes=([0], [1]))  # ket ind of density
                tmp = np.tensordot(d[d_slices[chg[1] * (-1)], d_slices[chg[1]]], tmp, axes=([1], [1]))  # bra ind of density
            elif frag_ind == 1:
                tmp = np.tensordot(d[d_slices[chg[1] * (-1)], d_slices[chg[1]]], tmp, axes=([1], [1]))  # ket ind of density
                tmp = np.tensordot(d[d_slices[chg[0]], d_slices[chg[0] * (-1)]], tmp, axes=([0], [1]))  # bra ind of density
            else:
                raise IndexError("dimer interaction only takes fragment indices 0 and 1")
            #elif frag_ind == 1:
            #    tmp = np.tensordot(d, tmp, dims=([1], [1]))
            #    tmp = np.tensordot(d, tmp, dims=([1], [1]))
            #else:
            #    raise IndexError("cannot request frag_ind > 1 for dimers")
            for i in range(state_dict[chg[0] * (-1)]):
                for j in range(state_dict[chg[1] * (-1)]):
                    ret[dens_key][(chg[0] * (-1), chg[1] * (-1))][(i, j)] = tl_tensor(tl.tensor(tmp[i][j], dtype=tl.float64))  # tl_tensor(tl.tensor(tmp[i][j].copy()))
    return density

# contract densities for frag B with d
B_dens_with_d = dens_contract_with_d(dens[1], 1)
"""

def state_gradients(frag_ind, ints, dens_builder_stuff, dens, monomer_charges, n_threads=1, xr_order=0):
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
    conf_dict = [{chg: len(dens_builder_stuff[i][0][chg].coeffs[0]) for chg in monomer_charges[i]} for i in range(2)]

    #def get_slices(dict, chgs):
    #    dummy_ind = 0
    #    ret = {}
    #    for chg in chgs:
    #        ret[chg] = slice(dummy_ind, dummy_ind+dict[chg])
    #        dummy_ind += dict[chg]
    #    return ret

    d_slices = [get_slices(state_dict[i], monomer_charges[i]) for i in range(2)]
    c_slices = [get_slices(conf_dict[i], monomer_charges[i]) for i in range(2)] 

    state_coeffs = [{chg: np.array(states.coeffs) for chg, states in dens_builder_stuff[frag][0].items()} for frag in range(len(dens_builder_stuff))]

    #gs_energy, gs_state = get_xr_states(ints, dens, xr_order)
    H1_for_d, H2_for_d = get_xr_H(ints, dens, xr_order)
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


    contract_mon_with_d(frag_map, dens_builder_stuff, d, state_coeffs, d_slices)  # reevaluating the densities is not necessary...better contract d with densities on frag B...therefore recycle dens_transform function
    
    # build new densities (slater det densities for fragment under optimization and contract_with_d densities for the other one)
    # decomposing the following densities sometimes yields errors, so dont decompose them for now
    print(f"build densities from states contracted with d on fragment {frag_map[1]}")
    #dens[frag_map[1]] = densities.build_tensors(*dens_builder_stuff[frag_map[1]][:-1], options=dens_builder_stuff[frag_map[1]][-1], n_threads=n_threads)
    dens[frag_map[1]] = densities.build_tensors(*dens_builder_stuff[frag_map[1]][:-1], n_threads=n_threads)
    print(f"build densities between slater determinant and state on fragment {frag_map[0]}")
    dens[frag_map[0]] = densities.build_tensors(*dens_builder_stuff[frag_map[0]][:-1], options=dens_builder_stuff[frag_map[0]][-1] + ["bra_det"], n_threads=n_threads)

    # build gradient
    H1, H2 = get_xr_H(ints, dens, xr_order, bra_det=True)
    print(H1[0].shape, H1[1].shape, H2.shape)
    H2 = H2.reshape((H1[0].shape[0], H1[1].shape[0]))#, H1[0].shape[1], H1[1].shape[1]))
    # H1 of frag A can be used as is and H1 of frag B needs to be contracted with the state coeffs of frag A. Note, that this is independent of the XR order 
    gradient_states = {}
    
    for chg in monomer_charges[frag_map[0]]:
        cs = state_coeffs[frag_map[0]][chg]
        gradient_states[chg] = H1[frag_map[0]][c_slices[frag_map[0]][chg], d_slices[frag_map[0]][chg]].T  # frag A monomer H term
        gradient_states[chg] -= E * cs  # E term
        if frag_ind == 0:
            #gradient_states[chg] += np.einsum("pkii->kp", H2[c_slices[frag_map[0]][chg], d_slices[frag_map[0]][chg], :, :])  # dimer H term
            gradient_states[chg] += H2[c_slices[frag_map[0]][chg], d_slices[frag_map[0]][chg]].T  # dimer H term
        else:
            #gradient_states[chg] += np.einsum("kpii->kp", H2[d_slices[frag_map[0]][chg], c_slices[frag_map[0]][chg], :, :])  # dimer H term
            gradient_states[chg] += H2[d_slices[frag_map[0]][chg], c_slices[frag_map[0]][chg]]  # dimer H term
        # this line also relies on equal charges on fragments A and B
        gradient_states[chg] += np.einsum("ip,ki->kp", cs, H1[frag_map[1]][d_slices[frag_map[0]][chg], d_slices[frag_map[0]][chg]])  # frag B monomer H term
        gradient_states[chg] *= 2
    
    return gs_energy, gradient_states

