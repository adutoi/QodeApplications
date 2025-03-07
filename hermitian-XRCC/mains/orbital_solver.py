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

from   get_ints import get_ints, tensorly_wrapper2, tens_diff
from   get_xr_result import get_xr_states, get_xr_H
from qode.math.tensornet import raw, tl_tensor, backend_contract_path
#import qode.util
from qode.util import timer, sort_eigen
from qode.util.dynamic_array import dynamic_array, wrap, cached
from qode.atoms.integrals.fragments import bra_transformed, ket_transformed, as_frag_blocked_mat, as_raw_mat, as_frag_blocked_U, as_raw_U, as_frag_blocked_V, as_raw_V
from state_gradients import get_slices
#from state_screening import state_screening, orthogonalize
from orb_grads import grads_and_hessian
#from orb_grads_old import grads_and_hessian as grads_and_hessian_wrong
from get_ints_Be import direct_Sinv
from state_solver import optimize_states
#from .state_solver import optimize_states

#import torch
import numpy as np
import scipy as sp
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


def optimize_orbs(displacement, max_iter, xr_order, conv_thresh=1e-6, dens_filter_thresh=1e-7, state_prep_guess=True):#, pen_start=10):
    ######################################################
    # Initialize integrals and density preliminaries
    ######################################################

    n_frag       = 2
    displacement = displacement
    project_core = True
    monomer_charges = [[0, +1, -1], [0, +1, -1]]
    #monomer_charges = [[0], [0]]
    density_options = ["compress=SVD,cc-aa"]
    #pen = pen_start
    #frozen_orbs = [0, 9]
    #n_orbs = 9
    #n_occ = [2, 2]  # currently only alpha beta separation, but generalize to frag level not done yet!!!
    #n_virt = [7, 7]
    XR_energies = []
    adap_xr_energies = []
    int_timer = timer()

    #ref_states = pickle.load(open("ref_states.pkl", mode="rb"))
    #ref_mos = pickle.load(open("ref_mos.pkl", mode="rb"))

    # "Assemble" the supersystem for the displaced fragments and get integrals
    if not state_prep_guess:
        BeN = []
        dens = []
        dens_builder_stuff = []
        state_coeffs_og = []
        #pre_opt_states = pickle.load(open("pre_opt_coeffs.pkl", mode="rb"))
        for m in range(int(n_frag)):
            state_obj, dens_var_1, dens_var_2, n_threads, Be = get_fci_states(displacement, n_state_list=[(1, 4), (0, 11), (-1, 8)])
            #Be.basis.MOcoeffs = ref_mos.copy()
            #pickle.dump(Be.basis.MOcoeffs, open(f"pre_opt_mos_{m}.pkl", mode="wb"))

            #print(Be.basis.MOcoeffs.shape)
            #raise ValueError("stop here")

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
            #del state_obj[+1]
            #del state_obj[-1]
            #dens.append(densities.build_tensors(state_obj, dens_var_1, dens_var_2, options=density_options, n_threads=n_threads))
            dens_builder_stuff.append([state_obj, dens_var_1, dens_var_2, density_options])
            state_coeffs_og.append({chg: state_obj[chg].coeffs for chg in state_obj})
            #print(Be.basis.MOcoeffs)
            #raise ValueError("stop here")
        #print(type(raw(dens[0]["a"][(+1,0)])), raw(dens[0]["a"][(+1,0)]).shape)

        ints = get_ints(BeN, project_core, int_timer)
        dens = [densities.build_tensors(*dens_builder_stuff[frag][:-1], options=density_options, n_threads=n_threads) for frag in range(2)]
    else:
        screening_energies, BeN, ints, dens, dens_builder_stuff = optimize_states(displacement, 0, xr_order)
        state_coeffs_og = [[dens_builder_stuff[frag][0][chg].coeffs for chg in monomer_charges[frag]] for frag in range(2)]

    n_states = [sum(len(state_coeffs_og[i][chg]) for chg in monomer_charges[i]) for i in range(2)]
    #n_confs = [sum(len(state_coeffs_og[i][chg][0]) for chg in monomer_charges[i]) for i in range(2)]

    state_dict = [{chg: len(state_coeffs_og[i][chg]) for chg in monomer_charges[i]} for i in range(2)]
    #conf_dict = [{chg: len(state_coeffs_og[i][chg][0]) for chg in monomer_charges[i]} for i in range(2)]

    d_slices = [get_slices(state_dict[i], monomer_charges[i]) for i in range(2)]
    #c_slices = [get_slices(conf_dict[i], monomer_charges[i]) for i in range(2)]

    def get_d(ints_):
        H1, H2 = get_xr_H(ints_, dens, xr_order, monomer_charges)
        
        full = H2.reshape(n_states[0], n_states[1],
                            n_states[0], n_states[1])
        
        #full = np.zeros_like(full)
        
        for chg0 in monomer_charges[0]:
            for chg1 in monomer_charges[1]:
                full[d_slices[0][chg0], d_slices[1][chg1], d_slices[0][chg0], d_slices[1][chg1]] +=\
                    np.einsum("ij,kl->ikjl", H1[0][d_slices[0][chg0], d_slices[0][chg0]], np.eye(state_dict[1][chg1])) +\
                    np.einsum("ij,kl->ikjl", np.eye(state_dict[0][chg0]), H1[1][d_slices[1][chg1], d_slices[1][chg1]])

        full = full.reshape(n_states[0] * n_states[1],
                            n_states[0] * n_states[1])

        full_eigvals_raw, full_eigvec_l_unsorted, full_eigvec_r_unsorted = sp.linalg.eig(full, left=True, right=True)
        full_eigvals, full_eigvec_l = sort_eigen((full_eigvals_raw, full_eigvec_l_unsorted))
        full_eigvals_check, full_eigvec_r = sort_eigen((full_eigvals_raw, full_eigvec_r_unsorted))
        if any(full_eigvals - full_eigvals_check):  # this check should be unnecessary
            raise ValueError("sorting went wrong with the sorting")
        XR_energies.append(full_eigvals[0])
        print(XR_energies)
        print("dropping imaginary part of non-diagonalized d with norm (left, right)",
            np.linalg.norm(np.imag(full_eigvec_l[0])), np.linalg.norm(np.imag(full_eigvec_r[0])))
        imag_norm = np.linalg.norm(np.imag(full_eigvec_l[0])) + np.linalg.norm(np.imag(full_eigvec_r[0]))
        dl = np.real(full_eigvec_l[0]).reshape(sum(state_dict[0].values()), sum(state_dict[1].values()))  # maybe 0 and 1 need to be swapped in reshape
        dr = np.real(full_eigvec_r[0]).reshape(sum(state_dict[0].values()), sum(state_dict[1].values()))
        return dl / np.linalg.norm(dl), dr / np.linalg.norm(dr), imag_norm
    
    n_occ, n_virt, n_frozen = [[4, 4], [14, 14], [2, 2]]
    n0 = n_occ[0] + n_virt[0]

    def off_diag_blocks(mat):
        ret = np.zeros_like(mat)
        ret[n0:,:n0] = mat[n0:,:n0]
        ret[:n0,n0:] = mat[:n0,n0:]
        return ret
    
    def diag_blocks(mat):
        ret = np.zeros_like(mat)
        ret[n0:,n0:] = mat[n0:,n0:]
        ret[:n0,:n0] = mat[:n0,:n0]
        return ret
    
    def off_diag_blocks_mo(mat):
        ret = np.zeros_like(mat)
        ret[n0:,:n0//2] = mat[n0:,:n0//2]
        ret[:n0,n0//2:] = mat[:n0,n0//2:]
        return ret

    def diag_inv(vec, set_one=False):
        ret = np.empty_like(vec)
        for i, el in enumerate(vec):
            if abs(el) < 1e-4:
                ret[i] = 0
            else:
                if set_one:
                    ret[i] = 1 #/ el
                else:
                    ret[i] = 1 / el
                #print(el)
        return ret
    
    def sequential_2b2_invert(mat):
        n_orb_tot = sum(n_occ) + sum(n_virt)
        ret = np.zeros_like(mat)
        for i in range(n_orb_tot):
            for j in range(n_orb_tot):
                if abs(mat[i * n_orb_tot + j, i * n_orb_tot + j]) < 1e-10:
                    continue
                submat = [[mat[i * n_orb_tot + j, i * n_orb_tot + j], mat[i * n_orb_tot + j, j * n_orb_tot + i]],
                          [mat[j * n_orb_tot + i, i * n_orb_tot + j], mat[j * n_orb_tot + i, j * n_orb_tot + i]]]
                #print(submat)
                e, v = sp.linalg.eigh(submat)
                #e_inv = [1 / x for x in e]
                #print(e)
                #e[1] = 1 / e[1]
                e_inv = diag_inv(e)
                submat_inv = v @ np.diag(e_inv) @ v.T
                #submat_inv = np.linalg.inv(submat)
                ret[i * n_orb_tot + j, i * n_orb_tot + j] = submat_inv[0, 0]
                ret[i * n_orb_tot + j, j * n_orb_tot + i] = submat_inv[0, 1]
                ret[j * n_orb_tot + i, i * n_orb_tot + j] = submat_inv[1, 0]
                ret[j * n_orb_tot + i, j * n_orb_tot + i] = submat_inv[1, 1]
                #print(submat_inv)
        return ret
    
    def blockwise_invert(mat):
        ret = np.zeros_like(mat)
        n_orb_tot = sum(n_occ) + sum(n_virt)
        n_orb_a = n_occ[0] + n_virt[0]
        n_orb_b = n_orb_tot - n_orb_a
        def symm_block_inv(submat):
            e, v = sp.linalg.eigh(submat)
            e_inv = diag_inv(e)
            return v @ np.diag(e_inv) @ v.T
        ret[:n_orb_a,:n_orb_a,:n_orb_a,:n_orb_a] = symm_block_inv(mat[:n_orb_a,:n_orb_a,:n_orb_a,:n_orb_a].reshape(n_orb_a**2,n_orb_a**2)).reshape(n_orb_a,n_orb_a,n_orb_a,n_orb_a)
        ret[n_orb_a:,n_orb_a:,n_orb_a:,n_orb_a:] = symm_block_inv(mat[n_orb_a:,n_orb_a:,n_orb_a:,n_orb_a:].reshape(n_orb_b**2,n_orb_b**2)).reshape(n_orb_b,n_orb_b,n_orb_b,n_orb_b)
        return ret

    
    #print("norm diff should be zero", np.linalg.norm(np.diag(hess_init.reshape(36*36,36*36))), np.linalg.norm(hess_init))
    #for i in range(36*36):
    #    for j in range(36*36):
    #        if abs(hess_init.reshape(36*36,36*36)[i,j]) > 0.03:
    #            print(i,j)

    #b = n_occ[0] * n_virt[0]
    #hess_inv = np.zeros((b*2, b*2))
    #hess_inv[:4,4:18,:4,4:18] = np.diag(diag_inv(np.diag(hess_init[:4,4:18,:4,4:18].reshape(b,b)))).reshape(4,14,4,14)  # ia,ia
    #hess_inv[4:18,:4,4:18,:4] = np.diag(diag_inv(np.diag(hess_init[4:18,:4,4:18,:4].reshape(b,b)))).reshape(14,4,14,4)  # ai,ai
    #hess_inv[:b,b:] = hess_init[:4,4:18,4:18,:4].reshape(b,b)  # ia,ai
    #hess_inv[b:,:b] = hess_init[4:18,:4,:4,4:18].reshape(b,b)  # ai,ia
    #for i in range(4):
    #    for a in range(4,18):
    #        hess_inv[i*14+a,a*4+i] = hess_init[i,a,a,i]
    #        hess_inv[a*4+i,i*14+a] = hess_init[a,i,i,a]
    #print(np.diag(hess_inv))
    #print(hess_inv[b:b+16,:16])

    #hess_inv = np.linalg.inv(hess_inv)

    #hess_inv0 = np.zeros((18,18,18,18))
    #hess_inv0[:4,4:18,:4,4:18] = hess_inv[:b,:b].reshape(4,14,4,14)
    #hess_inv0[4:18,:4,4:18,:4] = hess_inv[b:,b:].reshape(14,4,14,4)
    #hess_inv0[:4,4:18,4:18,:4] = hess_inv[:b,b:].reshape(4,14,14,4)
    #hess_inv0[4:18,:4,:4,4:18] = hess_inv[b:,:b].reshape(14,4,4,14)

    #hess_inv[18:22,22:,18:22,22:] = np.diag(diag_inv(np.diag(hess_init[18:22,22:,18:22,22:].reshape(b,b)))).reshape(4,14,4,14)  # ia,ia
    #hess_inv[22:,18:22,22:,18:22] = np.diag(diag_inv(np.diag(hess_init[22:,18:22,22:,18:22].reshape(b,b)))).reshape(14,4,14,4)  # ai,ai
    #print(np.linalg.norm(hess_inv))

    """
    # frag0
    hess_inv[:4,4:18,:4,4:18] += np.linalg.inv(hess_init[:4,4:18,:4,4:18].reshape(4*14,4*14)).reshape(4,14,4,14)  # ia,ia
    print(np.linalg.norm(hess_inv))
    hess_inv[4:18,:4,4:18,:4] += np.linalg.inv(hess_init[4:18,:4,4:18,:4].reshape(4*14,4*14)).reshape(14,4,14,4)  # ai,ai
    print(np.linalg.norm(hess_inv))
    hess_inv[:4,4:18,4:18,:4] += np.linalg.inv(hess_init[:4,4:18,4:18,:4].reshape(4*14,4*14)).reshape(4,14,14,4)  # ia,ai
    hess_inv[4:18,:4,:4,4:18] += np.linalg.inv(hess_init[4:18,:4,:4,4:18].reshape(4*14,4*14)).reshape(14,4,4,14)  # ai,ia
    print(np.linalg.norm(hess_inv))

    # frag1
    hess_inv[18:22,22:,18:22,22:] += np.linalg.inv(hess_init[18:22,22:,18:22,22:].reshape(4*14,4*14)).reshape(4,14,4,14)  # ia,ia
    print(np.linalg.norm(hess_inv))
    hess_inv[22:,18:22,22:,18:22] += np.linalg.inv(hess_init[22:,18:22,22:,18:22].reshape(4*14,4*14)).reshape(14,4,14,4)  # ai,ai
    print(np.linalg.norm(hess_inv))
    hess_inv[18:22,22:,22:,18:22] += np.linalg.inv(hess_init[18:22,22:,22:,18:22].reshape(4*14,4*14)).reshape(4,14,14,4)  # ia,ai
    hess_inv[22:,18:22,18:22,22:] += np.linalg.inv(hess_init[22:,18:22,18:22,22:].reshape(4*14,4*14)).reshape(14,4,4,14)  # ai,ia
    print(np.linalg.norm(hess_inv))
    """

    #print(x_prev[4:18,:4])
    # this would be the actual thing
    #x_prev = np.identity(sum(n_occ) + sum(n_virt))

    #for n in range(1, max_iter):
    #    grads = g_and_h.orb_grads(dl, dr, dens, ints)

    #print(x_prev[4:18,:4])
    #print(x_prev[:4,4:18])
    #U = np.identity(x_prev.shape[0]) + x_prev  # transformation matrix from e^x

    # not sure ... paper applies U to the right
    #print("previous MOs")
    #print(BeN[0].basis.MOcoeffs)
    #BeN[0].basis.MOcoeffs = BeN[0].basis.MOcoeffs @ U[:18, :18]

    # transform integrals instead of transforming orbitals and then reevaluate integrals
    # TODO: Behold the super ugly transformation routine of the integrals. Either don't look or clean up!
    for i in range(2):
        BeN[i].basis.MOcoeffs = np.concatenate((BeN[i].basis.MOcoeffs, BeN[i].basis.MOcoeffs))
        BeN[i].basis.n_spatial_orb *= 2
        BeN[i].basis.core = [0, 9]
    ints[0].fragments = BeN

    int_ranges = {2: ints[0].T.ranges, 3: ints[0].U.ranges, 4: ints[0].V.ranges}
    #orbs = np.concatenate((BeN[0].basis.MOcoeffs, BeN[1].basis.MOcoeffs))
    #U = U.T  # not sure if this is correct or wrong...
    #print("old MOs", orbs)
    #print("new MOs", U @ orbs)
    #print("MO diff new - old", U @ orbs - orbs)
    def transform_ints(_U, ints_):
        _U_T = as_frag_blocked_mat(_U.T, BeN)
        #U_T = as_frag_blocked_mat(np.linalg.inv(U), BeN)
        #_U = as_frag_blocked_mat(_U, BeN)
        #print(U[0,0][4:,:4])
        #print(U[0,1])
        #print(grads_prev[4:18,:4])
        #print(ints[0].S[0,0])
        #print(ints[0].S[0,1])
        #H_ = np.zeros(ints[0].T[1,0].shape)
        #print(ints[0].T[1,0], type(ints[0].T[1,0]))
        #print(U[0,0], type(U[0,0]))
        #print(H_)
        #print(np.dot(ints[0].T[1,0], U[0,0]))
        #H_ += np.dot(ints[0].T[1,0], U[0,0])
        #ints[0].T = as_raw_mat(ints[0].T, BeN)
        #print(type(ints[0].T))
        #ints[0].T = as_frag_blocked_mat(ints[0].T, BeN)
        #print(type(ints[0].T))
        try:
            tmp_ints = {"S":{}, "T":{}, "U":{}, "V":{}}
            for i in range(2):
                for j in range(2):
                    tmp_ints["S"][i,j] = raw(ints_[0].S[i,j])
                    tmp_ints["T"][i,j] = raw(ints_[0].T[i,j])
                    for k in range(2):
                        tmp_ints["U"][i,j,k] = raw(ints_[0].U[i,j,k])
                        for l in range(2):
                            tmp_ints["V"][i,j,k,l] = raw(ints_[0].V[i,j,k,l])
            #ints[0].S = as_frag_blocked_mat(as_raw_mat(tmp_ints["S"], BeN), BeN)
            ints_[0].S = tmp_ints["S"]
            ints_[0].T = tmp_ints["T"]
            ints_[0].U = tmp_ints["U"]
            ints_[0].V = tmp_ints["V"]
        except AttributeError:
            pass
        new_ints_symm = ket_transformed(_U_T, ints_[0], cache=True)
        new_ints_symm = bra_transformed(_U_T, new_ints_symm, cache=True)  # does this or above require U.T -> yes, because for the bra also U needs to be the hermitian conjugated
        #print(new_ints_symm.S[0,0])
        S_inv = direct_Sinv(BeN, new_ints_symm.S)
        new_ints_bior = bra_transformed(S_inv, new_ints_symm, cache=True)

        #print("is this still antisymmetric?", np.linalg.norm(new_ints_bior.V[0,0,0,0] + np.swapaxes(new_ints_bior.V[0,0,0,0], 2, 3)))
        #print("is this still antisymmetric?", np.linalg.norm(new_ints_bior.V[1,0,1,0] + np.swapaxes(new_ints_bior.V[1,0,0,1], 2, 3)))
        #print("is this still antisymmetric?", np.linalg.norm(new_ints_bior.V_half1[0,0,0,0] + np.swapaxes(new_ints_bior.V_half1[0,0,0,0], 2, 3)))

        # now the new integrals need to be wrapped as tensornet tensors again ...
        # if that works, uncomment the evaluation of the v terms in the gradient and hessian

        def tensornet_wrap(dyn_arr, ind_num):
            ret = {}
            #for key, ten in dict_.items():
            #    ret[key] = tl_tensor(tl.tensor(ten, dtype=tl.float64))
            for i in range(2):
                for j in range(2):
                    if ind_num == 2:
                        ret[i,j] = tl_tensor(tl.tensor(dyn_arr[i,j], dtype=tl.float64))
                    elif ind_num == 3:
                        for k in range(2):
                            ret[i,j,k] = tl_tensor(tl.tensor(dyn_arr[i,j,k], dtype=tl.float64))
                    elif ind_num == 4:
                        for k in range(2):
                            for l in range(2):
                                ret[i,j,k,l] = tl_tensor(tl.tensor(dyn_arr[i,j,k,l], dtype=tl.float64))
                    else:
                        raise ValueError(f"ind_num {ind_num} unknown")
            return ret
            #return dynamic_array([cached, tensorly_wrapper2(int_timer), ret], int_ranges[ind_num])
        
        new_ints_symm.S = tensornet_wrap(new_ints_symm.S, 2)
        new_ints_symm.T = tensornet_wrap(new_ints_symm.T, 2)
        new_ints_symm.U = tensornet_wrap(new_ints_symm.U, 3)
        new_ints_symm.V = tensornet_wrap(new_ints_symm.V, 4)
        new_ints_bior.S = tensornet_wrap(new_ints_bior.S, 2)
        new_ints_bior.T = tensornet_wrap(new_ints_bior.T, 2)
        new_ints_bior.U = tensornet_wrap(new_ints_bior.U, 3)
        new_ints_bior.V = tensornet_wrap(new_ints_bior.V, 4)
        new_ints_bior.V_half = tensornet_wrap(new_ints_bior.V_half, 4)
        new_ints_bior.V_half1 = tensornet_wrap(new_ints_bior.V_half1, 4)
        new_ints_bior.V_half2 = tensornet_wrap(new_ints_bior.V_half2, 4)

        #new_ints_bior.V_diff = dynamic_array([cached, tensorly_wrapper2(int_timer), tens_diff(new_ints_bior.V_half, new_ints_bior.V)], new_ints_bior.V.ranges)
        new_ints_bior.V_diff = dynamic_array([cached, tensorly_wrapper2(int_timer), tens_diff(new_ints_bior.V_half, new_ints_bior.V)], int_ranges[4])
        return new_ints_symm, new_ints_bior, ints_[2]

    #new_ints = transform_ints(U, ints)

    #for i in range(2):
    #    BeN[i].basis.MOcoeffs = np.concatenate((BeN[i].basis.MOcoeffs, BeN[i].basis.MOcoeffs))
    #    BeN[i].basis.MOcoeffs = np.einsum("ij,ik->kj", BeN[i].basis.MOcoeffs, U[i*18:(i+1)*18, i*18:(i+1)*18])  # this is the contraction from the paper, but maybe it's U @ phi instead of phi @ U
    #    BeN[i].basis.n_spatial_orb = 18
    #    BeN[i].basis.core = [0, 9]
    #new_ints = get_ints(BeN, project_core, int_timer, spin_ints=False)
    
    #_dl, _dr = get_d(new_ints)
    #dl, dr = get_d(new_ints)

    def apply_x(_x, _ints, scaling_factor, mo_prev):
        transform = np.identity(_x.shape[0]) + scaling_factor * _x  # transformation matrix from e^x
        #if iter == 2:
        #    U += x
        ret_ints_ = transform_ints(transform, _ints)
        dl_, dr_, d_imag_norm = get_d(ret_ints_)
        mo_new = transform @ mo_prev
        return dl_, dr_, ret_ints_, mo_new, d_imag_norm
    
    def lag_apply(dual_disp, new_ints, lag_prev, scale):
        safety_iter = 0
        while safety_iter < 10:
            safety_iter += 1
            lag_new = off_diag_blocks(dual_disp[dual_disp.shape[1]:,:])
            x_new = dual_disp[:dual_disp.shape[1],:]
            #upper_dual_grad = x_new + lag_prev + scale * lag_new
            dl_tmp, dr_tmp, ints_tmp = apply_x(x_new, new_ints, scale)
            #lag_new *= scale
            XR_energies[-1] += np.einsum("ij,ij->", lag_prev + scale * off_diag_blocks(lag_new), scale * off_diag_blocks(x_new))
            if XR_energies[-1] - XR_energies[-2] < 1e-8:
                break
            scale *= 0.2
            XR_energies.pop()
        return lag_prev + scale * off_diag_blocks(lag_new), scale * off_diag_blocks(x_new), dl_tmp, dr_tmp, ints_tmp
    
    def pen_apply(grad, new_ints, scale, mo_prev, mo_prevprev, pen=1e+2, hess=""):
        safety_iter = 0
        while safety_iter < 5:
            safety_iter += 1
            #lag_new = off_diag_blocks(dual_disp[dual_disp.shape[1]:,:])
            # for squared frob norm
            #pen_grad = diag_blocks(grad) + 2 * pen * off_diag_blocks()
            pen_term = 2 * pen * off_diag_blocks_mo(mo_prev) @ mo_prev.T
            pen_term = 1 * (pen_term - pen_term.T)
            pen_grad = grad + pen_term
            if type(hess) != str:
                #hess_pen_term = np.einsum("sq,pr->pqrs", mo_prevprev @ mo_prevprev.T, np.identity(pen_grad.shape[0]))  # not sure about this one
                #hess_pen_term = 1 * (hess_pen_term - np.transpose(hess_pen_term, (1,0,2,3)) - np.transpose(hess_pen_term, (0,1,3,2)) + np.transpose(hess_pen_term, (1,0,3,2)))
                pen_hess = hess #+ hess_pen_term
            print("grad and pen_grad", np.linalg.norm(grad), np.linalg.norm(pen_grad))
            # for linear penalty in frob norm
            #pen_grad = grad - (1/np.linalg.norm(off_diag_blocks_mo(mo_prev))) * pen * off_diag_blocks_mo(mo_prev) @ mo_prevprev.T
            if type(hess) == str:
                x_new = - pen_grad
            else:
                #x_new = - np.linalg.inv(pen_hess.reshape(hess.shape[0] * hess.shape[1], hess.shape[2] * hess.shape[3])).reshape(hess.shape[0] * hess.shape[1], hess.shape[2] * hess.shape[3]) @ pen_grad
                x_new = - np.einsum("ijkl,kl->ij", hess, pen_grad)  # this is only correct if hess has already been inverted
            #upper_dual_grad = x_new + lag_prev + scale * lag_new
            dl_tmp, dr_tmp, ints_tmp, mo_tmp, d_imag_norm = apply_x(x_new, new_ints, scale, mo_prev)
            #if safety_iter < 2:
            #    pen = (d_imag_norm)**(0.3)
            #lag_new *= scale
            # for squared frob norm
            off_diag_norm = np.linalg.norm(off_diag_blocks_mo(mo_tmp))**2
            # for linear penalty in frob norm
            #off_diag_norm = np.linalg.norm(off_diag_blocks_mo(mo_tmp))
            print("penalty factor and whole penalty term", pen, pen * off_diag_norm)
            adap_xr_energies.append(XR_energies[-1] + pen * off_diag_norm)
            if (XR_energies[-1] - XR_energies[-2] < 1e-8 and d_imag_norm < 1e-7) and np.max(off_diag_blocks_mo(mo_tmp)) < 1e-5:  # this is what we are looking for, but it might not be fetched out in the other if clause, because the penalty is too large
                break
            if adap_xr_energies[-1] - adap_xr_energies[-2] < 1e-6:
                if d_imag_norm < 1e-7:
                    break
                else:
                    if pen == 0:
                        scale *= 0.2
                    else:
                        pen_add = d_imag_norm / 1e-10
                        if pen_add < pen:
                            scale *= 0.2 #** (safety_iter - 1)
                        pen += pen_add
                        safety_iter = 0
                        #XR_energies.pop()
                        print("reset with higher penalty")
            else:
                scale *= 0.2
            if safety_iter < 5:
                adap_xr_energies.pop()
                XR_energies.pop()
        return dl_tmp, dr_tmp, ints_tmp, mo_tmp, pen
    
    class diis:
        def __init__(self, max_vec=4):  # value taken from paper
            self.x_history = []
            self.grad_history = []
            self.max_vec = max_vec

        def pop(self):
            if len(self.grad_history) > self.max_vec:
                self.grad_history.pop(0)
                self.x_history.pop(0)

        def add_vectors(self, grad, x):
            self.grad_history.append(grad)
            self.x_history.append(x)
            self.pop()

        def get_optimal_linear_combination(self):
            diis_size = len(self.grad_history) + 1
            diis_mat = np.zeros((diis_size, diis_size))
            diis_mat[:, 0] = -1.0
            diis_mat[0, :] = -1.0
            diis_mat[0, 0] = 0.
            for k, r1 in enumerate(self.x_history, 1):
                for ll, r2 in enumerate(self.x_history, 1):
                    diis_mat[k, ll] = np.einsum("ij,ij->", r1, r2)  #r1.dot(r2)
                    diis_mat[ll, k] = diis_mat[k, ll]
            diis_rhs = np.zeros(diis_size)
            diis_rhs[0] = -1.0
            weights = np.linalg.solve(diis_mat, diis_rhs)[1:]
            new_grad = np.zeros_like(self.grad_history[0])
            new_x = np.zeros_like(self.x_history[0])
            for ii, s in enumerate(self.grad_history):
                new_grad += weights[ii] * s
            for ii, s in enumerate(self.x_history):
                new_x += weights[ii] * s
            return new_x, new_grad

        def do_iteration(self, x, grad, hess_inv):
            #rnorm = np.sqrt(res.dot(res))
            self.add_vectors(grad, x)
            grad_norm = np.linalg.norm(grad)
            #t2 = t2new
            if len(self.grad_history) >= 2:# and rnorm <= 1.0:
                new_x, new_grad = self.get_optimal_linear_combination()
                print("diff norm grad_diis - grad", np.linalg.norm(new_grad - self.grad_history[-1]))
                grad_norm = np.linalg.norm(new_grad)
                #diff = new - self.grad_history[-1]
                #diff.evaluate()
                #rnorm = np.sqrt(diff.dot(diff))
                # the following is sum_i c_i x_i - H_n sum_i ci grad_i
                x = new_x - np.einsum("pqrs,rs->pq", hess_inv, new_grad)  # for bfgs update
                #x = new_x - new_grad  # for steepest descend update
            return x, new_grad, grad_norm
    

    dl, dr, d_imag_norm = get_d(ints)
    adap_xr_energies.append(XR_energies[-1])

    g_and_h = grads_and_hessian(n_occ, n_virt, n_frozen, d_slices)
    #hess_init = g_and_h.orb_hess_diag(dl, dr, dens, ints)

    # start from identity to perform gradient descend first...in case guess is too bad for quasi-newton
    #hess_inv = np.diag(diag_inv(np.diag(hess_init.reshape((sum(n_occ) + sum(n_virt)) ** 2, (sum(n_occ) + sum(n_virt)) ** 2)), set_one=True)).reshape(hess_init.shape)

    #hess_init_2 = hess_init.reshape((sum(n_occ) + sum(n_virt)) ** 2, (sum(n_occ) + sum(n_virt)) ** 2)
    #hess_inv = sequential_2b2_invert(hess_init_2).reshape(hess_init.shape)
    #hess_init_2 = hess_init.reshape((sum(n_occ) + sum(n_virt)) ** 2, (sum(n_occ) + sum(n_virt)) ** 2)
    #hess_inv = sp.linalg.inv(hess_init_2).reshape(hess_init.shape)

    #hess_inv = blockwise_invert(hess_init)

    grads_prev = g_and_h.orb_grads(dl, dr, dens, ints, off_diag=False)
    #grads_prev *= 0.02
    print("grads norm", np.linalg.norm(grads_prev))
    x_prev = - grads_prev
    #x_prev = - np.einsum("pqrs,rs->pq", hess_inv, grads_prev)
    print("x + x.T, x - x.T and norm(x)", np.linalg.norm(x_prev + x_prev.T), np.linalg.norm(x_prev - x_prev.T), np.linalg.norm(x_prev))
    #lag_prev = - off_diag_blocks(grads_prev)

    # here the update is lagrangian based
    #dual_grad = np.concatenate((grads_prev + off_diag_blocks(lag_prev), off_diag_blocks(x_prev)), axis=0)
    #grad_norm = np.linalg.norm(dual_grad)
    #lag_hess = 
    #dual_disp = - dual_grad
    
    scale = 1 #/ 2
    """
    #x_try, new_ints_try_t01 = x_new.copy(), raw(new_ints[0].T[0,1]).copy()
    dl_tmp, dr_tmp, ints_tmp = apply_x(x_prev, ints, scale)
    #print("diff x, diff t01", np.linalg.norm(x_try - x_new), np.linalg.norm(new_ints_try_t01 - new_ints[0].T[0,1]))
    while XR_energies[-1] > XR_energies[-2]:
        XR_energies.pop()
        scale *= 0.2
        dl_tmp, dr_tmp, ints_tmp = apply_x(x_prev, ints, scale)
        if XR_energies[-1] - XR_energies[-2] < 1e-8:
            break
    #_dl, _dr, new_ints = dl_tmp, dr_tmp, ints_tmp
    dl, dr, new_ints = dl_tmp, dr_tmp, ints_tmp
    """
    #lag_prev, x_new, dl, dr, new_ints = lag_apply(dual_disp, ints, scale)

    # here instead of a lagrangian a simple penalty is used for the squared frobenius norm of the off-diagonal blocks of the MO coeffs
    #pen = 1
    # here no previous x is known and initial mo coeffs are localized, so pen_grad is taken as the normal gradient in the first iteration
    #pen_grad = grads_prev   #+ pen * np.linalg.norm()
    mo_init_left = np.concatenate((BeN[0].basis.MOcoeffs, np.zeros_like(BeN[0].basis.MOcoeffs)))
    mo_init_right = np.concatenate((np.zeros_like(BeN[1].basis.MOcoeffs), BeN[1].basis.MOcoeffs))
    mo_init = np.concatenate((mo_init_left, mo_init_right), axis=1)
    dl, dr, new_ints, mo_prev, pen = pen_apply(grads_prev, ints, scale, mo_init, mo_init, pen=0)#, hess=hess_inv)
    mo_prevprev = mo_init

    # the following could also be implemented more efficiently using the
    # hessian update algorithm as introduced by Fischer and Almlöf using BFGS updates J. Phys. Chem. 1992, 96, 9768-9774
    # but even though the hessian is not small in terms of memory, it's still smaller than the 2p transition densities
    # one might also be able to speed up the convergence using a DIIS procedure including the updated hessians,
    # which is also described in the above mentioned paper.

    iter = 1  # because initialization is basically the first iteration
    x_diis = diis()#max_vec=6)
    x_diis.add_vectors(grads_prev, x_prev)
    # initialize lagrangian multiplier such that in first lagrangian iteration the x displacement is zero
    #lag_prev = - grads_prev
    while iter < max_iter:
        iter += 1
        scale = 1
        #print(iter)
        # naming convention taken from the paper mentioned above
        #hess_init = g_and_h.orb_hess_diag(dl, dr, dens, new_ints)
        #hess_inv = blockwise_invert(hess_init)
        #hess_init_2 = hess_init.reshape((sum(n_occ) + sum(n_virt)) ** 2, (sum(n_occ) + sum(n_virt)) ** 2)
        #hess_inv = sequential_2b2_invert(hess_init_2).reshape(hess_inv.shape)
        grads = g_and_h.orb_grads(dl, dr, dens, new_ints, off_diag=False)
        #if iter % 5 == 0:
        #    hess_init = g_and_h.orb_hess_diag(dl, dr, dens, new_ints)
        #    #if iter < 15:
        #    hess_inv = np.diag(diag_inv(np.diag(hess_init.reshape((sum(n_occ) + sum(n_virt)) ** 2, (sum(n_occ) + sum(n_virt)) ** 2)))).reshape(hess_init.shape)
            #else:  # do only gradient descent from here, to ensure correspondence
            #    hess_inv = np.diag(diag_inv(np.diag(hess_init.reshape((sum(n_occ) + sum(n_virt)) ** 2, (sum(n_occ) + sum(n_virt)) ** 2)), set_one=True)).reshape(hess_init.shape)
        """
        grad_norm = np.linalg.norm(grads)
        if iter % 5 == 0:
            hess_init = g_and_h.orb_hess_diag(dl, dr, dens, new_ints)
            if iter < 15:
                hess_inv = np.diag(diag_inv(np.diag(hess_init.reshape((sum(n_occ) + sum(n_virt)) ** 2, (sum(n_occ) + sum(n_virt)) ** 2)))).reshape(hess_init.shape)
            else:  # do only gradient descent from here, to ensure correspondence
                hess_inv = np.diag(diag_inv(np.diag(hess_init.reshape((sum(n_occ) + sum(n_virt)) ** 2, (sum(n_occ) + sum(n_virt)) ** 2)), set_one=True)).reshape(hess_init.shape)
        #if np.linalg.norm(grads) > 0.3:
        #    grads *= 0.02
        # Here different literature says to use grads for the gradient and some other would use grads_prev
        Delta_g = grads_prev  #- grads_prev is not given because the derivatives are taken with respect to previous x = 0
        delta_x = x_prev  # instead of x_prev - x_prevprev
        v = np.einsum("abcd,cd->ab", hess_inv, Delta_g)
        # since x_{n-1} is set to zero, delta_n = x_n
        alpha = 1 / np.einsum("ij,ij->", delta_x, Delta_g)
        hess_update1 = (1 + alpha * np.einsum("ij,ij->", Delta_g, v)) * alpha * np.einsum("ij,kl->ijkl", delta_x, delta_x)
        hess_update2 = alpha * (np.einsum("ij,kl->ijkl", delta_x, v) + np.einsum("ij,kl->ijkl", v, delta_x))
        #hess_inv += hess_update1 - hess_update2  # without updating this is steepest descend ... maybe useful for checks
        #x_new = - grads
        x_new = - np.einsum("pqrs,rs->pq", hess_inv, grads)
        #x = np.random.rand(*grads.shape)
        #x = x - x.T
        #x = 0.01 * x / np.linalg.norm(x)
        print("x + x.T, norm(x), norm(grad) and diff norm of grads", np.linalg.norm(x_new + x_new.T), np.linalg.norm(x_new), grad_norm, np.linalg.norm(grads - grads_prev))
        print("norm of x - x_prev", np.linalg.norm(x_new - x_prev))
        #x_new, grads_diis, grad_norm = x_diis.do_iteration(x_new, grads, hess_inv)
        #print("x + x.T, norm(x) and norm(grad) from diis", np.linalg.norm(x_new + x_new.T), np.linalg.norm(x_new), grad_norm)

        scale = 1 #/ 2
        #x_try, new_ints_try_t01 = x_new.copy(), raw(new_ints[0].T[0,1]).copy()
        dl_tmp, dr_tmp, ints_tmp = apply_x(x_new, new_ints, scale)
        #print("diff x, diff t01", np.linalg.norm(x_try - x_new), np.linalg.norm(new_ints_try_t01 - new_ints[0].T[0,1]))
        while XR_energies[-1] > XR_energies[-2]:
            XR_energies.pop()
            scale *= 0.2
            dl_tmp, dr_tmp, ints_tmp = apply_x(x_new, new_ints, scale)
            if XR_energies[-1] - XR_energies[-2] < 1e-8:
                break
        #_dl, _dr, new_ints = dl_tmp, dr_tmp, ints_tmp
        dl, dr, new_ints = dl_tmp, dr_tmp, ints_tmp
        """
        """
        # here the update is lagrangian based
        dual_grad = np.concatenate((grads + lag_prev, x_prev), axis=0)
        grad_norm = np.linalg.norm(dual_grad)
        #lag_hess = 
        dual_disp = - dual_grad

        def lag_apply(dual_disp, new_ints, scale):
            safety_iter = 0
            while safety_iter < 10:
                safety_iter += 1
                x_new = dual_disp[:dual_disp.shape[1],:]
                lag_new = dual_disp[dual_disp.shape[1]:,:]
                dl_tmp, dr_tmp, ints_tmp = apply_x(x_new, new_ints, scale)
                #lag_new *= scale
                XR_energies[-1] += np.einsum("ij,ij->", off_diag_blocks(lag_prev + scale * lag_new), off_diag_blocks(scale * x_new))
                if XR_energies[-1] - XR_energies[-2] < 1e-8:
                    break
                scale *= 0.2
                XR_energies.pop()
            return lag_prev + scale * lag_new, scale * x_new, dl_tmp, dr_tmp, ints_tmp

        lag_new, x_new, dl, dr, new_ints = lag_apply(dual_disp, new_ints, scale)
        """

        # here instead of a lagrangian a simple penalty is used for the squared frobenius norm of the off-diagonal blocks of the MO coeffs
        #pen = 1
        # here no previous x is known and initial mo coeffs are localized, so pen_grad is taken as the normal gradient in the first iteration
        #pen_grad = grads + 2 * pen * off_diag_blocks_mo(mo_prev) @ mo_prevprev.T
        #grad_norm = np.linalg.norm(pen_grad)
        dl, dr, new_ints, mo_new, pen = pen_apply(grads, new_ints, scale, mo_prev, mo_prevprev, pen=pen)#, hess=hess_inv)
        #pen *= 0.5
        grad_norm = np.linalg.norm(grads)

        #x_prev = x_new
        #lag_prev = lag_new
        grads_prev = grads#_diis
        mo_prev = mo_new
        mo_prevprev = mo_prev
        #print(XR_energies)
        try:
            if abs(XR_energies[-4] - XR_energies[-1]) < 1e-6:
                if grad_norm < 1e-4:
                    print("minimum reached")
                    #break
                else:
                    print(f"the energy is hardly getting lower anymore, but the norm of the gradient is still large {grad_norm}...")
                break
            else:
                pass
        except IndexError:
            pass
    return XR_energies[-1]
    


#print(optimize_orbs(4.5, 20, 0))
print([optimize_orbs(4.0 + r / 2, 20, 0) for r in range(1)])
