from   get_ints import get_ints
from   get_xr_result import get_xr_states, get_xr_H, get_xr_states_from_H
from qode.math.tensornet import raw, tl_tensor
import qode.util

#import torch
import numpy as np
import tensorly as tl
import pickle

#torch.set_num_threads(4)
#tl.set_backend("pytorch")
#tl.set_backend("numpy")

# choose which density routine to use in the following
# old and fast
#from   build_fci_states_fast import get_fci_states
#import densities_old as densities

# newer, more flexible, but somehow slower version
from   build_fci_states import get_fci_states
import densities


def optimize_states(displacement, max_iter):
    ######################################################
    # Initialize integrals and densities
    ######################################################

    n_frag       = 2
    displacement = displacement
    project_core = True
    monomer_charges = [[0, +1, -1], [0, +1, -1]]
    

    # "Assemble" the supersystem for the displaced fragments and get integrals
    coeffs_grads = pickle.load(open("coeffs_grads.pkl", mode="rb"))
    state_coeffs = coeffs_grads[0]
    gradient_states = coeffs_grads[1]
    BeN = []
    dens = []
    dens_builder_stuff = []
    #print(len(state_coeffs[0][0]))
    #n = sum(len(state_coeffs[0][chg]) for chg in monomer_charges[0])
    
    for m in range(int(n_frag)):
        state_obj, dens_var_1, dens_var_2, n_threads, Be = get_fci_states(displacement)
        for elem,coords in Be.atoms:  coords[2] += m * displacement    # displace along z
        for chg in monomer_charges[m]:
            state_obj[chg].coeffs = [i.copy() for i in state_coeffs[m][chg]]
            #state_obj[chg].bra_coeffs = [i.copy() for i in state_coeffs[m][chg]]
            #state_obj[chg].ket_coeffs = [i.copy() for i in state_coeffs[m][chg]]
        BeN.append(Be)
        dens.append(densities.build_tensors(state_obj, dens_var_1, dens_var_2, n_threads=n_threads))
        dens_builder_stuff.append([state_obj, dens_var_1, dens_var_2])

    ints = get_ints(BeN, project_core)
    
    state_dict = [{chg: len(dens_builder_stuff[i][0][chg].coeffs) for chg in monomer_charges[i]} for i in range(2)]
    conf_dict = [{chg: len(dens_builder_stuff[i][0][chg].coeffs[0]) for chg in monomer_charges[i]} for i in range(2)]

    def get_slices(dict, chgs, type="standard"):
        dummy_ind = 0
        ret = {}
        for chg in chgs:
            if type == "standard":
                ret[chg] = slice(dummy_ind, dummy_ind+dict[chg])
                dummy_ind += dict[chg]
            elif type == "double":
                ret[chg] = slice(dummy_ind, dummy_ind + 2 * dict[chg])
                dummy_ind += 2 * dict[chg]
            elif type == "first":
                ret[chg] = slice(dummy_ind, dummy_ind + dict[chg])
                dummy_ind += 2 * dict[chg]
            elif type == "latter":
                ret[chg] = slice(dummy_ind + dict[chg], dummy_ind + 2 * dict[chg])
                dummy_ind += 2 * dict[chg]
            else:
                raise ValueError(f"type {type} is unknown")
        return ret

    d_slices = [get_slices(state_dict[i], monomer_charges[i]) for i in range(2)]
    c_slices = [get_slices(conf_dict[i], monomer_charges[i]) for i in range(2)]

    #d_slices_double = [get_slices(state_dict[i], monomer_charges[i], type="double") for i in range(2)]
    #c_slices_double = [get_slices(conf_dict[i], monomer_charges[i], type="double") for i in range(2)]

    d_slices_first = [get_slices(state_dict[i], monomer_charges[i], type="first") for i in range(2)]
    #c_slices_first = [get_slices(conf_dict[i], monomer_charges[i], type="first") for i in range(2)]

    d_slices_latter = [get_slices(state_dict[i], monomer_charges[i], type="latter") for i in range(2)]
    #c_slices_latter = [get_slices(conf_dict[i], monomer_charges[i], type="latter") for i in range(2)]
    
    def orthogonalize(U, eps=1e-15):  # with the transpose commented out, it orthogonalizes rows instead of columns
        n = len(U)
        V = U#.T
        for i in range(n):
            prev_basis = V[0:i]     # orthonormal basis before V[i]
            coeff_vec = np.dot(prev_basis, V[i].T)  # each entry is np.dot(V[j], V[i]) for all j < i
            # subtract projections of V[i] onto already determined basis V[0:i]
            V[i] -= np.dot(coeff_vec, prev_basis).T
            if np.linalg.norm(V[i]) < eps:
                V[i][V[i] < eps] = 0.   # set the small entries to 0
            else:
                V[i] /= np.linalg.norm(V[i])
        return V#.T

    """
    gs_energy, gs_state = get_xr_states(ints, dens, 0)

    for chg in monomer_charges[0]:
        #tmp = np.array(state_coeffs[0][chg]) - 2 * gradient_states[chg]  # factor of 2 is still from gradients (no empirical parameter for better convergence or similar stuff)
        tmp = np.array(state_coeffs[0][chg])
        tmp[-1] -= 2 * gradient_states[chg][-1]
        dens_builder_stuff[0][0][chg].coeffs = [i for i in orthogonalize(tmp)]
        #dens_builder_stuff[1][0][chg].coeffs = state_coeffs[1][chg]

    dens[0] = densities.build_tensors(*dens_builder_stuff[0], n_threads=n_threads, bra_det=False)
    #dens[1] = densities.build_tensors(*dens_builder_stuff[1], n_threads=n_threads, bra_det=False)      
    gs_energy_new, gs_state_new = get_xr_states(ints, dens, 0)

    print(gs_energy, gs_energy_new)
    """
    #print(type(state_coeffs[0][0]), type(gradient_states[0]))
    a_coeffs = {chg: np.array(tens) for chg, tens in state_coeffs[0].items()}
    grad_coeffs = {chg: val / np.linalg.norm(val) for chg, val in gradient_states.items()}
    #grad_coeffs = {chg: val for chg, val in gradient_states.items()}

    print([[a_coeffs[chg][i].T @ grad_coeffs[chg][i] for i in range(len(a_coeffs[chg]))] for chg in a_coeffs.keys()])
    #print(a_coeffs[0][0].T @ grad_coeffs[0][0])
    print(a_coeffs[1][0].T)
    print(grad_coeffs[1][0])

    # for Hamiltonian evaluation in extended state basis, i.e. states + gradients, an orthonormal set for each fragment
    # is required, but this choice is not unique. One could e.g. normalize, orthogonalize and then again normalize,
    # or orthogonalize and normalize without previous normalization.

    tot_a_coeffs = {chg: orthogonalize(np.vstack((a_coeffs[chg], grad_coeffs[chg]))) for chg in monomer_charges[0]}
    #tot_a_coeffs = orthogonalize(tot_a_coeffs)
    """
    dummy_ind = 0
    for chg in monomer_charges[0]:
        print("chg = ", chg)
        dummy_ind = 0
        for i in range(len(tot_a_coeffs[chg])):
            print(np.linalg.norm(tot_a_coeffs[chg][i]), dummy_ind)
            dummy_ind += 1

    for i in grad_coeffs[-1]:
        print(np.linalg.norm(i))
    """
    #print(grad_coeffs[-1][0])
    #for frag_ind in range(2):
    #    for chg in monomer_charges[frag_ind]:
    #        dens[frag_ind][chg].coeffs = state_coeffs[frag_ind][chg]
    #n = sum(len(state_coeffs[0][chg]) for chg in monomer_charges[0])
    n = sum(state_dict[0].values())
    n_conf = sum(conf_dict[0].values())
    """
    ortho_grad_coeffs = {chg: [i for i in tens[len(a_coeffs[chg]):]] for chg, tens in tot_a_coeffs.items()}
    #print(ortho_grad_coeffs[1])
    #print("lens for ortho grads", [len(i) for i in ortho_grad_coeffs.values()])
    if sum([len(i) for i in ortho_grad_coeffs.values()]) != n:
        raise IndexError("number of new orthogonalized gradients is not equal to the number of states, which is required by the following code!")
    
    # the following should be build with one XR calculation, instead of 4!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #(0,0)
    H1, H2 = get_xr_H(ints, dens, 0)
    #gs_en, gs_vec = get_xr_states(ints, dens, 0)
    #print(gs_en)
    H_00 = H2.reshape((n, n, n, n)) + np.einsum("ij,kl->ikjl", H1[0], np.eye(n)) + np.einsum("ij,kl->ikjl", np.eye(n), H1[1])
    eigvals, eigvecs = np.linalg.eig(H_00)
    print(np.real(np.min(eigvals)), np.imag(np.min(eigvals)))
    
    #(0,1)
    #for chg in monomer_charges[0]:
    #    dens_builder_stuff[0][0][chg].ket_coeffs = ortho_grad_coeffs[chg]
    dens[0] = densities.build_tensors(*dens_builder_stuff[0], n_threads=n_threads,
                                      coeffs=[{}, ortho_grad_coeffs])
    H1, H2 = get_xr_H(ints, dens, 0)

    #overlap_psi_grad = np.zeros((n, n))
    #for chg in monomer_charges[0]:
    #    overlap_psi_grad[d_slices[chg], d_slices[chg]] = np.einsum("ik,jk->ij", tot_a_coeffs[chg][:state_dict[chg]],
    #                                                                            tot_a_coeffs[chg][state_dict[chg]:])
    H_01 = H2.reshape((n, n, n, n)) + np.einsum("ij,kl->ikjl", H1[0], np.eye(n))  # second term is zero, because overlap between psi_A and grad_A is zero
    
    #(1,0)
    # here one could make use of the hermitian conjugation for the densities taken from (0,1)
    dens[0] = densities.build_tensors(*dens_builder_stuff[0], n_threads=n_threads,
                                      coeffs=[ortho_grad_coeffs, {}])
    H1, H2 = get_xr_H(ints, dens, 0)
    H_10 = H2.reshape((n, n, n, n)) + np.einsum("ij,kl->ikjl", H1[0], np.eye(n))  # second term is zero, because overlap between psi_A and grad_A is zero
    #(1,1)
    dens[0] = densities.build_tensors(*dens_builder_stuff[0], n_threads=n_threads,
                                      coeffs=[ortho_grad_coeffs, ortho_grad_coeffs])
    H1, H2 = get_xr_H(ints, dens, 0)
    H_11 = H2.reshape((n, n, n, n)) + np.einsum("ij,kl->ikjl", H1[0], np.eye(n)) + np.einsum("ij,kl->ikjl", np.eye(n), H1[1])

    # put it all together
    upper = np.concatenate((H_00, H_01), axis=2)
    lower = np.concatenate((H_10, H_11), axis=2)
    full = np.concatenate((upper, lower), axis=0)
    full = full.reshape((2 * n**2, 2 * n**2))
    """

    og_en, og_vec = get_xr_states(ints, dens, 0)
    """
    H1, H2 = get_xr_H(ints, dens, 0)
    og_H = H2.reshape(n, n, n, n)
    for chg0 in monomer_charges[0]:
        for chg1 in monomer_charges[1]:
            og_H[d_slices[0][chg0], d_slices[1][chg1], d_slices[0][chg0], d_slices[1][chg1]] +=\
                np.einsum("ij,kl->ikjl", H1[0][d_slices[0][chg0], d_slices[0][chg0]], np.eye(state_dict[1][chg1])) +\
                np.einsum("ij,kl->ikjl", np.eye(state_dict[0][chg0]), H1[1][d_slices[1][chg1], d_slices[1][chg1]])

    og_en_diag, _ = np.linalg.eig(og_H.reshape(n**2, n**2))
    print("original energies from FCI vs custom diag", og_en, np.min(og_en_diag))
    """
    for chg in monomer_charges[0]:
        dens_builder_stuff[0][0][chg].coeffs = [i.copy() for i in tot_a_coeffs[chg]]
    dens[0] = densities.build_tensors(*dens_builder_stuff[0], n_threads=n_threads)
    H1, H2 = get_xr_H(ints, dens, 0)

    H2_ = H2.reshape(2* n, n, 2 * n, n)

    test_mat = np.zeros((n, n, n, n))
    for chg0 in monomer_charges[0]:
        for chg1 in monomer_charges[1]:
            test_mat[d_slices[0][chg0], d_slices[1][chg1], d_slices[0][chg0], d_slices[1][chg1]] +=\
                H2_[d_slices_first[0][chg0], d_slices[1][chg1], d_slices_first[0][chg0], d_slices[1][chg1]] +\
                np.einsum("ij,kl->ikjl", H1[0][d_slices_first[0][chg0], d_slices_first[0][chg0]], np.eye(state_dict[1][chg1])) +\
                np.einsum("ij,kl->ikjl", np.eye(state_dict[0][chg0]), H1[1][d_slices[1][chg1], d_slices[1][chg1]])
        
    gs_en_check, gs_vec_check = np.linalg.eig(test_mat)
    print("og en", og_en, "custom H build with full diag", np.real(np.min(gs_en_check)))
        
    H2_reduced = np.zeros((n, n, n, n))
    for chg0 in monomer_charges[0]:
        for chg1 in monomer_charges[1]:
            H2_reduced[d_slices[0][chg0], d_slices[1][chg1], d_slices[0][chg0], d_slices[1][chg1]] +=\
                H2_[d_slices_first[0][chg0], d_slices[1][chg1], d_slices_first[0][chg0], d_slices[1][chg1]]
        
    H1_reduced = [np.zeros((n, n)), np.zeros((n, n))]
    for m in range(2):
        for chg in monomer_charges[m]:
            if m == 0:
                H1_reduced[m][d_slices[m][chg], d_slices[m][chg]] = H1[m][d_slices_first[m][chg], d_slices_first[m][chg]]
            else:
                H1_reduced[m][d_slices[m][chg], d_slices[m][chg]] = H1[m][d_slices[m][chg], d_slices[m][chg]]

    H2_reduced = H2_reduced.reshape(n**2, n**2)
    gs_en, gs_vec = get_xr_states_from_H(H1_reduced, H2_reduced)
        
    print("excitonic.fci vs custom diag vs original en", gs_en, np.real(np.min(gs_en_check)), og_en)

    full = H2.reshape(2 * n, n, 2 * n, n)
    for chg0 in monomer_charges[0]:
        for chg1 in monomer_charges[1]:
            #(0,0)
            full[d_slices_first[0][chg0], d_slices[1][chg1], d_slices_first[0][chg0], d_slices[1][chg1]] +=\
                np.einsum("ij,kl->ikjl", H1[0][d_slices_first[0][chg0], d_slices_first[0][chg0]], np.eye(state_dict[1][chg1])) +\
                np.einsum("ij,kl->ikjl", np.eye(state_dict[0][chg0]), H1[1][d_slices[1][chg1], d_slices[1][chg1]])
            #(0,1)
            full[d_slices_first[0][chg0], d_slices[1][chg1], d_slices_latter[0][chg0], d_slices[1][chg1]] +=\
                np.einsum("ij,kl->ikjl", H1[0][d_slices_first[0][chg0], d_slices_latter[0][chg0]], np.eye(state_dict[1][chg1]))
            #(1,0)
            full[d_slices_latter[0][chg0], d_slices[1][chg1], d_slices_first[0][chg0], d_slices[1][chg1]] +=\
                np.einsum("ij,kl->ikjl", H1[0][d_slices_latter[0][chg0], d_slices_first[0][chg0]], np.eye(state_dict[1][chg1]))
            #(1,1)
            full[d_slices_latter[0][chg0], d_slices[1][chg1], d_slices_latter[0][chg0], d_slices[1][chg1]] +=\
                np.einsum("ij,kl->ikjl", H1[0][d_slices_latter[0][chg0], d_slices_latter[0][chg0]], np.eye(state_dict[1][chg1])) +\
                np.einsum("ij,kl->ikjl", np.eye(state_dict[0][chg0]), H1[1][d_slices[1][chg1], d_slices[1][chg1]])

    full = full.reshape(2 * n**2, 2 * n**2)
    full_eigvals, full_eigvec = np.linalg.eig(full)
    print(np.real(np.min(full_eigvals)), np.imag(np.min(full_eigvals)))

    # no determine which elements of the eigvec to keep
    full_gs_vec = full_eigvec[0].reshape(2 * n, n)
    dens_mat_a = np.einsum("ij,kj->ik", full_gs_vec, full_gs_vec)  # contract over frag_b part
    dens_eigvals, dens_eigvecs = qode.util.sort_eigen(np.linalg.eig(dens_mat_a))  # this sorts from low to high
    print(dens_eigvals)
    # choosing more states here at higher iterations, depending on the diagonal values might be a good way to slightly enlarge the state space if needed
    new_large_vecs = np.real(dens_eigvecs[n:])
    print("dropped imag part of eigvecs for new coeffs is", np.linalg.norm(np.imag(dens_eigvecs[n:])))
    """
    large_vec_map = np.zeros((n, n_conf))
    for chg in monomer_charges[0]:
        large_vec_map[d_slices[0][chg], c_slices[0][chg]] = state_coeffs[0][chg]
    large_vec_map = np.vstack((large_vec_map, large_vec_map))
    """
    large_vec_map = np.zeros((2 * n, n_conf))
    for chg in monomer_charges[0]:
        large_vec_map[d_slices_first[0][chg], c_slices[0][chg]] = state_coeffs[0][chg]
        large_vec_map[d_slices_latter[0][chg], c_slices[0][chg]] = state_coeffs[0][chg]
    new_vecs = np.einsum("ij,jp->ip", new_large_vecs, large_vec_map)
    print(new_vecs.shape)
    new_coeffs = {chg: new_vecs[d_slices[0][chg], c_slices[0][chg]] for chg in monomer_charges[0]}
    #print(new_coeffs[1])
    for chg in monomer_charges[0]:
        dens_builder_stuff[0][0][chg].coeffs = [i.copy() for i in new_coeffs[chg]]
    dens[0] = densities.build_tensors(*dens_builder_stuff[0], n_threads=n_threads)#,
                                      #coeffs=[new_coeffs, {}])
    new_gs_en, new_gs_vec = get_xr_states(ints, dens, 0)
    print(new_gs_en)

optimize_states(4.5, 1)
