from   get_ints import get_ints
from   get_xr_result import get_xr_states, get_xr_H
from qode.math.tensornet import raw, tl_tensor
from state_gradients import state_gradients

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


def optimize_states(displacement, max_iter, xr_order):
    ######################################################
    # Initialize integrals and densities
    ######################################################

    n_frag       = 2
    displacement = displacement
    project_core = True
    monomer_charges = [[0, +1, -1], [0, +1, -1]]

    # "Assemble" the supersystem for the displaced fragments and get integrals
    BeN = []
    dens = []
    dens_builder_stuff = []
    state_coeffs = []
    for m in range(int(n_frag)):
        state_obj, dens_var_1, dens_var_2, n_threads, Be = get_fci_states(displacement)
        for elem,coords in Be.atoms:  coords[2] += m * displacement    # displace along z
        BeN.append(Be)
        dens.append(densities.build_tensors(state_obj, dens_var_1, dens_var_2, n_threads=n_threads))
        dens_builder_stuff.append([state_obj, dens_var_1, dens_var_2])
        state_coeffs.append({chg: state_obj[chg].coeffs for chg in state_obj})

    ints = get_ints(BeN, project_core)

    ######################################################
    # iterative procedure
    ######################################################
    iter = 0
    previous_energy = None

    while iter < max_iter:
        iter += 1

        if False:  # keeping old code here for now
            # Get d from xr calc, which is the ground state vector
            
            gs_energy, gs_state = get_xr_states(ints, dens, 0)
            #print("norm of stripped complex part of ground state vector", torch.linalg.norm(torch.imag(gs_state)))
            #d = torch.real(gs_state)
            #d = gs_state.detach().clone()
            E = np.real(gs_energy)
            d = gs_state.copy()
            print("dropping imaginary part of non-diagonalized d with norm", np.linalg.norm(np.imag(d)))
            #print(len(d))
            #n_s = int(np.sqrt(len(d)))
            d = np.real(d).reshape(n_s, n_s)

            # normalize d
            # careful, this only works with S = 1, because then <Psi_D|Psi_D> = d * d * <Psi_A|Psi_A> <Psi_B|Psi_B> = d * d * 1 * 1
            # with XR' Psi_D needs to be normalized differently
            d = d / np.linalg.norm(d)

            #print(d)
            #print(d_slices)
            #for chg in monomer_charges:
            #    print(chg, chg * (-1), np.linalg.norm(d[d_slices[chg], d_slices[chg * (-1)]]))
            #    print(d[d_slices[chg], d_slices[chg * (-1)]])
            norm_of_blocks = np.sqrt(sum([np.linalg.norm(d[d_slices[chg], d_slices[chg * (-1)]])**2 for chg in monomer_charges]))
            #print("norm of blocks", norm_of_blocks)
            #print("total norm - norm of blocks", np.linalg.norm(d) - norm_of_blocks)
            print("relative norm of d, which is not contained in the 0 0, 1 -1, and -1 1 blocks", (np.linalg.norm(d) - norm_of_blocks) / np.linalg.norm(d))
            

            """
            # Change representation of states, such that d is diagonal
            n_s = int(np.sqrt(len(d)))
            d = d.reshape(n_s, n_s)  # substitute this with the actual fragment specific numbers of states
            d, d_eigvecs = np.linalg.eig(d)
            print(d)
            #U = numpy.array(d_eigvecs)  # numpy because fci density backend uses numpy
            U = d_eigvecs
            #print(U)
            #print(torch.linalg.inv(U) @ gs_state.reshape(n_s, n_s) @ U)  # builds diagonal representation
            print("norm of neglected imaginary part", np.linalg.norm(np.imag(U)))
            #U = numpy.real(U)
            #print(numpy.linalg.norm(U))

            # transform as c_pi d_ij c_jq = c_pi U U^-1 d_ij U U^-1 c_jq = c_pi U d_diag U^-1 c_jq
            # -> frag A (c_pi) needs U, to transform ket, while frag B needs U^-1
            # this can also transform the densities, e.g. frag A: (U^-1)_ki <psi_i| ccaa |psi_j> U_jl
            """
            """
            # However, for now it is easier to simply adapt the coefficients and rebuild the densities
            # Therefore in build_adc_densities the coeffs are now U @ coeffs_ip
            # requires U = U.T for frag A and U = U^-1 for frag B
            BeN[0].load_states(states, n_states, coeff_trans_mat=U.T)  # if they would be real and unitary, both would be equal
            BeN[1].load_states(states, n_states, coeff_trans_mat=numpy.linalg.inv(U))
            densities = [BeN[0].rho, BeN[1].rho]
            """
            """
            # transform densities
            def dens_transform(density, frag_ind):
                for dens_key in density:
                    if dens_key not in ("ca", "ccaa"):
                        continue
                    for chg in density[dens_key]:
                        if chg != (0,0):
                            continue
                        print(dens_key, chg, type(density[dens_key][chg][0][0]))
                        tmp = torch.tensor(density[dens_key][chg], dtype=torch.double)
                        print(tmp.shape)
                        if frag_ind == 0:
                            tmp = torch.tensordot(torch.linalg.inv(U), tmp, dims=([1], [0]))
                            tmp = torch.tensordot(U, tmp, dims=([0], [1]))
                        elif frag_ind == 1:
                            tmp = torch.tensordot(U, tmp, dims=([0], [0]))
                            tmp = torch.tensordot(torch.linalg.inv(U), tmp, dims=([1], [1]))
                        else:
                            raise IndexError("cannot request frag_ind > 1 for dimers")
                        for i in n_s:
                            for j in n_s:
                                density[dens_key][chg][i][j] = tmp[i][j]
                return density
            
            densities = [dens_transform(densities[0], 0), dens_transform(densities[1], 1)]
            """
            
            # optimize only fragment A
            # contract state coeff of B with d
            def contract_mon_with_d(frag_ind):  # updates states.coeffs in the dens_builder_stuff object...original coeffs can be found in state_coeffs
                """
                full_coeffs = []
                for chg in monomer_charges:  # same ordering for monomer charges required as for d
                    full_coeffs += dens_builder_stuff[frag_ind][0][chg].coeffs
                print(type(full_coeffs))
                print(type(full_coeffs[0]))
                full_coeffs = np.array(full_coeffs)
                #full_coeffs = np.array([*states[1][chg] for chg in monomer_charges])
                #diag_block_norm = np.linalg.norm(d[11:15, 11:15]) + np.linalg.norm(d[:11, :11]) + np.linalg.norm(d[15:, 15:])
                #print(diag_block_norm)
                #print(np.linalg.norm(d) - diag_block_norm)
                #adapted_coeffs = d @ full_coeffs
                print(d.shape, full_coeffs.shape)
                print(type(full_coeffs[0]))
                adapted_coeffs = np.tensordot(d, full_coeffs, axes=([frag_ind], [0]))  # better do this block-wise for d (0 0, +1 -1, -1 +1)
                #dummy_ind = 0
                for chg in monomer_charges:
                    #dens_builder_stuff[frag_ind][0][chg].coeffs = adapted_coeffs[dummy_ind:dummy_ind+state_dict[chg]].tolist()
                    dens_builder_stuff[frag_ind][0][chg].coeffs = adapted_coeffs[d_slices[chg]].tolist()
                    #dummy_ind += state_dict[chg]
                #states[frag_ind][0].coeffs = adapted_coeffs[:11].tolist()
                #states[frag_ind][1].coeffs = adapted_coeffs[11:15].tolist()
                #states[frag_ind][-1].coeffs = adapted_coeffs[15:].tolist()
                """
                #for chg, data in dens_builder_stuff[frag_ind][0].items():
                #    print("chg, len(data.coeffs before reassignment of coeffs)", chg, len(data.coeffs))
                #    print(type(data.coeffs), print(type(data.coeffs[0])))
                for chg in monomer_charges:
                    if frag_ind == 0:
                        adapted_coeffs = np.tensordot(d[d_slices[chg], d_slices[chg * (1)]], np.array(state_coeffs[frag_ind][chg]), axes=([0], [0]))
                    elif frag_ind == 1:
                        adapted_coeffs = np.tensordot(d[d_slices[chg * (-1)], d_slices[chg]], np.array(state_coeffs[frag_ind][chg]), axes=([1], [0]))
                    else:
                        raise IndexError("for dimers only fragment indices 0 and 1 are valid")
                    #for i in range(len(adapted_coeffs)):
                    #    dens_builder_stuff[frag_ind][0][chg * (-1)].coeffs[i] = adapted_coeffs[i]  #[i for i in adapted_coeffs]
                    # use the following with the old code
                    #for chg, data in dens_builder_stuff[frag_ind][0].items():  # no clue why, but without this loop I obtain an index error
                    #    pass
                    dens_builder_stuff[frag_ind][0][chg * (-1)].coeffs = [i for i in adapted_coeffs]
                    #for chg, data in dens_builder_stuff[frag_ind][0].items():
                    #    print("chg, len(data.coeffs after reassignment of coeffs)", chg, len(data.coeffs))

            contract_mon_with_d(1)  # reevaluating the densities is not necessary...better contract d with densities on frag B...therefore recycle dens_transform function
            # build new densities for fragment B
            print("build densities from d * Psi for fragment B")
            dens[1] = densities.build_tensors(*dens_builder_stuff[1], n_threads=n_threads)

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

            # get densities for A involving a slater determinant in the bra state
            # build states, which map only on one configuration

            print("build slater det densities for fragment A")
            dens[0] = densities.build_tensors(*dens_builder_stuff[0], n_threads=n_threads, bra_det=True)
            #E, dens = state_gradients(0, ints, dens_builder_stuff, dens, n_threads=n_threads, xr_order=xr_order)

            #norm_list = np.empty((560, 11))
            #small_count = 0
            #for ind_key, tens in dens[0]["cca"][(-1, 0)].items():
                #norm_list.append([])
                #for ket_ind, tens in enumerate(tens_pre):
            #    norm = np.linalg.norm(raw(tens))
            #    norm_list[ind_key[0], ind_key[1]] = norm
            #    if norm < 1e-14:
            #        small_count += 1
            #print(small_count)
            #print(small_count / (560 * 11))
            #print(norm_list)
            #for ind_key, tens in dens[0]["cca"][(0, 1)].items():
            #    #norm_list.append([])
            #    #for ket_ind, tens in enumerate(tens_pre):
            #    norm_list[ind_key[0], ind_key[1]] = np.linalg.norm(raw(tens))
            #print(norm_list)

            # build gradient
            # start with E term, which for basic XR is just energy times state
            #E_term = [E * i for i in state_coeffs[0]]
            # H term
            """
            H1, H2 = get_xr_H(ints, dens, xr_order, bra_det=True)
            print(H1[0].shape, H1[1].shape, H2.shape)
            H2 = H2.reshape((H1[0].shape[0], H1[1].shape[0], H1[0].shape[1], H1[1].shape[1]))
            # H1 of frag A can be used as is and H1 of frag B needs to be contracted with the state coeffs of frag A. Note, that this is independent of the XR order 
            gradient_states = {}
            #cs = []
            for chg in monomer_charges:
                cs = np.array(state_coeffs[0][chg])
                #cs = np.array(cs)
                gradient_states[chg] = H1[0][c_slices[chg], d_slices[chg]].T  # frag A monomer H term
                gradient_states[chg] -= E * cs  # E term
                gradient_states[chg] += np.einsum("pkii->kp", H2)[d_slices[chg], c_slices[chg]]  # dimer H term
                #for chg_ in monomer_charges:
                #    cs_ = np.array(state_coeffs[0][chg_])
                gradient_states[chg] += np.einsum("ip,ki->kp", cs, H1[1][d_slices[chg], d_slices[chg]])  # frag B monomer H term
                gradient_states[chg] *= 2
            """
            """
            monomer_charges = [monomer_charges, monomer_charges]
            d_slices = [d_slices, d_slices]
            c_slices = [c_slices, c_slices]

            H1, H2 = get_xr_H(ints, dens, xr_order, bra_det=True)
            print(H1[0].shape, H1[1].shape, H2.shape)
            H2 = H2.reshape((H1[0].shape[0], H1[1].shape[0], H1[0].shape[1], H1[1].shape[1]))
            # H1 of frag A can be used as is and H1 of frag B needs to be contracted with the state coeffs of frag A. Note, that this is independent of the XR order 
            gradient_states = {}
            #cs = []
            for chg in monomer_charges[0]:
                cs = np.array(state_coeffs[0][chg])
                #cs = np.array(cs)
                gradient_states[chg] = H1[0][c_slices[0][chg], d_slices[0][chg]].T  # frag A monomer H term
                gradient_states[chg] -= E * cs  # E term
                gradient_states[chg] += np.einsum("pkii->kp", H2)[d_slices[0][chg], c_slices[0][chg]]  # dimer H term
                #for chg_ in monomer_charges:
                #    cs_ = np.array(state_coeffs[0][chg_])
                gradient_states[chg] += np.einsum("ip,ki->kp", cs, H1[1][d_slices[0][chg], d_slices[0][chg]])  # frag B monomer H term
                gradient_states[chg] *= 2
            """
            #grads = {chg: gradient_states[d_slices[chg], :] for chg in monomer_charges}
            #for chg in monomer_charges:
            #    grads[chg] = 

            # gradient_states can be seen as a dict, that stores the state_coefficients for the gradient states wrt their corresponding charge

            #for chg in monomer_charges:
            #    dens_builder_stuff[0][0][chg].coeffs = state_coeffs[0][chg]
            #    dens_builder_stuff[1][0][chg].coeffs = state_coeffs[1][chg]

            #coeffs_grads = [state_coeffs, gradient_states]
            #pickle.dump(coeffs_grads, open("coeffs_grads.pkl", mode="wb"))
            #self.rho = pickle.load(open("dumped_densities.pkl", mode="rb"))

        gs_energy, gradient_states = state_gradients(0, ints, dens_builder_stuff, dens, monomer_charges, n_threads=n_threads, xr_order=xr_order)

        #coeffs_grads = [state_coeffs, gradient_states]
        #pickle.dump(coeffs_grads, open("coeffs_grads.pkl", mode="wb"))

        ################################
        # Application of the derivatives
        ################################

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


        # for now apply all the gradients to the subset of states and diagonalize
        # frag A
        for chg in monomer_charges[0]:
            tmp = np.array(state_coeffs[0][chg])
            tmp[-1] -= 0.2 * gradient_states[chg][-1]
            dens_builder_stuff[0][0][chg].coeffs = [i for i in orthogonalize(tmp)]
            dens_builder_stuff[1][0][chg].coeffs = state_coeffs[1][chg]

        # frag B
        #for chg in monomer_charges[1]:
        #    tmp = np.array(state_coeffs[1][chg])
        #    tmp[-1] -= 0.2 * gradient_states[chg][-1]
        #    dens_builder_stuff[1][0][chg].coeffs = [i for i in orthogonalize(tmp)]
        #    dens_builder_stuff[0][0][chg].coeffs = state_coeffs[0][chg]

        dens[0] = densities.build_tensors(*dens_builder_stuff[0], n_threads=n_threads)
        dens[1] = densities.build_tensors(*dens_builder_stuff[1], n_threads=n_threads)    
        gs_energy_new, gs_state_new = get_xr_states(ints, dens, xr_order)

        print(gs_energy, gs_energy_new)

        print(gradient_states[-1][0])







optimize_states(4.5, 1, 0)













