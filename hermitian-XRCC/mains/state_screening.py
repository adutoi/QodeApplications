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
 
from qode.math.tensornet import raw, tl_tensor

#import torch
import numpy as np
#import tensorly as tl

import densities
import pickle


def orthogonalize(U, eps=1e-15):  # with the transpose commented out, it orthogonalizes rows instead of columns
    # one should play with eps a little here
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


def get_large(ten):
    ten = abs(np.array(ten))
    thresh = np.max(ten) / 3  # for the integrals this equals for t_01 roughly 1e-2 to 1e-3
    ret = []
    reduced_ret = {i: [] for i in range(len(ten.shape))}
    if thresh < 1e-10:  # filter out zero vectors
        return reduced_ret
    def rec_loop(ten, *args):
        for i in range(len(ten)):# // 2):
            if type(ten[i]) != np.float64:
                rec_loop(ten[i], *args, i)
            else:
                if ten[i] >= thresh:
                    ret.append((*args, i))
        #for j in range(len(ten[i])):#range(i):  # is this always symmetric?
        #    if ten[i,j] >= thresh:
        #        ret.append((i,j))
    #return ret
    rec_loop(ten)
    #reduced_ret = {i: [] for i in range(len(ret[0]))}#{"bra": [], "ket": []}
    for tup in ret:
        for i in range(len(tup)):
            if tup[i] not in reduced_ret[i]:
                reduced_ret[i].append(tup[i])
        #if tup[0] not in reduced_ret["bra"]:
        #    reduced_ret["bra"].append(tup[0])
        #if tup[1] not in reduced_ret["ket"]:
        #    reduced_ret["ket"].append(tup[1])
    return {key: sorted(val) for key, val in reduced_ret.items()}


def dens_looper(ten):
    ret = {}
    for bra_ind in range(len(ten)):
        for ket_ind in range(len(ten[bra_ind])):
            #if ket_ind == bra_ind:
            #    print("alpha", ten[bra_ind][ket_ind][:9, :9])
            #    print("beta", ten[bra_ind][ket_ind][9:, 9:])
            ret[(bra_ind, ket_ind)] = get_large(ten[bra_ind][ket_ind])
    return ret


def missing_orbs(ref, big_nums, ref_ind):
    covered_elems = []
    for subdict in big_nums.values():
        for numlist in subdict.values():
            for number in numlist:
                if number not in covered_elems:
                    covered_elems.append(number)
    return [i for i in ref[ref_ind] if i not in covered_elems]


def state_screening(dens_builder_stuff, ints, monomer_charges, n_orbs, frozen, n_occ, n_threads=1):
    # since in this procedure all elements of the integrals are looped over, one could make use of
    # this also build an object storing information on which elements of a certain density to compute
    # and which to neglect, which would speed up the density builder. Making use of sparse backend ontop
    # of that would then also speed up the contractions required to build the XR Hamiltonian.
    h01 = raw(ints[0]("T")._as_tuple()[0][(0,1)])
    h01 += raw(ints[0]("U")._as_tuple()[0][(0,1,0)])
    h01 += raw(ints[0]("U")._as_tuple()[0][(1,1,0)])
    h01 = get_large(h01)

    h10 = raw(ints[0]("T")._as_tuple()[0][(1,0)])
    h10 += raw(ints[0]("U")._as_tuple()[0][(0,0,1)])
    h10 += raw(ints[0]("U")._as_tuple()[0][(1,0,1)])
    h10 = get_large(h10)
    # it seems like screening over V is not necessary, since h01 captures basically all contributions already...
    # this needs to be further investigated for a larger example, where not all contributions are relevant.
    # maybe instead of screening V one could also screen over the fock operator...
    # one should also screen over 1010 integrals, but this can be neglected, if the integrals are almost hermitian.
    # for strongly interacting fragments this might lead to missing contributions, so one probably needs to revisit this setup.
    v0101 = get_large(raw(ints[0]("V")._as_tuple()[0][(0,1,0,1)]))

    dens_arr = [densities.build_tensors(*dens_builder_stuff[m][:-1], options=[], n_threads=n_threads, screen=True) for m in range(2)]
    #for i in range(len(raw(dens_arr[0]["ca"][(0,0)]))):
    #    print(raw(dens_arr[0]["ca"][(0,0)][i, i, :9, :9]))
    #    print(raw(dens_arr[0]["ca"][(0,0)][i, i, 9:, 9:]))

    # building missing states initially from just the determinants seems to not work well...
    # Therefore here is some heuristic...
    # It is based on the Be FCI example, where one can see, that the states are not purely alpha
    # or beta, but rather mixed (often 50/50), so the mixing is determined from the provided
    # lowest energy states, averaged and then applied. (<--should not be necessary, since as long as
    # basis functions are provided, the algorithm will build linear combinations itself)
    # Furthermore, it can be seen, that some
    # orbitals have large partial shares (due to multi reference character) in many
    # excitations, so the partial occupancies of the occupied orbitals is also determined, averaged
    # and then applied, such that the determinant excitations are now only partial excitations as
    # well. (<--this effect cannot be reintroduced with linear combinations!!!)
    # It seems appropriate to take an other look at the integrals again though, as this
    # scheme seems awfully arbitrary. Also double check which index of the u integrals is actually the core!!!!!!!!!!!!!!!!!!!!!!!!
    # To be done...
    """
    mr_occs = [{}, {}]
    for frag in range(2):
        dens = raw(dens_arr[frag]["ca"][(0,0)])  # get multi reference character only from neutral one particle densities
        alpha = [i for i in range(n_occ[0]) if i not in frozen]  # should be enough to only look for valence orbitals
        beta = [i for i in range(n_orbs, n_occ[1] + n_orbs) if i not in frozen]  # should be enough to only look for valence orbitals
        val_orbs = alpha + beta
        for orb in val_orbs:
            diags = [dens[i, i, orb, orb] for i in range(len(dens))
                     if dens[i, i, orb, orb] > 0.05 and dens[i, i, orb, orb] < 0.95]  # take this as mr criterion for now
            if len(diags) < 3:  # this might require quite a lot of initial neutral states to be provided here
                continue
            print(diags)
            mr_occs[frag][orb] = sum(diags) / len(diags)
        print(f"multi reference contribution for fragment {frag} detected in orbitals {mr_occs[frag]}")
    """
    # here only the densities are taken into account, for which the initial state (ket) is the neutral state
    missing_states = [{0: {}, -1: {}}, {0: {}, -1: {}}]  # note, that within the following procedure all missing contributions are captured without appending the most positively charged states

    # ionization contributions
    beta_gs_config = sum([2**(n_orbs + i) for i in range(n_occ[1])])
    total_gs_config_neutral = beta_gs_config + sum([2**(i) for i in range(n_occ[0])])
    for frag in range(2):
        for chg in range(min(monomer_charges[frag]), max(monomer_charges[frag])):  # loops over -1 and 0
            # this currently only works with charges +1,0,-1 and densities only with initial neutral state
            if chg == 0:
                dens = dens_looper(raw(dens_arr[frag]["a"][(1,0)]))
                ref_ind = 1
            elif chg == -1:
                dens = dens_looper(raw(dens_arr[frag]["c"][(-1,0)]))
                ref_ind = 0
                gs = total_gs_config_neutral
            else:
                raise ValueError("invalid charge provided")
            if frag == ref_ind:
                int = h01
            else:
                int = h10 
            for elem in missing_orbs(int, dens, ref_ind):
                #for chg in range(min(monomer_charges[0]), max(monomer_charges[0])):
                # skip frozen core contributions
                if elem in frozen:
                    continue
                # get gs for required spin and assume more positively charged state to be gs
                if chg == 0:
                    # positively charged reference ground state is ground state of corresponding spin
                    if elem >= n_orbs:  # appending beta needs alpha ref (except neutral gs)
                        #gs = min(dens_builder_stuff[frag][0][chg + 1].configs)
                        gs = total_gs_config_neutral - 2**(n_orbs - 1 + n_occ[1])  # one should rather sweep over the available ionized contributions
                    else:  # appending alpha needs beta ref (except neutral gs)
                        #gs = min([abs(i - beta_gs_config) for i in dens_builder_stuff[frag][0][chg + 1].configs]) + beta_gs_config
                        gs = total_gs_config_neutral - 2**(-1 + n_occ[0])  # one should rather sweep over the available ionized contributions
                ex = gs + 2**elem
                #if chg == -1:
                #    print(elem, ex)
                if ex not in missing_states[frag][chg].keys():
                    #det_state = np.zeros_like(dens_builder_stuff[frag][0][chg].configs)
                    #det_state[dens_builder_stuff[frag][0][chg].configs.index(ex)] = 1.
                    #missing_states[frag][chg][ex] = det_state
                    missing_states[frag][chg][ex] = dens_builder_stuff[frag][0][chg].configs.index(ex)

    # neutral contributions
    """
    gs = total_gs_config_neutral
    for frag in range(2):
        # the following loop is unique here, because one should also loop over charge 1, but for
        # Be with frozen core only the frozen electron is left for the cation for one spin contribution.
        # That means, one could only excite the electron of the other spin, corresponding to a shake up
        # state over two different spins...These contributions are expected to be small though and due to
        # the ambiguity of choosing states, which form densities covering the remaining large contributions
        # of the integrals, only the strongest expected contributions are accounted for, which would be
        # an ionization and excitation in the same spin, which is not possible though for Be with frozen core.
        for chg in range(min(monomer_charges[frag]), max(monomer_charges[frag])):  # loops over -1 and 0
            dens = dens_looper(raw(dens_arr[frag]["ca"][(chg,chg)]))
            int = v0101
            for elem in missing_orbs(int, dens, frag):  # ref_inds 2 and 3 should be equal to 0 and 1
                if elem in frozen:
                    continue
                # build excitation from gs det into singly excited det
                if elem >= n_orbs:  # beta requires beta excitation
                    #gs = min([abs(i - beta_gs_config) for i in dens_builder_stuff[0][0][0].configs]) + beta_gs_config
                    ion = gs - 2**(n_orbs - 1 + n_occ[1])
                else:  # alpha requires alpha excitation
                    #gs = min(dens_builder_stuff[0][0][0].configs)  # this works, if spin flip is not allowed
                    ion = gs - 2**(n_occ[0] - 1)
                if chg == 0:
                    ex = ion + 2**elem
                elif chg == -1:
                    ex = gs + 2**elem
                else:
                    raise ValueError(f"chg {chg} not accepted")
                if ex not in missing_states[frag][chg].keys():
                    #det_state = np.zeros_like(dens_builder_stuff[frag][0][chg].configs)
                    #det_state[dens_builder_stuff[frag][0][chg].configs.index(ex)] = 1.
                    #missing_states[frag][chg][ex] = det_state
                    missing_states[frag][chg][ex] = dens_builder_stuff[frag][0][chg].configs.index(ex)
    """

    
    # neutral spin flip contributions (only for chg 0)
    """
    gs = total_gs_config_neutral
    for frag in range(2):
        #for chg in range(min(monomer_charges[frag]), max(monomer_charges[frag])):  # loops over -1 and 0
        chg = 0
        dens = dens_looper(raw(dens_arr[frag]["ca"][(chg,chg)]))
        int = v0101
        for elem in missing_orbs(int, dens, frag):  # ref_inds 2 and 3 should be equal to 0 and 1
            if elem in frozen:
                continue
            # build excitation from gs det into singly excited det
            if elem >= n_orbs:  # spin flip beta requires alpha hole
                ion = gs - 2**(n_occ[0] - 1)
            else:  # spin flip alpha requires beta hole
                ion = gs - 2**(n_orbs + n_occ[1] - 1)
            #if chg == 0:
            ex = ion + 2**elem
            #elif chg == -1:
            #    ex = gs + 2**elem
            #else:
            #    raise ValueError(f"chg {chg} not accepted")
            if ex not in missing_states[frag][chg].keys():
                #det_state = np.zeros_like(dens_builder_stuff[frag][0][chg].configs)
                #det_state[dens_builder_stuff[frag][0][chg].configs.index(ex)] = 1.
                #missing_states[frag][chg][ex] = det_state
                missing_states[frag][chg][ex] = dens_builder_stuff[frag][0][chg].configs.index(ex)
    """
    #print(missing_states[0][0].keys(), missing_states[0][-1].keys())

    def conf_decoder(conf):
        ret = []
        for bit in range(n_orbs * 2, -1, -1):
            if conf - 2**bit < 0:
                continue
            conf -= 2**bit
            ret.append(bit)
        return sorted(ret)

    """
    for frag in range(2):
        for chg in range(1):#range(min(monomer_charges[frag]), max(monomer_charges[frag])):
            det_states = []
            for det, ind in missing_states[frag][chg].items():
                det_states.append(np.zeros_like(dens_builder_stuff[frag][0][chg].coeffs[0]))
                conf = conf_decoder(det)
                missing_mr_orbs = [i for i in mr_occs[frag] if not all(np.array(conf) - i)]
                #print(np.sqrt(mr_occs[frag][missing_mr_orbs[0]]), np.sqrt(1 - mr_occs[frag][missing_mr_orbs[0]]))
                if len(missing_mr_orbs) == 0:
                    det_states[-1][ind] = 1
                    #print(np.linalg.norm(det_states[-1]))
                    continue
                elif len(missing_mr_orbs) > 1:
                    raise NotImplementedError("too many mr orbs encountered. This code needs to be adapted to be able to deal with more than 1")
                else:                
                    det_states.append(np.zeros_like(dens_builder_stuff[frag][0][chg].coeffs[0]))
                    # maybe also allow for lin comb with opposite signs?
                    # also the following probably doesnt work for all examples, since it employs the mr character for all mr orbs for each guess state
                    det_states[-1][dens_builder_stuff[frag][0][chg].configs.index(total_gs_config_neutral)] = np.sqrt(mr_occs[frag][missing_mr_orbs[0]])
                    det_states[-1][ind] = np.sqrt(1 - mr_occs[frag][missing_mr_orbs[0]])
                    #det_states[-1][dens_builder_stuff[frag][0][chg].configs.index(total_gs_config_neutral)] = 1#np.sqrt(mr_occs[frag][missing_mr_orbs[0]])
                    #det_states[-1][ind] = 1#np.sqrt(1 - mr_occs[frag][missing_mr_orbs[0]])
                    #print(det_states[-1])
                    #print("mr", np.linalg.norm(det_states[-1]))
            new_states = dens_builder_stuff[frag][0][chg].coeffs + det_states #list(missing_states[frag][chg].values())
            new_states = orthogonalize(np.array(new_states))
            dens_builder_stuff[frag][0][chg].coeffs = [i for i in new_states]
    """
    for frag in range(2):
        for chg in range(min(monomer_charges[frag]), max(monomer_charges[frag])):
            det_states = []
            for det, ind in missing_states[frag][chg].items():
                det_states.append(np.zeros_like(dens_builder_stuff[frag][0][chg].coeffs[0]))
                det_states[-1][ind] = 1.
            new_states = dens_builder_stuff[frag][0][chg].coeffs + det_states #list(missing_states[frag][chg].values())
            new_states = orthogonalize(np.array(new_states))
            dens_builder_stuff[frag][0][chg].coeffs = [i for i in new_states]

    """
    for i, vec in enumerate(dens_builder_stuff[0][0][0].coeffs):
        big_inds = {ind: elem for ind, elem in enumerate(vec) if abs(elem) > 1e-1}
        #print(i, [ind for ind, elem in enumerate(vec) if abs(elem) > 1e-1])
        print(i, {tuple(conf_decoder(dens_builder_stuff[frag][0][chg].configs[j])): val for j, val in big_inds.items()})

    print("now for opt states")
    for chg in monomer_charges[0]:
        coeffs = np.load(f"../../../QodeApplications_old/hermitian-XRCC/atomic_states/states/16-115-550/load=states:16-115-550:thresh=1e-6:4.5:u.pickle/4.5/Z_{2 - chg}e.npy")
        for i, vec in enumerate(coeffs.T):
            big_inds = {ind: elem for ind, elem in enumerate(vec) if abs(elem) > 1e-1}
            print(chg, i, {tuple(conf_decoder(dens_builder_stuff[frag][0][chg].configs[j])): val for j, val in big_inds.items()})
    """
    """
    opt_dens = pickle.load(open("../../../QodeApplications_old/hermitian-XRCC/density_c_a_ca.pkl", mode="rb"))

    new_dens_arr = [densities.build_tensors(*dens_builder_stuff[m][:-1], options=[], n_threads=n_threads, screen=True) for m in range(2)]

    
    chg = -1
    for bra in range(len(raw(new_dens_arr[0]["ca"][(chg,chg)]))):
        #for ket in range(2):
        print(bra, "alpha", raw(new_dens_arr[0]["ca"][(chg,chg)])[bra, bra][:9, :9])
        print(bra, "beta", raw(new_dens_arr[0]["ca"][(chg,chg)])[bra, bra][9:, 9:])
    for bra in range(len(opt_dens[2][(chg,chg)])):
        #for ket in range(4):
        print(bra, "alpha", np.array(opt_dens[2][(chg,chg)][bra][bra])[:9, :9])
        print(bra, "beta", np.array(opt_dens[2][(chg,chg)][bra][bra])[9:, 9:])
    
    #print(h01)
    
    raise ValueError("stop here")
    """
    #raise ValueError("stop here")
    #return dens_builder_stuff
