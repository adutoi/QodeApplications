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
from itertools import combinations

import densities
import pickle


def orthogonalize(U, eps=1e-6, normalize=True):
    # gram schmidt orthogonalizer for orthogonalizing rows of a matrix
    # one might has to play with eps a little here
    n = len(U)
    V = U
    for i in range(n):
        #print("norm before orthogonalization", np.linalg.norm(V[i]))
        prev_basis = V[0:i]     # orthonormal basis before V[i]
        coeff_vec = np.dot(prev_basis, V[i].T)  # each entry is np.dot(V[j], V[i]) for all j < i
        # subtract projections of V[i] onto already determined basis V[0:i]
        V[i] -= np.dot(coeff_vec, prev_basis).T
        if normalize:
            if np.linalg.norm(V[i]) < eps:
                #V[i][V[i] < eps] = 0.   # set the small entries to 0
                V[i] = np.zeros_like(V[i])
                #print("zero vector encountered!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            else:
                V[i] /= np.linalg.norm(V[i])
        else:
            #print("norm after orthogonalization", np.linalg.norm(V[i]))
            if np.linalg.norm(V[i]) < eps:
                V[i] = np.zeros_like(V[i])
    return V


def get_large(ten, thresh_frac=1 / 3, compress_ouput=True):
    ten = abs(np.array(ten))
    thresh = np.max(ten) * thresh_frac  # for the integrals this equals for t_01 roughly 1e-2 to 1e-3
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
    if compress_ouput == False:
        return ret
    else:
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

def conf_decoder(conf, n_orbs):
    ret = []
    for bit in range(n_orbs * 2, -1, -1):
        if conf - 2**bit < 0:
            continue
        conf -= 2**bit
        ret.append(bit)
    return sorted(ret)

def is_singlet(det, n_orbs):
    alpha_det = [occ for occ in det if occ < n_orbs]
    beta_det = det[len(alpha_det):]
    pair = sum([2**(i - n_orbs) for i in beta_det] + [2**(i + n_orbs) for i in alpha_det])
    return len(alpha_det) == len(beta_det), pair


def state_screening(dens_builder_stuff, ints, monomer_charges, n_orbs, frozen, n_occ, n_threads=1,
                    single_thresh=1/5, double_thresh=1/3.5, triple_thresh=1/2.5, sp_thresh=1/1.1):
    # since in this procedure all elements of the integrals are looped over, one could make use of
    # this to also build an object storing information on which elements of a certain density to compute
    # and which to neglect, which would speed up the density builder. Making use of sparse backend on top
    # of that would then also speed up the contractions required to build the XR Hamiltonian.
    # An other thing to investigate is how the filtered information is converted...Currently if one orbital
    # transition is found to be important, it will simply be added to a list of important transitions,
    # from which the determinants are build, but one could include information on which pairs of orbital
    # transitions are important, i.e. according to the screened integral information build pairs of determinants
    # which should be applied together on the corresponding fragment when appending the spaces with the
    # gradient free solver type.

    #h01 = raw(ints[0]("T")._as_tuple()[0][(0,1)])
    #h01 += raw(ints[0]("U")._as_tuple()[0][(0,0,1)])
    #h01 += raw(ints[0]("U")._as_tuple()[0][(1,0,1)])
    #h01 = get_large(h01)

    #h10 = raw(ints[0]("T")._as_tuple()[0][(1,0)])
    #h10 += raw(ints[0]("U")._as_tuple()[0][(0,1,0)])
    #h10 += raw(ints[0]("U")._as_tuple()[0][(1,1,0)])
    #h10 = get_large(h10)
    symm_ints, bior_ints, nuc_rep = ints
    # for the following to work for more than two fragments, one needs to track the fragment indices, such that
    # e.g. if the dimer contribution between fragment 3 and 7 is required [3, 7] is requested (see XR term)
    h01 = raw(bior_ints.T[0, 1])
    h01 += raw(bior_ints.U[0, 0, 1])
    h01 += raw(bior_ints.U[1, 0, 1])

    h10 = raw(bior_ints.T[1, 0])
    h10 += raw(bior_ints.U[0, 1, 0])
    h10 += raw(bior_ints.U[1, 1, 0])

    h00 = raw(bior_ints.T[0, 0])
    h00 += raw(bior_ints.U[0, 0, 0])
    h00 += raw(bior_ints.U[1, 0, 0])

    h11 = raw(bior_ints.T[1, 1])
    h11 += raw(bior_ints.U[0, 1, 1])
    h11 += raw(bior_ints.U[1, 1, 1])
    # it seems like screening over V is not necessary, since h01 captures basically all contributions already...
    # this needs to be further investigated for a larger example, where not all contributions are relevant.
    # maybe instead of screening V one could also screen over the fock operator...
    # one should also screen over 1010 integrals, but this can be neglected, if the integrals are almost hermitian.
    # for strongly interacting fragments this might lead to missing contributions, so one probably needs to revisit this setup.
    #v0101 = get_large(raw(ints[0]("V")._as_tuple()[0][(0,1,0,1)]))
    #v0101 = raw(ints[0]("V")._as_tuple()[0][(0,1,0,1)])
    v0101 = raw(bior_ints.V[0,1,0,1])

    # the following are required to screen for equations B30 and B31
    def get_v(frag_tuple):
        #return raw(ints[0]("V")._as_tuple()[0][frag_tuple])
        return raw(bior_ints.V[frag_tuple])

    #dens_arr = [densities.build_tensors(*dens_builder_stuff[m][:-1], options=[], n_threads=n_threads, screen=True) for m in range(2)]
    #for i in range(len(raw(dens_arr[0]["ca"][(0,0)]))):
    #    print(raw(dens_arr[0]["ca"][(0,0)][i, i, :9, :9]))
    #    print(raw(dens_arr[0]["ca"][(0,0)][i, i, 9:, 9:]))

    # Furthermore, it can be seen, that some
    # orbitals have large partial shares (due to multi reference character) in many
    # excitations, so the partial occupancies of the occupied orbitals is also determined, averaged
    # and then applied, such that the determinant excitations are now only partial excitations as
    # well. (<--this effect cannot be reintroduced with linear combinations!!!)
    # It seems appropriate to take an other look at the integrals again though, as this
    # scheme seems awfully arbitrary.
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

    missing_states = [{chg: {} for chg in monomer_charges[frag]} for frag in range(2)]

    beta_gs_config = sum([2**(n_orbs + i) for i in range(n_occ[1])])
    total_gs_config_neutral = beta_gs_config + sum([2**(i) for i in range(n_occ[0])])

    # add the ground state (and maybe also multi reference contributions)
    for frag in range(2):
        missing_states[frag][0][total_gs_config_neutral] = dens_builder_stuff[frag][0][0].configs.index(total_gs_config_neutral)

    
    # add monomer contributions
    for frag in range(2):
        chg = 0
        if frag == 0:
            one_e_int = h00
        else:
            one_e_int = h11
        two_el_int = get_v((frag, frag, frag, frag))
        # single excitations
        for elem in get_large(one_e_int, thresh_frac=single_thresh)[frag] + get_large(two_el_int, thresh_frac=single_thresh)[frag]:  # last two are equal to first two indices of v
            if elem in frozen:
                continue
            if elem in conf_decoder(total_gs_config_neutral, n_orbs):  # filter out occupied orbitals
                continue
            # positively charged reference ground state is ground state of corresponding spin
            if elem >= n_orbs:  # appending beta needs alpha ref
                gs = total_gs_config_neutral - 2**(n_orbs - 1 + n_occ[1])  # one should rather sweep over the available ionized contributions
            else:  # appending alpha needs beta ref
                gs = total_gs_config_neutral - 2**(-1 + n_occ[0])  # one should rather sweep over the available ionized contributions
            ex = gs + 2**elem
            if ex not in missing_states[frag][chg].keys():
                missing_states[frag][chg][ex] = dens_builder_stuff[frag][0][chg].configs.index(ex)
        # double excitations
        for comb in get_large(two_el_int, thresh_frac=double_thresh, compress_ouput=False):  # one might also want to build combinations from h00 and h11 here
            if any(elem in frozen for elem in comb):
                continue
            if comb[2] == comb[3]:
                continue
            if comb[0] == comb[1]:
                continue
            if comb[0] in conf_decoder(total_gs_config_neutral, n_orbs) and not comb[0] in comb[2:]:
                continue
            if comb[1] in conf_decoder(total_gs_config_neutral, n_orbs) and not comb[1] in comb[2:]:
                continue
            if comb[2] not in conf_decoder(total_gs_config_neutral, n_orbs):
                continue
            if comb[3] not in conf_decoder(total_gs_config_neutral, n_orbs):
                continue
            ex = total_gs_config_neutral + 2**comb[0] + 2**comb[1] - 2**comb[2] - 2**comb[3]
            missing_states[frag][0][ex] = dens_builder_stuff[frag][0][0].configs.index(ex)
    
    #print(missing_states[0])


    # ionization contributions form one el ints for single excitations without spin flip
    for frag in range(2):
        for chg in range(min(monomer_charges[frag]), max(monomer_charges[frag])):  # loops over -1 and 0
            # this currently only works with charges +1,0,-1 and densities only with initial neutral state
            if chg == 0:
                #dens = dens_looper(raw(dens_arr[frag]["a"][(1,0)]))
                ref_ind = 1
                thresh=single_thresh
            elif chg == -1:
                #dens = dens_looper(raw(dens_arr[frag]["c"][(-1,0)]))
                ref_ind = 0
                gs = total_gs_config_neutral
                thresh=single_thresh #/ 2
            else:
                raise ValueError("invalid charge provided")
            if frag == ref_ind:
                int = h01
            else:
                int = h10
            for elem in get_large(int, thresh_frac=thresh)[ref_ind]:
                #for chg in range(min(monomer_charges[0]), max(monomer_charges[0])):
                # skip frozen core contributions
                if elem in frozen:
                    continue
                if elem in conf_decoder(total_gs_config_neutral, n_orbs):  # filter out occupied orbitals
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
                    #print(chg, conf_decoder(ex))
                    missing_states[frag][chg][ex] = dens_builder_stuff[frag][0][chg].configs.index(ex)

    #print(missing_states[0])

    # neutral contributions from two el ints for single excitations (less important, but still relevant for high precision...)
    
    gs = total_gs_config_neutral
    for frag in range(2):
        # the following loop is unique here, because one should also loop over charge 1, but for
        # Be with frozen core only the frozen electron is left for the cation for one spin contribution.
        # That means, one could only excite the electron of the other spin, corresponding to a shake up
        # state over two different spins...These contributions are expected to be small though and due to
        # the ambiguity of choosing states, which form densities covering the remaining large contributions
        # of the integrals, only the strongest expected contributions are accounted for, which would be
        # an ionization and excitation in the same spin, which is not possible though for Be with frozen core.
        for chg in [0]:#range(min(monomer_charges[frag]), max(monomer_charges[frag])):  # loops over -1 and 0
            #dens = dens_looper(raw(dens_arr[frag]["ca"][(chg,chg)]))
            int = v0101
            #for elem in missing_orbs(int, dens, frag):  # ref_inds 2 and 3 should be equal to 0 and 1
            for elem in get_large(int, thresh_frac=single_thresh)[frag]:
                if elem in frozen:
                    continue
                if elem in conf_decoder(total_gs_config_neutral, n_orbs):  # filter out occupied orbitals
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

    #print(missing_states[0])

    # charged contributions from two el ints for single and double excitations
    for frag in range(2):
        frag_inds = [1-frag, 1-frag, 1-frag, 1-frag]
        for special_ind in [0,2]:
            frag_inds[special_ind] = frag
            int = get_v(tuple(frag_inds))
            for comb in get_large(int, thresh_frac=(1/1.5) * single_thresh * double_thresh * triple_thresh, compress_ouput=False):
                # to understand the following sorting you must know, that according to the working equaitons B30 and B31
                # the last two indices of the two el int always refer to annihilations and the first two always to creations
                if any(elem in frozen for elem in comb):
                    continue
                """
                # filter out single excitations with ionizations, which are actually only ionizations or completely wrong
                # ...again I'm not entirely sure, we might miss contributions this way
                if comb[0] in conf_decoder(total_gs_config_neutral):
                    continue
                if comb[1] in conf_decoder(total_gs_config_neutral):
                    continue
                if comb[2] not in conf_decoder(total_gs_config_neutral):
                    continue
                if comb[3] not in conf_decoder(total_gs_config_neutral):
                    continue
                if (special_ind == 0 and comb[2] == comb[3]) or (special_ind == 2 and comb[0] == comb[1]):
                    continue
                """
                if special_ind == 0:
                    if comb[2] == comb[3]:
                        continue
                    if comb[0] in conf_decoder(total_gs_config_neutral, n_orbs):
                        continue
                    if comb[1] in conf_decoder(total_gs_config_neutral, n_orbs) and not comb[1] in comb[2:]:
                        continue
                    if comb[2] not in conf_decoder(total_gs_config_neutral, n_orbs):# and comb[2] != comb[1]:
                        continue
                    if comb[3] not in conf_decoder(total_gs_config_neutral, n_orbs):# and comb[3] != comb[1]:
                        continue
                else:
                    if comb[0] == comb[1]:
                        continue
                    if comb[2] not in conf_decoder(total_gs_config_neutral, n_orbs):
                        continue
                    if comb[3] not in conf_decoder(total_gs_config_neutral, n_orbs):# and not comb[3] in comb[:2]:
                        continue
                    if comb[0] in conf_decoder(total_gs_config_neutral, n_orbs) and comb[0] != comb[3]:
                        continue
                    if comb[1] in conf_decoder(total_gs_config_neutral, n_orbs) and comb[1] != comb[3]:
                        continue
                #print("valid ", frag, special_ind, comb)
                if special_ind == 0:
                    ex_minus = total_gs_config_neutral + 2**comb[0]
                    ex_plus = total_gs_config_neutral + 2**comb[1] - 2**comb[2] - 2**comb[3]
                else:
                    ex_plus = total_gs_config_neutral - 2**comb[2]
                    ex_minus = total_gs_config_neutral + 2**comb[0] + 2**comb[1] - 2**comb[3]
                #print(conf_decoder(ex_minus), conf_decoder(ex_plus))
                missing_states[frag][-1][ex_minus] = dens_builder_stuff[frag][0][-1].configs.index(ex_minus)
                missing_states[frag][1][ex_plus] = dens_builder_stuff[frag][0][1].configs.index(ex_plus)

    """
    # check which double excitations are covered already for neutral and anion
    occ_orbs = [i for i in range(n_occ[0])] + [i + n_orbs for i in range(n_occ[1])]
    for frag in range(2):
        for chg in range(min(monomer_charges[frag]), max(monomer_charges[frag])):  # loops over -1 and 0
            dens = raw(dens_arr[frag]["ca"][(chg,chg)])
            covered_double_exs = []
            for bra in range(dens.shape[0]):
                diag = np.diag(dens[bra, bra, :, :])
                if sum([diag[i] for i in occ_orbs]) < 2.9:  # a little double exc character is ok, since we are looking for dominant double exc contributions
                    covered_double_exs.append(bra)
            if len(covered_double_exs) > 0:
                # if there are any encountered already, one doesnt have account for them anymore in the following,
                # but instead provide these contributions as singly excited states
                print("double excitation contributions already encountered in initially provided state"
                      f"numbers {covered_double_exs} of chg {chg} in fragment {frag}."
                      "Note, that for now this is not covered")
    """
    
    # ionization contributions from one el ints for double excitations without spin flip
    for frag in range(2):
        for chg in range(min(monomer_charges[frag]), max(monomer_charges[frag])):  # loops over -1 and 0
            # this currently only works with charges +1,0,-1 and densities only with initial neutral state
            if chg == 0:
                ref_ind = 1
                # one should actually sweep over all possible reference states
                gs = total_gs_config_neutral - 2**(n_orbs - 1 + n_occ[1]) - 2**(-1 + n_occ[0])
                thresh = double_thresh
            elif chg == -1:
                ref_ind = 0
                thresh = double_thresh#1/5#single_thresh
            else:
                raise ValueError("invalid charge provided")
            if frag == ref_ind:
                int = h01
            else:
                int = h10 
            for pair in combinations(get_large(int, thresh_frac=thresh)[ref_ind], 2):  # we cannot provide whole double space, so adjust thresh_frac (maybe iteratively)
                if pair[0] == pair[1]:  # cant put two electrons in the same spin orbital
                    continue
                #if any(elem in frozen for elem in pair):  # filter out forbidden and single excitations
                #    continue
                if (pair[0] < n_orbs and pair[1] < n_orbs) or (pair[0] >= n_orbs and pair[1] >= n_orbs):  # filter out spin flip
                    continue
                if any(elem in conf_decoder(total_gs_config_neutral, n_orbs) for elem in pair):  # filter out forbidden and single excitations
                    continue
                # get gs for required spin and assume more positively charged state to be gs
                if chg == -1:
                    gs = total_gs_config_neutral - 2**(n_orbs - 1 + n_occ[1])  # one should rather sweep over the available ionized contributions
                    ex = gs + 2**pair[0] + 2**pair[1]
                    #print(conf_decoder(total_gs_config_neutral), conf_decoder(gs), conf_decoder(ex))
                    #print(pair)
                    if ex not in missing_states[frag][chg].keys():
                        missing_states[frag][chg][ex] = dens_builder_stuff[frag][0][chg].configs.index(ex)
                    gs = total_gs_config_neutral - 2**(-1 + n_occ[0])  # one should rather sweep over the available ionized contributions
                ex = gs + 2**pair[0] + 2**pair[1]
                if ex not in missing_states[frag][chg].keys():
                    #det_state = np.zeros_like(dens_builder_stuff[frag][0][chg].configs)
                    #det_state[dens_builder_stuff[frag][0][chg].configs.index(ex)] = 1.
                    #missing_states[frag][chg][ex] = det_state
                    missing_states[frag][chg][ex] = dens_builder_stuff[frag][0][chg].configs.index(ex)
    
    #print(missing_states[0])

    # ionization contributions from one el ints for anionic triple excitations without spin flip
    for frag in range(2):
        for chg in [-1]:#range(min(monomer_charges[frag]), max(monomer_charges[frag])):  # loops over -1 and 0
            # this currently only works with charges +1,0,-1 and densities only with initial neutral state
            #elif chg == -1:
            ref_ind = 0
            # one should actually sweep over all possible reference states
            gs = total_gs_config_neutral - 2**(n_orbs - 1 + n_occ[1]) - 2**(-1 + n_occ[0])
            #else:
            #    raise ValueError("invalid charge provided")
            if frag == ref_ind:
                int = h01
            else:
                int = h10 
            for pair in combinations(get_large(int, thresh_frac=triple_thresh)[ref_ind], 3):  # we cannot provide whole triple space, so adjust thresh_frac (maybe iteratively)
                # skip frozen core contributions
                if pair[0] == pair[1] or (pair[0] == pair[2] or pair[1] == pair[2]):  # cant put two electrons in the same spin orbital
                    continue
                #if any(elem in frozen for elem in pair):  # filter out forbidden and single excitations
                #    continue
                if all(elem < n_orbs for elem in pair) or all(elem >= n_orbs for elem in pair):  # filter out spin flip
                    continue
                if any(elem in conf_decoder(total_gs_config_neutral, n_orbs) for elem in pair):  # filter out forbidden, single and double excitations
                    continue
                ex = gs + 2**pair[0] + 2**pair[1] + 2**pair[2]
                if ex not in missing_states[frag][chg].keys():
                    #det_state = np.zeros_like(dens_builder_stuff[frag][0][chg].configs)
                    #det_state[dens_builder_stuff[frag][0][chg].configs.index(ex)] = 1.
                    #missing_states[frag][chg][ex] = det_state
                    missing_states[frag][chg][ex] = dens_builder_stuff[frag][0][chg].configs.index(ex)

    #print(missing_states[0])
    
    # singly and doubly excited anionic determinants from double ionization term
    gs_list = [2**n_occ[i] for i in range(2)]
    for frag in range(2):
        if frag == 0:
            int = get_v((0, 0, 1, 1))
        else:
            int = get_v((1, 1, 0, 0))
        for comb in get_large(int, thresh_frac=double_thresh, compress_ouput=False):
            if any(elem in frozen for elem in comb):
                continue
            if comb[2] == comb[3]:
                continue
            if comb[0] in conf_decoder(total_gs_config_neutral, n_orbs) and not comb[0] in gs_list:
                continue
            if comb[1] in conf_decoder(total_gs_config_neutral, n_orbs) and not comb[1] in gs_list:# and not comb[1] in comb[2:]:
                continue
            if comb[2] not in conf_decoder(total_gs_config_neutral, n_orbs):# and comb[2] != comb[1]:
                continue
            if comb[3] not in conf_decoder(total_gs_config_neutral, n_orbs):# and comb[3] != comb[1]:
                continue
            gs1 = total_gs_config_neutral - 2**(n_orbs - 1 + n_occ[1])  # one should rather sweep over the available ionized contributions
            gs2 = total_gs_config_neutral - 2**(-1 + n_occ[0])  # one should rather sweep over the available ionized contributions
            ex1 = gs1 + 2**comb[0] + 2**comb[1]
            ex2 = gs2 + 2**comb[0] + 2**comb[1]
            #print(conf_decoder(ex_minus), conf_decoder(ex_plus))
            missing_states[frag][-1][ex1] = dens_builder_stuff[frag][0][-1].configs.index(ex1)
            missing_states[frag][-1][ex2] = dens_builder_stuff[frag][0][-1].configs.index(ex2)

    # neutral spin flip contributions (only for chg 0) from two el ints for single excitations (seems like these are only necessary for 1e-6 Hartree precision)
    gs = total_gs_config_neutral
    for frag in range(2):
        #for chg in range(min(monomer_charges[frag]), max(monomer_charges[frag])):  # loops over -1 and 0
        chg = 0
        #dens = dens_looper(raw(dens_arr[frag]["ca"][(chg,chg)]))
        int = v0101
        #for elem in missing_orbs(int, dens, frag):  # ref_inds 2 and 3 should be equal to 0 and 1
        # TODO: something needs to be done with this threshold, which needs to be chosen ridiculously small to not incorporate all determinants
        for elem in get_large(int, thresh_frac=sp_thresh)[frag]:  # spin flip is reduced to lower threshold than single here!!!!!!!!!!!!!!!!!!!!!!!!!
            if elem in frozen:
                continue
            if elem in conf_decoder(total_gs_config_neutral, n_orbs):  # filter out occupied orbitals
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
        
    #print(missing_states[0])

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
    ret = {}
    for frag in range(2):
        ret[frag] = {}
        for chg in monomer_charges[frag]:#range(min(monomer_charges[frag]), max(monomer_charges[frag])):
            det_states = []
            already_included = []
            for det, ind in missing_states[frag][chg].items():
                if det in already_included:
                    continue
                det_dec = conf_decoder(det, n_orbs)
                print(chg, det_dec)
                det_states.append(np.zeros_like(dens_builder_stuff[frag][0][chg].coeffs[0]))
                det_states[-1][ind] = 1.
                already_included.append(det)
                # the following part "sorts" the filtered determinants such that every non-singlet determinant is followed by its counterpart,
                # e.g. the alpha HOMO -> beta LUMO determinant is followed by the beta HOMO -> alpha LUMO determinant for a singlet reference system.
                # This is important for the stability of the solver later, where these determinants are explicitly provided, but cannot be taken into
                # the model state space all at once. However, it was found for Be 6-31g that no sorting makes the result much worse for gradient based opt!
                """
                it_is_singlet, pair = is_singlet(det_dec, n_orbs)
                if pair in already_included:
                    continue
                #if it_is_singlet:
                #    continue
                try:
                    mirror_det_ind = missing_states[frag][chg][pair]
                    print(chg, conf_decoder(pair, n_orbs))
                    det_states.append(np.zeros_like(dens_builder_stuff[frag][0][chg].coeffs[0]))
                    det_states[-1][mirror_det_ind] = 1.
                    already_included.append(pair)
                except KeyError:
                    continue
                """
            #if len(det_states) >= 150:
            #    print(len(det_states))
            #    raise RuntimeError(f"for fragment {frag} with charge {chg} the additionally screened states exceed 150...this will take forever")
            print(f"for fragment {frag} with charge {chg} {len(det_states)} determinants are taken into account")
            #new_states = dens_builder_stuff[frag][0][chg].coeffs + det_states #list(missing_states[frag][chg].values())
            #new_states = orthogonalize(np.array(new_states))
            #ret[frag][chg] = [i for i in new_states]
            #print(frag, chg, len(det_states))
            ret[frag][chg] = det_states#[i for i in det_states]

    # the upper layer expects lists for every charge, but they can be empty of course
    #for frag in range(2):
    #    pos_chg = max(monomer_charges[frag])
    #    ret[frag][pos_chg] = []

    #for chg in monomer_charges[0]:
    #    for i, vec in enumerate(ret[0][chg]):
    #        big_inds = {ind: elem for ind, elem in enumerate(vec) if abs(elem) > 1e-1}
    #        #print(i, [ind for ind, elem in enumerate(vec) if abs(elem) > 1e-1])
    #        print(chg, i, {tuple(conf_decoder(dens_builder_stuff[frag][0][chg].configs[j])): val for j, val in big_inds.items()})

    #raise ValueError("stop here")
    """
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
    return ret, missing_states
