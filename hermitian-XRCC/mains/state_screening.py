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
    h01 += raw(ints[0]("U")._as_tuple()[0][(0,1,1)])
    h01 = get_large(h01)

    h10 = raw(ints[0]("T")._as_tuple()[0][(1,0)])
    h10 += raw(ints[0]("U")._as_tuple()[0][(1,0,0)])
    h10 += raw(ints[0]("U")._as_tuple()[0][(1,0,1)])
    h10 = get_large(h10)
    # it seems like screening over V is not necessary, since h01 captures basically all contributions already...
    # this needs to be further investigated for a larger example, where not all contributions are relevant.
    # maybe instead of screening V one could also screen over the fock operator...
    # one should also screen over 1010 integrals, but this can be neglected, if the integrals are almost hermitian.
    # for strongly interacting fragments this might lead to missing contributions, so one probably needs to revisit this setup.
    v0101 = get_large(raw(ints[0]("V")._as_tuple()[0][(0,1,0,1)]))

    dens_arr = [densities.build_tensors(*dens_builder_stuff[m][:-1], options=[], n_threads=n_threads, screen=True) for m in range(2)]

    # here only the densities are taken into account, for which the initial state (ket) is the neutral state
    #p0_c_anion = dens_looper(raw(dens[0]["c"][(-1,0)]))
    #p1_a_cation = dens_looper(raw(dens[1]["a"][(1,0)]))
    #p1_c_anion = dens_looper(raw(dens[1]["c"][(-1,0)]))
    #p0_a_cation = dens_looper(raw(dens[0]["a"][(1,0)]))
    #p0_ca_neutral = dens_looper(raw(dens[0]["ca"][(0,0)]))
    #p1_ca_neutral = dens_looper(raw(dens[1]["ca"][(0,0)]))

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
                        gs = total_gs_config_neutral - 2**(n_orbs - 1 + n_occ[1])
                    else:  # appending alpha needs beta ref (except neutral gs)
                        #gs = min([abs(i - beta_gs_config) for i in dens_builder_stuff[frag][0][chg + 1].configs]) + beta_gs_config
                        gs = total_gs_config_neutral - 2**(-1 + n_occ[0])
                ex = gs + 2**elem
                #if chg == -1:
                #    print(elem, ex)
                if ex not in missing_states[frag][chg].keys():
                    det_state = np.zeros_like(dens_builder_stuff[frag][0][chg].configs)
                    det_state[dens_builder_stuff[frag][0][chg].configs.index(ex)] = 1.
                    missing_states[frag][chg][ex] = det_state
 
    # neutral contributions
    gs = total_gs_config_neutral
    for frag in range(2):
        dens = dens_looper(raw(dens_arr[frag]["ca"][(0,0)]))
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
            ex = ion + 2**elem
            if ex not in missing_states[frag][0].keys():
                det_state = np.zeros_like(dens_builder_stuff[frag][0][0].configs)
                det_state[dens_builder_stuff[frag][0][0].configs.index(ex)] = 1.
                missing_states[frag][0][ex] = det_state

    #print(missing_states[0][0].keys(), missing_states[0][-1].keys())

    for frag in range(2):
        for chg in missing_states[frag].keys():
            new_states = dens_builder_stuff[frag][0][chg].coeffs + list(missing_states[frag][chg].values())
            new_states = orthogonalize(np.array(new_states))
            dens_builder_stuff[frag][0][chg].coeffs = [i for i in new_states]
    #raise ValueError("stop here")

    #return dens_builder_stuff
