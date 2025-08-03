import adcc
import numpy as np
import psi4
from adc_density_equations.tdm_ip import tdm_1p_ip, tdm_2p_ip, tdm_3p_ip
from adc_density_equations.tdm_ea import tdm_1p_ea, tdm_2p_ea, tdm_3p_ea
from adc_density_equations.tdm_pp import tdm_2p_pp
from adc_density_equations.s2s_pp import s2s_2p_pp
from adc_density_equations.gs_pp import gs_2p_pp
#from adc_density_equations.s2s_2ea import s2s_2p_2ea, s2s_3p_2ea
from adc_density_equations.s2s_2ip import s2s_2p_2ip, s2s_3p_2ip
from adc_density_equations.s2s_ip_0 import s2s_2p_ip_0
from adc_density_equations.s2s_ea_0 import s2s_2p_ea_0
from adc_density_equations.s2s_ip import s2s_1p_ip, s2s_2p_ip, s2s_3p_ip
from adc_density_equations.s2s_ea import s2s_1p_ea, s2s_2p_ea, s2s_3p_ea
from qode.math.tensornet import evaluate, raw, tl_tensor  #, increment, raw, scalar_value, contract, tensor_sum
#import pickle
import time
#import ray
import tensorly as tl
import torch
import itertools

if True:
    tl.set_backend("pytorch")
    torch.set_num_threads(4)
    #ray.init(num_cpus=1)
    #########
    # Get SCF result
    #########

    mol = psi4.geometry("""
        Be 0 0 0
        symmetry c1
        units angstrom
        no_reorient
    """)
    #psi4.set_num_threads(adcc.get_n_threads())
    psi4.core.be_quiet()
    psi4.set_options({'basis': "6-31g",
                    #'num_frozen_docc': 1,#})#,
                    'freeze_core': True,
                    #'cholesky_tolerance': 1e-6,
                    #'scf_type': "pk",
                    'guess': "read",
                    #'reference': "rhf",
                    'e_convergence': 1e-10,
                    'd_convergence': 1e-8})#,
                    #'scf_type': 'df'})
    scf_e, wfn = psi4.energy('scf', return_wfn=True, restart_file="psi4_ref_wfn_from_Be_C.npy") #restart_file="psi4_ref_wfn_monomer.npy")

    #pickle.dump(wfn, open("psi4_wfn_object_Be.pkl", mode="wb"))

    #print(np.asarray(wfn.Ca()))

    #def get_psi4_Ca():
    #    return tl.tensor(np.asarray(wfn.Ca()), dtype=tl.float64)

    #########
    # Get ADC excited states objects for IP, EA and PP
    #########

    # interesting states, according to Be2_best_states, with unmapped states after #, for minimum energy diff of 1e-2
    #interesting_pp_singlets = [5, 6, 7, 8, 9, 17, 22, 25, 28, 31, 32]  # + [23, 24, 33]
    #interesting_pp_triplets = [6, 7, 8, 14, 20, 24, 26, 27]  # + [18, 21]
    #interesting_ip_doublets = [1, 7]  # actually just 7, for min en diff 3e-2
    #interesting_ea_doublets = [0, 6, 7, 8, 9, 10, 11, 22, 24, 36, 38, 40, 43, 46, 50, 51, 52, 54]  # + [14, 15, 23, 25, 26, 27, 28, 29, 35, 37, 39, 53]

    # interesting states, according to Be2_best_states, with unmapped states after +, for minimum energy diff of 3e-2
    # this is with consistent orbital guess parsing
    # also the unmapped states are the few states, which can never be mapped properly
    #interesting_pp_singlets = [5, 22, 23, 28, 31, 32, 33] + [7, 24]
    #interesting_pp_triplets = [6, 24, 26, 27] + [21]
    # since ip_doublets are very few states, the following have minimum en diff of 1e-3
    #interesting_ip_doublets = [1, 4, 5, 6, 7]
    #interesting_ip_doublets = [1, 7]  # min en diff 1e-2
    #interesting_ea_doublets = [6, 32, 35, 43, 50, 51, 52, 53, 54] + [15, 25, 37, 39]
    #interesting_ea_doublets = [6, 43, 51, 52, 54] + [15, 25, 37, 39]  # min en diff 3.25e-2

    #states_pp = adcc.adc2(wfn, n_singlets=35, frozen_core=1)#adcc.adc2(wfn, n_states=7, frozen_core=1)
    #states_pp_triplet = adcc.adc2(wfn, n_triplets=28, frozen_core=1)
    #states_ip = adcc.ip_adc2(wfn, n_doublets=8, frozen_core=1)
    #states_ea = adcc.ea_adc2(wfn, n_doublets=56, frozen_core=1)
    #states_pp = adcc.adc1(wfn, n_singlets=2, frozen_core=1)#adcc.adc2(wfn, n_states=7, frozen_core=1)
    #states_pp_triplet = adcc.adc1(wfn, n_triplets=2, frozen_core=1)
    #states_ip = adcc.ip_adc1(wfn, n_doublets=2, frozen_core=1)
    #states_ea = adcc.ea_adc1(wfn, n_doublets=2, frozen_core=1)
    #states_pp = adcc.adc0(wfn, n_singlets=14)#, frozen_core=1)#adcc.adc2(wfn, n_states=7, frozen_core=1)
    #states_pp_triplet = adcc.adc0(wfn, n_triplets=14)#, frozen_core=1)
    #states_ip = adcc.ip_adc0(wfn, n_doublets=1)#, frozen_core=1)
    #states_ea = adcc.ea_adc0(wfn, n_doublets=1)#, frozen_core=1)
    states_pp = adcc.adc1(wfn, n_singlets=7, frozen_core=1)#adcc.adc2(wfn, n_states=7, frozen_core=1)
    #states_pp_triplet = adcc.adc1(wfn, n_triplets=14)#, frozen_core=1)
    states_ip = adcc.ip_adc1(wfn, is_alpha=True, n_doublets=1, frozen_core=1)
    states_ea = adcc.ea_adc1(wfn, is_alpha=True, n_doublets=7, frozen_core=1)
    #states_ip_beta = adcc.ip_adc1(wfn, is_alpha=False, n_doublets=2)#, frozen_core=1)
    #states_ea_beta = adcc.ea_adc1(wfn, is_alpha=False, n_doublets=7)#, frozen_core=1)
    #print(states_pp.describe_amplitudes())
    #print("first neutral excited state", states_pp.excitation_vector[0].ph)
    #print("anionic ground state", states_ea.excitation_vector[0].p)
    #print("neutral @ anionic = occ block of density", states_pp.excitation_vector[0].ph.to_ndarray() @ states_ea.excitation_vector[0].p.to_ndarray())

    

    #setattr(states_pp.ground_state.mospaces, "tmp_change_to_unfrozen_core", True)
    #setattr(states_pp_triplet.ground_state.mospaces, "tmp_change_to_unfrozen_core", True)
    #setattr(states_ip.ground_state.mospaces, "tmp_change_to_unfrozen_core", True)
    #setattr(states_ea.ground_state.mospaces, "tmp_change_to_unfrozen_core", True)

    
    ex_vecs_pp = states_pp.excitation_vector #+ states_pp_triplet.excitation_vector
    ex_vecs_ip = states_ip.excitation_vector #+ states_ip_beta.excitation_vector
    ex_vecs_ea = states_ea.excitation_vector #+ states_ea_beta.excitation_vector
    

    #ex_vecs_pp = [vec for i, vec in enumerate(states_pp.excitation_vector) if i in interesting_pp_singlets] + [vec for i, vec in enumerate(states_pp_triplet.excitation_vector) if i in interesting_pp_triplets]
    #ex_vecs_ip = [vec for i, vec in enumerate(states_ip.excitation_vector) if i in interesting_ip_doublets]
    #ex_vecs_ea = [vec for i, vec in enumerate(states_ea.excitation_vector) if i in interesting_ea_doublets]


    #states_pp_ph = pickle.load(open("states_pp_ph.pkl", mode="rb"))
    #states_ip_h = pickle.load(open("states_ip_h.pkl", mode="rb"))
    #states_ea_p = pickle.load(open("states_ea_p.pkl", mode="rb"))
    #states_pp_pphh = pickle.load(open("states_pp_pphh.pkl", mode="rb"))
    #states_ip_phh = pickle.load(open("states_ip_phh.pkl", mode="rb"))
    #states_ea_pph = pickle.load(open("states_ea_pph.pkl", mode="rb"))

    #print([np.linalg.norm(state - states_pp.excitation_vector[i].ph.to_ndarray()) for i, state in enumerate(states_pp_ph)])
    #print([np.linalg.norm(state - states_ip.excitation_vector[i].h.to_ndarray()) for i, state in enumerate(states_ip_h)])
    #print([np.linalg.norm(state - states_ea.excitation_vector[i].p.to_ndarray()) for i, state in enumerate(states_ea_p)])
    #print([np.linalg.norm(state - states_pp.excitation_vector[i].pphh.to_ndarray()) for i, state in enumerate(states_pp_pphh)])
    #print([np.linalg.norm(state - states_ip.excitation_vector[i].phh.to_ndarray()) for i, state in enumerate(states_ip_phh)])
    #print([np.linalg.norm(state - states_ea.excitation_vector[i].pph.to_ndarray()) for i, state in enumerate(states_ea_pph)])


    #pickle.dump([i.ph.to_ndarray() for i in states_pp.excitation_vector], open("states_pp_ph.pkl", mode="wb"))
    #pickle.dump([i.h.to_ndarray() for i in states_ip.excitation_vector], open("states_ip_h.pkl", mode="wb"))
    #pickle.dump([i.p.to_ndarray() for i in states_ea.excitation_vector], open("states_ea_p.pkl", mode="wb"))
    #pickle.dump([i.pphh.to_ndarray() for i in states_pp.excitation_vector], open("states_pp_pphh.pkl", mode="wb"))
    #pickle.dump([i.phh.to_ndarray() for i in states_ip.excitation_vector], open("states_ip_phh.pkl", mode="wb"))
    #pickle.dump([i.pph.to_ndarray() for i in states_ea.excitation_vector], open("states_ea_pph.pkl", mode="wb"))



    #########
    # Build OneParticleObject and extract all necessary information from it, to separate 
    # the full np tensors into their orbital space slices with a simple function call
    #########

    #gs_dm_1p_pp = adcc.LazyMp.density(states_pp.ground_state, level=2)
    gs_dm_1p_pp = adcc.LazyMp.density(states_pp.ground_state, level=1)

    frozen_core = False  # in principle one also needs to generalize to frozen virtuals, but that's usually not needed in practice

    for ss in gs_dm_1p_pp.orbital_subspaces:
        if not "o" in ss and not "v" in ss:
            raise KeyError("something else in orbital_subspace than o and v")
        if "o3" in ss:
            frozen_core = True
        if ss not in ["o3", "o1", "v1"]:
            raise KeyError(f"no recipe known to take care of orbital subspace of type {ss}, only o3, o1 and v1")

    #print("care that frozen_core is set to false always!")
    #print("is frozen core true?", frozen_core)
    #frozen_core = False


    offsets = {
        sp: sum(
            gs_dm_1p_pp.mospaces.n_orbs(ss)
            for ss in gs_dm_1p_pp.orbital_subspaces[:gs_dm_1p_pp.orbital_subspaces.index(sp)]
        )
        for sp in gs_dm_1p_pp.orbital_subspaces
    }

    # slices for each space
    slices = {
        sp: slice(offsets[sp], offsets[sp] + gs_dm_1p_pp.mospaces.n_orbs(sp))
        for sp in gs_dm_1p_pp.orbital_subspaces
    }

    #testtensor = np.diag([i for i in np.arange(mo_size)])

    orb_str_map = {"o":"o1", "v":"v1", "f":"o3"}

    # return requested subblock of a full tensor
    def subblock(tensor, mo_string):
        mos = list(mo_string)
        block = tuple(slices[orb_str_map[ss]] for ss in mos)
        #print(type(tensor))
        return tensor[block]


    # return full tensor from a dict of subblocks
    #@ray.remote
    def full_tensor(tensor):
        #print("all of this belongs to one tensor")
        #import numpy as np
        #from qode.math.tensornet import raw
        n_orb_total = sum(gs_dm_1p_pp.mospaces.n_orbs(i) for i in gs_dm_1p_pp.orbital_subspaces)
        ten_shape = tuple([n_orb_total] * len(list(tensor.keys())[0]))
        #ret = np.zeros(ten_shape)
        ret = tl.zeros(ten_shape, dtype=tl.float64)
        for key in tensor:
            mos = list(key)
            #if "f" in mos:
            #    print(key)
            #    print(raw(tensor[key]))
            block = tuple(slices[orb_str_map[ss]] for ss in mos)
            #print(key, mos)
            #print(block)
            #print(ret[block])
            #print(tensor[key])
            #print(type(ret))
            ret[block] = raw(tensor[key])
        return ret

    #@ray.remote
    #def full_tensor_task(tensor, n_orb_total):
    #    return full_tensor(tensor, n_orb_total)

    #def full_density(density):
        #n_orb_total = sum(gs_dm_1p_pp.mospaces.n_orbs(i) for i in gs_dm_1p_pp.orbital_subspaces)
        #return [ray.get([full_tensor.remote(j, n_orb_total) for j in i]) for i in density]
        #return [[full_tensor.remote(j, n_orb_total) for j in i] for i in density]
        #return [[full_tensor(j) for j in i] for i in density]


    #def full_density(density):
    #    full_dens = full_density_pre(density)
    #    ret = [[den.copy() for den in vec] for vec in full_dens]
    #    del full_dens
    #    return ret


    # Note, that the densities only need to be build for non-frozen orbitals
    # densities of frozen orbitals are trivial
    #d_vv = np_tensor(np.identity(gs_dm_1p_pp.mospaces.n_orbs("v1")))
    #d_oo = np_tensor(np.identity(gs_dm_1p_pp.mospaces.n_orbs("o1")))
    #if frozen_core:
    #    d_oo_frozen = np_tensor(np.identity(gs_dm_1p_pp.mospaces.n_orbs("o3")))
    d_vv = tl_tensor(tl.eye(gs_dm_1p_pp.mospaces.n_orbs("v1")))
    d_oo = tl_tensor(tl.eye(gs_dm_1p_pp.mospaces.n_orbs("o1")))
    if frozen_core:
        d_oo_frozen = tl_tensor(tl.eye(gs_dm_1p_pp.mospaces.n_orbs("o3")))

    #print(subblock(testtensor, "oo"))

    #print([(split_spaces(block), gs_dm_1p_pp[block].to_ndarray()) for block in gs_dm_1p_pp.blocks_nonzero])
    #print(gs_dm_1p_pp.to_ndarray())

    #print(slices)

    def get_psi4_Ca():
        print("beware that everything is done here in restricted mode and with the same orbital spaces on every fragment")
        Ca = tl.tensor(np.asarray(wfn.Ca()), dtype=tl.float64)
        n_orbs_restricted = {
            sp: gs_dm_1p_pp.mospaces.n_orbs(sp) // 2
            for sp in gs_dm_1p_pp.orbital_subspaces
        }
        return Ca, n_orbs_restricted
    


    """
    def frozen_blocks_2p_helper(orb_str_list, gs_2p_pp, d_oo_frozen, subblock_1p):
        delta_dispatch = {"ff": d_oo_frozen, "oo": subblock_1p, "vv": subblock_1p, "ov": subblock_1p, "vo": subblock_1p(1,0)}
        for elem in orb_str_list:
            if (elem[0] == elem[3] and elem[0] == "f") or (elem[1] == elem[2] and elem[1] == "f"):
                gs_2p_pp[''.join(elem)] = delta_dispatch[elem[0] + elem[3]](0,3) @ delta_dispatch[elem[1] + elem[2]](1,2)
            elif (elem[1] == elem[3] and elem[1] == "f") or (elem[0] == elem[2] and elem[0] == "f"):
                gs_2p_pp[''.join(elem)] = - delta_dispatch[elem[1] + elem[3]](1,3) @ delta_dispatch[elem[0] + elem[2]](0,2)
            else:
                continue
    """


    # 2p frozen density blocks in zeroth order ... Not sure, if this is correct for higher perturbation orders too
    #def gs_2p_pp_frozen_blocks(gs_2p_pp, d_oo_frozen, d_oo, d_vv):
    #    gs_2p_pp["ffff"] = d_oo_frozen(0,3) @ d_oo_frozen(1,2) - d_oo_frozen(1,3) @ d_oo_frozen(0,2)
    #    possible_ffoo = list(itertools.permutations(["f", "f", "o", "o"]))
    #    frozen_blocks_2p_helper(possible_ffoo, gs_2p_pp, d_oo_frozen, d_oo, d_vv)


    # here we are facing an inconvenience, since only blocks without frozen core orbital are calculated,
    # but for the 2p objects only the ffff block is trivial. The blocks consisting of 2 occ and to f_occ orbs
    # are not trivial though, but one can obtain them from the corresponding 1p objects. The problem here is,
    # that they are not the same for all indexes, so dependent on the index we have to build a new gs_2p_pp
    # object, which provides the correct blocks containing frozen core blocks.

    """
    def build_2p_gs_object(gs_2p_initial, dens_1p, gs_1p=None):
        if frozen_core is False:
            return gs_2p_initial
        else:
            if gs_1p is not None:
                dens_1p.oo += gs_1p.oo
                dens_1p.vv += gs_1p.vv
                dens_1p.ov += gs_1p.ov
                # only symmetric densities are used here (same bra as ket state), so no .vo block needed
            gs_2p_initial["ffff"] = d_oo_frozen(0,3) @ d_oo_frozen(1,2) - d_oo_frozen(1,3) @ d_oo_frozen(0,2)
            possible_ffoo = list(itertools.permutations(["f", "f", "o", "o"]))
            frozen_blocks_2p_helper(possible_ffoo, gs_2p_initial, d_oo_frozen, tl_tensor(tl.tensor(dens_1p.oo.to_ndarray(), dtype=tl.float64)))
            possible_ffoo = list(itertools.permutations(["f", "f", "v", "v"]))
            frozen_blocks_2p_helper(possible_ffoo, gs_2p_initial, d_oo_frozen, tl_tensor(tl.tensor(dens_1p.vv.to_ndarray(), dtype=tl.float64)))
            possible_ffoo = list(itertools.permutations(["f", "f", "o", "v"]))
            frozen_blocks_2p_helper(possible_ffoo, gs_2p_initial, d_oo_frozen, tl_tensor(tl.tensor(dens_1p.ov.to_ndarray(), dtype=tl.float64)))
            #print(gs_2p_initial)
            return gs_2p_initial
    """


    def pp_2p_f_core_helper(tensor, orb_str_perms, lower_p_dens):
        def dispatch(elem0, elem1):
            #print(elem0 + elem1, subblock(lower_p_dens, elem0 + elem1))
            return tl_tensor(subblock(lower_p_dens, elem0 + elem1))
        for el in orb_str_perms:
            if el[0] == "f" and el[3] == "f":
                tensor[''.join(el)] = d_oo_frozen(0,3) @ dispatch(el[1], el[2])(1,2)
            elif el[1] ==  "f" and el[2] == "f":
                tensor[''.join(el)] = dispatch(el[0], el[3])(0,3) @ d_oo_frozen(1,2)
            elif el[1] == "f" and el[3] == "f":
                tensor[''.join(el)] = - d_oo_frozen(1,3) @ dispatch(el[0], el[2])(0,2)
            elif el[0] == "f" and el[2] == "f":
                tensor[''.join(el)] = - dispatch(el[1], el[3])(1,3) @ d_oo_frozen(0,2)
            else:
                continue

    
    def ip_2p_f_core_helper(tensor, orb_str_perms, lower_p_dens):
        def dispatch(elem0):
            #print(elem0 + elem1, subblock(lower_p_dens, elem0 + elem1))
            return tl_tensor(subblock(lower_p_dens, elem0))
        for el in orb_str_perms:
            if el[0] == "f" and el[1] == "f":
                tensor[''.join(el)] = d_oo_frozen(0,1) @ dispatch(el[2])(2)
            elif el[0] ==  "f" and el[2] == "f":
                tensor[''.join(el)] = - dispatch(el[1])(1) @ d_oo_frozen(0,2)
            else:
                continue


    def ea_2p_f_core_helper(tensor, orb_str_perms, lower_p_dens):
        def dispatch(elem0):
            #print(elem0 + elem1, subblock(lower_p_dens, elem0 + elem1))
            return tl_tensor(subblock(lower_p_dens, elem0))
        for el in orb_str_perms:
            if el[1] == "f" and el[2] == "f":
                tensor[''.join(el)] = d_oo_frozen(1,2) @ dispatch(el[0])(0)
            elif el[0] ==  "f" and el[2] == "f":
                tensor[''.join(el)] = - dispatch(el[1])(1) @ d_oo_frozen(0,2)
            else:
                continue


    def ip2_3p_f_core_helper(tensor, orb_str_perms, lower_p_dens):
        def dispatch(elem0, elem1):
            #print(elem0 + elem1, subblock(lower_p_dens, elem0 + elem1))
            return tl_tensor(subblock(lower_p_dens, elem0 + elem1))
        for el in orb_str_perms:
            if el[0] == "f" and el[1] == "f":
                tensor[''.join(el)] = d_oo_frozen(0,1) @ dispatch(el[2], el[3])(2,3)
            elif el[0] ==  "f" and el[2] == "f":
                tensor[''.join(el)] = - d_oo_frozen(0,2) @ dispatch(el[1], el[3])(1,3)
            elif el[0] == "f" and el[3] == "f":
                tensor[''.join(el)] = d_oo_frozen(0,3) @ dispatch(el[1], el[2])(1,2)
            else:
                continue


    def ea2_3p_f_core_helper(tensor, orb_str_perms, lower_p_dens):
        def dispatch(elem0, elem1):
            #print(elem0 + elem1, subblock(lower_p_dens, elem0 + elem1))
            return tl_tensor(subblock(lower_p_dens, elem0 + elem1))
        for el in orb_str_perms:
            if el[2] == "f" and el[3] == "f":
                tensor[''.join(el)] = d_oo_frozen(2,3) @ dispatch(el[0], el[1])(0,1)
            elif el[1] ==  "f" and el[3] == "f":
                tensor[''.join(el)] = - d_oo_frozen(1,3) @ dispatch(el[0], el[2])(0,2)
            elif el[0] == "f" and el[3] == "f":
                tensor[''.join(el)] = d_oo_frozen(0,3) @ dispatch(el[1], el[2])(1,2)
            else:
                continue


    def ip_3p_f_core_helper(tensor, orb_str_perms, lower_p_dens):
        def dispatch(elem0, elem1, elem2):
            #print(elem0 + elem1, subblock(lower_p_dens, elem0 + elem1))
            return tl_tensor(subblock(lower_p_dens, elem0 + elem1 + elem2))
        for el in orb_str_perms:
            if el[0] == "f" and el[3] == "f":
                tensor[''.join(el)] = d_oo_frozen(0,3) @ dispatch(el[1], el[2], el[4])(1,2,4)
            elif el[0] ==  "f" and el[2] == "f":
                tensor[''.join(el)] = - d_oo_frozen(0,2) @ dispatch(el[1], el[3], el[4])(1,3,4)
            elif el[0] ==  "f" and el[4] == "f":
                tensor[''.join(el)] = - d_oo_frozen(0,4) @ dispatch(el[1], el[2], el[3])(1,2,3)
            elif el[1] == "f" and el[3] == "f":
                tensor[''.join(el)] = - d_oo_frozen(1,3) @ dispatch(el[0], el[2], el[4])(0,2,4)
            elif el[1] ==  "f" and el[2] == "f":
                tensor[''.join(el)] = d_oo_frozen(1,2) @ dispatch(el[0], el[3], el[4])(0,3,4)
            elif el[1] ==  "f" and el[4] == "f":
                tensor[''.join(el)] = d_oo_frozen(1,4) @ dispatch(el[0], el[2], el[3])(0,2,3)
            else:
                continue


    def ea_3p_f_core_helper(tensor, orb_str_perms, lower_p_dens):
        def dispatch(elem0, elem1, elem2):
            #print(elem0 + elem1, subblock(lower_p_dens, elem0 + elem1))
            return tl_tensor(subblock(lower_p_dens, elem0 + elem1 + elem2))
        for el in orb_str_perms:
            if el[0] == "f" and el[3] == "f":
                tensor[''.join(el)] = d_oo_frozen(0,3) @ dispatch(el[1], el[2], el[4])(1,2,4)
            elif el[1] == "f" and el[3] == "f":
                tensor[''.join(el)] = - d_oo_frozen(1,3) @ dispatch(el[0], el[2], el[4])(0,2,4)
            elif el[2] ==  "f" and el[3] == "f":
                tensor[''.join(el)] = d_oo_frozen(2,3) @ dispatch(el[0], el[1], el[4])(0,1,4)
            elif el[0] == "f" and el[4] == "f":
                tensor[''.join(el)] = - d_oo_frozen(0,4) @ dispatch(el[1], el[2], el[3])(1,2,3)
            elif el[1] == "f" and el[4] == "f":
                tensor[''.join(el)] = d_oo_frozen(1,4) @ dispatch(el[0], el[2], el[3])(0,2,3)
            elif el[2] ==  "f" and el[4] == "f":
                tensor[''.join(el)] = - d_oo_frozen(2,4) @ dispatch(el[0], el[1], el[3])(0,1,3)
            else:
                continue

    
    def full_density(density, dens_key=None, lower_dens=None):  # this function has only been properly tested for ccaa!!!!!!!!!!
        # lower_dens needs to be the full density, not just the difference density
        if frozen_core:
            if dens_key == None or lower_dens == None:
                return [[full_tensor(j) for j in i] for i in density]
            #    raise ValueError("full_density requires a dens_key and a lower_dens with frozen core True")
            dens_key_map = {
                "ccaa": pp_2p_f_core_helper, "caa": ip_2p_f_core_helper, "cca": ea_2p_f_core_helper,
                "ccaaa": ip_3p_f_core_helper, "cccaa": ea_3p_f_core_helper,
                "caaa": ip2_3p_f_core_helper, "ccca": ea2_3p_f_core_helper
            }
            orb_type_combs = ["f", "f"]
            if len(dens_key) == 4:
                possible_orb_types = [["o", "o"], ["v", "v"], ["o", "v"]]
            if len(dens_key) == 3:
                possible_orb_types = [["o"], ["v"]]
            if len(dens_key) == 5:
                possible_orb_types = [[i[0] + i[1] for i in list(itertools.product([["f", "f"], ["o", "o"], ["v", "v"]], [["o"], ["v"]]))]]
            if len(dens_key) <= 2 or len(dens_key) >= 6:
                # dens_key of len 2 and lower doens't contain extra blocks with f type
                raise NotImplementedError(f"dens_key only implemented up to 5, but {dens_key} was given")
            def correct_for_frozen_core(i,j):
                #if i == 0 and j == 0:
                for sub_dens_comb in possible_orb_types:
                    possible_orb_type_perms = list(itertools.permutations(orb_type_combs + sub_dens_comb))
                    dens_key_map[dens_key](density[i][j], possible_orb_type_perms, lower_dens[i][j])  # updates tensor in place
                return density[i][j]
            return [[full_tensor(correct_for_frozen_core(i,j)) for j in range(len(density[i]))] for i in range(len(density))]
        else:
            return [[full_tensor(j) for j in i] for i in density]


    #########
    # Get densities, which are already implemented from adcc
    # (for IP and EA only final pole strengths are available, so we can't extract them from adcc)
    #########

    # TODO: read in all tensors blockwise to build a dict like for the remaining densities and make the blocks np_tensor
    #gs_1p_pp = np_tensor(gs_dm_1p_pp.to_ndarray())
    #tdm_1p_pp = np_tensor(adcc.adc_pp.transition_dm("adc2", states_pp.ground_state, states_pp.excitation_vector[0]).to_ndarray())
    #s2s_1p_pp = np_tensor(adcc.adc_pp.state2state_transition_dm("adc2", states_pp.ground_state, states_pp.excitation_vector[0], states_pp.excitation_vector[0]).to_ndarray())
    # no tdm (neutral ground to charged excited state), only pole strengths, so we have to build them too later
    # check if states_pp.ground_state also works
    # the zero indicates, that these terms provide a charge difference between bra and ket of zero
    #s2s_1p_ip_0 = np_tensor(adcc.adc_ip.state2state_transition_dm("ip_adc2", states_ip.ground_state, states_ip.excitation_vector[0], states_ip.excitation_vector[0]).to_ndarray())
    #s2s_1p_ea_0 = np_tensor(adcc.adc_ea.state2state_transition_dm("ea_adc2", states_ea.ground_state, states_ea.excitation_vector[0], states_ea.excitation_vector[0]).to_ndarray())

    # for now don't extract the densities blockwise, but return them as the full final numpy tensor
    """
    gs_1p_pp = tl.tensor(gs_dm_1p_pp.to_ndarray(), dtype=tl.float64)
    tdm_1p_pp = [[tl.tensor(adcc.adc_pp.transition_dm("adc2", states_pp.ground_state, i).to_ndarray(), dtype=tl.float64)]
                for i in ex_vecs_pp]
    #tdm_1p_pp_herm_conj = [i[0].T for i in tdm_1p_pp]
    tdm_1p_pp_herm_conj = [tl.transpose(i[0]) for i in tdm_1p_pp]
    s2s_1p_pp = [[tl.tensor(adcc.adc_pp.state2state_transition_dm("adc2", states_pp.ground_state, i, j).to_ndarray(), dtype=tl.float64) for j in ex_vecs_pp]
                for i in ex_vecs_pp]
    full_1p_pp = [[gs_1p_pp] + tdm_1p_pp_herm_conj] + [i+j for i, j in zip(tdm_1p_pp, s2s_1p_pp)]
    s2s_1p_ip_0 = [[tl.tensor(adcc.adc_ip.state2state_transition_dm("ip_adc2", states_ip.ground_state, i, j).to_ndarray(), dtype=tl.float64) for j in states_ip.excitation_vector]
                for i in states_ip.excitation_vector]
    s2s_1p_ea_0 = [[tl.tensor(adcc.adc_ea.state2state_transition_dm("ea_adc2", states_ea.ground_state, i, j).to_ndarray(), dtype=tl.float64) for j in states_ea.excitation_vector]
                for i in states_ea.excitation_vector]
    """


    ###########################################################################
    ### Attention !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ###########################################################################

    # Note that the left vector (bra), i.e. i here, is in the outer loop
    # This is true for self build densities, where the bra is always on the left in the function call
    # However, in adcc it was decided to swap them in the dispatching routine!!!!!!!!!!
    # Hence, this does not affect the gs or the tdms, but the s2s densities need to be transposed!!!!!!!!!!!
    # This is only true for the densities taken from adcc!!!!!!!
    # This is why in the following few s2s densities j is on the left and i on the right of the function call.
    
    gs_1p_pp = tl.tensor(gs_dm_1p_pp.to_ndarray(), dtype=tl.float64)
    tdm_1p_pp = [[tl.tensor(adcc.adc_pp.transition_dm("adc1", states_pp.ground_state, i).to_ndarray(), dtype=tl.float64)]
                for i in ex_vecs_pp]
    #tdm_1p_pp_herm_conj = [i[0].T for i in tdm_1p_pp]
    tdm_1p_pp_herm_conj = [tl.transpose(i[0]) for i in tdm_1p_pp]
    s2s_1p_pp = [[tl.tensor(adcc.adc_pp.state2state_transition_dm("adc1", states_pp.ground_state, j, i).to_ndarray(), dtype=tl.float64) for j in ex_vecs_pp]
                for i in ex_vecs_pp]
    full_1p_pp = [[gs_1p_pp] + tdm_1p_pp_herm_conj] + [i+j for i, j in zip(tdm_1p_pp, s2s_1p_pp)]
    s2s_1p_ip_0 = [[tl.tensor(adcc.adc_ip.state2state_transition_dm("ip_adc1", states_ip.ground_state, j, i).to_ndarray(), dtype=tl.float64) for j in ex_vecs_ip]
                for i in ex_vecs_ip]
    s2s_1p_ea_0 = [[tl.tensor(adcc.adc_ea.state2state_transition_dm("ea_adc1", states_ea.ground_state, j, i).to_ndarray(), dtype=tl.float64) for j in ex_vecs_ea]
                for i in ex_vecs_ea]
    


    for i in range(1, len(full_1p_pp)):
        #if i > 0:
        full_1p_pp[i][i] += gs_1p_pp

    for i in range(len(s2s_1p_ip_0)):
        s2s_1p_ip_0[i][i] += gs_1p_pp

    for i in range(len(s2s_1p_ea_0)):
        s2s_1p_ea_0[i][i] += gs_1p_pp

    #print(gs_1p_pp[:4, :4])


    #print(s2s_1p_ip_0[0][0])

    #print(gs_1p_pp)

    #print(full_1p_pp[0][0])
    #print(full_1p_pp[1][1])
    #print(s2s_1p_ip_0[0][0])

    #########
    # Extract all objects required to build the missing density tensors
    #########

    # TODO: make these np_tensor now, so we don't have to convert them for every density we build
    #ex_vecs_pp = states_pp.excitation_vector
    #ex_vecs_ip = states_ip.excitation_vector
    #ex_vecs_ea = states_ea.excitation_vector

    t2 = tl_tensor(tl.tensor(states_pp.ground_state.t2(adcc.block.oovv).to_ndarray(), dtype=tl.float64))

    mp2_diffdm_ov = None#tl_tensor(tl.tensor(states_pp.ground_state.mp2_diffdm.ov.to_ndarray(), dtype=tl.float64))

    # We build the one particle operators in IP and EA consistently up to second order,
    # while building all higher particle operators consistently to first order.

    #########
    # Build the remaining densities
    #########

    def herm_conj(tensor):
        ret = [[0] * len(tensor) for i in np.arange(len(tensor[0]))]  # initialize list of lists with independent objects
        for i in np.arange(len(tensor)):
            for j in np.arange(len(tensor[0])):
                tmp_dict = {}
                for key in tensor[i][j]:
                    rev_order = tuple([i for i in range(len(key) - 1, -1, -1)])  # reversed orbital index ordering
                    rev_orb_string = key[::-1]  # reversed orbital string ordering
                    tmp_dict[rev_orb_string] = tensor[i][j][key](*rev_order)
                ret[j][i] = tmp_dict  # reversed bra and ket
        return ret


    def herm_conj_tdm(tensor):
        ret = []
        for i in np.arange(len(tensor)):
            tmp_dict = {}
            for key in tensor[i][0]:
                rev_order = tuple([i for i in range(len(key) - 1, -1, -1)])  # reversed orbital index ordering
                rev_orb_string = key[::-1]  # reversed orbital string ordering
                tmp_dict[rev_orb_string] = tensor[i][0][key](*rev_order)
            ret.append(tmp_dict)  # reversed bra and ket
        return ret

    # Note, that these are build without the frozen core orbitals.
    # If the frozen core orbitals are present, also include the all frozen occupied block for symmetric densities,
    # which can be build from d_oo_frozen, and the remaining blocks, including at least one
    # frozen core orbital space, which are always zero blocks
    # However, this is only relevant for the gs gs densities, where the block consisting of only
    # frozen core, e.g. o3o3o3o3 for ccaa, is diagonal. The diagonal contributions are then
    # only added to the diagonal s2s pp and ip/ea_0 tensors, since they are build as difference tensors with a non-zero gs contribution.
    # This also needs to be done for the diagonal elements of s2s_1p_pp.

    # inner list in tdm terms are due to the gs in the ket (initial state), since they are merged later
    # with s2s contributions, where ket is an excited state. It also simplifies the herm_conj function

    """
    # this is necessary for the parallelization for building the explicit tensors
    tdm_1p_ip_ = [[tdm_1p_ip(mp2_diffdm_ov, t2, vec)] for vec in ex_vecs_ip]
    s2s_1p_ip_ = [[s2s_1p_ip(t2, mp2_diffdm_ov, vec_left, vec_right) for vec_right in ex_vecs_pp] for vec_left in ex_vecs_ip]
    full_1p_ip_ = [i+j for i, j in zip(tdm_1p_ip_, s2s_1p_ip_)]
    tdm_2p_ip_ = [[tdm_2p_ip(d_oo, t2, vec)] for vec in ex_vecs_ip]
    s2s_2p_ip_ = [[s2s_2p_ip(d_oo, t2, vec_left, vec_right) for vec_right in ex_vecs_pp] for vec_left in ex_vecs_ip]
    full_2p_ip_ = [i+j for i, j in zip(tdm_2p_ip_, s2s_2p_ip_)]
    tdm_3p_ip_ = [[tdm_3p_ip(d_oo, t2, vec)] for vec in ex_vecs_ip]
    s2s_3p_ip_ = [[s2s_3p_ip(d_oo, t2, vec_left, vec_right) for vec_right in ex_vecs_pp] for vec_left in ex_vecs_ip]
    full_3p_ip_ = [i+j for i, j in zip(tdm_3p_ip_, s2s_3p_ip_)]
    tdm_1p_ea_ = [[tdm_1p_ea(mp2_diffdm_ov, t2, vec)] for vec in ex_vecs_ea]
    s2s_1p_ea_ = [[s2s_1p_ea(t2, mp2_diffdm_ov, vec_left, vec_right) for vec_right in ex_vecs_pp] for vec_left in ex_vecs_ea]
    full_1p_ea_ = [i+j for i, j in zip(tdm_1p_ea_, s2s_1p_ea_)]
    tdm_2p_ea_ = [[tdm_2p_ea(d_oo, t2, vec)] for vec in ex_vecs_ea]
    s2s_2p_ea_ = [[s2s_2p_ea(d_oo, t2, vec_left, vec_right) for vec_right in ex_vecs_pp] for vec_left in ex_vecs_ea]
    full_2p_ea_ = [i+j for i, j in zip(tdm_2p_ea_, s2s_2p_ea_)]
    tdm_3p_ea_ = [[tdm_3p_ea(d_oo, t2, vec)] for vec in ex_vecs_ea]
    s2s_3p_ea_ = [[s2s_3p_ea(d_oo, t2, vec_left, vec_right) for vec_right in ex_vecs_pp] for vec_left in ex_vecs_ea]
    full_3p_ea_ = [i+j for i, j in zip(tdm_3p_ea_, s2s_3p_ea_)]
    """

    # all of these have a charge diff of 1
    tdm_1p_ip_ = [[tdm_1p_ip(mp2_diffdm_ov, t2, vec)] for vec in ex_vecs_ip]
    s2s_1p_ip_ = [[s2s_1p_ip(t2, mp2_diffdm_ov, vec_left, vec_right) for vec_right in ex_vecs_pp] for vec_left in ex_vecs_ip]
    full_1p_ip = [i+j for i, j in zip(tdm_1p_ip_, s2s_1p_ip_)]
    tdm_2p_ip_ = [[tdm_2p_ip(d_oo, t2, vec)] for vec in ex_vecs_ip]
    s2s_2p_ip_ = [[s2s_2p_ip(d_oo, t2, vec_left, vec_right) for vec_right in ex_vecs_pp] for vec_left in ex_vecs_ip]
    full_2p_ip = [i+j for i, j in zip(tdm_2p_ip_, s2s_2p_ip_)]
    tdm_3p_ip_ = [[tdm_3p_ip(d_oo, t2, vec)] for vec in ex_vecs_ip]
    s2s_3p_ip_ = [[s2s_3p_ip(d_oo, t2, vec_left, vec_right) for vec_right in ex_vecs_pp] for vec_left in ex_vecs_ip]
    full_3p_ip = [i+j for i, j in zip(tdm_3p_ip_, s2s_3p_ip_)]
    tdm_1p_ea_ = [[tdm_1p_ea(mp2_diffdm_ov, t2, vec)] for vec in ex_vecs_ea]
    s2s_1p_ea_ = [[s2s_1p_ea(t2, mp2_diffdm_ov, vec_left, vec_right) for vec_right in ex_vecs_pp] for vec_left in ex_vecs_ea]
    full_1p_ea = [i+j for i, j in zip(tdm_1p_ea_, s2s_1p_ea_)]
    tdm_2p_ea_ = [[tdm_2p_ea(d_oo, t2, vec)] for vec in ex_vecs_ea]
    s2s_2p_ea_ = [[s2s_2p_ea(d_oo, t2, vec_left, vec_right) for vec_right in ex_vecs_pp] for vec_left in ex_vecs_ea]
    full_2p_ea = [i+j for i, j in zip(tdm_2p_ea_, s2s_2p_ea_)]
    tdm_3p_ea_ = [[tdm_3p_ea(d_oo, t2, vec)] for vec in ex_vecs_ea]
    s2s_3p_ea_ = [[s2s_3p_ea(d_oo, t2, vec_left, vec_right) for vec_right in ex_vecs_pp] for vec_left in ex_vecs_ea]
    full_3p_ea = [i+j for i, j in zip(tdm_3p_ea_, s2s_3p_ea_)]


    # all of the above hermitian conjugated:
    #tdm_1p_ip_herm_conj = [[tdm_1p_ip_herm_conj(d_oo, mp2_diffdm_ov, t2, vec)] for vec in ex_vecs_ip]  # herm_conj func
    # s2s_1p_ip_herm_conj up to 2nd order  # here we can actually use the function, because no d_oo or d_vv is used
    #tdm_2p_ip_herm_conj = [[tdm_2p_ip_herm_conj(d_oo, t2, vec)] for vec in ex_vecs_ip]
    # s2s_2p_ip_herm_conj
    #tdm_3p_ip_herm_conj = [[tdm_3p_ip_herm_conj(d_oo, t2, vec)] for vec in ex_vecs_ip]
    # s2s_3p_ip_herm_conj
    #tdm_1p_ea_herm_conj = [[tdm_1p_ea_herm_conj(d_vv, mp2_diffdm_ov, t2, vec)] for vec in ex_vecs_ea]  # herm_conj func
    # s2s_1p_ea_herm_conj up to 2nd order  # herm_conj func
    #tdm_2p_ea_herm_conj = [[tdm_2p_ea_herm_conj(d_oo, t2, vec)] for vec in ex_vecs_ea]
    # s2s_2p_ea_herm_conj
    #tdm_3p_ea_herm_conj = [[tdm_3p_ea_herm_conj(d_oo, t2, vec)] for vec in ex_vecs_ea]
    # s2s_3p_ea_herm_conj


    # all of these have a charge diff of 0
    gs_2p_pp_ = gs_2p_pp(d_oo, t2)
    dummy_gs_2p_pp = gs_2p_pp(d_oo, t2)
    if frozen_core:  # manually added pure "f" blocks (see full_density function)
        gs_2p_pp_["ffff"] = d_oo_frozen(0,3) @ d_oo_frozen(1,2) - d_oo_frozen(1,3) @ d_oo_frozen(0,2)
        dummy_gs_2p_pp["ffff"] = d_oo_frozen(0,3) @ d_oo_frozen(1,2) - d_oo_frozen(1,3) @ d_oo_frozen(0,2)
        dummy_2p_neutral = {"ffff": d_oo_frozen(0,3) @ d_oo_frozen(1,2) - d_oo_frozen(1,3) @ d_oo_frozen(0,2)}
    #    build_2p_gs_object(_gs_2p_pp, d_oo_frozen, d_oo, d_vv)  # updates gs_2p_pp in place
    #    build_2p_gs_object(_gs_2p_pp, adcc.LazyMp.density(states_pp.ground_state, level=1))  # updates _gs_2p_pp in place
    tdm_2p_pp_ = [[tdm_2p_pp(d_oo, t2, vec)] for vec in ex_vecs_pp]
    s2s_2p_pp_ = [[s2s_2p_pp(d_oo, t2, vec_left, vec_right) for vec_right in ex_vecs_pp] for vec_left in ex_vecs_pp]
    full_2p_pp = [[gs_2p_pp_] + herm_conj_tdm(tdm_2p_pp_)] + [i+j for i, j in zip(tdm_2p_pp_, s2s_2p_pp_)]
    s2s_2p_ip_0_ = [[s2s_2p_ip_0(d_oo, t2, vec_left, vec_right) for vec_right in ex_vecs_ip] for vec_left in ex_vecs_ip]
    s2s_2p_ea_0_ = [[s2s_2p_ea_0(d_oo, t2, vec_left, vec_right) for vec_right in ex_vecs_ea] for vec_left in ex_vecs_ea]
    # all of these have a charge diff of 2
    #s2s_2p_2ea = [[s2s_2p_2ea(vec_left, vec_right) for vec_right in ex_vecs_ip] for vec_left in ex_vecs_ea]  # here we can also apply herm_conj
    #s2s_3p_2ea = [[s2s_3p_2ea(d_oo, t2, vec_left, vec_right) for vec_right in ex_vecs_ip] for vec_left in ex_vecs_ea]  # here we can also apply herm_conj
    # the following are required for the parallelization
    #s2s_2p_2ip_ = [[s2s_2p_2ip(vec_left, vec_right) for vec_right in ex_vecs_ea] for vec_left in ex_vecs_ip]
    #s2s_3p_2ip_ = [[s2s_3p_2ip(d_oo, t2, vec_left, vec_right) for vec_right in ex_vecs_ea] for vec_left in ex_vecs_ip]
    s2s_2p_2ip_ = [[s2s_2p_2ip(vec_left, vec_right) for vec_right in ex_vecs_ea] for vec_left in ex_vecs_ip]
    s2s_3p_2ip_ = [[s2s_3p_2ip(d_oo, t2, vec_left, vec_right) for vec_right in ex_vecs_ea] for vec_left in ex_vecs_ip]


    start_time = time.time()
    full_1p_ip_herm_conj = full_density(herm_conj(full_1p_ip))  # 0 +1
    full_2p_ip_herm_conj = full_density(herm_conj(full_2p_ip), dens_key="cca", lower_dens=full_1p_ip_herm_conj)  # 0 +1
    #full_2p_ip_herm_conj = full_density(herm_conj(full_2p_ip))  # 0 +1
    #full_3p_ip_herm_conj = full_density(herm_conj(full_3p_ip), dens_key="cccaa", lower_dens=full_2p_ip_herm_conj)  # 0 +1
    print(time.time() - start_time)
    full_1p_ea_herm_conj = full_density(herm_conj(full_1p_ea))  # 0 -1
    full_2p_ea_herm_conj = full_density(herm_conj(full_2p_ea), dens_key="caa", lower_dens=full_1p_ea_herm_conj)  # 0 -1
    #full_2p_ea_herm_conj = full_density(herm_conj(full_2p_ea))  # 0 -1
    #full_3p_ea_herm_conj = full_density(herm_conj(full_3p_ea), dens_key="ccaaa", lower_dens=full_2p_ea_herm_conj)  # 0 -1
    print(time.time() - start_time)
    full_1p_ip = full_density(full_1p_ip)  # +1 0
    full_2p_ip = full_density(full_2p_ip, dens_key="caa", lower_dens=full_1p_ip)  # +1 0
    #full_2p_ip = full_density(full_2p_ip)  # +1 0
    #full_3p_ip = full_density(full_3p_ip, dens_key="ccaaa", lower_dens=full_2p_ip)  # +1 0
    print(time.time() - start_time)
    full_1p_ea = full_density(full_1p_ea)  # -1 0
    full_2p_ea = full_density(full_2p_ea, dens_key="cca", lower_dens=full_1p_ea)  # -1 0
    #full_2p_ea = full_density(full_2p_ea)  # -1 0
    #full_3p_ea = full_density(full_3p_ea, dens_key="cccaa", lower_dens=full_2p_ea)  # -1 0
    print(time.time() - start_time)
    #gs_2p_pp = full_density(gs_2p_pp)
    full_2p_pp = full_density(full_2p_pp, dens_key="ccaa", lower_dens=full_1p_pp)  # 0 0
    full_2p_ip_0 = full_density(s2s_2p_ip_0_, dens_key="ccaa", lower_dens=s2s_1p_ip_0)  # 0 0
    full_2p_ea_0 = full_density(s2s_2p_ea_0_, dens_key="ccaa", lower_dens=s2s_1p_ea_0)  # 0 0
    # if we had 3p objects of this kind as well, we need to add the gs contribution, before
    # building the 3p density!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    print(time.time() - start_time)
    full_2p_2ea = full_density(herm_conj(s2s_2p_2ip_))  # -1 +1
    #full_3p_2ea = full_density(herm_conj(s2s_3p_2ip), dens_key="ccca", lower_dens=full_2p_2ea)  # -1 +1
    full_3p_2ea = full_density(herm_conj(s2s_3p_2ip_))  # -1 +1
    full_2p_2ip = full_density(s2s_2p_2ip_)  # +1 -1
    #full_3p_2ip = full_density(s2s_3p_2ip, dens_key="caaa", lower_dens=full_2p_2ip)  # +1 -1
    full_3p_2ip = full_density(s2s_3p_2ip_)  # +1 -1
    print(time.time() - start_time)

    #ray.shutdown()


    for i in range(1, len(full_2p_pp)):
        #full_2p_pp[i][i] = ray.put(np.copy(ray.get(full_2p_pp[i][i])) + np.copy(ray.get(full_2p_pp[0][0])))
        full_2p_pp[i][i] += full_tensor(dummy_gs_2p_pp) #full_tensor(build_2p_gs_object(gs_2p_pp(d_oo, t2), 
                            #                               adcc.adc_pp.state2state_transition_dm("adc0", states_pp.ground_state, 
                            #                                                                     ex_vecs_pp[i-1], ex_vecs_pp[i-1])
                            #                               + adcc.LazyMp.density(states_pp.ground_state, level=1)))
    if frozen_core:  # add only ffff block, because other gs part is state specific and therefore handled in the density equations already
        for i in range(len(s2s_2p_ip_0_)):
            #full_2p_ip_0[i][i] = ray.put(np.copy(ray.get(full_2p_ip_0[i][i])) + np.copy(ray.get(full_2p_pp[0][0])))
            full_2p_ip_0[i][i] += full_tensor(dummy_2p_neutral) #full_tensor(build_2p_gs_object(gs_2p_pp(d_oo, t2), 
                                #                             adcc.adc_ip.state2state_transition_dm("ip_adc0", states_ip.ground_state, 
                                #                                                                   ex_vecs_ip[i], ex_vecs_ip[i]),
                                #                             gs_1p=adcc.LazyMp.density(states_pp.ground_state, level=1)))

        for i in range(len(s2s_2p_ea_0_)):
            #full_2p_ea_0[i][i] = ray.put(np.copy(ray.get(full_2p_ea_0[i][i])) + np.copy(ray.get(full_2p_pp[0][0])))
            full_2p_ea_0[i][i] += full_tensor(dummy_2p_neutral) #full_tensor(build_2p_gs_object(gs_2p_pp(d_oo, t2), 
                                #                             adcc.adc_ea.state2state_transition_dm("ea_adc0", states_ea.ground_state, 
                                #                                                                   ex_vecs_ea[i], ex_vecs_ea[i]),
                                #                             gs_1p=adcc.LazyMp.density(states_pp.ground_state, level=1)))
        
    #print(full_tensor(dummy_gs_2p_pp)[:4, :4, :4, :4])
    #print("something similar has to be done for all other 2p densities, starting with full_2p_pp[0][4] and herm. conj., which are tdms")

    """
    #full_1p_ip = full_density(full_1p_ip)  # +1 0
    #full_1p_ea_herm_conj = full_density(herm_conj(full_1p_ea))  # 0 -1
    #full_1p_pp = [[gs_1p_pp]]
    #full_2p_pp = [[full_2p_pp[0][0]]]
    #print(4 * np.einsum("ij,ij->", np.asarray(wfn.Da()), np.asarray(wfn.Fa())))
    #custom_Da = np.zeros_like(np.asarray(wfn.Fa()))
    #custom_Da[:2,:2] = np.identity(2)
    #print(4 * np.einsum("ij,ij->", custom_Da, np.asarray(wfn.Fa())))
    #print(np.asarray(wfn.Da()))
    #print(full_2p_pp[0][0])
    #for i in range(len(gs_1p_pp)):
    #    for j in range(len(gs_1p_pp)):
    #        for k in range(len(gs_1p_pp)):
    #            for l in range(len(gs_1p_pp)):
    #                if full_2p_pp[0][0][i,j,k,l] != 0:
    #                    print((i,j,k,l), full_2p_pp[0][0][i,j,k,l])
    #print(np.asarray(wfn.Da()))
    #print(np.linalg.eigh(np.asarray(wfn.Da()))[0])
    #print(np.asarray(wfn.aotoso(wfn.Da())))
    #print(np.einsum("ki,ij,lj->kl", np.asarray(wfn.Ca()), np.asarray(wfn.Da()), np.asarray(wfn.Ca())))
    print(gs_1p_pp)
    mints = psi4.core.MintsHelper(wfn)
    T = np.asarray(mints.ao_kinetic())
    V_1p = np.asarray(mints.ao_potential())
    h = T + V_1p
    Ca = np.asarray(wfn.Ca())
    h_mo = np.einsum("ij,ik->jk", Ca, np.einsum("ij,ki->kj", Ca, h))  # this should be the correct ao to mo transformation
    T_mo = np.einsum("ij,ik->jk", Ca, np.einsum("ij,ki->kj", Ca, T))
    print(h_mo.shape, np.asarray(gs_1p_pp).shape)
    #print(h)
    #print(h_mo)
    gs_1p_custom = np.zeros_like(h)
    gs_1p_custom[0,0] = 1.0
    gs_1p_custom[1,1] = 1.0
    #gs_1p_custom[2,2] = 1.0
    #gs_1p_custom[3,3] = 1.0
    gs_1p_custom_ao = np.einsum("ji,ik->jk", Ca, np.einsum("ji,ki->kj", Ca, gs_1p_custom))  # this is the correct mo to ao transformation
    print(np.linalg.norm(gs_1p_custom_ao - np.asarray(wfn.Da())))
    #print(np.asarray(wfn.Da()))
    print(4 * np.einsum("ij,ij->", h, gs_1p_custom_ao))
    print(4 * np.einsum("ij,ij->", h_mo, gs_1p_custom))
    print("this is T_mo")
    print(T_mo)
    print(4 * np.einsum("ij,ij->", T_mo, gs_1p_custom))
    #print("this is T_ao")
    #print(T)
    """

    #print(full_2p_pp[0][0])
    #print(full_2p_pp[1][1])

    # now for each density we have to build the full state list, so the states for bra and ket look like this:
    #    0 +1 -1
    #  0
    # +1
    # -1

    """
    def blocked_density(tensor_dict):
        n_pp = len(ex_vecs_pp) + 1  # add ground state
        n_ip = len(ex_vecs_ip)
        n_ea = len(ex_vecs_ea)
        n_tot = n_pp + n_ip + n_ea
        # we refer 0 to 0, 1 to +1 and 2 to -1
        offset_dict = {"0": 0, "1": n_pp, "2": n_pp + n_ip}
        # initialize with zero tensors
        ten_shape = tensor_dict[list(tensor_dict.keys())[0]][0][0].shape
        ret = [[np.zeros(ten_shape)] * n_tot for i in range(n_tot)]
        for key in tensor_dict:
            block_ind = list(key)
            for sub_i in range(len(tensor_dict[key])):
                for sub_j in range(len(tensor_dict[key][0])):
                    ret[offset_dict[block_ind[0]] + sub_i][offset_dict[block_ind[1]] + sub_j] = tensor_dict[key][sub_i][sub_j]
        return ret


    # denotes the charge difference from ket (initial) to bra (final), like chg_diff in build_adc_density_tensors
    full_1p_plus1 = blocked_density({"10": full_1p_ip, "02": full_1p_ea_herm_conj})
    full_2p_plus1 = blocked_density({"10": full_2p_ip, "02": full_2p_ea_herm_conj})
    full_3p_plus1 = blocked_density({"10": full_3p_ip, "02": full_3p_ea_herm_conj})
    full_1p_minus1 = blocked_density({"20": full_1p_ea, "01": full_1p_ip_herm_conj})
    full_2p_minus1 = blocked_density({"20": full_2p_ea, "01": full_2p_ip_herm_conj})
    full_3p_minus1 = blocked_density({"20": full_3p_ea, "01": full_3p_ip_herm_conj})
    full_1p_zero = blocked_density({"00": full_1p_pp, "11": s2s_1p_ip_0, "22": s2s_1p_ea_0})
    full_2p_zero = blocked_density({"00": full_2p_pp, "11": full_2p_ip_0, "22": full_2p_ea_0})
    full_2p_plus2 = blocked_density({"12": full_2p_2ip})
    full_3p_plus2 = blocked_density({"12": full_3p_2ip})
    full_2p_minus2 = blocked_density({"21": full_2p_2ea})
    full_3p_minus2 = blocked_density({"21": full_3p_2ea})

    #print(full_1p_plus1)
    """

    full_1p_plus1 = {(1, 0): full_1p_ip, (0, -1): full_1p_ea_herm_conj}
    full_2p_plus1 = {(1, 0): full_2p_ip, (0, -1): full_2p_ea_herm_conj}
    #full_3p_plus1 = {(1, 0): full_3p_ip, (0, -1): full_3p_ea_herm_conj}
    full_1p_minus1 = {(-1, 0): full_1p_ea, (0, 1): full_1p_ip_herm_conj}
    full_2p_minus1 = {(-1, 0): full_2p_ea, (0, 1): full_2p_ip_herm_conj}
    #full_3p_minus1 = {(-1, 0): full_3p_ea, (0, 1): full_3p_ip_herm_conj}
    full_1p_zero = {(0, 0): full_1p_pp, (1, 1): s2s_1p_ip_0, (-1, -1): s2s_1p_ea_0}
    full_2p_zero = {(0, 0): full_2p_pp, (1, 1): full_2p_ip_0, (-1, -1): full_2p_ea_0}
    full_2p_plus2 = {(1, -1): full_2p_2ip}
    full_3p_plus2 = {(1, -1): full_3p_2ip}
    full_2p_minus2 = {(-1, 1): full_2p_2ea}
    full_3p_minus2 = {(-1, 1): full_3p_2ea}


    #print(full_2p_ip_0[0][0][:4, :4, :4, :4])
    #print(s2s_1p_ip_0[0][0][:4, :4])

    #print(full_1p_ea[0][0])
    #print(full_1p_ea[0][1])
    #print(full_1p_ip[0][0])
    #print(full_1p_ip[0][1])
    #print(full_1p_ip[0][0])
    #print(full_1p_pp[4][4])
    #print(full_1p_pp[0][0])
    #print(full_2p_pp[0][0])
    """
    tensor = torch.tensor
    compare_vals = {(0, 4, 0, 1): tensor(-0.7071, dtype=torch.float64), (0, 4, 1, 0): tensor(0.7071, dtype=torch.float64), (0, 11, 0, 3): tensor(-0.7071, dtype=torch.float64), (0, 11, 3, 0): tensor(0.7071, dtype=torch.float64), (1, 11, 1, 3): tensor(-0.7071, dtype=torch.float64), (1, 11, 3, 1): tensor(0.7071, dtype=torch.float64), (2, 4, 1, 2): tensor(0.7071, dtype=torch.float64), (2, 4, 2, 1): tensor(-0.7071, dtype=torch.float64), (2, 11, 2, 3): tensor(-0.7071, dtype=torch.float64), (2, 11, 3, 2): tensor(0.7071, dtype=torch.float64), (3, 4, 1, 3): tensor(0.7071, dtype=torch.float64), (3, 4, 3, 1): tensor(-0.7071, dtype=torch.float64), (4, 0, 0, 1): tensor(0.7071, dtype=torch.float64), (4, 0, 1, 0): tensor(-0.7071, dtype=torch.float64), (4, 2, 1, 2): tensor(-0.7071, dtype=torch.float64), (4, 2, 2, 1): tensor(0.7071, dtype=torch.float64), (4, 3, 1, 3): tensor(-0.7071, dtype=torch.float64), (4, 3, 3, 1): tensor(0.7071, dtype=torch.float64), (11, 0, 0, 3): tensor(0.7071, dtype=torch.float64), (11, 0, 3, 0): tensor(-0.7071, dtype=torch.float64), (11, 1, 1, 3): tensor(0.7071, dtype=torch.float64), (11, 1, 3, 1): tensor(-0.7071, dtype=torch.float64), (11, 2, 2, 3): tensor(0.7071, dtype=torch.float64), (11, 2, 3, 2): tensor(-0.7071, dtype=torch.float64)}
    #to_print = []
    compare_map = {i: i for i in range(len(gs_1p_pp))}
    compare_map[1] = 2
    compare_map[2] = 1
    to_print = {}
    summed_val_diff = 0
    for i in range(len(gs_1p_pp)):
        for j in range(len(gs_1p_pp)):
            for k in range(len(gs_1p_pp)):
                for l in range(len(gs_1p_pp)):
                    val = full_2p_pp[1][0][i,j,k,l]
                    if val != 0:
                        print((i,j,k,l), val)
                        to_print[(i,j,k,l)] = val
                        #val_diff = compare_vals[(compare_map[i], compare_map[j], compare_map[k], compare_map[l])] - val
                        #summed_val_diff += abs(val_diff)

    print(to_print)
    #print([compare_vals[i] - to_print[i] for i in range(len(compare_vals))])
    print("sum of all difference values", summed_val_diff)
    """

    #print(full_2p_ip[0][0][:4, :4, :4])
    

def build_adc_density_tensors(z_lists):
    densities = {
          'n_elec':{}, 'n_states':{},
          'aa':{}, 'caaa':{},
          'a':{}, 'caa':{}, 'ccaaa':{},
          'ca':{}, 'ccaa':{},
          'c':{}, 'cca':{}, 'cccaa':{},
          'cc':{}, 'ccca':{}
    }

    for (chg,states) in z_lists.items():
        # Store also for use at higher levels
        n_elec_neutral = offsets["v1"]
        #print(states.coeffs.shape[0])
        #print(states.configs.shape[1])
        #print(chg, type(chg))
        #print(states_ip.excitation_vector[0].phh.to_ndarray())
        chg_adc_type_map = {0: ex_vecs_pp, 1: ex_vecs_ip, -1: ex_vecs_ea}
        densities['n_states'][chg] = len(chg_adc_type_map[chg])
        densities['n_elec'  ][chg] = n_elec_neutral - chg

    for bra_chg in z_lists:
        for ket_chg in z_lists:
            chg_diff = bra_chg - ket_chg
            #
            if chg_diff == +2:
                #aa
                densities['aa'][bra_chg,ket_chg] = full_2p_plus2[bra_chg,ket_chg]
                #caaa
                densities['caaa'][bra_chg,ket_chg] = full_3p_plus2[bra_chg,ket_chg]
            elif chg_diff == +1:
                #a
                densities['a'][bra_chg,ket_chg] = full_1p_plus1[bra_chg,ket_chg]
                #caa
                densities['caa'][bra_chg,ket_chg] = full_2p_plus1[bra_chg,ket_chg]
                #ccaaa
                #densities['ccaaa'][bra_chg,ket_chg] = full_3p_plus1[bra_chg,ket_chg]
            elif chg_diff == 0:
                #ca
                densities['ca'][bra_chg,ket_chg] = full_1p_zero[bra_chg,ket_chg]
                #ccaa
                densities['ccaa'][bra_chg,ket_chg] = full_2p_zero[bra_chg,ket_chg]
            elif chg_diff == -1:
                #c
                densities['c'][bra_chg,ket_chg] = full_1p_minus1[bra_chg,ket_chg]
                #cca
                densities['cca'][bra_chg,ket_chg] = full_2p_minus1[bra_chg,ket_chg]
                #cccaa
                #densities['cccaa'][bra_chg,ket_chg] = full_3p_minus1[bra_chg,ket_chg]
            elif chg_diff == -2:
                #cc
                densities['cc'][bra_chg,ket_chg] = full_2p_minus2[bra_chg,ket_chg]
                #ccca
                densities['ccca'][bra_chg,ket_chg] = full_3p_minus2[bra_chg,ket_chg]
            else:
                raise ValueError(f"chg_diff is {chg_diff}, and not an integer between -2 and 2")
    return densities


#print(tdm_2p_ip[0][0]["ooo"][1,1,0])
#print(tdm_2p_ip[0][0]["ooo"](2,1,0)[0,1,1])
#print(s2s_2p_pp[1][0]["oooo"][1,0,1,0])
#print(s2s_2p_pp[1][0]["oooo"](3,2,1,0)[0,1,0,1])
#l_2d_t = [list(x) for x in zip(*s2s_2p_pp)]
#print(l_2d_t[0][1]["oooo"][1,0,1,0])
#index_tuple = tuple([3,2,1,0])
#print(s2s_2p_pp[1][0]["oooo"](*index_tuple)[0,1,0,1])


#print(evaluate(s2s_3p_2ea[0][0]["ovoo"]))
#print(evaluate(herm_conj(s2s_3p_2ip)[0][0]["ovoo"]))



