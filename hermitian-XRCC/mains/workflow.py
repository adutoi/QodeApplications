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

from get_ints import get_ints
from get_xr_result import get_xr_states#, get_xr_H
from state_solver import optimize_states
from orbital_solver import optimize_orbs
#from qode.math.tensornet import backend_contract_path#, raw, tl_tensor
import qode.math.tensornet as tensornet
#import qode.util
from qode.util import timer, sort_eigen
#from state_gradients import state_gradients, get_slices, get_adapted_overlaps
from state_screening import state_screening#, orthogonalize

#import torch
import numpy as np
import tensorly as tl
import pickle
#import scipy as sp

#torch.set_num_threads(4)
#tl.set_backend("pytorch")
#tl.set_backend("numpy")

from   build_fci_states import get_fci_states
#from build_Be_rho import build_Be_rho
import densities

tl.plugins.use_opt_einsum()
tensornet.backend_contract_path(True)

class empty(object):  pass  # for pickle load initialization without get_fci_states

def run_xr(displacement, max_iter, xr_order_final, xr_order_solver=0, dens_filter_thresh_solver=1e-7, orb_max_iter=0, target_state=0,
           single_thresh=1/5, double_thresh=1/3.5, triple_thresh=1/2.5, sp_thresh=1/1.1, grad_level="herm", state_prep=False):#, n_threads):
    tensornet.initialize_timer()
    tensornet.tensorly_backend.initialize_timer()

    #global_timings = timer()
    #global_timings.start()

    n_frag       = 2
    displacement = displacement
    project_core = True
    monomer_charges = [[0, +1, -1], [0, +1, -1]]
    density_options = []#["compress=SVD,cc-aa"]

    #ref_states = pickle.load(open("ref_states.pkl", mode="rb"))
    #ref_mos = pickle.load(open("ref_mos.pkl", mode="rb"))

    # "Assemble" the supersystem for the displaced fragments and get integrals
    BeN = []
    #dens = []
    dens_builder_stuff = []
    state_coeffs_og = []
    #pre_opt_states = pickle.load(open("pre_opt_coeffs.pkl", mode="rb"))
    #ref_state_coeffs_configs = pickle.load(open("opt_state_coeffs_configs.pkl", mode="rb"))
    for m in range(int(n_frag)):
        state_obj, dens_var_1, dens_var_2, n_threads, Be = get_fci_states(displacement, n_state_list=[(1, 2), (0, 10), (-1, 10)])
        #state_obj, dens_var_1, dens_var_2, n_threads, Be = build_Be_rho(("6-31g", 9), displacement, n_state_list=[(1, 2), (0, 10), (-1, 10)])
        #Be.basis.MOcoeffs = ref_mos.copy()
        #pickle.dump(Be.basis.MOcoeffs, open(f"check_mos_{m}.pkl", mode="wb"))
        # the following provide two possibilities to start from a dumped reference
        """
        frag0 = empty()
        frag0.atoms = [("Be",[0,0,0])]
        frag0.n_elec_ref = 4	# The number of electrons in the reference state of the monomer ("cation (+1)" and "anion (-1)" and technically interpreted relative to the reference, not zero, as would be the chemical definition)
        frag0.basis = empty()
        frag0.basis.AOcode = "6-31g"#basis_label
        frag0.basis.n_spatial_orb = 9#n_spatial_orb
        frag0.basis.MOcoeffs = np.identity(frag0.basis.n_spatial_orb)    # rest of code assumes spin-restricted orbitals
        frag0.basis.core = [0]	# indices of spatial MOs to freeze in CI portions
        Be = frag0
        dens_var_1 = 2 * Be.basis.n_spatial_orb
        dens_var_2 = Be.n_elec_ref
        n_threads = 4
        Be.basis.MOcoeffs = pickle.load(open(f"check_mos_0.pkl", mode="rb")).copy()
        state_obj = {}
        for chg in monomer_charges[m]:
            #pickle.dump(state_obj[chg].coeffs, open(f"check_states_{m}_{chg}.pkl", mode="wb"))
            state_obj[chg] = empty()
            state_obj[chg].coeffs = pickle.load(open(f"check_states_0_{chg}.pkl", mode="rb")).copy()
            #pickle.dump(state_obj[chg].configs, open(f"check_configs_{m}_{chg}.pkl", mode="wb"))
            state_obj[chg].configs = pickle.load(open(f"check_configs_0_{chg}.pkl", mode="rb")).copy()
        #raise ValueError("stop here")
        """
        """
        ref_data = pickle.load(open("../ref_state_data/state-0.pkl", mode="rb"))
        ref_states = ref_data.states
        Be.basis.MOcoeffs = ref_data.basis.MOcoeffs.copy()
        #Be.states[0].configs   # for charge=0, an array of integers whose binary rep gives the electron configurations
        #Be.states[0].coeffs[n]    # for charge=0, the coefficients of said configurations for the n-th most important monomer state

        #Be.basis.MOcoeffs = pickle.load(open(f"pre_opt_mos_{m}.pkl", mode="rb"))
        for chg in monomer_charges[m]:
        #    state_obj[chg].coeffs = ref_state_coeffs_configs[0][m][chg].copy()
        #    state_obj[chg].configs = ref_state_coeffs_configs[1][m][chg].copy()

        #    state_obj[chg].coeffs = pre_opt_states[m][chg].copy()

            state_obj[chg].coeffs = ref_states[chg].coeffs.copy()
            state_obj[chg].configs = ref_states[chg].configs.copy()
            #state_obj[chg].coeffs = [i[:448] for i in state_obj[chg].coeffs]
            #if m == 0:
            #    state_obj[chg].coeffs = ref_states[chg].coeffs.copy()
        #    state_obj[chg].coeffs = ref_states[chg].coeffs[:1]
        """
        # TODO: the following and the rest of the code currently only work for equal fragments
        n_orbs = Be.basis.n_spatial_orb
        n_occ = [Be.n_elec_ref // 2, Be.n_elec_ref // 2]  # alpha and beta
        frozen_orbs = Be.basis.core.copy()
        frozen_orbs += [i + n_orbs for i in Be.basis.core]  # append by beta frozen orb numbers

        for elem,coords in Be.atoms:  coords[2] += m * displacement    # displace along z
        BeN.append(Be)
        #dens.append(densities.build_tensors(state_obj, dens_var_1, dens_var_2, options=density_options, n_threads=n_threads))
        dens_builder_stuff.append([state_obj, dens_var_1, dens_var_2, density_options])
        state_coeffs_og.append({chg: state_obj[chg].coeffs for chg in state_obj})
        """
        def conf_decoder(conf):
            ret = []
            for bit in range(n_orbs * 2, -1, -1):
                if conf - 2**bit < 0:
                    continue
                conf -= 2**bit
                ret.append(bit)
            return sorted(ret)
        
        for chg in monomer_charges[0]:
            for i, vec in enumerate(state_obj[chg].coeffs):
                big_inds = {ind: elem for ind, elem in enumerate(vec) if abs(elem) > 1e-1}
                print(chg, i, {tuple(conf_decoder(dens_builder_stuff[0][0][chg].configs[j])): val for j, val in big_inds.items()})
        """

    #global_timings.record("initialize state space")
    #global_timings.start()

    int_timer = timer()
    ints = get_ints(BeN, project_core, int_timer)

    relevant_determinants, confs_and_inds = state_screening(dens_builder_stuff, ints, monomer_charges, n_orbs, frozen_orbs, n_occ, n_threads=n_threads,
                                                        single_thresh=single_thresh, double_thresh=double_thresh, triple_thresh=triple_thresh, sp_thresh=sp_thresh)
    #raise ValueError("stop here")
    energies, dens_builder_stuff = optimize_states(max_iter, xr_order_solver, dens_builder_stuff, ints, n_occ, n_orbs, frozen_orbs, relevant_determinants,
                                                   dens_filter_thresh=dens_filter_thresh_solver, grad_level=grad_level, begin_from_state_prep=state_prep,
                                                   monomer_charges=monomer_charges, density_options=density_options, n_threads=n_threads, target_state=target_state)
    
    # rebuild densities for orb solver order
    #dens = [densities.build_tensors(*dens_builder_stuff[frag][:-1], options=dens_builder_stuff[frag][-1], n_threads=n_threads, xr_order=xr_order_solver) for frag in range(2)]
    
    #final_orb_opt_en, ints = optimize_orbs(orb_max_iter, xr_order_solver, BeN, ints, dens, dens_builder_stuff, monomer_charges=monomer_charges)

    # TODO: remove this, when asymmetric diagonalization is activated
    for chg in monomer_charges[0]:
        len0 = len(dens_builder_stuff[0][0][chg].coeffs)
        len1 = len(dens_builder_stuff[1][0][chg].coeffs)
        if len0 > len1:
            dens_builder_stuff[0][0][chg].coeffs = dens_builder_stuff[0][0][chg].coeffs[:len1]
        elif len0 < len1:
            dens_builder_stuff[1][0][chg].coeffs = dens_builder_stuff[1][0][chg].coeffs[:len0]
        else:  # equal
            continue

    # rebuild densities for appropriate order
    dens = [densities.build_tensors(*dens_builder_stuff[frag][:-1], options=dens_builder_stuff[frag][-1], n_threads=n_threads, xr_order=xr_order_final) for frag in range(2)]

    #pickle.dump([dens, ints], open("dens_ints_prep_orb_opt.pkl", mode="wb"))
    #pickle.dump(dens, open("dens_opt_states.pkl", mode="wb"))
    #dump_states = [{}, {}]
    #for frag in range(2):
    #    for chg in monomer_charges[frag]:
    #        dump_states[frag][chg] = dens_builder_stuff[frag][0][chg].coeffs
    #pickle.dump(dump_states, open("dens_opt_states.pkl", mode="wb"))

    final_en, final_state = get_xr_states(ints, dens, xr_order_final, monomer_charges, target_state)

    #return energies[-1], final_orb_opt_en, final_en
    return energies[-1], final_en



print(run_xr(3.5, 0, 0, single_thresh=1/6, double_thresh=1/4, triple_thresh=1/2.5,# sp_thresh=1/1.005,
             grad_level="herm", state_prep=True, target_state=[0, 1], dens_filter_thresh_solver=1e-5))
