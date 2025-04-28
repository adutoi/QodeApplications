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
#from qode.math.tensornet import backend_contract_path#, raw, tl_tensor
import qode.math.tensornet as tensornet
#import qode.util
from qode.util import timer, sort_eigen
#from state_gradients import state_gradients, get_slices, get_adapted_overlaps
from state_screening import state_screening#, orthogonalize

#import torch
#import numpy as np
import tensorly as tl
import pickle
#import scipy as sp

#torch.set_num_threads(4)
#tl.set_backend("pytorch")
#tl.set_backend("numpy")

from   build_fci_states import get_fci_states
import densities

tl.plugins.use_opt_einsum()
tensornet.backend_contract_path(True)

#class empty(object):  pass

def run_xr(displacement, max_iter, xr_order_final, xr_order_solver=0, dens_filter_thresh_solver=1e-7,
           single_thresh=1/5, double_thresh=1/3.5, triple_thresh=1/2.5, grad_level="herm", state_prep=False):#, n_threads):
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
        #Be.basis.MOcoeffs = ref_mos.copy()
        #pickle.dump(Be.basis.MOcoeffs, open(f"pre_opt_mos_{m}.pkl", mode="wb"))

        #Be.basis.MOcoeffs = pickle.load(open(f"pre_opt_mos_{m}.pkl", mode="rb"))
        #for chg in monomer_charges[m]:
        #    state_obj[chg].coeffs = ref_state_coeffs_configs[0][m][chg].copy()
        #    state_obj[chg].configs = ref_state_coeffs_configs[1][m][chg].copy()

        #    state_obj[chg].coeffs = pre_opt_states[m][chg].copy()

            #state_obj[chg].coeffs = ref_states[chg].coeffs.copy()
        #    state_obj[chg].configs = ref_states[chg].configs.copy()
            #state_obj[chg].coeffs = [i[:448] for i in state_obj[chg].coeffs]
            #if m == 0:
            #    state_obj[chg].coeffs = ref_states[chg].coeffs.copy()
        #    state_obj[chg].coeffs = ref_states[chg].coeffs[:1]

        # TODO: the following and the rest of the code currently only work for equal fragments
        n_orbs = Be.basis.n_spatial_orb
        n_occ = [Be.n_elec_ref // 2, Be.n_elec_ref // 2]  # alpha and beta
        frozen_orbs = Be.basis.core.copy()
        frozen_orbs += [i + n_orbs for i in Be.basis.core]  # append by beta frozen orb numbers

        for elem,coords in Be.atoms:  coords[2] += m * displacement    # displace along z
        BeN.append(Be)
        #dens.append(densities.build_tensors(state_obj, dens_var_1, dens_var_2, options=density_options, n_threads=n_threads))
        dens_builder_stuff.append([state_obj, dens_var_1, dens_var_2, density_options])
        #state_coeffs_og.append({chg: state_obj[chg].coeffs for chg in state_obj})
        state_coeffs_og.append({chg: state_obj[chg].coeffs for chg in state_obj})
        #print(Be.basis.MOcoeffs)
        #raise ValueError("stop here")
    #print(type(raw(dens[0]["a"][(+1,0)])), raw(dens[0]["a"][(+1,0)]).shape)

    #global_timings.record("initialize state space")
    #global_timings.start()

    int_timer = timer()
    ints = get_ints(BeN, project_core, int_timer)

    relevant_determinants, confs_and_inds = state_screening(dens_builder_stuff, ints, monomer_charges, n_orbs, frozen_orbs, n_occ, n_threads=n_threads,
                                                        single_thresh=single_thresh, double_thresh=double_thresh, triple_thresh=triple_thresh)
    
    energies, dens_builder_stuff = optimize_states(max_iter, xr_order_solver, dens_builder_stuff, ints, n_occ, n_orbs, frozen_orbs, relevant_determinants,
                                                   dens_filter_thresh=dens_filter_thresh_solver, grad_level=grad_level, begin_from_state_prep=state_prep,
                                                   monomer_charges=monomer_charges, density_options=density_options, n_threads=n_threads)
    
    # rebuild densities for appropriate order
    dens = [densities.build_tensors(*dens_builder_stuff[frag][:-1], options=dens_builder_stuff[frag][-1], n_threads=n_threads, xr_order=xr_order_final) for frag in range(2)]
    
    final_en, final_state = get_xr_states(ints, dens, xr_order_final, monomer_charges)

    return energies[-1], final_en
    #return final_en



print(run_xr(2.5, 50, 0, single_thresh=1/5, double_thresh=1/3.5, triple_thresh=1/2.5, grad_level="full", state_prep=False))
