#    (C) Copyright 2026 Marco Bauer
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

# Spin-adapted counterpart of workflow.py, based on xr0-ccsd-from-hummr-hdf5.py.
#
# Usage:
#     python [-u] workflow_spin_adapt.py

from get_ints import get_ints
from get_xr_result_spin_adapt import get_xr_states, build_frag_densities
from state_solver_csf import optimize_states, count_states, unit_vector_pool
import qode.math.tensornet as tensornet
from qode.util import timer
from dens_provider import DensBackend

import numpy as np
import tensorly as tl

tl.plugins.use_opt_einsum()
tensornet.backend_contract_path(False)

np.set_printoptions(legacy="1.21")  # this gets rid of the print update in numpy 2

class empty(object):  pass



def run_xr(displacement, max_iter, xr_order_final=0, xr_order_solver=0, dens_filter_thresh_solver=1e-7, target_state=0,
           target_multiplicity=1, target_charge=0, n_init_states=3, max_states_per_sector=15, n_pool_roots=30, conv_thresh=1e-6, eval_reduced=False,
           hummr_input="/home/marco/hummr_tests/scratch/be_631g_fci.inp", pool="roots", n_csf_passes=1,
           use_gradients=False, grad_level="herm", grad_conv_thresh=1e-5):
    """
    xr_order_final:        XR order (or list of orders) of the final evaluation in the optimized state space
    n_init_states:         number of (lowest) hummr roots per (chg, mult) sector used as initial guess
    pool:                  "roots": the remaining ones of n_pool_roots hummr roots per sector serve as pool of additional
                                    states for the solver (instead of the state screening)
                           "csf":   all CSFs of every sector (unit vectors) serve as pool, which is passed through
                                    completely (only n_init_states hummr roots are computed, max_iter has to be large enough),
                                    n_csf_passes times
                           "none":  no pool (only sensible with use_gradients)
    use_gradients:         gradient based solver variant (fragments alternately enlarged by their state gradients,
                           see state_gradients_csf.py), with grad_level "herm" (right eigenvector, the fixed point of
                           the solver) or "full" (left and right eigenvectors, exact gradient of the eigenvalue)
    n_pool_roots:          number of hummr roots per sector computed in total for pool="roots"
    max_states_per_sector: maximum number of states per sector during the solver iterations
    """
    if pool not in ("roots", "csf", "none"):
        raise ValueError(f"unknown pool {pool!r}")
    tensornet.initialize_timer()
    tensornet.tensorly_backend.initialize_timer()

    n_frag = 2
    project_core = True
    monomer_charges = [[0, +1, -1], [0, +1, -1]]
    sectors = {0: [1, 3], +1: [2], -1: [2, 4]}   # chg: [mult]

    #########
    # initial MOs and states
    #########

    dens_builder = DensBackend("hummr")
    roots = {chg: {mult: (n_pool_roots if pool == "roots" else n_init_states) for mult in mults} for chg, mults in sectors.items()}
    frag_name = "Be"
    # compute only unique fragments
    final_orders = [xr_order_final] if isinstance(xr_order_final, int) else list(xr_order_final)
    (mo_coeffs, hummr_states) = dens_builder.init_backend([[frag_name, hummr_input, roots, max(final_orders + [xr_order_solver])]])[0]

    states = [{chg: {mult: [np.array(vec) for vec in vecs[:n_init_states]] for mult, vecs in mult_dict.items()}
               for chg, mult_dict in hummr_states.items()} for _ in range(n_frag)]
    if pool == "roots":
        additional_states = [{chg: {mult: [np.array(vec) for vec in vecs[n_init_states:]] for mult, vecs in mult_dict.items()}
                              for chg, mult_dict in hummr_states.items()} for _ in range(n_frag)]
    elif pool == "csf":
        additional_states = [unit_vector_pool(states[frag], n_passes=n_csf_passes) for frag in range(n_frag)]
    else:
        additional_states = [{} for _ in range(n_frag)]

    #########
    # "Assemble" the supersystem for the displaced fragments and get integrals
    #########

    BeN = []
    for m in range(n_frag):
        Be = empty()
        Be.atoms = [["Be", [0, 0, 0]]]
        Be.charge = 0
        Be.n_elec_ref = 4
        Be.basis = empty()
        Be.basis.n_spatial_orb = 9
        Be.basis.AOcode = "6-31G"
        # frozen core of the hummr input (empty for all-electron inputs), projected out of the valence in get_ints;
        # the densities already include the doubly occupied core (density_providers/hummr_dens embed_frozen_core)
        Be.basis.core = dens_builder.frozen_core(frag_name)
        Be.basis.MOcoeffs = mo_coeffs
        for elem,coords in Be.atoms:  coords[2] += m * displacement    # displace along z
        BeN.append(Be)

    int_timer = timer()
    ints = get_ints(BeN, project_core, int_timer, spin_ints=False, backend="lible")
    nuc_rep = ints[2]
    E_nuc = sum(nuc_rep[m1,m2] for m1 in range(n_frag) for m2 in range(m1+1))

    def build_dens(frag, frag_states, xr_level=xr_order_solver, bra_states=None):
        # both fragments are the same molecule, so they share the hummr instance
        return build_frag_densities(dens_builder, frag_name, frag_states, xr_level, BeN[frag].n_elec_ref, bra_states=bra_states)

    #########
    # Optimize the fragment states
    #########

    energies, reduced_energies, states = optimize_states(max_iter, xr_order_solver, states, additional_states, ints, build_dens,
                                                         monomer_charges, target_multiplicity, target_charge=target_charge,
                                                         max_states_per_sector=max_states_per_sector,
                                                         conv_thresh=conv_thresh, dens_filter_thresh=dens_filter_thresh_solver,
                                                         target_state=target_state, eval_reduced=eval_reduced, exhaust_pool=(pool == "csf"),
                                                         use_gradients=use_gradients, grad_level=grad_level, grad_conv_thresh=grad_conv_thresh)

    #########
    # Final evaluation in the optimized state space
    #########

    # (for several final XR orders, e.g. [1, 2], with the same optimized states; the densities are built once
    #  for the highest order, since they contain all lower ones)
    print("final state space:", [count_states(states[frag]) for frag in range(n_frag)])
    dens = [build_dens(frag, states[frag], xr_level=max(final_orders)) for frag in range(n_frag)]
    final_en = {}
    for order in final_orders:
        E, _ = get_xr_states(ints, dens, order, monomer_charges, target_multiplicity, target_state=target_state, target_charge=target_charge)
        final_en[order] = np.real(E) + E_nuc
        print(f"final XR[{order}] energy: {final_en[order]}")
    if isinstance(xr_order_final, int):
        final_en = final_en[xr_order_final]

    dens_builder.finalize_all()

    return [np.real(E) + E_nuc for E in energies], [np.real(E) + E_nuc for E in reduced_energies], final_en


if __name__ == "__main__":
    # frozen 1s, (2,8) active spaces, gradient based solver at XR[0] starting from the lowest 3 states per sector,
    # then XR[0], XR[1] and XR[2] in the optimized state space
    # (pool="csf", n_csf_passes=2, use_gradients=False is the gradient free variant passing through all CSFs)
    solver_en, reduced_en, final_en = run_xr(4.5, 50, xr_order_final=[0, 1], n_init_states=8, max_states_per_sector=20,
                                             eval_reduced=True, hummr_input="/home/marco/hummr_tests/scratch/be_631g_fci_fcore.inp",
                                             pool="csf", use_gradients=False, dens_filter_thresh_solver=1e-9)
    print("solver energies (enlarged state spaces):  ", solver_en)
    print("solver energies (compressed state spaces):", reduced_en)
    for order, E in final_en.items():
        print(f"final XR[{order}] energy:", E)
    #print("ref from HCI with frozen core = ", -29.21086186)
    print("ref from psi4 with frozen core = ", -29.225805355910)
