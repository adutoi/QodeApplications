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
from hummr_dens.space_selection import selected_csf_indices, project_states, select_spaces, n_csfs

import os
import numpy as np
import tensorly as tl

tl.plugins.use_opt_einsum()
tensornet.backend_contract_path(False)

np.set_printoptions(legacy="1.21")  # this gets rid of the print update in numpy 2

class empty(object):  pass



def run_xr(displacement, max_iter, xr_order_final=0, xr_order_solver=0, dens_filter_thresh_solver=1e-7, target_state=0,
           target_multiplicity=1, target_charge=0, n_init_states=3, max_states_per_sector=15, n_pool_roots=30, conv_thresh=1e-6, eval_reduced=False,
           hummr_input="/home/marco/hummr_tests/scratch/be_631g_fci.inp", basis="6-31G", pool="roots", n_csf_passes=1,
           use_gradients=False, grad_level="herm", grad_conv_thresh=1e-5, selected_cfgs=None, selected_engine="hci",
           hci_epsilon=None, selection_roots=3, field_strength=1e-3, selection_dir=None, streamed=False,
           memory_budget="auto"):
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
    selected_cfgs:         optional selected CI space {chg: {mult: set of CFGs}} (hummr_dens.space_selection), used
                           according to selected_engine (the number of initial roots per sector is capped at the number
                           of CSFs of its selected space)
    hci_epsilon:           if given (and selected_cfgs is None), the CI spaces are selected first with this HCI threshold
                           as the union of independent selections (hummr_dens.space_selection.select_spaces) for the
                           actual dimer geometry: the bare fragment, the point charge of the charge-transfer partner
                           (charged sectors) and the six weak fields +-field_strength along x, y and z
    selection_roots:       number of (lowest) roots per sector each selection run is done for
    field_strength:        strength F (a.u.) of the uniform fields of the selection runs
    selection_dir:         directory of the selection runs and the CFG files of the union (union_c<chg>_m<mult>.cfgs),
                           default: selection_<input name>_<hci_epsilon>_<displacement>
    selected_engine:       "hci": every sector is solved in its selected space (HCI without further selection, hummr's
                                  HCIManager), the states, the pool and the densities live in the selected spaces only
                           "fci": the states and the pool are restricted to the selected CSFs (the FCI roots projected
                                  onto them and orthonormalized), while the densities are still built by the
                                  complete-space (FCI) engine (validation, requires the complete spaces)
    streamed:              the large density types (4 and more operators, e.g. ccaa) are not built, their
                           contractions with the integrals are evaluated on demand, see hummr_dens/streamed.py
    memory_budget:         bytes for the cached intermediates of every XR Hamiltonian build ("auto" or ("auto", node
                           memory[, fraction]): as large as the node memory allows, None: 8 GB), see
                           get_xr_result_spin_adapt.resolve_memory_budget
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
    if selected_engine not in ("hci", "fci"):
        raise ValueError(f"unknown selected_engine {selected_engine!r}")
    if selected_cfgs is None and hci_epsilon is not None:
        # the unique fragment (in the frame of its hummr input) occurs at z = 0 and at z = displacement, its
        # environment is the other one (no partial charges for the neutral Be neighbour, it is nonpolar)
        environments = [[{"atoms": [("Be", (0.0, 0.0, +displacement))], "partial_charges": None}],
                        [{"atoms": [("Be", (0.0, 0.0, -displacement))], "partial_charges": None}]]
        n_roots = {chg: {mult: selection_roots for mult in mults} for chg, mults in sectors.items()}
        scratch = selection_dir or f"selection_{os.path.splitext(os.path.basename(hummr_input))[0]}_{hci_epsilon:.0e}_{displacement}"
        selected_cfgs = select_spaces(hummr_input, sectors, n_roots, hci_epsilon, environments, field_strength=field_strength,
                                      scratch_dir=scratch, include_fields=True)
        print("selected CFGs per sector:", {chg: {mult: len(cfgs) for mult, cfgs in by_mult.items()} for chg, by_mult in selected_cfgs.items()})
    use_hci = selected_cfgs is not None and selected_engine == "hci"
    if use_hci:
        # at most as many roots as the selected space has CSFs
        roots = {chg: {mult: min(n, n_csfs(selected_cfgs[chg][mult], mult)) for mult, n in by_mult.items()}
                 for chg, by_mult in roots.items()}
    (mo_coeffs, hummr_states) = dens_builder.init_backend([[frag_name, hummr_input, roots, max(final_orders + [xr_order_solver]),
                                                            selected_cfgs if use_hci else None]])[0]
    csf_indices = None
    if selected_cfgs is not None and not use_hci:
        csf_indices = selected_csf_indices(dens_builder.csf_layout(frag_name), selected_cfgs)
        hummr_states = {chg: {mult: project_states(vecs, csf_indices[chg][mult], len(vecs)) for mult, vecs in mult_dict.items()}
                        for chg, mult_dict in hummr_states.items()}
        print("selected CSFs per sector:", {chg: {mult: len(idx) for mult, idx in mult_dict.items()} for chg, mult_dict in csf_indices.items()})

    states = [{chg: {mult: [np.array(vec) for vec in vecs[:n_init_states]] for mult, vecs in mult_dict.items()}
               for chg, mult_dict in hummr_states.items()} for _ in range(n_frag)]
    if pool == "roots":
        additional_states = [{chg: {mult: [np.array(vec) for vec in vecs[n_init_states:]] for mult, vecs in mult_dict.items()}
                              for chg, mult_dict in hummr_states.items()} for _ in range(n_frag)]
    elif pool == "csf":
        additional_states = [unit_vector_pool(states[frag], n_passes=n_csf_passes, csf_indices=csf_indices) for frag in range(n_frag)]
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
        Be.basis.n_spatial_orb = mo_coeffs.shape[1]
        Be.basis.AOcode = basis
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
        return build_frag_densities(dens_builder, frag_name, frag_states, xr_level, BeN[frag].n_elec_ref, bra_states=bra_states,
                                    streamed=streamed, memory_budget=memory_budget)

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
    # CI spaces: union of the HCI selections (bare, charge-transfer point charges, +-F fields) at eps = 1e-3, solved
    # with HCI in these spaces (hummr's HCIManager); streamed density contractions with an 8 GB cache budget
    solver_en, reduced_en, final_en = run_xr(4.5, 50, xr_order_final=[0, 1, 2], n_init_states=8, max_states_per_sector=20,
                                             eval_reduced=True, hummr_input="/home/marco/hummr_tests/scratch/be_631g_fci_fcore.inp",
                                             basis="6-31G", pool="csf", use_gradients=False, dens_filter_thresh_solver=1e-7,
                                             hci_epsilon=1e-3, selected_engine="hci", streamed=True, memory_budget=8e9)
    print("solver energies (enlarged state spaces):  ", solver_en)
    print("solver energies (compressed state spaces):", reduced_en)
    for order, E in final_en.items():
        print(f"final XR[{order}] energy:", E)
    #print("ref from HCI with frozen core 6-31G = ", -29.21086186)
    print("ref from psi4 with frozen core 6-31G = ", -29.225805355910)
