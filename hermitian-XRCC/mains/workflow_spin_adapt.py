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
from get_xr_result_spin_adapt import get_xr_states, get_leakage_metric_states, build_frag_densities, check_overlap, overlap_matrices, overlap_truncation, local_sectors, dimer_labels
from state_solver_csf import optimize_states, count_states, unit_vector_pool, save_states, load_states
import qode.math.tensornet as tensornet
from qode.util import timer
from dens_provider import DensBackend
from hummr_dens.space_selection import selected_csf_indices, project_states, select_spaces, n_csfs, read_spaces

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
           memory_budget="auto", pool_order="given", use_davidson=False, dm_perturbation=1e-3, davidson_tol=1e-5,
           davidson_preconditioner=True, dm_vectors="right", overlap_check_order=4, selected_spaces_dir=None,
           states_in=None, states_out=None, overlap_min_eigval=0.15, overlap_max_weight=1e-5, overlap_reoptimize=True,
           overlap_max_cycles=4, leakage_metric_order=None, leakage_pair_thresh=None, metric_pair_thresh=None,
           leakage_v_thresh={2: 1e-5}, leakage_pair_slots=2.5, pair_rotation=False, bridge_strong=0, bridge_out=None,
           bridge_mode="glued", bridge_pair_thresh=None):
    """
    xr_order_final:        XR order (or list of orders) of the final evaluation in the optimized state space
    leakage_metric_order:  if given (2, 3 or 4), the final evaluation also includes the leakage-metric ansatz
                           H0 + (S2^[n])^-1 L^[2] with n = leakage_metric_order and L^[2] = S2H2^[2] - S2^[2] H0
                           (get_xr_result_spin_adapt.get_leakage_metric_H), as final_en["metric"] (final_en is then
                           a dict, also for an integer xr_order_final); its state space is truncated with the overlap
                           criterion of S2^[n] (see overlap_min_eigval); requires the densities of xr_level 2
    The approximations leakage_pair_thresh, metric_pair_thresh and leakage_v_thresh apply to the entire computation:
    to the XR orders n >= 1 of the final evaluation (their orders k <= n; XR[n] is then evaluated in the equivalent
    leakage form H0 + (S2^[n])^-1 L^[n], get_xr_result_spin_adapt.get_xr_states), to the leakage-metric ansatz, and
    metric_pair_thresh also to the overlap truncation and the overlap check.  Only leakage_metric_order selects the
    leakage-metric ansatz itself (a metric of a higher order than the leakage).
    leakage_pair_thresh:   {order: threshold}: the order-k increment of the leakage L^[k] - L^[k-1] (k = 1, 2) is
                           evaluated with the inter-fragment overlap restricted to the corresponding-orbital pairs with
                           a singular value above the threshold (consistently, i.e. also in the biorthogonalization,
                           get_ints pair_thresh); orders without a threshold (or None) are exact.  Be2 cc-pVDZ: order 2
                           with 0.02 (0.1) within 10 (50) uEh of the exact one, a compressed order 1 fails (1e2 to 1e3
                           uEh)
    metric_pair_thresh:    {order: threshold} as leakage_pair_thresh for the order-k terms of the overlap S2 (k = 1 ...
                           4; the metric of the leakage-metric ansatz and of the XR orders, the overlap truncation and
                           the overlap check).  Be2 cc-pVDZ: orders 3 and 4 with 0.1 change the energy by less than
                           8 uEh (+50 uEh for a nearly singular metric, lowest eigenvalue 0.02)
    leakage_v_thresh:      {2: threshold}: the order-2 diagrams of the order-2 increment of the leakage (XR[2] and the
                           leakage-metric ansatz) are evaluated with the inter-fragment blocks of V factorized by
                           fragment (singular values above the threshold, precontract_pairs.V_factorized_precontract);
                           XR[0], L^[1] and the lower-order terms of the increment use the complete V ({} or
                           {2: None}: complete V everywhere).  Be2 cc-pVDZ, 3.4 / 4.0 A: 1e-5 changes the energy by -0.2 / 0.0 uEh, 1e-4 by
                           -0.6 / +0.1 uEh (about 25 % faster than the complete V at 4.0 A), 1e-3 by +17 / +8 uEh
    leakage_pair_slots:    the free density slots that only meet overlap factors in the diagrams of the leakage (L^[1]
                           and the order-2 increment, i.e. XR[1]/XR[2] with leakage_pair_thresh and the leakage-metric
                           ansatz) in the basis of the K corresponding-orbital pairs of leakage_pair_thresh
                           (diagrams/pair_slots.py, exact): a number r uses it if r K <= n_orb, True always, False
                           never.  With few pairs it brings a speed up (Be2 cc-pVDZ, K = 5 of 14: 41.5 -> 36.4 s) and
                           lowers the peak memory; with many pairs it is slower but still lowers the peak memory
                           (cc-pVTZ at 3.0 A, K = 19 of 30, overlap_min_eigval 0.05: 370 -> 432 s, peak 21.1 -> 16.5 GB),
                           so the default 2.5 switches it off there, and True trades the runtime for the memory
    pair_rotation:         the final evaluation in the corresponding orbitals of the core-projected valence overlap
                           (S_AB diagonal, strongest pairs as the last active orbitals; fragment A rotated by U, B by V
                           of S_AB = U s V^T), with the optimized states transformed exactly to the rotated orbitals
                           (pair_rotation.py; complete CSF spaces only).  XR is invariant under this rotation; it is the
                           basis of the exact-bridge ansatz
    bridge_strong:         number K of strong pairs of the exact-bridge ansatz (0: none); implies pair_rotation: hummr
                           is initialized with the sectors needed to split the states by the occupation of the K strong
                           orbitals (strong_split.required_sectors, not part of the XR basis), and the rotated final
                           states are split (strong_split.py; their weights are printed)
    bridge_out:            with bridge_strong = 1, a file (.npz) for the order-resolved overlap S^(k) (k = 0, 1, 2 in the weak
                           pairs) of the singlet product states of the final states (bridge_assembly.py), with their
                           labels (sector A, i_A, sector B, i_B); in factorized mode also the energies E(n) of the weak
                           orders n = 0, 1, 2 (lowest real eigenvalue of S_n^-1 H_n), returned as final_en["bridge[n]"]
                           (final_en is then a dict)
    bridge_mode:           "glued" (T_B on the bra, one rectangular XR build of all terms, bridge_assembly.GluedAssembly: S_A,
                           H_A), "factorized" (bridge tables x weak dimer terms, BridgeAssembly and bridge_mixed.
                           MixedEvaluator: S, H_weak, H_strong, H_mixed) or "both"
    bridge_pair_thresh:    factorized mode: the order-2 terms with the inter-fragment overlap restricted to the
                           corresponding-orbital pairs with a singular value above the threshold (as leakage_pair_thresh
                           for XR[2]: get_ints with pair_thresh, i.e. also in the biorthogonalization, and sigma of the
                           mixed terms; the weak overlap in the pair basis with leakage_pair_slots); orders 0 and 1 exact
    n_init_states:        number of (lowest) hummr roots per (chg, mult) sector used as initial guess
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
    selected_spaces_dir:   if given (and selected_cfgs is None), the selected CI spaces are read from the CFG files of a
                           previous selection in this directory (union_c<chg>_m<mult>.cfgs, see selection_dir) instead
                           of running the selections (hci_epsilon is then not used), e.g. to reuse them for the other
                           geometries of a scan
    states_in:             if given, file (written with states_out) with the fragment states to start from instead of the
                           n_init_states hummr roots, mapped onto the current CSF space via the CFGs (state_solver_csf.
                           load_states; sectors without states get the hummr roots), e.g. the optimized states of a
                           nearby geometry; with max_iter = 0 they are used for the final evaluation as they are
    states_out:            if given, file to which the final optimized fragment states are written (with their CSF
                           layout and the geometry, state_solver_csf.save_states)
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
    pool_order:            "given", "cipsi" or "residual": the remaining pool is reordered after every solve by the
                           first-order (Epstein-Nesbet, "cipsi", or residual only, "residual") importance from the
                           embedded XR[0] problem (state_solver_csf.optimize_states, embedded.py)
    use_davidson:          Davidson sweeps instead of the pool (state_solver_csf.optimize_states): every fragment is
                           optimized in its complete (selected) CSF space with the other fragment's states fixed
                           (XR[0], single target state; use pool="none"), with the density-matrix perturbation of
                           weight dm_perturbation (see state_solver_csf.optimize_states); per sector, the states with
                           a weight above dens_filter_thresh_solver in the fragment's (perturbed) reduced density
                           matrix are kept, at most max_states_per_sector
    davidson_tol:          residual norm at which the Davidson iterations of a sweep step are converged
    davidson_preconditioner: diagonal preconditioner for the Davidson iterations (diagonal of the embedded Hamiltonian)
    dm_vectors:            "right" or "left_right": eigenvectors of the embedded problem the reduced density matrix of
                           the Davidson sweeps is built from (see state_solver_csf.optimize_states)
    overlap_check_order:   maximum order n (1 ... 4, default 4) of the overlap check: for every final state space, the
                           lowest and largest eigenvalues of the product-state overlap S2^[n'] (S expansion up to order
                           n' = 1 ... n, the orders with metric_pair_thresh compressed) are printed, with warnings for
                           the final XR orders, if S2 is nearly singular, indefinite or not converged relative to order
                           n (get_xr_result_spin_adapt.check_overlap); the S2 terms are shared with the overlap
                           truncation and the metric (Be2 cc-pVDZ, 32 states per fragment: orders 0 - 4 in 1.2 s with
                           metric_pair_thresh {3: 0.1, 4: 0.1}, 4 s exact); None: no check
    overlap_min_eigval:    for every final order n >= 1, the optimized state space is truncated further, until the lowest
                           eigenvalue of the product-state overlap S2^[n] (target block) is at least this value: the
                           states are removed in the order of increasing weight in the reduced density matrix of the
                           solver (the XR[0] one of the final state space, if the solver did not provide them, e.g.
                           for states_in with max_iter = 0), as long as their weight is below overlap_max_weight and
                           every sector keeps one state (get_xr_result_spin_adapt.overlap_truncation); XR[0] is always
                           evaluated in the complete optimized state space.  A nearly singular or indefinite S2^[n]
                           makes XR[n] erratic (poles, spurious roots); in Be2 6-31G/cc-pVDZ, XR[1] is smooth for a
                           lowest eigenvalue above about 0.15.  None: no truncation
    overlap_max_weight:    states with a larger weight are never removed by the truncation (the largest
                           dens_filter_thresh_solver effectively used)
    overlap_reoptimize:    after a truncation, the states are reoptimized (Davidson sweeps, use_davidson only, starting
                           from the truncated states, at most max_iter sweeps) with dens_filter_thresh_solver raised to
                           the largest weight of the removed states, so that the smaller state space is optimal for its
                           size, and the overlap criterion is checked again, at most overlap_max_cycles times (with a
                           final plain truncation, if it is still not met); the filter threshold only grows, up to
                           overlap_max_weight
    """
    if pool not in ("roots", "csf", "none"):
        raise ValueError(f"unknown pool {pool!r}")
    tensornet.initialize_timer()
    tensornet.tensorly_backend.initialize_timer()

    n_frag = 2
    # the core projection (get_ints.direct_CoreProj) alone does not retain the orthonormality of the valence orbitals
    # within a fragment, which get_ints restores by default (orthonormalize_valence=True, Loewdin within each fragment)
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
    # the final evaluations: the XR orders and the leakage-metric ansatz ("metric"), with the order of the overlap S2
    # each one inverts (truncation criterion)
    evaluations = final_orders + (["metric"] if leakage_metric_order is not None else [])
    def overlap_order(ev):
        return leakage_metric_order if ev == "metric" else ev
    def ev_name(ev):
        return f"leakage metric (S2^[{leakage_metric_order}])" if ev == "metric" else f"XR[{ev}]"
    if leakage_metric_order is not None and not 2 <= leakage_metric_order <= 4:
        raise ValueError(f"leakage_metric_order has to be 2, 3 or 4, not {leakage_metric_order}")
    if selected_engine not in ("hci", "fci"):
        raise ValueError(f"unknown selected_engine {selected_engine!r}")
    if selected_cfgs is None and selected_spaces_dir is not None:
        selected_cfgs = read_spaces(selected_spaces_dir, sectors)
        print(f"selected CFGs per sector (read from {selected_spaces_dir}):",
              {chg: {mult: len(cfgs) for mult, cfgs in by_mult.items()} for chg, by_mult in selected_cfgs.items()})
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
    pair_rotation = pair_rotation or bridge_strong > 0
    split_sectors = None
    if bridge_strong > 0:
        from strong_split import required_sectors
        split_sectors = lambda engine: required_sectors(engine, sectors, bridge_strong, glue=bridge_out is not None)
    (mo_coeffs, hummr_states) = dens_builder.init_backend([[frag_name, hummr_input, roots, max(final_orders + [xr_order_solver]
                                                                                            + ([2] if leakage_metric_order is not None else [])),
                                                            selected_cfgs if use_hci else None,
                                                            split_sectors]])[0]
    csf_indices = None
    if selected_cfgs is not None and not use_hci:
        csf_indices = selected_csf_indices(dens_builder.csf_layout(frag_name), selected_cfgs)
        hummr_states = {chg: {mult: project_states(vecs, csf_indices[chg][mult], len(vecs)) for mult, vecs in mult_dict.items()}
                        for chg, mult_dict in hummr_states.items()}
        print("selected CSFs per sector:", {chg: {mult: len(idx) for mult, idx in mult_dict.items()} for chg, mult_dict in csf_indices.items()})

    states = [{chg: {mult: [np.array(vec) for vec in vecs[:n_init_states]] for mult, vecs in mult_dict.items()}
               for chg, mult_dict in hummr_states.items()} for _ in range(n_frag)]
    if states_in is not None:
        read_states, info = load_states(states_in, dens_builder.csf_layout(frag_name))
        for frag in range(n_frag):
            for chg, by_mult in read_states[frag].items():
                for mult, vecs in by_mult.items():
                    if chg in states[frag] and mult in states[frag][chg]:
                        states[frag][chg][mult] = vecs
        print(f"initial states read from {states_in} ({info}):", [count_states(states[frag]) for frag in range(n_frag)])
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
    pair_ints_cache = {}
    def pair_ints(thresh):
        """the integrals with the inter-fragment overlap restricted to the orbital pairs above thresh (None: ints)"""
        if thresh is None:
            return ints
        if thresh not in pair_ints_cache:
            pair_ints_cache[thresh] = get_ints(BeN, project_core, int_timer, spin_ints=False, backend="lible", pair_thresh=thresh)
        return pair_ints_cache[thresh]
    # the integrals of the approximated orders (see leakage_pair_thresh, metric_pair_thresh), for all evaluations
    leakage_ints = {k: pair_ints(th) for k, th in (leakage_pair_thresh or {}).items()}
    metric_ints = {k: pair_ints(th) for k, th in (metric_pair_thresh or {}).items()}
    nuc_rep = ints[2]
    E_nuc = sum(nuc_rep[m1,m2] for m1 in range(n_frag) for m2 in range(m1+1))

    def embedded(frag, dens, H1):
        from embedded import EmbeddedHamiltonian
        return EmbeddedHamiltonian(frag, ints, dens, dens_builder.engine.fragments[frag_name], monomer_charges,
                                   H1[1 - frag], target_multiplicity=target_multiplicity)

    def build_dens(frag, frag_states, xr_level=xr_order_solver, bra_states=None):
        # both fragments are the same molecule, so they share the hummr instance
        return build_frag_densities(dens_builder, frag_name, frag_states, xr_level, BeN[frag].n_elec_ref, bra_states=bra_states,
                                    streamed=streamed, memory_budget=memory_budget)

    #########
    # Optimize the fragment states
    #########

    def solve_states(start_states, dens_filter, n_iter, pool_states):
        """optimized states (start_states is not modified) and their weights (state_solver_csf.optimize_states)"""
        start_states = [{chg: {mult: list(vecs) for mult, vecs in by_mult.items()} for chg, by_mult in st.items()} for st in start_states]
        state_weights = [{}, {}]
        result = optimize_states(n_iter, xr_order_solver, start_states, pool_states, ints, build_dens,
                                 monomer_charges, target_multiplicity, target_charge=target_charge,
                                 max_states_per_sector=max_states_per_sector,
                                 conv_thresh=conv_thresh, dens_filter_thresh=dens_filter,
                                 target_state=target_state, eval_reduced=eval_reduced, exhaust_pool=(pool == "csf"),
                                 use_gradients=use_gradients, grad_level=grad_level, grad_conv_thresh=grad_conv_thresh,
                                 pool_order=pool_order, embedded=embedded, use_davidson=use_davidson, davidson_tol=davidson_tol,
                                 davidson_preconditioner=davidson_preconditioner,
                                 dm_perturbation=dm_perturbation, dm_vectors=dm_vectors, state_weights=state_weights)
        return result, state_weights

    (energies, reduced_energies, states), weights = solve_states(states, dens_filter_thresh_solver, max_iter, additional_states)

    #########
    # Final evaluation in the optimized state space
    #########

    # (for several final XR orders, e.g. [1, 2], with the same optimized states; the densities are built once
    #  for the highest order, since they contain all lower ones)
    print("final state space:", [count_states(states[frag]) for frag in range(n_frag)])
    if states_out is not None:
        save_states(states_out, states, dens_builder.csf_layout(frag_name), displacement=displacement, basis=basis,
                    hummr_input=hummr_input)
        print(f"final states written to {states_out}")
    if pair_rotation:
        from pair_rotation import corresponding_rotations, valence_overlap, rotated_mo_coeffs, StateRotation
        if selected_cfgs is not None:
            raise NotImplementedError("pair_rotation requires the complete CSF spaces (no selected configurations)")
        hummr_frag = dens_builder.engine.fragments[frag_name]
        n_core, n_act = hummr_frag.n_core, hummr_frag.n_act
        R_A, R_B, sigma = corresponding_rotations(valence_overlap(ints, (0, 1), n_core, n_act))
        for frag, R in enumerate((R_A, R_B)):
            BeN[frag].basis.MOcoeffs = rotated_mo_coeffs(BeN[frag].basis.MOcoeffs, n_core, R)
            states[frag] = StateRotation(hummr_frag, R)(states[frag])
        ints = get_ints(BeN, project_core, int_timer, spin_ints=False, backend="lible")
        pair_ints_cache.clear()
        leakage_ints = {k: pair_ints(th) for k, th in (leakage_pair_thresh or {}).items()}
        metric_ints = {k: pair_ints(th) for k, th in (metric_pair_thresh or {}).items()}
        dev = np.abs(valence_overlap(ints, (0, 1), n_core, n_act) - np.diag(sigma)).max()
        print(f"pair rotation: sigma = {np.array2string(sigma[::-1], precision=4)} (strongest first, the last active "
              f"orbitals), max deviation of the rotated S_AB from diag(sigma) {dev:.1e}")
    bridge_en = {}     # final_en["bridge[n]"]: the energies of the exact-bridge ansatz (bridge_out, factorized mode)
    if bridge_strong > 0:
        from strong_split import StrongSplit
        splitter = StrongSplit(hummr_frag, range(n_act - bridge_strong, n_act))
        # {(chg, mult): {path: (sector, weak parts (n_states, dim))}} per fragment (the input of the bridge densities)
        split_states = [{(chg, mult): splitter(chg, mult, vecs) for chg, by_mult in states[frag].items()
                         for mult, vecs in by_mult.items() if len(vecs)} for frag in range(n_frag)]
        for frag in range(n_frag):
            for (chg, mult), parts in sorted(split_states[frag].items()):
                w = {path: np.einsum("ij,ij->i", chi, chi) for path, (_, chi) in parts.items()}
                total = sum(w.values())
                vacant = [path for path in w if all(occ == 0 for _, occ, _ in path)][0]
                i = int(np.argmin(w[vacant]))                       # the state with the largest weight on the strong orbitals
                label = lambda path: "".join(str(occ) for _, occ, _ in path) + f"/{path[-1][2]}"
                print(f"strong split fragment {frag}, sector ({chg:+d}, {mult}): norm sum rule {np.abs(total - 1).max():.1e}; "
                      f"state {i} (occupations of the strong orbitals / multiplicity of the weak part): "
                      + ", ".join(f"{label(path)} {w[path][i]:.4f}" for path in sorted(w)))
        # orthonormal bases of the weak parts per weak sector; their densities rho_W (the densities of the weak parts are
        # X_bra^T rho_W X_ket, strong_split.part_density) are built on demand by the bridge assembly, so that they do
        # not stay in memory next to the densities of the final evaluation
        from strong_split import WeakBasis
        weak_bases = [WeakBasis(split_states[frag]) for frag in range(n_frag)]
        def weak_densities(frag, xr_level):
            return build_dens(frag, weak_bases[frag].states(), xr_level=xr_level)
        print("weak bases (charge: {multiplicity: size}):", [weak_bases[frag].sizes() for frag in range(n_frag)])
        if bridge_out is not None:
            if bridge_strong != 1:
                raise NotImplementedError("the bridge assembly is implemented for bridge_strong = 1")
            from bridge_assembly import BridgeAssembly, product_states
            products = product_states(states)
            from bridge_assembly import add_extra_densities, GluedAssembly
            out = {}
            if bridge_mode in ("factorized", "both"):
                from bridge_assembly import active_integrals
                from bridge_mixed import WeakOperators, MixedEvaluator
                weak_dens = [add_extra_densities(hummr_frag, weak_bases[frag], weak_densities(frag, 2)) for frag in range(n_frag)]
                nmax_mixed = int(os.environ.get("BRIDGE_MIXED_NMAX", 2))
                from get_xr_result_spin_adapt import resolve_memory_budget
                budget = resolve_memory_budget(memory_budget, hummr_frag.orbitals.shape[1], n_act, 2, weak_bases[0].states())
                # the order 2 with the overlap restricted to the pairs above bridge_pair_thresh (all of its terms, also
                # the biorthogonalization and sigma of the mixed terms), the orders 0 and 1 exact
                runs = [(ints, sigma, None)] if bridge_pair_thresh is None else \
                       [(ints, sigma, (0, 1)), (pair_ints(bridge_pair_thresh), np.where(sigma > bridge_pair_thresh, sigma, 0.0), (2,))]
                for run_ints, run_sigma, orders in runs:
                    assembly = BridgeAssembly(split_states, weak_bases, weak_dens, run_ints, n_core + n_act - 1, sigma[-1], 2,
                                              hummr_frag._n_act_elec + hummr_frag._input_charge, pair_slots=leakage_pair_slots)
                    mats = assembly.compute(products, n_core=n_core, n_act=n_act, orders=orders)
                    M_act, G_act = active_integrals(run_ints, n_core, n_act)
                    mixed = MixedEvaluator(assembly, [WeakOperators(hummr_frag, weak_bases[frag], assembly.n_val) for frag in range(n_frag)],
                                           M_act, G_act, run_sigma, n_act, nmax_mixed, memory_budget=budget,
                                           orders=orders).compute(products)
                    for name, mat in (("S", mats["S"]), ("H_weak", mats["H_weak"]), ("H_strong", mats["H_strong"]), ("H_mixed", mixed)):
                        mat = np.array(mat)
                        out[name] = mat if name not in out else out[name] + mat
                    assembly = mixed = mats = None
                weak_dens = None
            if bridge_mode in ("glued", "both"):
                def build_rect(frag, ket, bra):
                    return build_frag_densities(dens_builder, frag_name, ket, 2, BeN[frag].n_elec_ref, bra_states=bra,
                                                streamed=streamed, memory_budget=memory_budget)
                glued = GluedAssembly(splitter, split_states, weak_bases, states, build_rect, ints, n_core + n_act - 1,
                                      sigma[-1], 2, hummr_frag._n_act_elec + hummr_frag._input_charge).compute(products)
                out.update(S_A=np.array(glued["S"]), H_A=np.array(glued["H"]))
                glued = None
            np.savez(bridge_out, **out, products=np.array([(kA[0], kA[1], iA, kB[0], kB[1], iB) for kA, iA, kB, iB in products]))
            print(f"bridge overlap S^(k), k = 0..2, of {len(products)} singlet product states written to {bridge_out}")
            if "H_mixed" in out:
                # E(n): the lowest real eigenvalue of S_n^-1 H_n, S_n and H_n summed up to the weak order n
                H = out["H_weak"] + out["H_strong"] + out["H_mixed"]
                for n in range(len(H)):
                    w_n = np.linalg.eigvals(np.linalg.solve(out["S"][:n + 1].sum(0), H[:n + 1].sum(0)))
                    w_n = w_n[np.abs(w_n.imag) < 1e-8].real
                    bridge_en[f"bridge[{n}]"] = (w_n.min() if len(w_n) else np.nan) + E_nuc
                    print(f"final bridge (K = 1) energy, weak order {n}: {bridge_en[f'bridge[{n}]']}")
            assembly = weak_dens = mats = None
    # the state space of every final order: the complete one, or the one truncated (and reoptimized) by the overlap
    # criterion
    def sector_list(st, frag):
        return local_sectors({chg: {mult: len(vecs) for mult, vecs in by_mult.items()} for chg, by_mult in st[frag].items()},
                             monomer_charges[frag])

    def subset(st, keep):
        """the states of st with keep[frag] (boolean masks over the local states)"""
        ret = []
        for frag in range(n_frag):
            sub = {}
            for chg, mult, sl in sector_list(st, frag):
                sub.setdefault(chg, {})[mult] = [vec for vec, k in zip(st[frag][chg][mult], keep[frag][sl]) if k]
            ret.append(sub)
        return ret

    # The order-k terms of S2 (with metric_ints) of every state space, shared by the overlap truncation, the overlap
    # check and the final evaluations, and the densities of the last state space (one set at a time, released before
    # the next one is built), shared by the overlap truncation and the final evaluation of the same states.
    terms_of = {}    # id(states): (states, {k: order-k term of S2})
    def s_terms(st):
        if id(st) not in terms_of:
            terms_of[id(st)] = (st, {})
        return terms_of[id(st)][1]
    def inherit_terms(st_new, st_old, keep):
        """the S2 terms of the subset st_new of st_old (keep[frag] masks): principal submatrices (orthonormal states)"""
        old = s_terms(st_old)
        if old:
            labels = dimer_labels([{chg: {mult: len(vecs) for mult, vecs in by_mult.items()} for chg, by_mult in st_old[frag].items()}
                                   for frag in range(n_frag)], monomer_charges)
            rows = keep[0][labels[:, 0]] & keep[1][labels[:, 1]]
            s_terms(st_new).update({k: term[np.ix_(rows, rows)] for k, term in old.items()})
    dens_slot = {"states": None, "level": -1, "dens": None}
    def release_dens():
        dens_slot.update(states=None, level=-1, dens=None)
    def shared_dens(st, xr_level):
        """the densities of the state space st (of at least xr_level), built only if the last ones do not fit"""
        if dens_slot["states"] is not st or dens_slot["level"] < xr_level:
            release_dens()
            dens_slot.update(states=st, level=xr_level, dens=[build_dens(frag, st[frag], xr_level=xr_level) for frag in range(n_frag)])
        return dens_slot["dens"]

    def overlap_data(st, w, orders):
        """S2^[n] of the orders, their row labels, the sector slices and the local state weights (w, or the XR[0]
        reduced density matrix diagonal, if w does not cover all states) of the state space st"""
        # (S2^[n] requires the densities of xr_level n - 2)
        dens_t = shared_dens(st, max(0, max(orders) - 2))
        S2, labels = overlap_matrices(ints, dens_t, monomer_charges, target_multiplicity, target_charge, orders,
                                      order_ints=metric_ints, terms=s_terms(st))
        sectors_t = [sector_list(st, frag) for frag in range(n_frag)]
        if w is not None and all(mult in w[frag].get(chg, {}) and len(w[frag][chg][mult]) == len(st[frag][chg][mult])
                                 for frag in range(n_frag) for chg, mult, _ in sectors_t[frag]):
            w_loc = [np.concatenate([w[frag][chg][mult] for chg, mult, _ in sectors_t[frag]]) for frag in range(n_frag)]
        else:
            _, vecs = get_xr_states(ints, dens_t, 0, monomer_charges, target_multiplicity, target_state=target_state,
                                    target_charge=target_charge)
            vec = np.asarray(vecs).reshape(sectors_t[0][-1][2].stop, sectors_t[1][-1][2].stop)
            vec = np.real(vec * np.conj(vec.flat[np.argmax(np.abs(vec))]))
            vec /= np.linalg.norm(vec)
            w_loc = [np.sum(vec ** 2, axis=1), np.sum(vec ** 2, axis=0)]
            print("overlap truncation: no solver weights, the XR[0] reduced density matrices are used")
        return S2, labels, [[sl for _, _, sl in sectors_t[frag]] for frag in range(n_frag)], w_loc

    spaces = {ev: (None, states) for ev in evaluations}    # evaluation: (label, states)
    trunc_evs = [ev for ev in evaluations if overlap_order(ev) >= 1] if overlap_min_eigval is not None else []
    reoptimize = overlap_reoptimize and use_davidson and max_iter > 0
    if trunc_evs:
        full_data = overlap_data(states, weights, sorted(set(overlap_order(ev) for ev in trunc_evs)))
        for ev in trunc_evs:
            order = overlap_order(ev)
            st, w, thresh = states, weights, dens_filter_thresh_solver
            for cycle in range(overlap_max_cycles + 1):
                S2, labels, slices, w_loc = full_data if cycle == 0 else overlap_data(st, w, [order])
                keep, low, n_removed = overlap_truncation(S2[order], labels, w_loc, slices, overlap_min_eigval, overlap_max_weight)
                n_states = [len(k) for k in keep]
                print(f"overlap truncation for {ev_name(ev)} (cycle {cycle}, dens_filter_thresh {thresh:.1e}): lowest eigenvalue of "
                      f"S2^[{order}] {low:.5f} with {n_removed} of {n_states} states removed"
                      + ("" if low >= overlap_min_eigval else f" (below {overlap_min_eigval}, no more states with a weight below "
                                                              f"{overlap_max_weight:.0e})"))
                if n_removed == 0:
                    break
                removed = max(w_loc[frag][i] for frag in range(n_frag) for i in np.flatnonzero(~keep[frag]))
                if not reoptimize or cycle == overlap_max_cycles or removed * (1 + 1e-6) <= thresh:
                    st, parent = subset(st, keep), st
                    inherit_terms(st, parent, keep)
                    break
                thresh = min(overlap_max_weight, removed * (1 + 1e-6))
                print(f"\n###### reoptimization of the states for {ev_name(ev)} with dens_filter_thresh {thresh:.1e} ######")
                release_dens()
                (_, _, st), w = solve_states(subset(st, keep), thresh, max_iter, [{}, {}])
            if st is not states:
                spaces[ev] = (ev, st)
    final_en = {}
    for label in dict.fromkeys(label for label, _ in spaces.values()):
        evs = [ev for ev in evaluations if spaces[ev][0] == label]
        sub_states = spaces[evs[0]][1]
        check_order = overlap_check_order or None
        # (S2^[3] requires the densities of xr_level 1, S2^[4] and L^[2] the ones of xr_level 2)
        dens_level = max([2 if ev == "metric" else ev for ev in evs] + ([check_order - 2] if check_order else []))
        dens = shared_dens(sub_states, dens_level)
        terms = s_terms(sub_states)
        print(f"state space of {', '.join(ev_name(ev) for ev in evs)}:", [count_states(sub_states[frag]) for frag in range(n_frag)])
        if check_order:
            check_overlap(ints, dens, monomer_charges, target_multiplicity, target_charge, check_order,
                          xr_orders=sorted(set(overlap_order(ev) for ev in evs)), order_ints=metric_ints, terms=terms)
        approx = dict(leakage_ints=leakage_ints, metric_ints=metric_ints, leakage_v_thresh=leakage_v_thresh, metric_terms=terms,
                      leakage_pair_slots=leakage_pair_slots)
        for ev in evs:
            if ev == "metric":
                E, _ = get_leakage_metric_states(ints, dens, monomer_charges, target_multiplicity, target_state=target_state,
                                                 target_charge=target_charge, metric_order=leakage_metric_order, **approx)
            else:
                E, _ = get_xr_states(ints, dens, ev, monomer_charges, target_multiplicity, target_state=target_state,
                                     target_charge=target_charge, **approx)
            final_en[ev] = np.real(E) + E_nuc
            print(f"final {ev_name(ev)} energy: {final_en[ev]}")
        dens = terms = approx = None
        if all(spaces[ev][1] is not sub_states for ev in evaluations if spaces[ev][0] != label):
            terms_of.pop(id(sub_states), None)    # (5 matrices of the dimer dimension squared, not needed any more)
    release_dens()
    final_en = {ev: final_en[ev] for ev in evaluations}
    if isinstance(xr_order_final, int) and leakage_metric_order is None and not bridge_en:
        final_en = final_en[xr_order_final]
    else:
        final_en.update(bridge_en)

    dens_builder.finalize_all()

    return [np.real(E) + E_nuc for E in energies], [np.real(E) + E_nuc for E in reduced_energies], final_en


if __name__ == "__main__":
    # frozen 1s, state space optimization at XR[0] starting from the lowest 8 states per sector, then XR[0], XR[1] and
    # XR[2] in the optimized (overlap-truncated) state space and the exact-bridge ansatz of the optimized states;
    # streamed density contractions with an 8 GB cache budget
    # (for HCI spaces: hci_epsilon, selected_engine="hci", selected_spaces_dir; not yet with the bridge)
    # solver = "davidson": Davidson sweeps in the complete (selected) CSF space of one fragment with the other fragment's
    #                      states fixed; per sector, the states with a weight above dens_filter_thresh_solver in the
    #                      (perturbed, dm_perturbation) reduced density matrix are kept; max_iter is the maximum number
    #                      of sweeps. 6-31G, eps = 1e-3: 1e-8 and 1e-2 keep 22 states per fragment instead of 68 (all
    #                      states up to max_states_per_sector) with XR[1]/XR[2] within 6e-6/3e-6 Eh of the latter
    # solver = "cipsi":    passes through all CSFs (pool="csf"), the remaining CSFs reordered by their first-order
    #                      importance after every solve
    solver = "davidson"
    solver_args = {"davidson": dict(pool="none", use_davidson=True, dens_filter_thresh_solver=1e-8, dm_perturbation=1e-2,
                                     davidson_tol=1e-5),
                   "cipsi":    dict(pool="csf", pool_order="cipsi", dens_filter_thresh_solver=1e-7)}[solver]
    # Be2 cc-pVDZ at 3.0 A: Davidson-XR[0] state optimization in the complete (FCI) CSF spaces, final XR[0..2] in the
    # overlap-truncated state space, and the exact-bridge ansatz (K = 1, factorized) of the optimized states up to the weak
    # order 2, with the order-2 terms restricted to the corresponding-orbital pairs above 0.1 (results_xr_order_states.tex,
    # step 5: 11.3 min and 12.6 GB on 8 cores with these states, bridge order 2 within 29 uEh of the exact bridge)
    solver_en, reduced_en, final_en = run_xr(3.0, 50, xr_order_final=[0, 1, 2], n_init_states=8, max_states_per_sector=20,
                                             eval_reduced=True, hummr_input="/home/marco/hummr_tests/scratch/be_ccdz_fci_fcore.inp",
                                             basis="cc-pvdz", use_gradients=False, streamed=True, memory_budget=8e9,
                                             states_in=None, states_out=None,   # e.g. "states_3.0.pkl": read / write the optimized states
                                             overlap_min_eigval=0.15, overlap_max_weight=1e-4, overlap_reoptimize=False,
                                             overlap_check_order=4,
                                             # the bridge requires the complete CSF spaces (pair rotation), no HCI selection
                                             bridge_strong=1, bridge_out="bridge_ccdz_3.0.npz", bridge_mode="factorized",
                                             bridge_pair_thresh=0.1,
                                             **solver_args)
    print("solver energies (enlarged state spaces):  ", solver_en)
    print("solver energies (compressed state spaces):", reduced_en)
    for ev, E in final_en.items():
        print(f"final {'leakage-metric' if ev == 'metric' else ev if str(ev).startswith('bridge') else f'XR[{ev}]'} energy:", E)
    #print("ref from HCI with frozen core 6-31G = ", -29.21086186)
    print("ref from the brute-force product-space reference E_P (cc-pVDZ, 3.0 A, these settings) = ", -29.2336776283)
