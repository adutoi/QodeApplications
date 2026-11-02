#    (C) Copyright 2024, 2026 Marco Bauer
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

# Spin-adapted (CSF based) state solver.
#
# The fragment state spaces are stored as
#
#     states[frag][chg][mult] = list of CSF coefficient vectors
#
# which is the format provided and consumed by the hummr density backend.
#
# Currently only the gradient free variant is implemented: in every iteration the state space
# of both fragments is enlarged by states from a pool of candidate states (additional_states, e.g.
# further hummr roots or simply all CSFs as unit vectors, see unit_vector_pool),
# the XR Hamiltonian is built and diagonalized in this enlarged space and the fragment state
# spaces are compressed again by keeping the dominant eigenvectors of the fragment reduced
# density matrices of the target state(s), sector by sector.
#
# With use_gradients=True the fragments are optimized alternately with the state gradients of
# state_gradients_csf.py (gradient based variant, as in state_solver.py): the state space of the optimized
# fragment is enlarged by the gradients of its states (and, if a pool is given, the state space of the other
# fragment by states from the pool), followed by the same diagonalization and compression.

from get_xr_result_spin_adapt import get_xr_H, get_xr_states_from_H, local_sectors, drop_empty_sectors, product_basis_mask, \
                                     multiplicities_couple
from state_gradients_csf import state_gradients
from state_screening import orthogonalize
from qode.util import struct, timer
import excitonic

import pickle
import numpy as np
import scipy as sp



def count_states(states):
    return {chg: {mult: len(vecs) for mult, vecs in mult_dict.items()} for chg, mult_dict in states.items()}

def unit_vector_pool(states, n_passes=1, csf_indices=None):
    """
    Pool of candidate states consisting of all CSFs of every sector, i.e. unit vectors (zero vectors with a
    single 1) in the CSF basis of the sector.  The dimension is taken from the given states (at least one
    state per sector is required).  Candidates, which are (nearly) contained in the current state space,
    are skipped by the solver, so the pool can be passed through completely.
    n_passes:    the CSFs are offered this many times (CSFs discarded in an early compression can become
                 relevant again later)
    csf_indices: optional csf_indices[chg][mult] = CSF indices, which restrict the pool of the sector to these
                 CSFs (e.g. those of a selected CI space, see hummr_dens.space_selection)
    """
    pool = {}
    for chg, mult_dict in states.items():
        for mult, vecs in mult_dict.items():
            if len(vecs) == 0:
                raise ValueError(f"unit_vector_pool needs at least one state in sector (chg {chg}, mult {mult}) for its dimension")
            unit_vectors = np.eye(len(vecs[0]))
            if csf_indices is not None:
                unit_vectors = unit_vectors[csf_indices[chg][mult]]
            pool.setdefault(chg, {})[mult] = list(unit_vectors) * n_passes
    return pool

def real_eigenvectors(vals, vecs, degeneracy_tol=1e-7, imag_tol=1e-10):
    """
    real, unit-norm eigenvectors (columns) of a real non-symmetric matrix from the (complex) ones of numpy/scipy eig,
    for real eigenvalues vals: a non-degenerate eigenvector is real up to its phase, which is fixed; the eigenvectors
    of a degenerate cluster (|E_i - E_j| < degeneracy_tol, e.g. the two components of a Pi state) may be complex
    mixtures, which are replaced by an orthonormal real basis of the same space (the leading left singular vectors
    of their real and imaginary parts; any basis of an eigenspace is one of eigenvectors)
    """
    vals = np.real(np.asarray(vals))
    out = np.zeros(vecs.shape)
    i = 0
    while i < len(vals):
        j = i + 1
        while j < len(vals) and abs(vals[j] - vals[j - 1]) < degeneracy_tol:
            j += 1
        block = vecs[:, i:j]
        vec = block[:, 0] * np.conj(block[np.argmax(np.abs(block[:, 0])), 0])
        if j - i == 1 and np.linalg.norm(np.imag(vec)) <= imag_tol * np.linalg.norm(vec):
            out[:, i] = np.real(vec) / np.linalg.norm(np.real(vec))
        elif j - i == 1:
            # complex, but no degenerate partner among the given roots (e.g. the list ends within a degenerate
            # cluster): the dominant real direction of its span
            print(f"WARNING: the eigenvector of {vals[i]:.8f} is complex (degenerate with a root not solved for?), "
                  "its dominant real direction is used")
            U = np.linalg.svd(np.stack([np.real(vec), np.imag(vec)], axis=1), full_matrices=False)[0]
            out[:, i] = U[:, 0]
        else:
            U, s, _ = np.linalg.svd(np.concatenate([np.real(block), np.imag(block)], axis=1), full_matrices=False)
            if s[j - i - 1] < 1e-8 * s[0]:
                raise ValueError(f"the eigenvectors of the degenerate roots {vals[i:j]} do not span a real space of their dimension")
            out[:, i:j] = U[:, :j - i]
        i = j
    return out

def save_states(path, states, csf_layout, **info):
    """
    writes the fragment states states[frag][chg][mult] (lists of CSF vectors) together with the CSF layout of their
    space ({chg: {mult: [(cfg, pos, dim)]}}, dens_provider.csf_layout) and optional info (e.g. the geometry), so that
    load_states can map them onto another (selected) CSF space
    """
    with open(path, "wb") as f:
        pickle.dump({"states": [{chg: {mult: [np.asarray(v, dtype=float) for v in vecs] for mult, vecs in by_mult.items()}
                                 for chg, by_mult in frag_states.items()} for frag_states in states],
                     "csf_layout": csf_layout, "info": info}, f)

def map_states(states, old_layout, new_layout):
    """
    states[chg][mult] (CSF vectors of the space old_layout) on the CSF space new_layout via their CFGs (the CSFs of a
    CFG do not depend on the rest of the space), without orthonormalization: exact if new_layout contains all CFGs of
    old_layout (e.g. a selected space and its extension), CFGs missing in new_layout are dropped
    """
    out = {}
    for chg, by_mult in states.items():
        for mult, vecs in by_mult.items():
            old = {cfg: (pos, dim) for cfg, pos, dim in old_layout[chg][mult]}
            new = new_layout[chg][mult]
            n_new = sum(dim for _, _, dim in new)
            mapped = []
            for vec in vecs:
                v = np.zeros(n_new)
                for cfg, pos, dim in new:
                    if cfg in old:
                        v[pos:pos + dim] = vec[old[cfg][0]:old[cfg][0] + dim]
                mapped.append(v)
            out.setdefault(chg, {})[mult] = mapped
    return out

def load_states(path, csf_layout, thresh=1e-8):
    """
    reads the states written by save_states and maps them onto the CSF space with csf_layout via their CFGs (the
    CSFs of a CFG do not depend on the rest of the space; CFGs missing in the new space are dropped, new ones get zero
    coefficients), then orthonormalizes them per sector (vectors with a norm below thresh after the mapping are
    dropped). Returns (states[frag][chg][mult], info).
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    old_layout = data["csf_layout"]
    states = []
    for frag_states in data["states"]:
        mapped = {}
        for chg, by_mult in frag_states.items():
            for mult, vecs in by_mult.items():
                if chg not in csf_layout or mult not in csf_layout[chg]:
                    continue
                old = {cfg: (pos, dim) for cfg, pos, dim in old_layout[chg][mult]}
                new_layout = csf_layout[chg][mult]
                n_new = sum(dim for _, _, dim in new_layout)
                out = []
                for vec in vecs:
                    v = np.zeros(n_new)
                    for cfg, pos, dim in new_layout:
                        if cfg in old:
                            v[pos:pos + dim] = vec[old[cfg][0]:old[cfg][0] + dim]
                    for u in out:
                        v -= (u @ v) * u
                    if np.linalg.norm(v) > thresh:
                        out.append(v / np.linalg.norm(v))
                if out:
                    mapped.setdefault(chg, {})[mult] = out
        states.append(mapped)
    return states, data["info"]

def optimize_states(max_iter, xr_order, states, additional_states, ints, build_dens, monomer_charges, target_multiplicity,
                    target_charge=0, max_states_per_sector=15, conv_thresh=1e-6, dens_filter_thresh=1e-7, target_state=0, eval_reduced=False,
                    exhaust_pool=False, lin_dep_thresh=1e-6, use_gradients=False, grad_level="herm", grad_conv_thresh=1e-5,
                    pool_order="given", embedded=None, use_davidson=False, davidson_tol=1e-5, davidson_max_iter=100,
                    davidson_max_size=40, dm_perturbation=1e-3, davidson_preconditioner=True, dm_vectors="right",
                    state_weights=None, target_weights=None):
    """
    max_iter:              maximum number of enlarge-diagonalize-compress cycles
    target_state:          index (0: lowest root of the target block) or list of indices of the roots the states are
                           optimized for (state averaging): the compressions keep the dominant eigenvectors of the
                           averaged reduced density matrix sum_k w_k rho_k (the Davidson sweeps: of the target roots of
                           the embedded problem, with the perturbation averaged in the same way); the energies are then
                           arrays over the target roots, converged if all of them are.  The roots below the highest
                           target are always solved for (the eigenvectors are found by their order), whether they are
                           targets or not
    target_weights:        weights w_k of the target roots (normalized to one; default: equal weights)
    states:                states[frag][chg][mult] = list of CSF vectors (initial guess), modified in place
    additional_states:     same structure, pool of candidate states, which are added successively
    exhaust_pool:          only stop due to convergence once the pool has been passed through completely
                           (with a unit-vector pool the energy may stagnate for some iterations, before an
                           important CSF is reached)
    lin_dep_thresh:        candidates whose component orthogonal to the current state space is smaller are skipped
    use_gradients:         gradient based variant: alternately enlarge the state space of one fragment by its state
                           gradients (and the other one from the pool, if given)
    grad_level:            "herm" (right eigenvector only) or "full" (exact gradient of the non-hermitian eigenvalue,
                           left and right eigenvectors), see state_gradients_csf.py.  Since the compression keeps the
                           dominant states of the reduced density matrix of the right eigenvector, the fixed point of
                           this solver is the one of the "herm" gradient (the "full" gradient does not vanish there, so
                           the solver does not converge with it, although each step is a valid search direction).
    grad_conv_thresh:      the gradient based variant is converged, if additionally the norm of the gradients is below
    build_dens:            build_dens(frag, states[frag]) -> fragment densities as needed by get_xr_H
    max_states_per_sector: the enlarged state space contains at most this many states per (chg, mult) sector, unless
                           the sector already holds more states (after compression), in which case it may grow by 1/3
                           of its current size, i.e. the cap is max(max_states_per_sector, 4/3 * n_current)
    pool_order:            "given": the pool is passed through in its order, "cipsi": after every solve, the remaining
                           pool vectors v of each fragment are reordered by their first-order (Epstein-Nesbet)
                           importance sum_b (v^T r_b)^2 / (E - <v b|H^eff|v b>)^2, r = (H^eff - E) x the residual of
                           the target state x in the fragment's complete CSF space (embedded XR[0] problem, the
                           current states projected out, the diagonal element in the diagonal approximation
                           sum_p v_p^2 H^eff_{pb,pb}, exact for the CSF pool), most important first; "residual": the
                           same without the denominator, ||v^T r||^2
    embedded:              embedded(frag, dens, H1) -> embedded.EmbeddedHamiltonian of fragment frag (pool_order="cipsi",
                           use_davidson)
    use_davidson:          Davidson sweeps instead of the pool: alternately for both fragments, the target state of the
                           XR[0] problem with the fragment's complete (selected) CSF space and the other fragment's
                           states fixed (embedded.py) is solved with a non-symmetric Davidson (residual norm
                           davidson_tol, at most davidson_max_iter iterations, subspace restarted at
                           davidson_max_size), and the fragment's states are replaced by the eigenvectors of its
                           (perturbed, see dm_perturbation) reduced density matrix per sector with eigenvalues above
                           dens_filter_thresh (at least one, at most max_states_per_sector); XR[0] only, several target
                           roots with a block Davidson (all roots up to the highest target)
    davidson_preconditioner: diagonal (Davidson) preconditioner (D - theta)^-1 with the diagonal D of H^eff
    dm_perturbation:       weight alpha of the density-matrix perturbation of the Davidson sweeps (White's correction
                           of single-site DMRG): rho += alpha sum_t Y_t Y_t^T with Y_t the contributions of the dimer
                           terms to H^eff x, which contain the fragment's directions coupling the target state to
                           the other fragment's states outside its current set (without it, the number of states per
                           sector cannot grow beyond the other fragment's); these directions are the ones XR[1] and
                           XR[2] couple to, so alpha also decides how well the states describe the higher XR orders
    dm_vectors:            the vectors of the reduced density matrix of the Davidson sweeps: "right" (the right
                           eigenvector x) or "left_right" (additionally the left eigenvector l of the non-symmetric
                           H^eff, from a second Davidson with (H^eff)^T: rho = (X X^T + L L^T) / 2 with ||x|| = ||l|| = 1,
                           and the perturbation alpha/2 sum_t (Y_t Y_t^T + Z_t Z_t^T), Z_t the dimer-term contributions to
                           (H^eff)^T l); the error of the Ritz value in the truncated space is of the order of the
                           products of the parts of l and x outside of it
    state_weights:         if given (a list of two dicts), state_weights[frag][chg][mult] is set to the weights of the final
                           states in the reduced density matrix of their last compression (perturbed for the Davidson
                           sweeps), in the order of the states, e.g. to truncate the state space further afterwards
    eval_reduced:          additionally evaluate the energy in the compressed state space (debugging only,
                           requires an additional density and Hamiltonian build per iteration)
    returns the energy histories of the enlarged and of the compressed state spaces (the latter only if eval_reduced)
    and the optimized states
    """
    target_states = [int(target_state)] if np.ndim(target_state) == 0 else [int(k) for k in target_state]
    if len(set(target_states)) != len(target_states) or min(target_states) < 0:
        raise ValueError(f"target_state has to be a root index or a list of distinct root indices, not {target_state!r}")
    if target_weights is None:
        target_weights = np.full(len(target_states), 1 / len(target_states))
    else:
        target_weights = np.asarray(target_weights, dtype=float)
        if target_weights.shape != (len(target_states),) or np.any(target_weights < 0) or target_weights.sum() <= 0:
            raise ValueError(f"target_weights {target_weights} do not fit the target states {target_states}")
        target_weights = target_weights / target_weights.sum()
    n_roots = max(target_states) + 1     # the roots that are solved for

    pool_tracker = [{chg: {mult: 0 for mult in mult_dict} for chg, mult_dict in additional_states[frag].items()} for frag in range(2)]

    def pool_exhausted():
        return all(pool_tracker[frag][chg][mult] >= len(pool) for frag in range(2)
                   for chg, mult_dict in additional_states[frag].items() for mult, pool in mult_dict.items())

    def solve_left_right(dens):
        """target energy, right and left eigenvectors (reshaped to the product dimensions) and H1"""
        if len(target_states) > 1:
            raise NotImplementedError("the gradient based variant is only implemented for a single target state")
        H1, H2 = get_xr_H(ints, dens, xr_order, monomer_charges, target_multiplicity, target_charge=target_charge)
        mask = product_basis_mask(dens, monomer_charges, target_charge, target_multiplicity)
        out = struct(log=lambda *args: None)
        E, vec_l, vec_r = excitonic.fci((H1, [[None, H2], [None, None]]), out, timer(), target_state=target_states[0],
                                        get_left_and_right=True, basis_mask=mask)
        dims = [H.shape[0] for H in H1]
        def real(vec):
            vec = vec * np.conj(vec[np.argmax(np.abs(vec))]) / np.abs(vec[np.argmax(np.abs(vec))])
            if np.linalg.norm(np.imag(vec)) > 1e-10 * np.linalg.norm(vec):
                raise ValueError("imaginary part of eigenvector is not negligible")
            return np.real(vec)
        vec_r, vec_l = real(vec_r), real(vec_l)
        return np.real(E), (vec_r / np.linalg.norm(vec_r)).reshape(dims), vec_l.reshape(dims), H1

    def add_gradient_states(frag, gradients):
        """append the gradients (orthogonalized against the current states) to the state space of frag"""
        n_added, norms = 0, []
        for chg, mult_dict in gradients.items():
            for mult, grads in mult_dict.items():
                current = [np.array(vec, dtype=float) for vec in states[frag][chg][mult]]
                n_before = len(current)
                for g in grads:
                    norms.append(np.linalg.norm(g))
                    Q = np.array(current)
                    g = g - Q.T @ (Q @ g)
                    g = g - Q.T @ (Q @ g)
                    if np.linalg.norm(g) < grad_conv_thresh * 1e-2:    # (nearly) vanishing or already contained
                        continue
                    current.append(g / np.linalg.norm(g))
                n_added += len(current) - n_before
                states[frag][chg][mult] = current
                print(f"frag {frag}, chg {chg}, mult {mult}: {len(current) - n_before} gradient states added, "
                      f"gradient norms {np.array2string(np.array(norms[-len(grads):]), precision=1)}")
        return n_added, np.sqrt(np.sum(np.square(norms)))

    def enlarge_state_space(frag):
        """
        add linearly independent states from the pool until max(max_states_per_sector, 4/3 * n_current) is reached
        (candidates, which are already contained in the current state space, are skipped); returns the number of
        added states
        """
        n_added = 0
        for chg, mult_dict in additional_states[frag].items():
            for mult, pool in mult_dict.items():
                current = [np.array(vec, dtype=float) for vec in states[frag].setdefault(chg, {}).setdefault(mult, [])]
                n_before, n_skipped = len(current), 0
                # sectors, which require many states, may push beyond max_states_per_sector by a relative increase of 1/3
                # (the XR evaluation scales as the fourth order in the number of states, so this is roughly a factor of 3)
                max_states = max(max_states_per_sector, n_before * 4 // 3)
                Q = np.array(current) if len(current) > 0 else None    # orthonormal rows
                while len(current) < max_states and pool_tracker[frag][chg][mult] < len(pool):
                    vec = np.array(pool[pool_tracker[frag][chg][mult]], dtype=float)
                    pool_tracker[frag][chg][mult] += 1
                    if len(current) > 0:
                        vec = vec - Q.T @ (Q @ vec)
                        vec = vec - Q.T @ (Q @ vec)    # second pass for numerical orthogonality
                    norm = np.linalg.norm(vec)
                    if norm < lin_dep_thresh:
                        n_skipped += 1
                        continue
                    current.append(vec / norm)
                    Q = np.array(current)
                n_added += len(current) - n_before
                states[frag][chg][mult] = current
                if len(current) > n_before or n_skipped > 0:
                    print(f"frag {frag}, chg {chg}, mult {mult}: {len(current) - n_before} states added, {n_skipped} skipped "
                          f"({len(pool) - pool_tracker[frag][chg][mult]} left in pool)")
        return n_added

    last = struct(H1=None)

    def cipsi_reorder(dens, E, target_vec):
        """reorders the remaining pool vectors of both fragments by their first-order importance (see pool_order)"""
        if xr_order != 0 or len(target_states) > 1:
            raise NotImplementedError("pool_order='cipsi' is implemented for XR[0] and a single target state")
        for frag in range(2):
            remaining = {(chg, mult): pool_tracker[frag][chg][mult] for chg, mult_dict in additional_states[frag].items()
                         for mult, pool in mult_dict.items() if pool_tracker[frag][chg][mult] < len(pool)}
            if not remaining:
                continue
            emb = embedded(frag, dens, last.H1)
            d = target_vec if frag == 0 else target_vec.T          # (fragment's local states, other's local states)
            x = emb.states_to_csf(drop_empty_sectors(states[frag]), d)
            r = emb.sigma(x) - E * x
            allowed = allowed_pairs(emb)
            for (chg, mult), start in remaining.items():
                if (chg, mult) not in emb.rows:
                    continue
                r_s = r[emb.rows[(chg, mult)]]
                if len(states[frag][chg][mult]) > 0:
                    C = np.array(states[frag][chg][mult], dtype=float)
                    r_s = r_s - C.T @ (C @ r_s)
                pool = additional_states[frag][chg][mult]
                rest = pool[start:]
                if pool_order == "residual":
                    importance = [np.sum((np.asarray(v, dtype=float) @ r_s) ** 2) for v in rest]
                else:
                    # Epstein-Nesbet: sum_b (v^T r_b)^2 / (E - <v b|H^eff|v b>)^2, the diagonal element in the
                    # diagonal approximation sum_p v_p^2 D[p, b] (exact for the CSF pool)
                    D_s, cols = emb.diagonal()[emb.rows[(chg, mult)]], allowed[emb.rows[(chg, mult)]].any(axis=0)
                    importance = []
                    for v in rest:
                        v = np.asarray(v, dtype=float)
                        denom = E - (v ** 2) @ D_s[:, cols]
                        denom = np.where(np.abs(denom) < 1e-4, np.copysign(1e-4, denom), denom)
                        importance.append(np.sum(((v @ r_s[:, cols]) / denom) ** 2))
                order = np.argsort(importance)[::-1]
                pool[start:] = [rest[i] for i in order]
                print(f"frag {frag}, chg {chg}, mult {mult}: pool reordered by first-order importance "
                      f"(largest {importance[order[0]]:.2e}, {len(rest)} remaining)")

    def solve(dens, roots=None):
        """energies (a scalar for a single root) and eigenvectors (reshaped to the product dimensions) of the roots
        (default: the target states) in the product basis of the current states"""
        roots = target_states if roots is None else roots
        n_states = [count_states(drop_empty_sectors(states[frag])) for frag in range(2)]
        H1, H2 = get_xr_H(ints, dens, xr_order, monomer_charges, target_multiplicity, target_charge=target_charge)
        last.H1 = H1
        # an int target_state allows excitonic.fci to use the sparse eigensolver
        mask = product_basis_mask(dens, monomer_charges, target_charge, target_multiplicity)
        E, vecs = get_xr_states_from_H(H1, H2, target_state=roots if len(roots) > 1 else roots[0], basis_mask=mask)
        vecs = vecs.reshape(vecs.shape[0], -1)
        dims = [H.shape[0] for H in H1]
        real_vecs = real_eigenvectors(np.atleast_1d(E), vecs)
        target_vecs = [real_vecs[:, i].reshape(dims) for i in range(len(roots))]
        return np.real(E), target_vecs, n_states

    def compress_state_space(target_vecs, n_states):
        """
        Replace the states of every sector by the dominant eigenvectors of the (state-averaged) fragment
        reduced density matrix restricted to this sector. Contributions of the reduced density matrix
        between different sectors are neglected (they vanish for spin-pure target states, if the
        sectors differ in multiplicity, and always if they differ in charge).
        """
        dens_mats = [sum(w * vec @ vec.T for w, vec in zip(target_weights, target_vecs)),   # contract over frag 1
                     sum(w * vec.T @ vec for w, vec in zip(target_weights, target_vecs))]   # contract over frag 0
        for frag in range(2):
            for chg, mult, sl in local_sectors(n_states[frag], monomer_charges[frag]):
                rho = dens_mats[frag][sl, sl]
                eigvals, eigvecs = sp.linalg.eigh(0.5 * (rho + rho.T))
                order = np.argsort(eigvals)[::-1]
                eigvals, eigvecs = eigvals[order], eigvecs[:, order]
                keep = eigvals > dens_filter_thresh
                C = np.array(states[frag][chg][mult])   # n_sector_states x n_csf
                new_C = eigvecs[:, keep].T @ C
                if len(new_C) > 0:
                    new_C = orthogonalize(new_C, eps=dens_filter_thresh)
                states[frag][chg][mult] = [vec for vec in new_C]
                if state_weights is not None:
                    state_weights[frag].setdefault(chg, {})[mult] = eigvals[keep][:len(new_C)]
                print(f"frag {frag}, chg {chg}, mult {mult}: {len(new_C)} of {C.shape[0]} states kept, "
                      f"dens eigvals {np.array2string(eigvals[keep], precision=2)}, "
                      f"largest discarded {np.max(eigvals[~keep], initial=0.):.2e}")

    def allowed_pairs(emb):
        """mask of x[p, b]: the sector pairs with the target charge, whose multiplicities can couple to the target"""
        mask = np.zeros((emb.dim_a, emb.dim_b), dtype=bool)
        for sec_a, rows in emb.rows.items():
            for sec_b, cols in emb.cols.items():
                if sec_a[0] + sec_b[0] == target_charge and multiplicities_couple(sec_a[1], sec_b[1], target_multiplicity):
                    mask[rows, cols] = True
        return mask

    def davidson(emb, x0s, E0s, transpose=False):
        """the lowest len(x0s) (real) eigenpairs of H^eff (transpose: (H^eff)^T, i.e. the left eigenvectors) in the
        allowed sector pairs, ordered by their energy, starting from x0s (non-symmetric block Davidson; the converged
        roots get no further correction vectors).  Returns (thetas, xs) with unit-norm x[p, b]."""
        mask = allowed_pairs(emb)
        n = int(mask.sum())
        k = len(x0s)
        if n < k:
            raise ValueError(f"the embedded problem has {n} dimensions, fewer than the {k} roots solved for")
        diag = emb.diagonal()[mask] if davidson_preconditioner else None
        sigma = emb.sigma_T if transpose else emb.sigma
        max_size = min(max(davidson_max_size, 4 * k), n)
        def apply(v):
            x = np.zeros(mask.shape)
            x[mask] = v
            return sigma(x)[mask]
        def orthonormal_add(V, W, ts):
            """appends the components of ts orthogonal to V (normalized, linearly independent ones only)"""
            for t in ts:
                for _ in range(2):
                    if V:
                        t = t - np.array(V).T @ (np.array(V) @ t)
                norm = np.linalg.norm(t)
                if norm > 1e-10:
                    V.append(t / norm)
                    W.append(apply(V[-1]))
        def ritz(Vm, Wm):
            """the lowest k Ritz pairs by their real part, the vectors real (phase-fixed) and normalized"""
            vals, vecs = sp.linalg.eig(Vm.T @ Wm)
            order = np.argsort(vals.real)[:k]
            thetas, Y = vals[order], vecs[:, order]
            for j in range(k):
                if abs(thetas[j].imag) > 1e-8:
                    print(f"WARNING: Davidson Ritz value {thetas[j]:.8f} is complex, its real part is used")
            # (V is orthonormal, so the real basis of y is one of u = V y; complex conjugate Ritz pairs and the complex
            #  mixtures of degenerate ones are replaced by a real basis of their span)
            Y = real_eigenvectors(thetas.real, Y, degeneracy_tol=1e-8 + 1e-8 * np.abs(thetas.real).max())
            return thetas.real, Y
        V, W = [], []
        orthonormal_add(V, W, [x0[mask] for x0 in x0s])
        orthonormal_add(V, W, [np.random.default_rng(len(V)).standard_normal(n) for _ in range(k - len(V))])
        thetas, res = np.asarray(E0s, dtype=float), np.full(k, np.inf)
        for it in range(davidson_max_iter):
            Vm, Wm = np.array(V).T, np.array(W).T
            thetas, Y = ritz(Vm, Wm)
            U, R = Vm @ Y, Wm @ Y - (Vm @ Y) * thetas
            res = np.linalg.norm(R, axis=0)
            if np.all(res < davidson_tol):
                break
            if len(V) + int(np.sum(res >= davidson_tol)) > max_size:
                # restart with the current Ritz vectors (orthonormalized, W transformed alike)
                Q, Rq = np.linalg.qr(U)
                WQ = (Wm @ Y) @ np.linalg.inv(Rq)
                V, W = list(Q.T), list(WQ.T)
            ts = []
            for j in np.flatnonzero(res >= davidson_tol):
                if davidson_preconditioner:
                    # diagonal (Davidson) preconditioner, small denominators bounded
                    denom = diag - thetas[j]
                    ts.append(-R[:, j] / np.where(np.abs(denom) < 1e-4, np.copysign(1e-4, denom), denom))
                else:
                    ts.append(R[:, j])
            n_before = len(V)
            orthonormal_add(V, W, ts)
            if len(V) == n_before:
                break
        print(f"Davidson{' (left)' if transpose else ''}: E = {np.array2string(thetas, precision=10)}, residuals "
              f"{np.array2string(res, precision=2)} after {it + 1} iterations (dimension {n})")
        xs = []
        for j in range(k):
            x = np.zeros(mask.shape)
            x[mask] = U[:, j] / np.linalg.norm(U[:, j])
            xs.append(x)
        return thetas, xs

    def davidson_compress(frag, emb, xs, x_lefts=None):
        """the fragment's new states: dominant eigenvectors of its (perturbed) reduced density matrix per sector,
        averaged over the target roots with target_weights (of the right eigenvectors xs, and the left ones x_lefts,
        if given, see dm_vectors)"""
        vecs = []
        for w, x, xl in zip(target_weights, xs, x_lefts if x_lefts is not None else [None] * len(xs)):
            if xl is None:
                vecs.append((x, False, w))
            else:
                vecs += [(x, False, w / 2), (xl / np.linalg.norm(xl), True, w / 2)]
        rhos = {sec: sum(w * v[rows] @ v[rows].T for v, _, w in vecs) for sec, rows in emb.rows.items()}
        if dm_perturbation > 0:
            # the dimer-term contributions one at a time (all of them at once: one dim_a x dim_b array per term)
            for v, tr, w in vecs:
                for y in emb.dimer_terms(v, transpose=tr):
                    for sec, rows in emb.rows.items():
                        rhos[sec] += dm_perturbation * w * (y[rows] @ y[rows].T)
        for (chg, mult), rows in emb.rows.items():
            eigvals, eigvecs = sp.linalg.eigh(rhos[(chg, mult)])
            order = np.argsort(eigvals)[::-1]
            eigvals, eigvecs = eigvals[order], eigvecs[:, order]
            # states with a weight in the (perturbed) density matrix above dens_filter_thresh, at most max_states_per_sector
            n_keep = max(1, min(max_states_per_sector, int(np.sum(eigvals > dens_filter_thresh))))
            states[frag][chg][mult] = [eigvecs[:, i].copy() for i in range(n_keep)]
            if state_weights is not None:
                state_weights[frag].setdefault(chg, {})[mult] = eigvals[:n_keep]
            print(f"frag {frag}, chg {chg}, mult {mult}: {n_keep} states kept, dens eigvals "
                  f"{np.array2string(eigvals[:n_keep], precision=2)}, largest discarded "
                  f"{eigvals[n_keep] if n_keep < len(eigvals) else 0.:.2e}")

    energies, reduced_energies = [], []
    if use_davidson:
        if xr_order != 0:
            raise NotImplementedError("the Davidson sweeps are implemented for XR[0]")
        if dm_vectors not in ("right", "left_right"):
            raise ValueError(f"unknown dm_vectors {dm_vectors!r}")
        all_roots = list(range(n_roots))
        for it in range(max_iter):
            for frag in range(2):
                print(f"\n###### Davidson sweep {it}, fragment {frag} ######")
                dens = emb = None    # (the previous densities and H^eff are released before building the new ones)
                dens = [build_dens(f, states[f]) for f in range(2)]
                E, root_vecs, n_states = solve(dens, roots=all_roots)
                emb = embedded(frag, dens, last.H1)
                x0s = [emb.states_to_csf(drop_empty_sectors(states[frag]), d if frag == 0 else d.T) for d in root_vecs]
                thetas, xs = davidson(emb, x0s, np.atleast_1d(E))
                if dm_vectors == "left_right":
                    x_lefts = davidson(emb, xs, thetas, transpose=True)[1]
                    x_lefts = [x_lefts[k] for k in target_states]
                else:
                    x_lefts = None
                energies.append(thetas[target_states] if len(target_states) > 1 else thetas[target_states[0]])
                davidson_compress(frag, emb, [xs[k] for k in target_states], x_lefts)
                print(f"History of embedded XR[0] energies (without nuc_rep):", energies)
            if len(energies) > 2 and np.all(np.abs(energies[-1] - energies[-3]) < conv_thresh):
                print("Converged!!!")
                break
        else:
            print(f"WARNING: max_iter = {max_iter} reached before convergence")
        for frag in range(2):
            states[frag] = drop_empty_sectors(states[frag])
        return energies, reduced_energies, states

    if use_gradients:
        grad_norms = []
        for it in range(max_iter):
            for frag in range(2):
                print(f"\n###### state solver iteration {it}, gradients on fragment {frag} ######")
                enlarge_state_space(1 - frag)
                dens = [build_dens(f, states[f]) for f in range(2)]
                E, d, d_left, H1 = solve_left_right(dens)
                print(f"energy before adding the gradients: {E}")
                gradients = state_gradients(frag, states, dens, E, d, H1, ints, build_dens, monomer_charges, target_multiplicity,
                                            xr_order=xr_order, grad_level=grad_level, d_left=d_left)
                n_added, grad_norm = add_gradient_states(frag, gradients)
                grad_norms.append(grad_norm)
                dens[frag] = build_dens(frag, states[frag])
                E, target_vecs, n_states = solve(dens)
                energies.append(E)
                print(f"state space before compression: {n_states}")
                compress_state_space(target_vecs, n_states)
                print(f"state space after compression: {[count_states(drop_empty_sectors(states[f])) for f in range(2)]}")
                if eval_reduced:
                    dens = [build_dens(f, states[f]) for f in range(2)]
                    reduced_energies.append(solve(dens)[0])
                    print("History of XR energies in compressed state spaces:", reduced_energies)
                print(f"History of XR[{xr_order}] energies in enlarged state spaces (without nuc_rep):", energies)
                print("History of gradient norms:", np.array2string(np.array(grad_norms), precision=2))
            # converged, if a full cycle (both fragments) changes the energy by less than conv_thresh with small gradients
            if (len(energies) > 2 and np.all(np.abs(energies[-1] - energies[-3]) < conv_thresh)
                    and max(grad_norms[-2:]) < grad_conv_thresh and pool_exhausted()):
                print("Converged!!!")
                break
        else:
            print(f"WARNING: max_iter = {max_iter} reached before convergence")
        for frag in range(2):
            states[frag] = drop_empty_sectors(states[frag])
        return energies, reduced_energies, states

    for it in range(max_iter):
        print(f"\n###### state solver iteration {it} ######")
        n_added = sum(enlarge_state_space(frag) for frag in range(2))
        if n_added == 0 and it > 0:
            print("pool of additional states is exhausted" if pool_exhausted() else
                  "no room for additional states (no sector can grow after compression)")
            break
        dens = [build_dens(frag, states[frag]) for frag in range(2)]
        E, target_vecs, n_states = solve(dens)
        energies.append(E)
        if pool_order in ("cipsi", "residual"):
            cipsi_reorder(dens, E, target_vecs[0])
        print(f"state space before compression: {n_states}")
        compress_state_space(target_vecs, n_states)
        print(f"state space after compression: {[count_states(drop_empty_sectors(states[frag])) for frag in range(2)]}")
        if eval_reduced:
            dens = [build_dens(frag, states[frag]) for frag in range(2)]
            reduced_energies.append(solve(dens)[0])
            print("History of XR energies in compressed state spaces:", reduced_energies)
        print(f"History of XR[{xr_order}] energies in enlarged state spaces (without nuc_rep):", energies)
        if len(energies) > 1 and np.all(np.abs(energies[-1] - energies[-2]) < conv_thresh):
            if exhaust_pool and not pool_exhausted():
                print("energy change below threshold, but the pool is not exhausted yet")
            else:
                print("Converged!!!")
                break
    else:
        if not pool_exhausted():
            print(f"WARNING: max_iter = {max_iter} reached before the pool was passed through")

    for frag in range(2):
        states[frag] = drop_empty_sectors(states[frag])

    return energies, reduced_energies, states
