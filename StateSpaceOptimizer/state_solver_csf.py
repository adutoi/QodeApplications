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

from get_xr_result_spin_adapt import get_xr_H, get_xr_states_from_H, local_sectors, drop_empty_sectors, product_basis_mask
from state_gradients_csf import state_gradients
from state_screening import orthogonalize
from qode.util import struct, timer
import excitonic

import numpy as np
import scipy as sp



def count_states(states):
    return {chg: {mult: len(vecs) for mult, vecs in mult_dict.items()} for chg, mult_dict in states.items()}

def unit_vector_pool(states, n_passes=1):
    """
    Pool of candidate states consisting of all CSFs of every sector, i.e. unit vectors (zero vectors with a
    single 1) in the CSF basis of the sector.  The dimension is taken from the given states (at least one
    state per sector is required).  Candidates, which are (nearly) contained in the current state space,
    are skipped by the solver, so the pool can be passed through completely.
    n_passes: the CSFs are offered this many times (CSFs discarded in an early compression can become
              relevant again later)
    """
    pool = {}
    for chg, mult_dict in states.items():
        for mult, vecs in mult_dict.items():
            if len(vecs) == 0:
                raise ValueError(f"unit_vector_pool needs at least one state in sector (chg {chg}, mult {mult}) for its dimension")
            pool.setdefault(chg, {})[mult] = list(np.eye(len(vecs[0]))) * n_passes
    return pool

def optimize_states(max_iter, xr_order, states, additional_states, ints, build_dens, monomer_charges, target_multiplicity,
                    target_charge=0, max_states_per_sector=15, conv_thresh=1e-6, dens_filter_thresh=1e-7, target_state=0, eval_reduced=False,
                    exhaust_pool=False, lin_dep_thresh=1e-6, use_gradients=False, grad_level="herm", grad_conv_thresh=1e-5):
    """
    max_iter:              maximum number of enlarge-diagonalize-compress cycles
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
    max_states_per_sector: the enlarged state space contains at most this many states per (chg, mult) sector
    eval_reduced:          additionally evaluate the energy in the compressed state space (debugging only,
                           requires an additional density and Hamiltonian build per iteration)
    returns the energy histories of the enlarged and of the compressed state spaces (the latter only if eval_reduced)
    and the optimized states
    """
    target_states = [target_state] if isinstance(target_state, int) else list(target_state)

    pool_tracker = [{chg: {mult: 0 for mult in mult_dict} for chg, mult_dict in additional_states[frag].items()} for frag in range(2)]

    def pool_exhausted():
        return all(pool_tracker[frag][chg][mult] >= len(pool) for frag in range(2)
                   for chg, mult_dict in additional_states[frag].items() for mult, pool in mult_dict.items())

    def solve_left_right(dens):
        """target energy, right and left eigenvectors (reshaped to the product dimensions) and H1"""
        if len(target_states) > 1:
            raise NotImplementedError("the gradient based variant is only implemented for a single target state")
        H1, H2 = get_xr_H(ints, dens, xr_order, monomer_charges, target_multiplicity)
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
        add linearly independent states from the pool until max_states_per_sector is reached (candidates,
        which are already contained in the current state space, are skipped); returns the number of added states
        """
        n_added = 0
        for chg, mult_dict in additional_states[frag].items():
            for mult, pool in mult_dict.items():
                current = [np.array(vec, dtype=float) for vec in states[frag].setdefault(chg, {}).setdefault(mult, [])]
                n_before, n_skipped = len(current), 0
                Q = np.array(current) if len(current) > 0 else None    # orthonormal rows
                while len(current) < max_states_per_sector and pool_tracker[frag][chg][mult] < len(pool):
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

    def solve(dens):
        n_states = [count_states(drop_empty_sectors(states[frag])) for frag in range(2)]
        H1, H2 = get_xr_H(ints, dens, xr_order, monomer_charges, target_multiplicity)
        # an int target_state allows excitonic.fci to use the sparse eigensolver
        mask = product_basis_mask(dens, monomer_charges, target_charge, target_multiplicity)
        E, vecs = get_xr_states_from_H(H1, H2, target_state=target_states if len(target_states) > 1 else target_states[0], basis_mask=mask)
        vecs = vecs.reshape(vecs.shape[0], -1)
        dims = [H.shape[0] for H in H1]
        target_vecs = []
        for i in range(len(target_states)):
            vec = vecs[:, i]
            vec = vec * np.conj(vec[np.argmax(np.abs(vec))]) / np.abs(vec[np.argmax(np.abs(vec))])  # fix (complex) phase
            if np.linalg.norm(np.imag(vec)) > 1e-10:
                raise ValueError("imaginary part of eigenvector is not negligible")
            vec = np.real(vec)
            target_vecs.append((vec / np.linalg.norm(vec)).reshape(dims))
        return np.real(E), target_vecs, n_states

    def compress_state_space(target_vecs, n_states):
        """
        Replace the states of every sector by the dominant eigenvectors of the (state-averaged) fragment
        reduced density matrix restricted to this sector. Contributions of the reduced density matrix
        between different sectors are neglected (they vanish for spin-pure target states, if the
        sectors differ in multiplicity, and always if they differ in charge).
        """
        dens_mats = [sum(vec @ vec.T for vec in target_vecs) / len(target_vecs),   # contract over frag 1
                     sum(vec.T @ vec for vec in target_vecs) / len(target_vecs)]   # contract over frag 0
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
                print(f"frag {frag}, chg {chg}, mult {mult}: {len(new_C)} of {C.shape[0]} states kept, "
                      f"dens eigvals {np.array2string(eigvals[keep], precision=2)}, "
                      f"largest discarded {np.max(eigvals[~keep], initial=0.):.2e}")

    energies, reduced_energies = [], []
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
                  "no room for additional states (all sectors at max_states_per_sector after compression)")
            break
        dens = [build_dens(frag, states[frag]) for frag in range(2)]
        E, target_vecs, n_states = solve(dens)
        energies.append(E)
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
