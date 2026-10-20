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

# State gradients for the spin-adapted (CSF based) state solver (spin-adapted counterpart of state_gradients.py).
#
# The target state is expanded in the spin-coupled product basis of the fragment states,
#
#     |Psi> = sum_KL d_KL |K L; S M>,
#
# and the gradient of the energy with respect to the CSF coefficients c_K(p) of state K of the optimized
# fragment (here fragment A) is
#
#     g_K(p) = sum_L d_KL <p L| H - E |Psi>,
#
# where p runs over the CSFs of the sector of K.  The matrix elements <p L|H|K' L'> are obtained from the
# rectangular XR Hamiltonian with single CSFs as bra states on fragment A (densities <CSF|...|state>), i.e.
#
#     <p L|H|K' L'> = H1_A[p,K'] delta_LL' + O[p,K'] H1_B[L,L'] + H2[pL,K'L'],     O[p,K'] = <p|K'>,
#
# and <p L|Psi> = sum_K' O[p,K'] d_K'L.  For an eigenvector d of the Hamiltonian in the current state space,
# g_K is orthogonal to all current states of the sector.
#
# grad_level "herm": only the right eigenvector r is used (d = r, as in the "herm" variant of state_gradients.py).
#                    This is the exact gradient (up to a factor 2) only for a hermitian Hamiltonian.
# grad_level "full": exact derivative of the (non-hermitian) eigenvalue with left (l) and right (r) eigenvectors,
#
#     g_K(p) = [ sum_L l_KL <p L|H - E|Psi_r>  +  sum_L <Psi_l|H - E|p L> r_KL ] / <l|r>,
#
#                    where the second term requires the rectangular Hamiltonian with single CSFs as ket states.

import numpy as np
from get_xr_result_spin_adapt import get_xr_H, local_sectors, drop_empty_sectors



def unit_vectors(states):
    """single CSFs (unit vectors) of every sector of states (structure states[chg][mult] = list of CSF vectors)"""
    return {chg: {mult: list(np.eye(len(vecs[0]))) for mult, vecs in mult_dict.items() if len(vecs) > 0}
            for chg, mult_dict in states.items()}

def _block_overlap(rho, monomer_charges):
    """<bra|ket> of a fragment in the local ordering (block diagonal over the sectors)"""
    bra = local_sectors(rho['n_states_bra'], monomer_charges)
    ket = local_sectors(rho['n_states'], monomer_charges)
    O = np.zeros((bra[-1][2].stop, ket[-1][2].stop))
    for (chg, mult, sl_bra), (chg_k, mult_k, sl_ket) in zip(bra, ket):
        assert (chg, mult) == (chg_k, mult_k)
        O[sl_bra, sl_ket] = rho['overlap'][chg][mult]
    return O

def _rectangular_H(frag, H1_frag, H2, O, H1):
    """
    product-basis Hamiltonian between different bra and ket state sets on fragment frag (e.g. single CSFs as bra),
    assembled from the rectangular H1 of frag, the overlap O = <bra|ket> of frag and H1 of the other fragment;
    returns it together with the product-basis overlap
    """
    other = 1 - frag
    I_other = np.eye(H1[other].shape[0])
    if frag == 0:
        return np.kron(H1_frag, I_other) + np.kron(O, H1[1]) + H2, np.kron(O, I_other)
    else:
        return np.kron(H1[0], O) + np.kron(I_other, H1_frag) + H2, np.kron(I_other, O)

def _contract_with_d(frag, X, d, n_csf):
    """G[p, K] = sum_L X[p,L] d[K,L] (frag 0) or sum_K' X[K',p] d[K',L] (frag 1), X given in the product ordering"""
    if frag == 0:
        return X.reshape(n_csf, d.shape[1]) @ d.T
    else:
        return X.reshape(d.shape[0], n_csf).T @ d

def state_gradients(frag, states, dens, E, d, H1, ints, build_dens, monomer_charges, target_multiplicity, xr_order=0,
                    grad_level="herm", d_left=None):
    """
    frag:       fragment whose states are optimized
    states:     states[frag][chg][mult] = list of (orthonormal) CSF vectors
    dens:       square densities of both fragments for states
    E, d:       target energy and right eigenvector of the XR Hamiltonian in the current state space,
                d[K,L] in the tensor-product ordering of get_xr_H (zeros outside of the basis mask)
    d_left:     left eigenvector (same structure), required for grad_level="full"
    H1:         monomer Hamiltonians of the current state space (from get_xr_H)
    build_dens: build_dens(frag, states, bra_states=None) -> fragment densities
    returns {chg: {mult: gradient vectors (n_states_sector x n_csf_sector)}}
    """
    if xr_order != 0:
        raise NotImplementedError("state gradients are only implemented for XR[0] (no overlap terms)")
    if grad_level not in ("herm", "full"):
        raise ValueError(f"unknown grad_level {grad_level!r}")
    if grad_level == "full" and d_left is None:
        raise ValueError("grad_level 'full' requires the left eigenvector")
    other = 1 - frag
    frag_states = drop_empty_sectors(states[frag])
    csfs = unit_vectors(frag_states)

    # <p L| H - E |Psi_r>
    rect_dens = [None, None]
    rect_dens[other] = dens[other]
    rect_dens[frag] = build_dens(frag, frag_states, bra_states=csfs)
    H1r, H2r = get_xr_H(ints, rect_dens, xr_order, monomer_charges, target_multiplicity)
    O = _block_overlap(rect_dens[frag], monomer_charges[frag])
    Hr, Or = _rectangular_H(frag, H1r[frag], H2r, O, H1)
    n_csf = O.shape[0]
    d_bra = d if grad_level == "herm" else d_left
    G = _contract_with_d(frag, (Hr - E * Or) @ d.reshape(-1), d_bra, n_csf)

    if grad_level == "full":
        # <Psi_l| H - E |p L>
        rect_dens[frag] = build_dens(frag, csfs, bra_states=frag_states)
        H1c, H2c = get_xr_H(ints, rect_dens, xr_order, monomer_charges, target_multiplicity)
        Hc, Oc = _rectangular_H(frag, H1c[frag], H2c, O.T, H1)
        G += _contract_with_d(frag, d_left.reshape(-1) @ (Hc - E * Oc), d, n_csf)
        G /= d_left.reshape(-1) @ d.reshape(-1)

    gradients = {}
    for (chg, mult, sl_csf), (_, _, sl_state) in zip(local_sectors(rect_dens[frag]['n_states' if grad_level == "full" else 'n_states_bra'], monomer_charges[frag]),
                                                    local_sectors(dens[frag]['n_states'], monomer_charges[frag])):
        gradients.setdefault(chg, {})[mult] = G[sl_csf, sl_state].T
    return gradients
