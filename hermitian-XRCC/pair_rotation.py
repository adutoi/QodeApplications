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

"""
Corresponding-orbital rotation of the fragment orbitals and states of a dimer (exact-bridge ansatz, step 1).

The inter-fragment block of the core-projected valence overlap (orthonormal within each fragment) is S_AB = U diag(s) V^T;
fragment A is rotated by R_A = U, fragment B by R_B = V (columns: the new active orbitals in the old ones,
C_new = C_old R), so that S_AB becomes diagonal.  The pairs are ordered by increasing s, i.e. the strongest pairs are the
LAST active orbitals: hummr's CSFs are genealogical (Yamanouchi-Kotani) in increasing orbital order, so only then does
the weak remainder of every CSF have a good spin (the spin-adapted splitting of the states by the occupation of the
strong orbitals).  The core projection and the Loewdin orthonormalization within a fragment commute with a rotation of
the active orbitals, so the rotated MOs give exactly the rotated XR valence basis.

The states are transformed exactly within each (charge, multiplicity) CSF sector (complete spaces): with R = e^kappa
(proper rotation), the states in the new orbitals are e^{-K} |Psi>, K = sum_pq kappa_pq E_pq, and E_pq is the rank-0 "ca"
channel of hummr's densities between arbitrary vectors (factor 1).  An improper R is written as R_p D, D the sign flip of
the first (weakest) orbital, which multiplies every CSF by (-1)^{occupation of that orbital}.  Selected (HCI) spaces are
not closed under the rotation and are refused.
"""

import numpy as np
import scipy.linalg as sla
from qode.math.tensornet import raw

E_PQ_CHUNK_BYTES = 256 * 2**20      # bra chunk of the CSF matrix of E_pq (unit CSF vectors as bras)


def corresponding_rotations(S_AB):
    """(R_A, R_B, s): S_AB = R_A diag(s) R_B^T with s increasing (the strongest pair last)"""
    U, s, Vt = np.linalg.svd(S_AB)
    order = np.argsort(s, kind="stable")
    return U[:, order], Vt.T[:, order], s[order]


def valence_overlap(ints, frags, n_core, n_act):
    """the inter-fragment block of the core-projected valence overlap (active orbitals of frags[0] x frags[1])"""
    S = np.asarray(raw(ints[0].S[frags[0], frags[1]]))
    return S[n_core:n_core + n_act, n_core:n_core + n_act]


def rotated_mo_coeffs(mo_coeffs, n_core, R):
    """MO coefficients with the active block rotated, C_act R"""
    C = np.array(mo_coeffs, dtype=float, copy=True)
    n_act = R.shape[0]
    C[:, n_core:n_core + n_act] = C[:, n_core:n_core + n_act] @ R
    return C


def _proper(R):
    """(R_p, flip): R = R_p D with det R_p = +1, D the sign flip of orbital 0 if flip"""
    if np.linalg.det(R) > 0:
        return R, False
    R_p = R.copy()
    R_p[:, 0] *= -1
    return R_p, True


def _csf_occupations(layout, chg, mult, orbital):
    """occupation of one active orbital in the CFG of every CSF of the sector (CSF order of the states)"""
    occ = np.zeros(sum(n for _, _, n in layout[chg][mult]), dtype=int)
    for cfg, start, n in layout[chg][mult]:
        occ[start:start + n] = int(cfg[orbital])
    return occ


def csf_E_matrix(fragment, sector, kappa):
    """K = sum_pq kappa_pq E_pq in the CSF basis of sector (chg, mult): (CSF, CSF), from hummr's "ca" densities"""
    dim = sum(n for _, _, n in fragment.csf_layout()[sector[0]][sector[1]])
    unit = np.eye(dim)
    n = kappa.shape[0]
    chunk = max(1, int(E_PQ_CHUNK_BYTES // (8 * dim * n * n)))
    K = np.zeros((dim, dim))
    for b0 in range(0, dim, chunk):
        dens = fragment.density_sector(1, 1, 0, sector, sector, list(unit[b0:b0 + chunk]), list(unit))
        rank0 = [np.asarray(rho) for channel, rho in dens.items() if channel[2] == 0]
        if len(rank0) != 1:
            raise RuntimeError(f"expected one rank-0 ca channel for sector {sector}, found {len(rank0)}")
        K[b0:b0 + chunk] = np.einsum("bkpq,pq->bk", rank0[0], kappa, optimize=True)
    return K


class StateRotation(object):
    """the transformation of the CSF vectors of one fragment to the rotated orbitals C_old R"""
    def __init__(self, fragment, R):
        self.fragment, self.R = fragment, R
        R_p, self.flip = _proper(R)
        kappa = sla.logm(R_p)
        if np.abs(kappa.imag).max() > 1e-10:
            raise RuntimeError("no real logarithm of the orbital rotation")
        self.kappa = np.real(kappa)
        self.kappa = 0.5 * (self.kappa - self.kappa.T)
        dev = np.abs(sla.expm(self.kappa) - R_p).max()
        if dev > 1e-10:
            raise RuntimeError(f"orbital rotation not reproduced by exp(kappa) ({dev:.1e})")
        self.layout = fragment.csf_layout()
        self._T = {}
    def matrix(self, chg, mult):
        """the transformation of the CSF coefficients of the sector (exp(-K), then the sign flip)"""
        if (chg, mult) not in self._T:
            T = sla.expm(-csf_E_matrix(self.fragment, (chg, mult), self.kappa))
            if self.flip:
                T = ((-1.0) ** _csf_occupations(self.layout, chg, mult, 0))[:, None] * T
            self._T[chg, mult] = T
        return self._T[chg, mult]
    def __call__(self, states):
        """{chg: {mult: [vectors]}} in the rotated orbitals"""
        return {chg: {mult: [self.matrix(chg, mult) @ np.asarray(v) for v in vecs] if len(vecs) else []
                      for mult, vecs in by_mult.items()} for chg, by_mult in states.items()}
