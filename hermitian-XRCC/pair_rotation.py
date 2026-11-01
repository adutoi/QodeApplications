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
not closed under the rotation; for them the rotation is done first (rotated_input: the corresponding orbitals written as
the orbital guess of a hummr input, so that the selection, hummr and the state optimization work in them).
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
        self._T, self._channel = {}, {}
    def matrix(self, chg, mult):
        """the transformation of the CSF coefficients of the sector (exp(-K), then the sign flip)"""
        if (chg, mult) not in self._T:
            T = sla.expm(-csf_E_matrix(self.fragment, (chg, mult), self.kappa))
            if self.flip:
                T = ((-1.0) ** _csf_occupations(self.layout, chg, mult, 0))[:, None] * T
            self._T[chg, mult] = T
        return self._T[chg, mult]
    def apply_K(self, chg, mult, V):
        """K V for the CSF vectors V (dim, m) of the sector: hummr's open contraction of the rank-0 ca channel (exactly
        E_pq, factor 1) with kappa, a sigma build instead of the CSF matrix of K"""
        if self._channel.get((chg, mult)) is None:
            dim = V.shape[0]
            dens = self.fragment.density_sector(1, 1, 0, (chg, mult), (chg, mult), [np.eye(dim)[0]], [V[:, 0]])
            rank0 = [ch for ch in dens if ch[2] == 0]
            if len(rank0) != 1:
                raise RuntimeError(f"expected one rank-0 ca channel for sector {(chg, mult)}, found {len(rank0)}")
            self._channel[chg, mult] = list(rank0[0][4][0]) + list(rank0[0][4][1])
        sigma = self.fragment._engine.contract_density_open(1, 1, 0, chg, mult, chg, mult, list(V.T), True, 0,
                                                            self._channel[chg, mult], True,
                                                            np.ascontiguousarray(self.kappa))
        return np.asarray(sigma).reshape(V.shape)
    def apply(self, chg, mult, V):
        """exp(-K) V (then the sign flip) for the CSF vectors V (dim, m): Taylor series in s steps of -K/s, s from a
        power-iteration estimate of the spectral radius of K (antisymmetric), so that every step has a norm of about one
        and converges without cancellation; the cost is a few tens of sigma builds instead of the CSF matrix of K and its
        dense exponential (Be2 cc-pVTZ, anion doublet: 8120 CSFs)"""
        V = np.array(V, dtype=float)
        x = np.random.default_rng(0).normal(size=(V.shape[0], 1))
        rho = 0.0
        for _ in range(6):
            x /= np.linalg.norm(x)
            y = self.apply_K(chg, mult, x)
            rho = max(rho, np.linalg.norm(y))
            x = y
        steps = max(1, int(np.ceil(2.0 * rho)))
        for _ in range(steps):
            term, total = V, V.copy()
            for k in range(1, 200):
                term = -self.apply_K(chg, mult, term) / (steps * k)
                total += term
                if np.abs(term).max() < 1e-17 * max(1.0, np.abs(total).max()):
                    break
            else:
                raise RuntimeError(f"exp(-K) did not converge for sector {(chg, mult)}")
            V = total
        if self.flip:
            V = ((-1.0) ** _csf_occupations(self.layout, chg, mult, 0))[:, None] * V
        return V
    def __call__(self, states):
        """{chg: {mult: [vectors]}} in the rotated orbitals"""
        return {chg: {mult: list(self.apply(chg, mult, np.asarray(vecs).T).T) if len(vecs) else []
                      for mult, vecs in by_mult.items()} for chg, by_mult in states.items()}


# ---------------------------------------------------------------------------------------------------------------------
# rotation before hummr (selected CI spaces): the corresponding orbitals as hummr's orbital guess
#
def orbital_file(input_file):
    """the orbital guess (.C0 file) of a hummr input (OrbGuessName in the General block)"""
    from hummr_dens.space_selection import read_blocks, _get_keyword
    for name, lines in read_blocks(input_file):
        if name.lower() == "general":
            value = _get_keyword(lines, "OrbGuessName")
            if value:
                return value[0]
    raise ValueError(f"{input_file}: no OrbGuessName")


def read_orbitals(path):
    """MO coefficients of a .C0 file (int32 rows, cols, then the row-major doubles; with MaxIter 0 hummr keeps them)"""
    with open(path, "rb") as fh:
        shape = tuple(int(n) for n in np.fromfile(fh, dtype=np.int32, count=2))
        return np.fromfile(fh, dtype=float, count=shape[0] * shape[1]).reshape(shape)


def write_orbitals(path, C):
    with open(path, "wb") as fh:
        np.array(C.shape, dtype=np.int32).tofile(fh)
        np.ascontiguousarray(C, dtype=float).tofile(fh)


DAVIDSON_MAX_ITER_ROTATED = 500    # hummr's CI Davidson iterations in the rotated inputs (hummr default 50)


def rotated_input(input_file, C, out_dir, tag):
    """a copy of the hummr input in out_dir with the orbitals C as its orbital guess (<input name>_<tag>.inp/.C0)"""
    import os
    from hummr_dens.space_selection import read_blocks, write_blocks, _set_keyword
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(os.path.abspath(out_dir), f"{os.path.splitext(os.path.basename(input_file))[0]}_{tag}")
    write_orbitals(stem + ".C0", C)
    blocks = read_blocks(input_file)
    for name, lines in blocks:
        if name.lower() == "general":
            _set_keyword(lines, "OrbGuessName", stem + ".C0")
    # (hummr's CI Davidson converges more slowly in the corresponding orbitals, e.g. Be cc-pVTZ anion quartet in the
    # selected space: not within the default 50 iterations, and hummr then ends the process)
    if not any(name.lower() == "ci" for name, _ in blocks):
        blocks.append(("CI", []))
    for name, lines in blocks:
        if name.lower() == "ci":
            _set_keyword(lines, "DavidsonMaxIter", DAVIDSON_MAX_ITER_ROTATED)
    write_blocks(blocks, stem + ".inp")
    return stem + ".inp"


def mirror_deviation(S_AB, R_A, R_B, tol=1e-8):
    """
    max |R_B - D R_A diag(t)| over sign matrices: for a homodimer, S_AB^T = D S_AB D with the parities D of the orbitals
    under the mirror plane between the fragments, and the corresponding orbitals of B are the mirror images of those of
    A up to signs t (then one selected CI space, selected in A's orbitals with the environments on both sides, serves
    both fragments).  D is found from S_AB up to one sign per set of coupled orbitals, which commutes with the rotations.
    """
    n = S_AB.shape[0]
    d = np.zeros(n)
    for start in range(n):
        if d[start]:
            continue
        d[start], todo = 1.0, [start]
        while todo:
            i = todo.pop()
            for j in range(n):
                if i != j and abs(S_AB[i, j]) > tol and abs(S_AB[j, i]) > tol:
                    dj = d[i] * np.sign(S_AB[j, i] / S_AB[i, j])
                    if not d[j]:
                        d[j] = dj
                        todo.append(j)
                    elif d[j] != dj:
                        return np.inf                       # no mirror relation (not a homodimer)
    M = d[:, None] * R_A
    t = np.sign(np.einsum("ij,ij->j", M, R_B))
    return np.abs(R_B - M * t[None, :]).max()
