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
Spin-adapted splitting of fragment states by the occupation of the strong orbitals (exact-bridge ansatz, step 2).

In the corresponding-orbital basis with the strong orbitals last (pair_rotation.py), a state |Psi S S> with the last
strong orbital s splits as

    |Psi S S> = chi0 + a+_{s alpha} a+_{s beta} chi2 + sum_{S_w = S -+ 1/2} sum_m <S_w, S-m; 1/2, m|S S> a+_{s m} chi1[S_w]_{S-m}

with weak states (orbital s empty): chi0 (N, S), chi2 = a_{s beta} a_{s alpha} Psi (N - 2, S) and chi1[S_w] (N - 1, a
multiplet of spin S_w, given by its M = S_w component), a+_{s, +1/2} = a+_{s alpha}, a+_{s, -1/2} = a+_{s beta}.  Since
hummr's CSFs are genealogical in increasing orbital order, the weak parts are vectors of hummr's own CSF sectors (zero
on the CSFs with orbital s occupied), so their densities come from the same hummr instance.  They are obtained from
hummr's densities between the strong-empty unit CSFs of the target sector and the states, without assuming a CSF
convention:
  - single annihilator (stretched elements, at_{+1/2} = a_beta, at_{-1/2} = -a_alpha):
        <b S_w S_w| at_q(s) |Psi S S> = -<S_w S_w; 1/2 1/2|S S> chi1[S - 1/2][b]           (q = -1/2)
                                       =  <S_w S_w; 1/2 -1/2|S S> chi1[S + 1/2][b]          (q = +1/2)
  - pair annihilator, rank-0 channel at (s, s): -sqrt(2) <b| a_{s beta} a_{s alpha} |Psi>   (verified, Be 6-31G).
More strong orbitals (the last K active orbitals) are split recursively, the last one first: in a weak part all later
orbitals are empty, so the next strong orbital is again the last possible open shell of the genealogical coupling.
A component is labelled by the path of (orbital, occupation, multiplicity of the weak part after removing it).
"""

from functools import lru_cache
import numpy as np
from sympy import Rational
from sympy.physics.quantum.cg import CG


@lru_cache(maxsize=None)
def clebsch_gordan(twoj1, twom1, twoj2, twom2, twoJ, twoM):
    h = lambda x: Rational(x, 2)
    return float(CG(h(twoj1), h(twom1), h(twoj2), h(twom2), h(twoJ), h(twoM)).doit())


def required_sectors(fragment, sectors, n_strong, glue=False):
    """{chg: [mult]} of all sectors reached by removing up to 2 n_strong electrons from the sectors {chg: [mult]} (those
    with CSFs in the active space), i.e. the ones hummr has to be initialized with for the splitting; glue: also the
    sectors of the glued states (a strong configuration of up to 2 n_strong electrons on such a weak part, i.e. charge
    shifts down to -2 n_strong) and their weak parts"""
    out = {}
    shifts = range(-2 * n_strong, 4 * n_strong + 1) if glue else range(2 * n_strong + 1)
    for chg, mults in sectors.items():
        for dq in shifts:
            for m in fragment.multiplicities(chg + dq):
                if any(abs(m - m0) <= abs(dq) + (2 * n_strong if glue else 0) for m0 in mults):
                    out.setdefault(chg + dq, set()).add(m)
    return {chg: sorted(ms) for chg, ms in out.items()}


class StrongSplit(object):
    """splitting of the states of one fragment (hummr fragment, all sectors involved initialized) by the occupation of
    the strong active orbitals (indices, the last ones)"""
    def __init__(self, fragment, strong):
        self.fragment = fragment
        self.strong = sorted(strong, reverse=True)                  # the last one first
        self.layout = fragment.csf_layout()
    def dim(self, chg, mult):
        return sum(n for _, _, n in self.layout[chg][mult])
    def exists(self, chg, mult):
        """the sector has CSFs; a sector that has CSFs in the active space but was not initialized in hummr is refused
        (its parts would be lost silently; see required_sectors)"""
        if chg in self.layout and mult in self.layout[chg] and self.dim(chg, mult) > 0:
            return True
        if mult in self.fragment.multiplicities(chg):
            raise RuntimeError(f"sector (charge {chg}, multiplicity {mult}) is needed for the splitting but not "
                               f"initialized in hummr (request it in get_states, see required_sectors)")
        return False
    def empty_mask(self, chg, mult, orbitals):
        """CSFs of the sector with all the given orbitals empty"""
        mask = np.zeros(self.dim(chg, mult), dtype=bool)
        for cfg, start, n in self.layout[chg][mult]:
            mask[start:start + n] = all(cfg[o] == "0" for o in orbitals)
        return mask
    def split_one(self, chg, mult, vecs, s, done=()):
        """one strong orbital s (the orbitals done already empty): {(occ, mult_w): (sector, vectors (n_states, dim))};
        occ 0: chi0, 2: chi2, 1: chi1[S_w] with mult_w = 2 S_w + 1"""
        vecs = np.asarray(vecs, dtype=float).reshape(-1, self.dim(chg, mult))
        empty = tuple(done) + (s,)
        out = {}
        out[0, mult] = ((chg, mult), vecs * self.empty_mask(chg, mult, empty)[None, :])
        twoS = mult - 1
        for mult_w in (mult - 1, mult + 1):
            if mult_w < 1 or not self.exists(chg + 1, mult_w):
                continue
            mask = self.empty_mask(chg + 1, mult_w, empty)
            idx = np.flatnonzero(mask)
            if not len(idx):
                continue
            bras = list(np.eye(self.dim(chg + 1, mult_w))[idx])
            dens = self.fragment.density_sector(0, 1, 0, (chg + 1, mult_w), (chg, mult), bras, list(vecs))
            if len(dens) != 1:
                raise RuntimeError(f"expected one channel of a for {(chg + 1, mult_w)} <- {(chg, mult)}, found {len(dens)}")
            rho = np.asarray(next(iter(dens.values())))[:, :, s]                     # (bra, state)
            twoSw = mult_w - 1
            if mult_w == mult - 1:
                coef = -clebsch_gordan(twoSw, twoSw, 1, 1, twoS, twoS)
            else:
                coef = clebsch_gordan(twoSw, twoSw, 1, -1, twoS, twoS)
            chi = np.zeros((len(vecs), self.dim(chg + 1, mult_w)))
            chi[:, idx] = rho.T / coef
            out[1, mult_w] = ((chg + 1, mult_w), chi)
        if self.exists(chg + 2, mult):
            idx = np.flatnonzero(self.empty_mask(chg + 2, mult, empty))
            if len(idx):
                bras = list(np.eye(self.dim(chg + 2, mult))[idx])
                dens = self.fragment.density_sector(0, 2, 0, (chg + 2, mult), (chg, mult), bras, list(vecs))
                rank0 = [np.asarray(r) for ch, r in dens.items() if ch[2] == 0]
                if len(rank0) != 1:
                    raise RuntimeError(f"expected one rank-0 channel of aa for {(chg + 2, mult)} <- {(chg, mult)}")
                chi = np.zeros((len(vecs), self.dim(chg + 2, mult)))
                chi[:, idx] = -rank0[0][:, :, s, s].T / np.sqrt(2.0)
                out[2, mult] = ((chg + 2, mult), chi)
        return out
    def __call__(self, chg, mult, vecs):
        """all strong orbitals: {path: (sector, vectors (n_states, dim))}, path = ((orbital, occ, mult_w), ...) in the
        order of splitting (the last strong orbital first)"""
        parts = {(): ((chg, mult), np.asarray(vecs, dtype=float).reshape(-1, self.dim(chg, mult)))}
        done = []
        for s in self.strong:
            new = {}
            for path, ((c, m), v) in parts.items():
                for (occ, mult_w), (sector, chi) in self.split_one(c, m, v, s, done).items():
                    new[path + ((s, occ, mult_w),)] = (sector, chi)
            parts = new
            done.append(s)
        return parts


WEAK_BASIS_RTOL = 1e-12             # singular values (relative to the largest of the sector) kept in the weak basis


class WeakBasis(object):
    """
    orthonormal basis of the weak parts of one fragment, per weak sector: the weak parts (zero on the strong-occupied
    CSFs, neither normalized nor orthogonal, possibly linearly dependent) of all split states that fall into a sector
    are spanned by orthonormal vectors W (SVD of the stacked parts); every part is W^T X.  The densities of the weak
    parts are therefore X_bra^T rho_W X_ket with the densities rho_W between the basis vectors, which the standard
    density machinery builds (orthonormal states, identity overlaps as assumed by the frozen-core embedding and the
    spectator overlaps).
        split:   {(chg, mult): {path: ((chg_w, mult_w), parts (n_states, dim))}} (StrongSplit of all sectors)
        basis:   {chg_w: {mult_w: W (n_basis, dim)}}
        coef:    {((chg, mult), path): ((chg_w, mult_w), X (n_basis, n_states))}
    """
    def __init__(self, split, rtol=WEAK_BASIS_RTOL):
        stacks = {}
        for parent, parts in split.items():
            for path, (sector, chi) in parts.items():
                stacks.setdefault(sector, []).append(chi)
        self.basis = {}
        for (chg, mult), chis in stacks.items():
            A = np.vstack(chis)
            _, sv, Vt = np.linalg.svd(A, full_matrices=False)
            keep = sv > rtol * sv[0] if len(sv) and sv[0] > 0 else np.zeros(len(sv), dtype=bool)
            if keep.any():
                self.basis.setdefault(chg, {})[mult] = Vt[keep]
        self.coef = {}
        for parent, parts in split.items():
            for path, (sector, chi) in parts.items():
                if sector[1] in self.basis.get(sector[0], {}):
                    W = self.basis[sector[0]][sector[1]]
                    self.coef[parent, path] = (sector, W @ chi.T)
                    dev = np.abs(W.T @ self.coef[parent, path][1] - chi.T).max()
                    if dev > 1e-10 * max(1.0, np.abs(chi).max()):
                        raise RuntimeError(f"weak part {parent} {path} not spanned by the weak basis ({dev:.1e})")
    def states(self):
        """{chg: {mult: [basis vectors]}} (the input of build_frag_densities)"""
        return {chg: {mult: list(W) for mult, W in by_mult.items()} for chg, by_mult in self.basis.items()}
    def sizes(self):
        return {chg: {mult: len(W) for mult, W in by_mult.items()} for chg, by_mult in self.basis.items()}


def part_density(rho_W, X_bra, X_ket):
    """density between weak parts from the one between basis vectors (bra, ket, orbitals...): X_bra^T rho_W X_ket"""
    rho_W = np.asarray(rho_W)
    return np.tensordot(X_bra.T, np.tensordot(rho_W, X_ket, axes=([1], [0])), axes=([1], [0])).transpose(
        [0, rho_W.ndim - 1] + list(range(1, rho_W.ndim - 1)))
