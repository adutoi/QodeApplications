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
Assembly of the exact-bridge ansatz from the spin-adapted ingredients (step 5), K = 1 strong pair: the overlap.

A fragment state is |Psi_I S_I> = sum_parts [w_I(part)^{S_w} x s(occ)^{s}]^{S_I} (strong_split.py: weak part w first,
the strong configuration s of spin s = 0 (occ 0, 2) or 1/2 (occ 1) second, the strong creators left of the weak part).
For a product state, coupled A first to the total spin S (here the singlet), the strong creators of B are moved to the
left of the weak part of A (sign (-1)^{N(w_A) occ_B}) and the spins are recoupled (9j)

    [[w_A s_A]^{S_A} [w_B s_B]^{S_B}]^S = sum_{S_w S_s} [S_A S_B S_w S_s]^{1/2} {S_wA s_A S_A; S_wB s_B S_B; S_w S_s S}
                                           [[w_A w_B]^{S_w} [s_A s_B]^{S_s}]^S,

into a weak dimer state (XR's coupled product of the weak bases, A first) and a bridge state (bridge_tables.py).  The
overlap operator of the corresponding-orbital basis factorizes, S = T_B (x) S_w, T_B the exact one of the strong pair
(even, spin scalar, on the bridge) and S_w the one of the weak pairs, whose order-k terms are XR's own overlap terms
(get_xr_result_spin_adapt.overlap_terms) between the weak bases with the strong pair removed from the inter-fragment S.
For S = 0, S_w = S_s, and

    S^(k)[P, P'] = sum c_P(parts) c_P'(parts') t_B[b, b'](S_s) S_w^(k)[(w_A w_B), (w'_A w'_B)](S_w = S_s).
"""

import itertools
import numpy as np
from sympy import Rational
from sympy.physics.wigner import wigner_9j
from functools import lru_cache
from qode.util import struct, timer
from qode.util.dynamic_array import dynamic_array, wrap, cached
from qode.math.tensornet import raw
from get_ints import tensorly_wrapper, pair_compressed, factorized_wrapper
from get_xr_result_spin_adapt import S_term_blocks
import XR_term
import diagram_lists as D
from bridge_tables import BridgeTables
from strong_split import WEAK_BASIS_RTOL


@lru_cache(maxsize=None)
def ninej2(*args2):
    """9j symbol of the doubled arguments (row-wise)"""
    return float(wigner_9j(*[Rational(a, 2) for a in args2], prec=None))


def weak_overlap_ints(ints, strong_mo):
    """ints with the strong pair (MO index strong_mo on both fragments) removed from the inter-fragment blocks of the
    symmetric S (the S of the weak overlap series; in the corresponding-orbital basis S_AB is diagonal).  For a
    pair-compressed S (get_ints with pair_thresh) the weak S is the product of the pair factors without the strong pair,
    returned as S_factors (precontract_pairs.precontract_S, diagrams/pair_slots.py)"""
    S = ints[0].S
    factors = getattr(ints[0], "S_factors", None)
    if factors is None:
        def block(i, j):
            B = np.array(raw(S[i, j]), dtype=float, copy=True)
            if i != j:
                B[strong_mo, :] = 0.0
                B[:, strong_mo] = 0.0
            return B
        S_raw = dynamic_array(block, S.ranges)
        return (struct(S=wrap(S_raw, [cached, tensorly_wrapper(timer())])),) + tuple(ints[1:])
    def weak_factors(i, j):
        L, R = factors[i, j]
        # the strong pair is the factor column on the strong orbital (unit vectors in the corresponding orbitals)
        on_strong = np.abs(L[strong_mo]) * np.abs(R[strong_mo]) > 0.5 * np.linalg.norm(L, axis=0) * np.linalg.norm(R, axis=0)
        if on_strong.sum() != 1:
            raise RuntimeError(f"weak_overlap_ints: no unique strong pair among the pair factors of S[{i}, {j}]")
        keep = ~on_strong
        if not keep.any():
            return np.zeros((L.shape[0], 1)), R[:, :1]
        return L[:, keep], R[:, keep]
    w_factors = dynamic_array([cached, weak_factors], factors.ranges)
    S_raw = pair_compressed(dynamic_array(lambda i, j: np.asarray(raw(S[i, j])), S.ranges), w_factors)
    return (struct(S=wrap(S_raw, [cached, factorized_wrapper(w_factors, timer())]), S_factors=w_factors),) + tuple(ints[1:])


def product_states(states, target_mult=1):
    """the singlet product states (sector A, i_A, sector B, i_B) in the order of the prototype (bridge_det.setup):
    sector pairs sorted, total charge 0, equal multiplicities"""
    keys = [sorted((chg, mult) for chg, by_mult in states[f].items() for mult, vecs in by_mult.items() if len(vecs)) for f in range(2)]
    out = []
    for kA in keys[0]:
        for kB in keys[1]:
            if kA[0] + kB[0] == 0 and kA[1] == kB[1]:
                for iA in range(len(states[0][kA[0]][kA[1]])):
                    for iB in range(len(states[1][kB[0]][kB[1]])):
                        out.append((kA, iA, kB, iB))
    return out


def _spin2_config(occ):
    return 1 if occ == 1 else 0


def active_integrals(ints, n_core, n_act):
    """the biorthogonal integrals of the active orbitals with the frozen core folded in, per fragment block:
    M[f1, f2] (n_act x n_act) = T + sum_n U[n] + sum_core (2 <p c|q c> - <p c|c q>) and the physical
    G[f1..f4][p, q, r, s] = <p q|r s> (creators p, q, annihilators r, s, electron 1 p -> r, electron 2 q -> s;
    H = sum M a+_p a~_q + 1/2 sum G a+_p a+_q a~_s a~_r).  XR's spatial V is half of it, <p q|r s> = 2 V[p, q, r, s]
    (verified against the brute-force integrals, Be2 6-31G)"""
    bior = ints[1]
    act, core = slice(n_core, n_core + n_act), slice(0, n_core)
    blk = lambda X, *idx: np.asarray(raw(X[idx]))
    M, V = {}, {}
    for f1, f2 in itertools.product(range(2), repeat=2):
        m = blk(bior.T, f1, f2) + sum(blk(bior.U, nuc, f1, f2) for nuc in range(2))
        for g in range(2):
            m = m + 4 * np.einsum("pcqc->pq", blk(bior.V, f1, g, f2, g)[:, core, :, core]) \
                  - 2 * np.einsum("pccq->pq", blk(bior.V, f1, g, g, f2)[:, core, core, :])
        M[f1, f2] = m[act, act]
    for fs in itertools.product(range(2), repeat=4):
        V[fs] = 2 * blk(bior.V, *fs)[act, act, act, act]
    return M, V


class BridgeAssembly(object):
    """
    order-resolved overlap S^(k) and Hamiltonian H^(k) (k = 0 ... nmax, weak orders) of the singlet product states, K = 1
        split:        per fragment {(chg, mult): {path: (sector, parts)}} (strong_split.StrongSplit)
        bases:        per fragment strong_split.WeakBasis
        weak_dens:    per fragment the densities of the weak bases (build_frag_densities, xr_level 2 for the
                      Hamiltonian, with add_extra_densities)
        ints:         the integrals in the corresponding-orbital basis
        strong_mo:    MO index of the strong orbital (the same on both fragments), sigma its overlap
        n_val:        active electrons of the neutral fragment (N of a sector of charge chg: n_val - chg)
    The weak dimer matrices are built only for the pairs of weak sectors the product states reach, one weak spin S_w at a
    time (combined with the bridge tables and released before the next one).
    """
    def __init__(self, split, bases, weak_dens, ints, strong_mo, sigma, nmax, n_val, pair_slots=2.5):
        self.split, self.bases, self.nmax, self.n_val = split, bases, nmax, n_val
        self.tables = BridgeTables(sigma)
        self.TB = self.tables.table(0, 0, ((), ()), ())                   # {(bridge, bridge'): t}
        self.ints, self.ints_w, self.weak_dens = ints, weak_overlap_ints(ints, strong_mo), weak_dens
        self.n_states = [rho['n_states'] if rho is not None else None for rho in weak_dens]
        self.pair_slots = pair_slots        # (for a pair-compressed S, see get_leakage_metric_H, leakage_pair_slots)
    def components(self, f, sector, i):
        """[(occ, S2_w, weak sector, coefficients over the weak basis of that sector)] of state i of a parent sector"""
        out = []
        for path in self.split[f][sector]:
            if (sector, path) in self.bases[f].coef:
                wsec, X = self.bases[f].coef[sector, path]
                (_, occ, mult_w), = path
                out.append((occ, mult_w - 1, wsec, X[:, i]))
        return out
    def _expansion(self, products):
        """[(P, (S2_s, bridge state), (weak sector A, weak sector B), c, x_A, x_B)]: the product states in the recoupled
        basis [[w_A w_B]^{S_w} [s_A s_B]^{S_s}]^0 (S_w = S_s)"""
        out = []
        for P, (kA, iA, kB, iB) in enumerate(products):
            S2A, S2B = kA[1] - 1, kB[1] - 1
            for occA, S2wA, wsecA, xA in self.components(0, kA, iA):
                for occB, S2wB, wsecB, xB in self.components(1, kB, iB):
                    sA, sB = _spin2_config(occA), _spin2_config(occB)
                    sign = (-1) ** ((self.n_val - wsecA[0]) * occB)
                    for S2s in range(abs(sA - sB), sA + sB + 1, 2):
                        if not abs(S2wA - S2wB) <= S2s <= S2wA + S2wB:          # singlet: S_w = S_s
                            continue
                        nj = ninej2(S2wA, sA, S2A, S2wB, sB, S2B, S2s, S2s, 0)
                        if nj != 0.0:
                            c = sign * np.sqrt((S2A + 1) * (S2B + 1)) * (S2s + 1) * nj
                            out.append((P, (S2s, (occA, occB, S2s)), (wsecA, wsecB), c, xA, xB))
        return out
    def strong_table(self, n_core, n_act):
        """scalar bridge table of T_B X_b, X_b the terms of H with all indices on the strong orbitals (frozen core folded
        in, active_integrals), in the coupled bridge basis"""
        M, V = active_integrals(self.ints, n_core, n_act)
        s, sp, frags = n_act - 1, self.tables.space, ("A", "B")
        X = np.zeros((sp.dim, sp.dim))
        for f1, f2 in itertools.product(range(2), repeat=2):
            X += M[f1, f2][s, s] * sum(sp.op(frags[f1], sg, "c") @ sp.op(frags[f2], sg, "a") for sg in (0, 1))
        for fs in itertools.product(range(2), repeat=4):
            p, q, r, t = (frags[f] for f in fs)
            v = V[fs][s, s, s, s]
            if v != 0.0:
                X += 0.5 * v * sum(sp.op(p, s1, "c") @ sp.op(q, s2, "c") @ sp.op(t, s2, "a") @ sp.op(r, s1, "a")
                                   for s1 in (0, 1) for s2 in (0, 1))
        O = self.tables.TB @ X
        table = {}
        for b in self.tables.basis:
            for k in self.tables.basis:
                if b[2] == k[2]:
                    val = self.tables.vectors[b, b[2]] @ O @ self.tables.vectors[k, k[2]]
                    if abs(val) > 1e-14:
                        table[b, k] = val
        return table
    def compute(self, products, hamiltonian=True, n_core=None, n_act=None, orders=None):
        """{"S": [S^(k)], "H_weak": [H^(k)] of the terms without strong-orbital indices (E_core S^(k) contained),
        "H_strong": [H^(k)] of the terms with only strong-orbital indices (with n_core, n_act)}; orders: the weak
        orders k evaluated (default all, the others are zero)"""
        orders = sorted(range(self.nmax + 1) if orders is None else orders)
        expansion = self._expansion(products)
        n, t = len(products), timer()
        out = {"S": [np.zeros((n, n)) for _ in range(self.nmax + 1)]}
        if hamiltonian:
            out["H_weak"] = [np.zeros((n, n)) for _ in range(self.nmax + 1)]
            int_sets = delta_integrals(self.ints, self.ints_w[0].S)
            if n_core is not None:
                out["H_strong"] = [np.zeros((n, n)) for _ in range(self.nmax + 1)]
                strong_table = self.strong_table(n_core, n_act)
        for S2w in sorted({key[0] for _, key, _, _, _, _ in expansion}):
            part = [e for e in expansion if e[1][0] == S2w]
            # the pairs of weak sectors reached (dimer-sector ordering of XR_term.dimer_matrix restricted to them)
            pairs = sorted({e[2] for e in part})
            offset, start = {}, 0
            for wA, wB in pairs:
                offset[wA, wB] = start
                start += self.n_states[0][wA[0]][wA[1]] * self.n_states[1][wB[0]][wB[1]]
            V = {}
            for P, key, (wA, wB), c, xA, xB in part:
                if key not in V:
                    V[key] = np.zeros((start, n))
                o = offset[wA, wB]
                V[key][o:o + len(xA) * len(xB), P] += c * np.outer(xA, xB).ravel()
            S_blocks = S_term_blocks(self.weak_dens, self.ints_w[0], t, t, S2w + 1)
            weak = {"S": [XR_term.dimer_matrix(S_blocks, {0: D.S0[0]} if k == 0 else {2: D.S2[k]}, (0, 1), pairs, t, symmetric=True)
                          if k in orders else None for k in range(self.nmax + 1)]}
            if hamiltonian:
                weak["H_weak"] = weak_hamiltonian_terms(self.weak_dens, pairs, self.ints_w[0], int_sets, S2w + 1, self.nmax,
                                                        orders, self.pair_slots)
            tables = {"S": self.TB, "H_weak": self.TB}
            if hamiltonian and n_core is not None:
                weak["H_strong"], tables["H_strong"] = weak["S"], strong_table
            for name, terms in weak.items():
                table = tables[name]
                for key2, V2 in V.items():
                    MV = {k: terms[k] @ V2 for k in orders}
                    for key1, V1 in V.items():
                        tb = table.get((key1[1], key2[1]), 0.0)
                        if tb != 0.0:
                            for k in orders:
                                out[name][k] += tb * (V1.T @ MV[k])
            weak = V = None
        return out


# ---------------------------------------------------------------------------------------------------------------------
# Hamiltonian, part 1: the terms without strong-orbital indices
#
# <Phi| S H |Phi'> with S = T_B (x) S_w.  For a term without strong legs (and the frozen core), H acts on the weak dimer
# only (even), so it contributes t_B[b, b'] <W| S_w H_w |W'>, the latter XR's S H diagrams between the weak bases.
# XR's diagram with k overlap factors omits the contractions of an overlap annihilator with a creator of H on the same
# fragment (the delta terms, see results_xr_order_states.tex, "The code's order counting equals the oracle's"); the
# exact order-m term of the biorthogonal counting is therefore
#     H_w^(m) = D_m[bior] + D_{m-1}[delta1] + D_{m-2}[delta2],
# D_k[X] the diagrams with k (weak) overlap factors and the integrals X, delta1 the biorthogonal integrals with Delta_w
# (the inter-fragment weak overlap) applied to one creation index (T, U: the creation index, V: positions 0 and 1,
# summed), delta2 to both creation indices of V.
# ---------------------------------------------------------------------------------------------------------------------

import diagrammatic_expansion
import XR_term
import diagram_lists as D
from diagrams import ST_diagrams, SU_diagrams, SV_diagrams
from precontract_pairs import precontract_S
from get_ints import tens_wrap


def _plain(rule, ranges):
    return wrap(dynamic_array(rule, ranges), [cached, lambda r: (lambda *idx: tens_wrap(r(*idx)))])


def delta_integrals(ints, S_w):
    """{"bior": (T, U, V), "delta1": (T, U, V), "delta2": (None, None, V)} of the biorthogonal integrals of ints and
    the weak overlap S_w (dynamic array of the S blocks)"""
    bior = ints[1]
    Delta = {(i, 1 - i): np.asarray(raw(S_w[i, 1 - i])) for i in (0, 1)}
    blk = lambda X, *idx: np.asarray(raw(X[idx]))
    T1 = _plain(lambda i, j: Delta[i, 1 - i] @ blk(bior.T, 1 - i, j), bior.T.ranges)
    U1 = _plain(lambda n, i, j: Delta[i, 1 - i] @ blk(bior.U, n, 1 - i, j), bior.U.ranges)
    def V1(i, j, k, l):
        a = np.tensordot(Delta[i, 1 - i], blk(bior.V, 1 - i, j, k, l), axes=([1], [0]))
        b = np.tensordot(Delta[j, 1 - j], blk(bior.V, i, 1 - j, k, l), axes=([1], [1])).transpose(1, 0, 2, 3)
        return a + b
    def V2(i, j, k, l):
        X = np.tensordot(Delta[i, 1 - i], blk(bior.V, 1 - i, 1 - j, k, l), axes=([1], [0]))
        return np.tensordot(Delta[j, 1 - j], X, axes=([1], [1])).transpose(1, 0, 2, 3)
    return {"bior": (bior.T, bior.U, bior.V), "delta1": (T1, U1, _plain(V1, bior.V.ranges)),
            "delta2": (None, None, _plain(V2, bior.V.ranges))}


# Diagrams that diagram_lists.py leaves out because they transfer three electrons in one direction, which is impossible
# for fragment charges in {-1, 0, +1}; the weak parts reach charge +2 (the strong orbital's electrons removed from a
# cation), so the weak dimer needs them (Be2 6-31G: without them, errors of 3e-4 / 2e-3 at the orders 1 / 2 in the
# block (A anion, B cation) <-> (A cation, B anion)).
EXTRA_DIAGRAMS = {"T": {2: ["s01s01t01"]},
                  "U": {2: ["s01s01u001", "s01s01u101"]},
                  "V": {1: ["s01v0011"], 2: ["s01s01v0001", "s01s01v0010", "s01s01v0111", "s01s01v1011", "s01s01v0011"]}}


def weak_hamiltonian_terms(weak_dens, dimer_sectors, ints_w, int_sets, target_multiplicity, nmax, orders=None, pair_slots=2.5):
    """[H_w^(m)] (m = 0 ... nmax, those not in orders None) of the weak dimer in the ordering of XR_term.dimer_matrix
    over dimer_sectors; ints_w: the weak overlap (weak_overlap_ints), in the pair basis if it has pair factors"""
    t = timer()
    orders = range(nmax + 1) if orders is None else orders
    S_w = ints_w.S
    cache = precontract_S(weak_dens, ints_w, t)
    kinds = (("T", ST_diagrams, D.ST1, D.ST2), ("U", SU_diagrams, D.SU1, D.SU2), ("V", SV_diagrams, D.SV1, D.SV2))
    blocks = {}
    for cls, Xs in int_sets.items():
        for (name, diagrams, _, _), X in zip(kinds, Xs):
            if X is not None:
                blocks[cls, name] = diagrammatic_expansion.blocks(densities=weak_dens, integrals=struct(**{"S": S_w, name: X}),
                                                                   diagrams=diagrams, contract_cache=cache, timings=t,
                                                                   precon_timings=t, target_multiplicity=target_multiplicity)
                blocks[cls, name]._supersys_info.pair_slots = pair_slots
    out = []
    for m in range(nmax + 1):
        if m not in orders:
            out.append(None)
            continue
        H = 0
        for j, cls in enumerate(("bior", "delta1", "delta2")):
            k = m - j
            if k < 0:
                continue
            for name, _, X1, X2 in kinds:
                if (cls, name) in blocks:
                    terms = {1: X1[0], 2: X2[0]} if k == 0 else {2: X2[k] + EXTRA_DIAGRAMS[name].get(k, [])}
                    H = H + XR_term.dimer_matrix(blocks[cls, name], terms, (0, 1), dimer_sectors, t)
        out.append(H)
    return out


# density types of the diagrams in EXTRA_DIAGRAMS that the standard XR density build does not contain (lowering types;
# their raising partners ccc, cccc, cccca are generated by wrap_dens_and_build_transpose), with the restr_ca channels
EXTRA_DENSITY_TYPES = (("aaa", 0, 3, (0,)), ("aaaa", 0, 4, (0,)), ("caaaa", 1, 4, (0, 1)))


def add_extra_densities(fragment, basis, rho):
    """adds the types of EXTRA_DENSITY_TYPES (and their adjoint partners) between the weak basis vectors to the density
    dictionary rho of the weak bases (in place); they only connect weak sectors whose charges differ by n_a - n_c"""
    from hummr_dens.pybind_interface import wrap_dens_and_build_transpose
    states = basis.states()
    dens_inp = {}
    for name, n_c, n_a, restrs in EXTRA_DENSITY_TYPES:
        for chg_k, by_mult_k in states.items():
            chg_b = chg_k + n_a - n_c
            for mult_b, bra in states.get(chg_b, {}).items():
                for mult_k, ket in by_mult_k.items():
                    if abs(mult_b - mult_k) > n_c + n_a:
                        continue
                    for restr in restrs:
                        for channel, data in fragment.density_sector(n_c, n_a, restr, (chg_b, mult_b), (chg_k, mult_k),
                                                                     list(bra), list(ket)).items():
                            dens_inp.setdefault(name, {}).setdefault((chg_b, chg_k), {})[channel] = np.asarray(data)
    if dens_inp:
        extra = wrap_dens_and_build_transpose(dens_inp, n_core=fragment.n_core, n_mo=fragment.orbitals.shape[1])
        for name, by_chg in extra.items():
            for chg_pair, by_channel in by_chg.items():
                rho.setdefault(name, {}).setdefault(chg_pair, {}).update(by_channel)
    return rho


# ---------------------------------------------------------------------------------------------------------------------
# Option A: T_B on the bra, all terms by one rectangular XR build
#
#     <Phi| T_B S_w H |Phi'> = <T_B Phi| S_w H |Phi'>      (T_B Hermitian, commutes with S_w)
#
# T_B Phi is expanded in products of "glued" fragment states [w (x) s']^{S'} (a strong configuration s' on a weak part
# w, the adjoint of the splitting), recoupled from the bridge basis by the same 9j as in BridgeAssembly.  The glued
# states of every sector get an orthonormal basis (SVD); XR's S H diagrams between the glued bras and the original kets
# (rectangular densities, weak overlap S_w, the integrals of the delta terms) give all terms of H at once, the T_B part
# exactly (in the bra states, no contraction is omitted).
# ---------------------------------------------------------------------------------------------------------------------

def add_extra_densities_rect(fragment, bra, ket, rho):
    """add_extra_densities for different bra and ket states ({chg: {mult: [vectors]}}): the lowering types between bra
    and ket and the raising partners from the lowering types between ket and bra, with the bra-ket overlaps of the
    frozen-core embedding"""
    from hummr_dens.pybind_interface import wrap_dens_and_build_transpose
    def overlaps(B, K):
        return {(chg, mult): np.array(B[chg][mult]) @ np.array(K[chg][mult]).T for chg in B for mult in B[chg]
                if chg in K and mult in K[chg]}
    for (X, Y, parts) in ((bra, ket, "direct"), (ket, bra, "reversed")):
        dens_inp = {}
        for name, n_c, n_a, restrs in EXTRA_DENSITY_TYPES:
            for chg_k, by_mult_k in Y.items():
                chg_b = chg_k + n_a - n_c
                for mult_b, bvecs in X.get(chg_b, {}).items():
                    for mult_k, kvecs in by_mult_k.items():
                        if abs(mult_b - mult_k) > n_c + n_a:
                            continue
                        for restr in restrs:
                            for channel, data in fragment.density_sector(n_c, n_a, restr, (chg_b, mult_b), (chg_k, mult_k),
                                                                         list(bvecs), list(kvecs)).items():
                                dens_inp.setdefault(name, {}).setdefault((chg_b, chg_k), {})[channel] = np.asarray(data)
        if dens_inp:
            extra = wrap_dens_and_build_transpose(dens_inp, n_core=fragment.n_core, n_mo=fragment.orbitals.shape[1],
                                                  overlaps=overlaps(X, Y), parts=parts)
            for name, by_chg in extra.items():
                for chg_pair, by_channel in by_chg.items():
                    for channel, data in by_channel.items():
                        rho.setdefault(name, {}).setdefault(chg_pair, {}).setdefault(channel, data)
    return rho


class GluedAssembly(object):
    """
    order-resolved S^(k), H^(k) (k = 0 ... nmax, weak orders) of the singlet product states, K = 1, option A
        splitter:     strong_split.StrongSplit of the fragments (hummr fragment, the strong orbital)
        split, bases: as for BridgeAssembly;  states: the original (rotated) states per fragment
        build_rect:   build_rect(frag, ket_states, bra_states) -> rectangular densities (build_frag_densities, xr_level 2)
    """
    def __init__(self, splitter, split, bases, states, build_rect, ints, strong_mo, sigma, nmax, n_val):
        self.splitter, self.bases, self.states, self.nmax, self.n_val = splitter, bases, states, nmax, n_val
        self.helper = BridgeAssembly(split, bases, [None, None], ints, strong_mo, sigma, nmax, n_val)
        self.build_rect, self.ints = build_rect, ints
        self._glue = {}
    def glue_matrix(self, chg, mult, occ, mult_w):
        """G (CSFs of (chg, mult) x CSFs of the weak sector (chg + occ, mult_w)): the glued state of a weak vector w is
        G w (the adjoint of the splitting: <b|glue(w)> = <part(b)|w>)"""
        if (chg, mult) not in self._glue:
            dim = self.splitter.dim(chg, mult)
            self._glue[chg, mult] = self.splitter.split_one(chg, mult, np.eye(dim), self.splitter.strong[0])
        part = self._glue[chg, mult].get((occ, mult_w))
        return None if part is None else part[1]
    def _bra_expansion(self, products):
        """[(P, (sector A, glued vector A), (sector B, glued vector B), c)] of T_B Phi_P"""
        TB, out = self.helper.TB, []
        for P, (S2s, b), (wA, wB), c, xA, xB in self.helper._expansion(products):
            WA = self.bases[0].basis[wA[0]][wA[1]].T @ xA                     # the weak parts as CSF vectors
            WB = self.bases[1].basis[wB[0]][wB[1]].T @ xB
            S2wA, S2wB = wA[1] - 1, wB[1] - 1
            for (b2, b1), t in TB.items():
                if b1 != b or t == 0.0:
                    continue
                occA, occB, _ = b2
                sA, sB = _spin2_config(occA), _spin2_config(occB)
                sign = (-1) ** ((self.n_val - wA[0]) * occB)
                for S2A in range(abs(S2wA - sA), S2wA + sA + 1, 2):
                    S2B = S2A                                                    # singlet
                    if not abs(S2wB - sB) <= S2B <= S2wB + sB:
                        continue
                    nj = ninej2(S2wA, sA, S2A, S2wB, sB, S2B, S2s, S2s, 0)
                    if nj == 0.0:
                        continue
                    GA = self.glue_matrix(wA[0] - occA, S2A + 1, occA, wA[1])
                    GB = self.glue_matrix(wB[0] - occB, S2B + 1, occB, wB[1])
                    if GA is None or GB is None:
                        continue
                    cc = c * t * sign * np.sqrt((S2A + 1) * (S2B + 1)) * (S2s + 1) * nj
                    out.append((P, ((wA[0] - occA, S2A + 1), GA @ WA), ((wB[0] - occB, S2B + 1), GB @ WB), cc))
        return out
    def compute(self, products):
        """{"S": [S^(k)], "H": [H^(k)] (E_core S^(k) contained)}"""
        n = len(products)
        expansion = self._bra_expansion(products)
        # orthonormal bases of the glued bra states per fragment and sector
        stacks = [{}, {}]
        for _, gA, gB, _ in expansion:
            for f, (sec, v) in enumerate((gA, gB)):
                stacks[f].setdefault(sec, []).append(v)
        gbasis = [{}, {}]
        for f in range(2):
            for sec, vs in stacks[f].items():
                _, sv, Vt = np.linalg.svd(np.array(vs), full_matrices=False)
                keep = sv > WEAK_BASIS_RTOL * sv[0] if sv[0] > 0 else np.zeros(len(sv), dtype=bool)
                if keep.any():
                    gbasis[f][sec] = Vt[keep]
        # bra (glued bases) and ket (original states) with the same sectors (a zero vector where one side has none)
        bra, ket = [], []
        for f in range(2):
            secs = set(gbasis[f]) | {(c, m) for c, bm in self.states[f].items() for m, v in bm.items() if len(v)}
            b, k = {}, {}
            for (c, m) in secs:
                dim = self.splitter.dim(c, m)
                b.setdefault(c, {})[m] = list(gbasis[f][c, m]) if (c, m) in gbasis[f] else [np.zeros(dim)]
                kv = self.states[f].get(c, {}).get(m, [])
                k.setdefault(c, {})[m] = list(kv) if len(kv) else [np.zeros(dim)]
            bra.append(b); ket.append(k)
        dens = [add_extra_densities_rect(self.splitter.fragment, bra[f], ket[f], self.build_rect(f, ket[f], bra[f]))
                for f in range(2)]
        # sector pairs: the glued bra pairs and the original (singlet) ket pairs
        bra_pairs = {(gA[0], gB[0]) for _, gA, gB, _ in expansion}
        ket_pairs = {(kA, kB) for kA, _, kB, _ in products}
        pairs = sorted(bra_pairs | ket_pairs)
        n_bra = [{(c, m): len(v) for c, bm in bra[f].items() for m, v in bm.items()} for f in range(2)]
        n_ket = [{(c, m): len(v) for c, bm in ket[f].items() for m, v in bm.items()} for f in range(2)]
        boff, koff, Ib, Jk = {}, {}, 0, 0
        for pA, pB in pairs:
            boff[pA, pB], koff[pA, pB] = Ib, Jk
            Ib += n_bra[0][pA] * n_bra[1][pB]
            Jk += n_ket[0][pA] * n_ket[1][pB]
        # rows: the coefficients of T_B Phi_P over the glued product basis; columns: the original product states
        R = np.zeros((Ib, n))
        for P, (sA, vA), (sB, vB), c in expansion:
            xa, xb = gbasis[0][sA] @ vA, gbasis[1][sB] @ vB
            o = boff[sA, sB]
            R[o:o + len(xa) * len(xb), P] += c * np.outer(xa, xb).ravel()
        cols = np.array([koff[kA, kB] + iA * n_ket[1][kB] + iB for kA, iA, kB, iB in products])
        t = timer()
        ints_w = self.helper.ints_w
        S_blocks = S_term_blocks(dens, ints_w[0], t, t, 1)
        out = {"S": [], "H": []}
        for k in range(self.nmax + 1):
            M = XR_term.dimer_matrix(S_blocks, {0: D.S0[0]} if k == 0 else {2: D.S2[k]}, (0, 1), pairs, t)
            out["S"].append(R.T @ M[:, cols])
        int_sets = delta_integrals(self.ints, ints_w[0].S)
        for M in weak_hamiltonian_terms(dens, pairs, ints_w[0], int_sets, 1, self.nmax):
            out["H"].append(R.T @ M[:, cols])
        return out
