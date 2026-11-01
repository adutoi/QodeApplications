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
Spin-coupled bridge tables of the exact-bridge ansatz (step 4), K strong pairs (K <= 3: dense or sparse determinant
matrices over the 4K spin orbitals, 2^{4K} determinants).

Bridge space: the strong orbitals s_1 ... s_K of A and B (pair i with overlap sigma_i in the corresponding-orbital basis,
s_i in increasing active-orbital order), in the graded order A_{s_1} ... A_{s_K} B_{s_1} ... B_{s_K} (an operator on B
carries the parity of A), determinants as in the prototype (bridge_dens.Bridge).  The exact overlap operator of the
strong pairs (per pair and spin, all pairs mutually orthogonal, the factors commute)

    T_B = prod_i prod_spin (1 + sigma_i (a+_{A s_i} a~_{B s_i} + a+_{B s_i} a~_{A s_i}) - sigma_i^2 n_{A s_i} n_{B s_i})

is a spin scalar.  Strong configurations of a fragment (StrongConfigs): the occupations of its K strong orbitals,
coupled sequentially from the first strong orbital to the last, [[[s_1] s_2]^{t_2} ... s_K]^{s} (the creators of a
later orbital left of the earlier ones; a doubly occupied orbital a+_alpha a+_beta, a scalar), labelled by the
occupations and the doubled intermediate spins (2 t_1, ..., 2 t_K = 2 s).  For K = 1: |0> (S 0), a+_{s m}|0> (S 1/2,
a+_{s,+1/2} = a+_alpha) and a+_{s alpha} a+_{s beta}|0> (S 0).  The bridge basis couples A and B (A first):

    |c_A c_B; S_s M_s> = sum <S_A m_A S_B m_B|S_s M_s> X_A(c_A, m_A) X_B(c_B, m_B) |vac>     (creators of A left of B)

Operators: the strong parts of the Hamiltonian terms as the coupled tensors T^K_Q(path) of spin_tensors.py (type (n_c, n_a),
legs c_{+-1/2} = c_alpha/beta, at_{+1/2} = a_beta, at_{-1/2} = -a_alpha, legs in layout order c_1..c_n a_1..a_m), each leg
on s_A or s_B.  The tables are the reduced matrix elements in the CG convention of the densities,

    <b S M| T_B T^K_q(path) |b' S' M'> = <S' M' K q|S M> table[b, b'],

so that they combine with the weak densities by the same recoupling algebra.
"""

import itertools
import numpy as np
import scipy.sparse as sps
from spin_tensors import cg2, paths, components

FRAGS = ("A", "B")
DENSE_MAX_DIM = 256         # determinant matrices dense up to this dimension (K <= 2), sparse above (K = 3)


class FockSpace(object):
    """determinants (sorted tuples, all particle numbers) over n spin orbitals; creators/annihilators as matrices
    (dense up to DENSE_MAX_DIM, CSR above)"""
    def __init__(self, n):
        self.n = n
        self.dets = [tuple(sorted(s)) for N in range(n + 1) for s in itertools.combinations(range(n), N)]
        self.index = {d: k for k, d in enumerate(self.dets)}
        self.dim = len(self.dets)
        self.dense = self.dim <= DENSE_MAX_DIM
        self._ops = {}
    def so_op(self, i, kind):
        """a+ ("c") or a ("a") of spin orbital i"""
        key = (i, kind)
        if key not in self._ops:
            rows, cols, vals = [], [], []
            for col, d in enumerate(self.dets):
                if (kind == "c") == (i in d):
                    continue
                k = sum(1 for x in d if x < i)
                new = tuple(sorted(d + (i,))) if kind == "c" else tuple(x for x in d if x != i)
                rows.append(self.index[new]); cols.append(col); vals.append(float((-1) ** k))
            if self.dense:
                M = np.zeros((self.dim, self.dim))
                M[rows, cols] = vals
            else:
                M = sps.csr_matrix((vals, (rows, cols)), shape=(self.dim, self.dim))
            self._ops[key] = M
        return self._ops[key]
    def eye(self):
        return np.eye(self.dim) if self.dense else sps.identity(self.dim, format="csr")
    def vacuum(self):
        v = np.zeros(self.dim); v[self.index[()]] = 1.0
        return v


class StrongConfigs(object):
    """
    coupled strong configurations of one fragment (K strong orbitals, local spin orbitals 2 i + spin), as vectors of
    its strong Fock space (4^K determinants): label (occupations (o_1 .. o_K), doubled intermediate spins), see the
    module docstring; and the recoupling of the split fragment states (strong_split.StrongSplit, sequential coupling
    [[w s_K] s_{K-1} ...]^S, the last strong orbital split first and outermost) to [w [strong configuration]^{s}]^S
    """
    def __init__(self, K):
        self.K = K
        self.space = FockSpace(2 * K)
        self._labels, self._recoupled = {}, {}
        self.vectors = {}                   # {label: {2m: vector}}
    def creator(self, i, m2):
        return self.space.so_op(2 * i + (0 if m2 == 1 else 1), "c")
    def couple(self, comps, S2_in, i, occ, S2_out):
        """{2M: X} of [X_in (x) occ on strong orbital i]^{S_out} (creators of i left), X (.., strong Fock dim)"""
        if occ == 0:
            return comps if S2_in == S2_out else None
        if occ == 2:
            if S2_in != S2_out:
                return None
            P = self.creator(i, 1) @ self.creator(i, -1)
            return {M2: X @ P.T for M2, X in comps.items()}
        out = {}
        for M2 in range(-S2_out, S2_out + 1, 2):
            X = 0
            for m2 in (1, -1):
                c = cg2(S2_in, M2 - m2, 1, m2, S2_out, M2)
                if c != 0.0:
                    X = X + c * (comps[M2 - m2] @ self.creator(i, m2).T)
            out[M2] = X
        return out
    def labels(self, occs):
        """[(label, 2 s)] of the configurations with the occupations occs (local order) (vectors stored)"""
        occs = tuple(occs)
        if occs not in self._labels:
            states = [((), 0, {0: self.space.vacuum()})]
            for i, occ in enumerate(occs):
                new = []
                for twos, S2, comps in states:
                    for S2_out in ((S2 - 1, S2 + 1) if occ == 1 else (S2,)):
                        if S2_out < 0:
                            continue
                        c = self.couple(comps, S2, i, occ, S2_out)
                        if c is not None:
                            new.append((twos + (S2_out,), S2_out, c))
                states = new
            out = []
            for twos, S2, comps in states:
                label = (occs, twos)
                self.vectors[label] = comps
                out.append((label, S2))
            self._labels[occs] = out
        return self._labels[occs]
    def recouple(self, path, S2):
        """[(label, coefficient)]: the split component path (StrongSplit, ((orbital, occ, mult_w), ...) with the last
        strong orbital first) of a state of spin S2 / 2 as sum_c coefficient [w [c]^{s_c}]^S (w the weak part)"""
        key = (tuple((occ, mult) for _, occ, mult in path), S2)
        if key in self._recoupled:
            return self._recoupled[key]
        K = self.K
        occs = tuple(path[K - 1 - i][1] for i in range(K))
        S2w = path[-1][2] - 1
        nV = S2w + 1
        # the weak multiplet as unit vectors (rows 2 m_w = S2w, S2w - 2, ...), times the strong vacuum
        comps = {M2: np.outer(np.eye(nV)[(S2w - M2) // 2], self.space.vacuum()) for M2 in range(-S2w, S2w + 1, 2)}
        S2_in = S2w
        for j in range(K - 1, -1, -1):
            S2_out = path[j - 1][2] - 1 if j > 0 else S2
            comps = self.couple(comps, S2_in, K - 1 - j, path[j][1], S2_out)
            S2_in = S2_out
        target = comps[S2]
        out, norm = [], 0.0
        for label, s2 in self.labels(occs):
            if not abs(S2w - s2) <= S2 <= S2w + s2:
                continue
            Y = self.vectors[label]
            T = 0
            for M2w in range(-S2w, S2w + 1, 2):
                c = cg2(S2w, M2w, s2, S2 - M2w, S2, S2)
                if c != 0.0:
                    T = T + c * np.outer(np.eye(nV)[(S2w - M2w) // 2], Y[S2 - M2w])
            coef = float(np.sum(T * target))
            norm += coef ** 2
            if abs(abs(coef) - 1.0) < 1e-12:
                coef = float(np.sign(coef))
            if abs(coef) > 1e-14:
                out.append((label, coef))
        if abs(norm - 1.0) > 1e-10:
            raise RuntimeError(f"StrongConfigs.recouple: path {path} (2S = {S2}) not spanned by the coupled strong "
                               f"configurations (norm {norm})")
        self._recoupled[key] = out
        return out


def occupation(label):
    """number of electrons of a strong configuration"""
    return sum(label[0])


def spin2(label):
    """doubled spin of a strong configuration"""
    return label[1][-1] if label[1] else 0


class BridgeSpace(FockSpace):
    """determinants of the bridge space of K strong pairs: 4K spin orbitals in the graded order A_s, B_s; spin orbital
    (frag, strong orbital i, spin) -> 2K frag + 2 i + spin"""
    def __init__(self, K=1):
        self.K = K
        FockSpace.__init__(self, 4 * K)
    def op(self, frag, spin, kind, i=0):
        """a+ ("c") or a ("a") of the spin orbital (frag in "A"/"B", strong orbital i, spin 0 alpha / 1 beta)"""
        return self.so_op(2 * self.K * FRAGS.index(frag) + 2 * i + spin, kind)
    def leg(self, frag, kind, m2, i=0):
        """the spherical legs c_m, at_m (spin_tensors.py) of the strong orbital i of frag"""
        if kind == "c":
            return self.op(frag, 0 if m2 == 1 else 1, "c", i)
        return self.op(frag, 1, "a", i) if m2 == 1 else -self.op(frag, 0, "a", i)
    def overlap_operator(self, sigma):
        """T_B of the strong pairs (sigma: the K overlaps, or a number for K = 1)"""
        sigma = np.atleast_1d(sigma)
        T = self.eye()
        for i in range(self.K):
            for spin in (0, 1):
                cA, aA, cB, aB = (self.op(f, spin, k, i) for f in FRAGS for k in "ca")
                T = T @ (self.eye() + sigma[i] * (cA @ aB + cB @ aA) - sigma[i] ** 2 * (cA @ aA) @ (cB @ aB))
        return T
    def so(self, frag, spin, i=0):
        """the index of the spin orbital (frag, strong orbital i, spin)"""
        return 2 * self.K * FRAGS.index(frag) + 2 * i + spin
    def sector_dets(self, sector):
        """the determinants (indices) of the sector (n_alpha, n_beta)"""
        if not hasattr(self, "_sectors"):
            self._sectors = {}
            for k, d in enumerate(self.dets):
                na = sum(1 for x in d if x % 2 == 0)
                self._sectors.setdefault((na, len(d) - na), []).append(k)
            self._sectors = {sec: np.array(idx) for sec, idx in self._sectors.items()}
        return self._sectors.get(sector)
    @staticmethod
    def sector_of(N, M2):
        """the sector (n_alpha, n_beta) of N electrons with 2 M_s = M2"""
        return ((N + M2) // 2, (N - M2) // 2)
    def block(self, O, row_sector, col_sector):
        """the dense block of the determinant matrix O between two sectors (None if one of them is empty)"""
        r, c = self.sector_dets(row_sector), self.sector_dets(col_sector)
        if r is None or c is None:
            return None
        return O[np.ix_(r, c)] if self.dense else O[r][:, c].toarray()
    def row_block(self, so, kind, row_sector):
        """(column sector, dense block) of a+ / a of spin orbital so acting on row vectors of row_sector (v -> v O), or
        None if it vanishes there"""
        key = (so, kind, row_sector)
        if not hasattr(self, "_row_blocks"):
            self._row_blocks = {}
        if key not in self._row_blocks:
            d = (1 if kind == "c" else -1) * np.array((1, 0) if so % 2 == 0 else (0, 1))
            col_sector = tuple(int(x) for x in np.array(row_sector) - d)
            B = self.block(self.so_op(so, kind), row_sector, col_sector)
            self._row_blocks[key] = None if B is None or not B.any() else (col_sector, B)
        return self._row_blocks[key]
    def place(self, yA, yB):
        """the bridge vector of the product X_A X_B |vac> of two strong-configuration vectors (fragment Fock spaces)"""
        if not hasattr(self, "_place"):
            frag_dets = FockSpace(2 * self.K).dets
            self._place = np.array([[self.index[dA + tuple(2 * self.K + x for x in dB)] for dB in frag_dets]
                                    for dA in frag_dets])
        v = np.zeros(self.dim)
        v[self._place.ravel()] = np.outer(yA, yB).ravel()
        return v


class _Vectors(dict):
    """{(bridge state, 2M): vector}, built on demand"""
    def __init__(self, tables):
        dict.__init__(self)
        self.tables = tables
    def __missing__(self, key):
        v = self.tables._vector(*key)
        self[key] = v
        return v


class BridgeTables(object):
    """spin-coupled bridge tables of K strong pairs with the overlaps sigma (a number for K = 1); bridge states
    (c_A, c_B, 2 S_s) with c_A, c_B labels of StrongConfigs; their vectors are built on demand"""
    def __init__(self, sigma, K=1):
        self.sigma, self.K = sigma, K
        self.space = BridgeSpace(K)
        self.configs = StrongConfigs(K)
        self.TB = self.space.overlap_operator(sigma)
        self.vectors = _Vectors(self)
        self._basis = None
    def config_labels(self):
        """all strong configurations of a fragment"""
        return [lab for occs in itertools.product(range(3), repeat=self.K) for lab, _ in self.configs.labels(occs)]
    @property
    def basis(self):
        """all bridge states (A first: for K = 1 (occ_A, occ_B) in the order 0, 1, 2)"""
        if self._basis is None:
            self._basis = []
            for cA, cB in itertools.product(self.config_labels(), repeat=2):
                sA, sB = spin2(cA), spin2(cB)
                for S2 in range(abs(sA - sB), sA + sB + 1, 2):
                    self._basis.append((cA, cB, S2))
        return self._basis
    def _vector(self, b, M2):
        cA, cB, S2 = b
        sA, sB = spin2(cA), spin2(cB)
        self.configs.labels(cA[0]); self.configs.labels(cB[0])
        YA, YB = self.configs.vectors[cA], self.configs.vectors[cB]
        v = np.zeros(self.space.dim)
        for mA in range(-sA, sA + 1, 2):
            mB = M2 - mA
            c = cg2(sA, mA, sB, mB, S2, M2)
            if abs(mB) <= sB and c != 0.0:
                v += c * self.space.place(YA[mA], YB[mB])
        return v
    def reduced(self, O):
        """the reduced elements of a spin-scalar operator O (determinant matrix) between the bridge states, on demand
        (ReducedTable)"""
        return ReducedTable(self, O)
    def operator(self, n_c, n_a, path, Q2, frags, orbs=None):
        """the determinant matrix of T^K_Q(path) of the type (n_c, n_a) with the legs (layout order) on frags (and the
        strong orbitals orbs, default 0)"""
        kinds = ["c"] * n_c + ["a"] * n_a
        orbs = orbs or (0,) * len(frags)
        O = 0 * self.space.eye()
        for m, coef in components(n_c, n_a, path, Q2).items():
            X = self.space.eye()
            for kind, m2, f, i in zip(kinds, m, frags, orbs):
                X = X @ self.space.leg(f, kind, m2, i)
            O = O + coef * X
        return O
    def table(self, n_c, n_a, path, frags, with_overlap=True, orbs=None):
        """reduced matrix elements {(bra, ket): value} of T_B T^K(path) (with_overlap) or of T^K(path) alone over all
        bridge states, from the stretched-bra element (M = S) of the first ket projection with a nonzero CG coefficient"""
        K2 = path[1][-1] if path[1] else 0
        out = {}
        for b in self.basis:
            for k in self.basis:
                S2, Sk2 = b[2], k[2]
                for Mk2 in range(-Sk2, Sk2 + 1, 2):
                    Q2 = S2 - Mk2
                    cg = cg2(Sk2, Mk2, K2, Q2, S2, S2)
                    if abs(Q2) > K2 or abs(cg) < 1e-12:
                        continue
                    O = self.operator(n_c, n_a, path, Q2, frags, orbs)
                    if with_overlap:
                        O = self.TB @ O
                    val = self.vectors[b, S2] @ O @ self.vectors[k, Mk2] / cg
                    if abs(val) > 1e-14:
                        out[b, k] = val
                    break
        return out
    def all_types(self):
        """all (n_c, n_a, path, frags) of up to two creators and two annihilators on the bridge (strong orbital 0)"""
        for n_c, n_a in itertools.product(range(3), range(3)):
            for K2 in range(0, n_c + n_a + 1):
                for path in paths(n_c, n_a, K2, 0):
                    for frags in itertools.product(FRAGS, repeat=n_c + n_a):
                        yield n_c, n_a, path, frags


class ReducedTable(object):
    """dict-like {(bridge, bridge'): reduced element} of a spin-scalar operator O (T_B, T_B X_strong), computed on demand
    from the stretched components (zero between different spins and below 1e-14)"""
    def __init__(self, tables, O):
        self.tables, self.O, self.data = tables, O, {}
    def get(self, key, default=0.0):
        if key not in self.data:
            b, k = key
            val = 0.0
            if b[2] == k[2]:
                val = self.tables.vectors[b, b[2]] @ self.O @ self.tables.vectors[k, k[2]]
                if not abs(val) > 1e-14:
                    val = 0.0
            self.data[key] = val
        val = self.data[key]
        return default if val == 0.0 else val
