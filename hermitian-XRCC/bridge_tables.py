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
Spin-coupled bridge tables of the exact-bridge ansatz (step 4), K = 1 strong pair.

Bridge space: the strong orbitals s_A, s_B of the pair (overlap sigma in the corresponding-orbital basis), in the graded
order A_s, B_s (an operator on B_s carries the parity of A_s), determinants as in the prototype (bridge_dens.Bridge).
The exact overlap operator of the pair (per spin, all pairs mutually orthogonal)

    T_B = prod_spin (1 + sigma (a+_{A s} a~_{B s} + a+_{B s} a~_{A s}) - sigma^2 n_{A s} n_{B s})

is a spin scalar.  Strong configurations of a fragment, as in strong_split.py: |0> (S 0), a+_{s m}|0> (S 1/2,
a+_{s,+1/2} = a+_alpha) and a+_{s alpha} a+_{s beta}|0> (S 0); the bridge basis couples A and B (A first):

    |c_A c_B; S_s M_s> = sum <S_A m_A S_B m_B|S_s M_s> X_A(c_A, m_A) X_B(c_B, m_B) |vac>     (creators of A left of B)

Operators: the strong parts of the Hamiltonian terms as the coupled tensors T^K_Q(path) of spin_tensors.py (type (n_c, n_a),
legs c_{+-1/2} = c_alpha/beta, at_{+1/2} = a_beta, at_{-1/2} = -a_alpha, legs in layout order c_1..c_n a_1..a_m), each leg
on s_A or s_B.  The tables are the reduced matrix elements in the CG convention of the densities,

    <b S M| T_B T^K_q(path) |b' S' M'> = <S' M' K q|S M> table[b, b'],

so that they combine with the weak densities by the same recoupling algebra.
"""

import itertools
import numpy as np
from spin_tensors import cg2, paths, components

FRAGS = ("A", "B")


class BridgeSpace(object):
    """determinants of the bridge space of one strong pair: 4 spin orbitals (A alpha, A beta, B alpha, B beta) in the
    graded order A_s, B_s; spin orbital (frag, spin) -> index 2 frag + spin"""
    def __init__(self):
        self.dets = [tuple(sorted(s)) for N in range(5) for s in itertools.combinations(range(4), N)]
        self.index = {d: k for k, d in enumerate(self.dets)}
        self.dim = len(self.dets)
        self._ops = {}
    def op(self, frag, spin, kind):
        """a+ ("c") or a ("a") of the spin orbital (frag in "A"/"B", spin 0 alpha / 1 beta), as a dense matrix"""
        key = (frag, spin, kind)
        if key not in self._ops:
            i = 2 * FRAGS.index(frag) + spin
            M = np.zeros((self.dim, self.dim))
            for col, d in enumerate(self.dets):
                if (kind == "c") == (i in d):
                    continue
                k = sum(1 for x in d if x < i)
                new = tuple(sorted(d + (i,))) if kind == "c" else tuple(x for x in d if x != i)
                M[self.index[new], col] = (-1) ** k
            self._ops[key] = M
        return self._ops[key]
    def leg(self, frag, kind, m2):
        """the spherical legs c_m, at_m (spin_tensors.py) of the strong orbital of frag"""
        if kind == "c":
            return self.op(frag, 0 if m2 == 1 else 1, "c")
        return self.op(frag, 1, "a") if m2 == 1 else -self.op(frag, 0, "a")
    def vacuum(self):
        v = np.zeros(self.dim); v[self.index[()]] = 1.0
        return v
    def overlap_operator(self, sigma):
        """T_B of the pair"""
        T = np.eye(self.dim)
        for spin in (0, 1):
            cA, aA, cB, aB = (self.op(f, spin, k) for f in FRAGS for k in "ca")
            T = T @ (np.eye(self.dim) + sigma * (cA @ aB + cB @ aA) - sigma ** 2 * (cA @ aA) @ (cB @ aB))
        return T


# strong configurations of one fragment: (occupation, 2 S) and their creators X(c, m2) in terms of legs
CONFIGS = ((0, 0), (1, 1), (2, 0))


def config_creator(space, frag, occ, m2):
    if occ == 0:
        return np.eye(space.dim)
    if occ == 1:
        return space.leg(frag, "c", m2)
    return space.op(frag, 0, "c") @ space.op(frag, 1, "c")


class BridgeTables(object):
    """spin-coupled bridge tables of one strong pair with overlap sigma"""
    def __init__(self, sigma):
        self.sigma = sigma
        self.space = BridgeSpace()
        self.TB = self.space.overlap_operator(sigma)
        # coupled bridge basis: (occ_A, occ_B, 2 S_s) and its vectors |..; S_s M_s> for all M_s
        self.basis, self.vectors = [], {}
        vac = self.space.vacuum()
        for (oA, sA), (oB, sB) in itertools.product(CONFIGS, CONFIGS):
            for S2 in range(abs(sA - sB), sA + sB + 1, 2):
                b = (oA, oB, S2)
                self.basis.append(b)
                for M2 in range(-S2, S2 + 1, 2):
                    v = np.zeros(self.space.dim)
                    for mA in range(-sA, sA + 1, 2):
                        mB = M2 - mA
                        c = cg2(sA, mA, sB, mB, S2, M2)
                        if abs(mB) <= sB and c != 0.0:
                            v += c * (config_creator(self.space, "A", oA, mA) @ config_creator(self.space, "B", oB, mB) @ vac)
                    self.vectors[b, M2] = v
    def operator(self, n_c, n_a, path, Q2, frags):
        """the determinant matrix of T^K_Q(path) of the type (n_c, n_a) with the legs (layout order) on frags"""
        kinds = ["c"] * n_c + ["a"] * n_a
        O = np.zeros((self.space.dim, self.space.dim))
        for m, coef in components(n_c, n_a, path, Q2).items():
            X = np.eye(self.space.dim)
            for kind, m2, f in zip(kinds, m, frags):
                X = X @ self.space.leg(f, kind, m2)
            O += coef * X
        return O
    def table(self, n_c, n_a, path, frags, with_overlap=True):
        """reduced matrix elements {(bra, ket): value} of T_B T^K(path) (with_overlap) or of T^K(path) alone, from the
        stretched-bra element (M = S) of the first ket projection with a nonzero CG coefficient"""
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
                    O = self.operator(n_c, n_a, path, Q2, frags)
                    if with_overlap:
                        O = self.TB @ O
                    val = self.vectors[b, S2] @ O @ self.vectors[k, Mk2] / cg
                    if abs(val) > 1e-14:
                        out[b, k] = val
                    break
        return out
    def all_types(self):
        """all (n_c, n_a, path, frags) of up to two creators and two annihilators on the bridge"""
        for n_c, n_a in itertools.product(range(3), range(3)):
            for K2 in range(0, n_c + n_a + 1):
                for path in paths(n_c, n_a, K2, 0):
                    for frags in itertools.product(FRAGS, repeat=n_c + n_a):
                        yield n_c, n_a, path, frags
