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
The terms of H with legs on the strong and on the weak orbitals (option B of the exact-bridge ansatz, K = 1).

A term (with k weak overlap factors of (1/k!) :X^k:, X = sum_i sigma_i sum_spin (a+_{A i} a~_{B i} + a+_{B i} a~_{A i}),
and the H legs, normal-ordered without contractions; the contractions of an overlap annihilator with an H creator, the
delta terms, move that creator to the paired orbital of the other fragment with the weight sigma) is a product of
spin-orbital legs.  Reordered into (bridge legs)(weak A legs)(weak B legs), each block normal-ordered (sign of the
permutation), its matrix element between the states (bridge creators)(weak A)(weak B)|vac> is

    (-1)^{p_w n(b')} (-1)^{p_B N(w'_A)} <b|T_B X_b|b'> <w_A|X_A|w'_A> <w_B|X_B|w'_B>      (p: parities of the blocks)

with M-resolved elements: the bridge from its determinant matrices (bridge_tables.py), the weak fragments from hummr's
reduced densities of the active orbitals (all paths, restr 0) by spin_tensors.decompose and Wigner-Eckart (strings with
more creators than annihilators by their adjoint).  The product states are summed over their M components (CG).

Evaluation (MixedEvaluator): the spin, M and bridge algebra only gives scalar coefficients of the density paths; the
integrals and overlap weights of all topologies with the same weak leg kinds are combined with them into effective
tensors and contracted once per pair of density paths, either with both layout densities open or, for a large
density, with its H slots contracted with the integrals inside hummr (contract_density_sector).
"""

import itertools
from collections import OrderedDict
from functools import lru_cache
import numpy as np


class ByteLRU(object):
    """dict-like cache of arrays with a total size limit (least recently used entries dropped)"""
    def __init__(self, max_bytes):
        self.max_bytes, self.bytes, self.data = max_bytes, 0, OrderedDict()
    def get(self, key, make):
        if key in self.data:
            self.data.move_to_end(key)
            return self.data[key]
        value = make()
        size = self._size(value)
        if size <= self.max_bytes:
            self.data[key] = value
            self.bytes += size
            while self.bytes > self.max_bytes:
                _, old = self.data.popitem(last=False)
                self.bytes -= self._size(old)
        return value
    @staticmethod
    def _size(value):
        if value is None:
            return 0
        return sum(v.nbytes for v in value.values()) if isinstance(value, dict) else value.nbytes
from spin_tensors import cg2, decompose, _adjoint_leg

# spin-orbital legs as spherical legs: c_alpha = c_{+1/2}, c_beta = c_{-1/2}, a_alpha = -at_{-1/2}, a_beta = at_{+1/2}
SPHERICAL = {("c", 0): (1, 1.0), ("c", 1): (-1, 1.0), ("a", 0): (-1, -1.0), ("a", 1): (1, 1.0)}


@lru_cache(maxsize=None)
def _decomposition(n_c, n_a, m2s):
    """{(rank2, path): coefficient} of the product of spherical legs (layout order) in the tensors of the type"""
    Q2 = sum(m2s)
    out = {}
    for path, c in decompose({tuple(m2s): 1.0}, n_c, n_a, 0, Q2).items():
        out[path[1][-1] if path[1] else 0, path] = c
    return out


class WeakOperators(object):
    """M-resolved matrix elements of normal-ordered spin-orbital strings between the weak basis vectors of one fragment"""
    def __init__(self, fragment, basis, n_val, cache_bytes=2 * 2**30):
        self.fragment, self.basis, self.n_val = fragment, basis, n_val
        self._reduced = ByteLRU(cache_bytes)
    def vectors(self, sector):
        return self.basis.basis[sector[0]][sector[1]]
    def reduced(self, n_c, n_a, bra_sector, ket_sector, cache=True):
        """{path: rho (bra, ket, active orbitals in layout order)} in the CG convention, lowering types (n_a >= n_c)"""
        def make():
            from hummr_dens.pybind_interface import cg_factor
            dens = self.fragment.density_sector(n_c, n_a, 0, bra_sector, ket_sector, list(self.vectors(bra_sector)),
                                                list(self.vectors(ket_sector)))
            return {channel[4]: np.asarray(rho) / cg_factor(channel) for channel, rho in dens.items()}
        return self._reduced.get((n_c, n_a, bra_sector, ket_sector), make) if cache else make()
    def element(self, legs, bra_sector, ket_sector, M2b, M2k):
        """<w_bra S M2b/2| prod legs |w_ket S' M2k/2> (bra, ket, active orbital of every leg), legs [(kind, spin)] in
        layout order (all c before all a)"""
        n_c = sum(1 for k, _ in legs if k == "c"); n_a = len(legs) - n_c
        if not legs:
            # no operator on this fragment: the overlap of the (orthonormal) weak basis vectors
            nb = len(self.vectors(bra_sector))
            same = bra_sector == ket_sector and M2b == M2k and abs(M2b) <= bra_sector[1] - 1
            return np.eye(nb) if same else np.zeros((nb, len(self.vectors(ket_sector))))
        if n_c <= n_a:
            return self._lowering([SPHERICAL[l] for l in legs], n_c, n_a, bra_sector, ket_sector, M2b, M2k)
        # more creators: <b|X|k> = <k|X^+|b>, X^+ = the adjoint legs in reversed order (again c's before a's)
        sph = [SPHERICAL[l] for l in legs]
        kinds = [k for k, _ in legs]
        adj, phase = [], 1.0
        for (m2, ph), kind in zip(reversed(sph), reversed(kinds)):
            p, _, m2_new = _adjoint_leg(kind, m2)
            adj.append((m2_new, 1.0)); phase *= p * ph
        X = self._lowering(adj, n_a, n_c, ket_sector, bra_sector, M2k, M2b)
        L = len(legs)
        return phase * X.transpose([1, 0] + [2 + (L - 1 - j) for j in range(L)])
    def _lowering(self, sph, n_c, n_a, bra_sector, ket_sector, M2b, M2k):
        m2s = tuple(m for m, _ in sph)
        phase = np.prod([p for _, p in sph]) if sph else 1.0
        Q2 = sum(m2s)
        S2b, S2k = bra_sector[1] - 1, ket_sector[1] - 1
        nb, nk = len(self.vectors(bra_sector)), len(self.vectors(ket_sector))
        n_act = self.fragment.n_act
        out = np.zeros((nb, nk) + (n_act,) * len(sph))
        if M2b != M2k + Q2 or abs(M2b) > S2b or abs(M2k) > S2k:
            return out
        # (zero without hummr if the annihilators empty the ket or the creators overfill: hummr does not handle these)
        N_ket = self.n_val - ket_sector[0]
        if N_ket < n_a or N_ket - n_a + n_c > 2 * n_act:
            return out
        red = self.reduced(n_c, n_a, bra_sector, ket_sector)
        for (K2, path), c in _decomposition(n_c, n_a, m2s).items():
            cg = cg2(S2k, M2k, K2, Q2, S2b, M2b)
            if cg != 0.0 and path in red:
                out += (phase * c * cg) * red[path]
        return out


def _parity(perm):
    sign = 1
    perm = list(perm)
    for i in range(len(perm)):
        for j in range(i + 1, len(perm)):
            if perm[i] > perm[j]:
                sign = -sign
    return sign


def mixed_topologies(nmax, include_all=False):
    """
    [(m, weight, legs, spin_groups, (kind, integral legs))]: the terms with legs on the strong and on the weak orbitals,
    with k overlap factors and delta moves, weak order m = k + number of delta moves <= nmax.  A leg is
    (sub, kind, label, frag): sub "b" (bridge, frag of the strong orbital), "A"/"B" (weak; label: the orbital index
    variable; the S legs i0, i1, the H legs p, q, r, s).  integral legs: per H leg (label or None, fragment of the
    original orbital, strong); spin_groups: pairs of legs with the same spin.
    """
    out = []
    terms = (("M", 1.0, [("c", "p"), ("a", "q")], [(0, 1)]),
             ("G", 0.5, [("c", "p"), ("c", "q"), ("a", "s"), ("a", "r")], [(0, 3), (1, 2)]))
    classes = [("s", "A"), ("s", "B"), ("w", "A"), ("w", "B")]
    other = {"A": "B", "B": "A"}
    for kind, factor, hlegs, hgroups in terms:
        for cls in itertools.product(classes, repeat=len(hlegs)):
            n_s = sum(1 for c, _ in cls if c == "s")
            if not include_all and (n_s == 0 or n_s == len(hlegs)):
                continue
            weak_creators = [j for j, ((k, _), (c, _)) in enumerate(zip(hlegs, cls)) if k == "c" and c == "w"]
            for n_delta in range(len(weak_creators) + 1):
                for moved in itertools.combinations(weak_creators, n_delta):
                    for k in range(nmax - n_delta + 1):
                        for dirs in itertools.product(("AB", "BA"), repeat=k):
                            legs, groups = [], []
                            for t, d in enumerate(dirs):
                                legs += [(d[0], "c", f"i{t}", None), (d[1], "a", f"i{t}", None)]
                                groups.append((2 * t, 2 * t + 1))
                            off = len(legs)
                            ints = []
                            for j, ((lk, lab), (c, f)) in enumerate(zip(hlegs, cls)):
                                if c == "s":
                                    legs.append(("b", lk, None, f)); ints.append((None, f, True))
                                else:
                                    legs.append((other[f] if j in moved else f, lk, lab, None)); ints.append((lab, f, False))
                            groups += [(off + a, off + b) for a, b in hgroups]
                            delta_labels = [hlegs[j][1] for j in moved]
                            out.append((k + n_delta, factor / np.prod(range(1, k + 1)) if k else factor, legs, groups,
                                        (kind, ints), [f"i{t}" for t in range(k)], delta_labels))
    return out


@lru_cache(maxsize=None)
def weak_components(legs, S2b, S2k, M2b, M2k):
    """((path, coefficient), ...): <S2b M2b| prod legs |S2k M2k> = sum coefficient D[path] with D the layout densities
    (WeakOperators.layout_density) of the leg kinds; legs ((kind, spin), ...) in layout order; path "id" for no legs"""
    if not legs:
        return (("id", 1.0),) if S2b == S2k and M2b == M2k and abs(M2b) <= S2b else ()
    n_c = sum(1 for k, _ in legs if k == "c"); n_a = len(legs) - n_c
    sph = [SPHERICAL[l] for l in legs]
    if n_c > n_a:
        # <b|X|k> = <k|X^+|b>, X^+ = the adjoint legs in reversed order (the transposition is in the layout density)
        adj, phase = [], 1.0
        for (m2, ph), (kind, _) in zip(reversed(sph), reversed(legs)):
            p, _, m2_new = _adjoint_leg(kind, m2)
            adj.append((m2_new, 1.0)); phase *= p * ph
        return tuple((path, phase * c) for path, c in _lowering_components(tuple(m for m, _ in adj), n_a, n_c, S2k, S2b,
                                                                           M2k, M2b))
    phase = float(np.prod([p for _, p in sph]))
    return tuple((path, phase * c) for path, c in _lowering_components(tuple(m for m, _ in sph), n_c, n_a, S2b, S2k,
                                                                       M2b, M2k))


def _lowering_components(m2s, n_c, n_a, S2b, S2k, M2b, M2k):
    Q2 = sum(m2s)
    if M2b != M2k + Q2 or abs(M2b) > S2b or abs(M2k) > S2k:
        return ()
    out = {}
    for (K2, path), c in _decomposition(n_c, n_a, m2s).items():
        cg = cg2(S2k, M2k, K2, Q2, S2b, M2b)
        if cg != 0.0:
            out[path] = out.get(path, 0.0) + c * cg
    return tuple(out.items())


def _layout_density(op, kinds, bsec, ksec, n_w):
    """{path: D (bra, ket, weak orbital of every slot in layout order)} of the strings with these leg kinds, D in the
    convention of weak_components; {} if the strings vanish (hummr does not handle these)"""
    L = len(kinds)
    if not L:
        return {"id": np.eye(len(op.vectors(bsec)))} if bsec == ksec else {}
    n_c = kinds.count("c"); n_a = L - n_c
    n_act = op.fragment.n_act
    lowering = n_c <= n_a
    nc, na, b, k = (n_c, n_a, bsec, ksec) if lowering else (n_a, n_c, ksec, bsec)
    N_ket = op.n_val - k[0]
    if N_ket < na or N_ket - na + nc > 2 * n_act:
        return {}
    cut = (slice(None), slice(None)) + (slice(0, n_w),) * L
    red = op.reduced(nc, na, b, k, cache=False)
    if lowering:
        return {path: np.ascontiguousarray(rho[cut]) for path, rho in red.items()}
    return {path: np.ascontiguousarray(rho[cut].transpose([1, 0] + [2 + (L - 1 - j) for j in range(L)]))
            for path, rho in red.items()}


def _contracted_density(op, kinds, roles, J, path, bsec, ksec, n_w):
    """F (bra, ket, weak orbital of every S slot (layout order), axes of J after the H slots): the layout density
    (path) of the strings with these leg kinds, its H slots contracted with the first axes of J (weak orbitals) inside
    hummr, without forming the density; None if the strings vanish"""
    from hummr_dens.pybind_interface import cg_factor
    L = len(kinds)
    n_c = kinds.count("c"); n_a = L - n_c
    n_act = op.fragment.n_act
    lowering = n_c <= n_a
    nc, na, b, k = (n_c, n_a, bsec, ksec) if lowering else (n_a, n_c, ksec, bsec)
    N_ket = op.n_val - k[0]
    if N_ket < na or N_ket - na + nc > 2 * n_act:
        return None
    n_H = sum(1 for r in roles if r[0] == "H")
    J = np.pad(J, [(0, n_act - n_w)] * n_H + [(0, 0)] * (J.ndim - n_H))      # (the strong orbital is not a weak index)
    axis, h = [], 0
    for r in roles:
        axis.append(h if r[0] == "H" else -1)
        h += r[0] == "H"
    channel = (b[1], k[1], path[1][-1] if path[1] else 0, 0, path)
    F = op.fragment.contract_density_sector(nc, na, 0, b, k, list(op.vectors(b)), list(op.vectors(k)), channel,
                                            axis if lowering else axis[::-1], J)
    F = np.asarray(F) / cg_factor(channel)
    n_S, n_free = L - n_H, J.ndim - n_H
    F = F[(slice(None), slice(None)) + (slice(0, n_w),) * n_S]
    if not lowering:
        # hummr's slots are the adjoint's: reversed layout order, bra and ket swapped
        F = F.transpose([1, 0] + [2 + n_S - 1 - j for j in range(n_S)] + [2 + n_S + j for j in range(n_free)])
    return np.ascontiguousarray(F)


# the per-structure route of a large density forms it open (with numpy) if it serves at least this many structures per
# path (DENSE_MIN_REBUILT if it does not fit where it is kept, so that it is rebuilt for every transition of the other
# fragment), otherwise it contracts each structure inside hummr (about 0.1 s per structure and path for ccaa, Be2
# cc-pVDZ, against about 0.3 s per path for building the open density)
DENSE_MIN_STRUCTURES, DENSE_MIN_REBUILT = 3, 8


class MixedEvaluator(object):
    """
    order-resolved H^(m) of the mixed terms (mixed_topologies) of the singlet product states
        assembly:   bridge_assembly.BridgeAssembly (expansion of the product states, bridge tables, n_val)
        weak_ops:   [WeakOperators] of the fragments;  M, G: bridge_assembly.active_integrals;  sigma: increasing, the
                    strong pair last (the weak pair j of the active orbital index j)
        orders:     the weak orders m evaluated (default all, the others are zero)
        memory_budget: bytes (as for the XR builds); 1/8 for the cached (open and contracted) densities and the W,
                    1/32 for the intermediates formed at a time.  A fragment's density of more than 1/512 per path is
                    contracted per structure over its H slots (the S slots stay open; cached): with numpy from the open
                    density if that has at most 1/32 per path (built once for all structures), otherwise inside hummr
                    without forming it (Be2 cc-pVDZ: the four-operator densities, 0.25 GB per path, with numpy instead
                    of 1371 hummr contractions of 0.12 s; the five-operator ones, 2.8 GB per path, in hummr)
    The topologies are grouped by their leg structure (structure key: the legs without labels); the integrals and
    overlap weights of the topologies of one structure are summed into W (weak slots of A, then of B).  Per pair of
    weak sector pairs (quad), the spin, M and bridge algebra gives scalar coefficients of (structure, density path of
    A, density path of B); the W of all structures with the same weak leg kinds are combined with these coefficients
    and contracted once with the two layout densities.
    """
    def __init__(self, assembly, weak_ops, M, G, sigma, n_act, nmax, memory_budget=8e9, orders=None):
        self.asm, self.ops, self.M, self.G, self.nmax = assembly, weak_ops, M, G, nmax
        self.n_act, self.n_w, self.sw = n_act, n_act - 1, np.asarray(sigma[:n_act - 1])
        self.space, self.tables = assembly.tables.space, assembly.tables
        self.orders = set(range(nmax + 1) if orders is None else orders)
        self.chunk_bytes, self.open_bytes, self.dense_bytes = memory_budget / 32, memory_budget / 512, memory_budget / 32
        self._dens = ByteLRU(memory_budget / 16)
        self._fcache = ByteLRU(memory_budget / 32)
        self._wcache = ByteLRU(memory_budget / 32)
        self._tb = {}
    def _bridge(self, lb, b, M2b, bk, M2k):
        key = (lb, b, M2b, bk, M2k)
        if key not in self._tb:
            X = self.tables.TB
            for (f, kind), spin in lb:
                X = X @ self.space.op(f, spin, kind)
            self._tb[key] = self.tables.vectors[b, M2b] @ X @ self.tables.vectors[bk, M2k]
        return self._tb[key]
    def _density(self, f, kinds, bsec, ksec):
        return self._dens.get((f, kinds, bsec, ksec), lambda: _layout_density(self.ops[f], kinds, bsec, ksec, self.n_w))
    def _weights(self, top):
        """J (the H slots of A, then of B, in layout order) of a topology: weight x integral x delta weights"""
        m, weight, legs, sgroups, (ikind, ints), slabels, dlabels = top
        order = sorted(range(len(legs)), key=lambda j: ("bAB".index(legs[j][0]), legs[j][1] != "c", j))
        fr = {"A": 0, "B": 1}
        if ikind == "M":
            I = self.M[fr[ints[0][1]], fr[ints[1][1]]]
        else:
            # the legs are in operator order (p, q, s, r), G[p, q, r, s]: axes brought into the leg order
            p, q, s_, r = ints
            I = self.G[fr[p[1]], fr[q[1]], fr[r[1]], fr[s_[1]]].transpose(0, 1, 3, 2)
        I = I[tuple(self.n_act - 1 if strong else slice(0, self.n_w) for _, _, strong in ints)]
        letters = {"p": "p", "q": "q", "r": "r", "s": "s"}
        S_letters = {"i0": "ij", "i1": "kl"}            # (on A, on B): sum_i sigma_i on both fragments
        ops, specs = [I], ["".join(letters[lab] for lab, _, strong in ints if not strong)]
        for d in dlabels:
            ops.append(self.sw); specs.append(letters[d])
        H_out = "".join(letters[legs[j][2]] for sub in "AB" for j in order if legs[j][0] == sub
                        and legs[j][2] not in S_letters)
        return weight * np.einsum(",".join(specs) + "->" + H_out, *ops)
    def _full_weights(self, key, st, m):
        """W (all weak slots of A, then of B, in layout order) of the structure: J x the S factors
        sigma_i delta(i_A, i_B) (formed when needed, cached)"""
        def make():
            roles = st["roles"]
            S_letters = {"i0": "ij", "i1": "kl"}
            ops, specs = [st["J"][m]], ["".join(r[1] for f in (0, 1) for r in roles[f] if r[0] == "H")]
            for t in sorted({r[1] for r in roles[0] + roles[1] if r[0] == "S"}):
                ops.append(np.diag(self.sw)); specs.append(S_letters[t])
            out = "".join(S_letters[r[1]][f] if r[0] == "S" else r[1] for f in (0, 1) for r in roles[f])
            return np.einsum(",".join(specs) + "->" + out, *ops).reshape(-1)
        return self._wcache.get((key, m), make)
    def _structures(self, select):
        """{structure key: structure data with J per weak order}"""
        structs = {}
        for top in mixed_topologies(self.nmax, include_all=select is not None):
            if (select is not None and not select(top)) or top[0] not in self.orders:
                continue
            m, _, legs, sgroups = top[:4]
            key = (tuple((l[0], l[1], l[3]) for l in legs), tuple(sgroups))
            if key not in structs:
                order = sorted(range(len(legs)), key=lambda j: ("bAB".index(legs[j][0]), legs[j][1] != "c", j))
                blocks = {sub: [j for j in order if legs[j][0] == sub] for sub in "bAB"}
                structs[key] = dict(
                    J={}, roles=tuple(tuple(("S", legs[j][2]) if legs[j][2] in ("i0", "i1") else ("H", legs[j][2])
                                            for j in blocks[sub]) for sub in "AB"),
                    legs=[(l[0], l[1], l[3]) for l in legs], sgroups=sgroups, blocks=blocks, eps=_parity(order),
                    kinds=tuple(tuple(legs[j][1] for j in blocks[sub]) for sub in "AB"),
                    dN=tuple(sum(1 if legs[j][1] == "c" else -1 for j in blocks[sub]) for sub in "AB"),
                    docc=sum(1 if legs[j][1] == "c" else -1 for j in blocks["b"]))
                st = structs[key]
                st["spins"] = self._spin_assignments(st["legs"], sgroups, blocks)
            J = structs[key]["J"]
            J[m] = J[m] + self._weights(top) if m in J else self._weights(top)
        return structs
    def _coefficients(self, key, st, bkey, kkey):
        """{(path A, path B): coefficient} of the structure between the bridge x weak states bkey, kkey (cached by the
        spins, the bridge states and the ket charge parity of A)"""
        S2s, b, wA, wB = bkey
        S2k, bk, wAk, wBk = kkey
        ckey = (key, S2s, b, wA[1], wB[1], S2k, bk, wAk[1], wBk[1], (self.asm.n_val - wAk[0]) % 2)
        if ckey in self._coef:
            return self._coef[ckey]
        legs, blocks = st["legs"], st["blocks"]
        pA, pB = len(blocks["A"]) % 2, len(blocks["B"]) % 2
        sign_states = st["eps"] * (-1) ** (((pA + pB) % 2) * (bk[0] + bk[1]) + pB * (self.asm.n_val - wAk[0]))
        S2A, S2B, S2Ak, S2Bk = wA[1] - 1, wB[1] - 1, wAk[1] - 1, wBk[1] - 1
        acc = {}
        for QA, QB, Qb, lA, lB, lb in st["spins"]:
            for M2wk in range(-S2k, S2k + 1, 2):
                cgk0 = cg2(S2k, M2wk, S2k, -M2wk, 0, 0)
                for M2Ak in range(-S2Ak, S2Ak + 1, 2):
                    M2Bk = M2wk - M2Ak
                    if abs(M2Bk) > S2Bk:
                        continue
                    cgk = cgk0 * cg2(S2Ak, M2Ak, S2Bk, M2Bk, S2k, M2wk)
                    if cgk == 0.0:
                        continue
                    M2A, M2B, M2b = M2Ak + QA, M2Bk + QB, -M2wk + Qb
                    M2w = M2A + M2B
                    if abs(M2A) > S2A or abs(M2B) > S2B or abs(M2w) > S2s or M2b != -M2w:
                        continue
                    cgb = cg2(S2s, M2w, S2s, -M2w, 0, 0) * cg2(S2A, M2A, S2B, M2B, S2s, M2w)
                    if cgb == 0.0:
                        continue
                    tb = self._bridge(lb, b, M2b, bk, -M2wk)
                    if tb == 0.0:
                        continue
                    cA = weak_components(lA, S2A, S2Ak, M2A, M2Ak)
                    cB = weak_components(lB, S2B, S2Bk, M2B, M2Bk)
                    pref = sign_states * cgk * cgb * tb
                    for qA, xa in cA:
                        for qB, xb in cB:
                            acc[qA, qB] = acc.get((qA, qB), 0.0) + pref * xa * xb
        out = {k: v for k, v in acc.items() if v != 0.0}
        self._coef[ckey] = out
        return out
    @staticmethod
    def _spin_assignments(legs, sgroups, blocks):
        """[(Q2 of A, Q2 of B, Q2 of the bridge, legs of A, legs of B, legs of the bridge)] for all spins of the groups"""
        out = []
        for spins in itertools.product((0, 1), repeat=len(sgroups)):
            spin = {}
            for (x, y), s in zip(sgroups, spins):
                spin[x] = spin[y] = s
            QA, QB = (sum(SPHERICAL[legs[j][1], spin[j]][0] for j in blocks[sub]) for sub in "AB")
            Qb = sum((1 if spin[j] == 0 else -1) * (1 if legs[j][1] == "c" else -1) for j in blocks["b"])
            out.append((QA, QB, Qb, tuple((legs[j][1], spin[j]) for j in blocks["A"]),
                        tuple((legs[j][1], spin[j]) for j in blocks["B"]),
                        tuple(((legs[j][2], legs[j][1]), spin[j]) for j in blocks["b"])))
        return out
    def compute(self, products, select=None):
        """[H^(m)]; select(topology) -> bool restricts the terms (tests)"""
        n = len(products)
        out = [np.zeros((n, n)) for _ in range(self.nmax + 1)]
        # the product states grouped by (2 S_s, bridge state, weak sector A, weak sector B): C (products, n_A n_B)
        groups = {}
        for P, (S2s, b), (wA, wB), c, xA, xB in self.asm._expansion(products):
            key = (S2s, b, wA, wB)
            if key not in groups:
                groups[key] = np.zeros((n, len(xA) * len(xB)))
            groups[key][P] += c * np.outer(xA, xB).ravel()
        quads = {}
        for bkey in groups:
            for kkey in groups:
                quads.setdefault((bkey[2], kkey[2], bkey[3], kkey[3]), []).append((bkey, kkey))
        structs = self._structures(select)
        by_kinds = {}
        for key, st in structs.items():
            by_kinds.setdefault(st["kinds"], []).append(key)
        self._coef = {}
        # fragment A's transitions outside, B's inside (as XR_term.dimer_matrix): the large open densities of A are held
        # for all of B's transitions (_held), those of B are left to the cache
        self._held, self._held_at = {}, None
        for (wA, wAk, wB, wBk), pairs in sorted(quads.items(), key=lambda q: repr(q[0])):
            if self._held_at != (wA, wAk):
                self._held, self._held_at = {}, (wA, wAk)
            dN = (wAk[0] - wA[0], wBk[0] - wB[0])
            Y = {}
            for kinds, keys in by_kinds.items():
                keys = [k for k in keys if structs[k]["dN"] == dN]
                if not keys:
                    continue
                # C[(pair, m, path A, path B)][(structure, m)]
                C, wid = {}, {}
                for key in keys:
                    st = structs[key]
                    for pi, (bkey, kkey) in enumerate(pairs):
                        if bkey[1][0] + bkey[1][1] != kkey[1][0] + kkey[1][1] + st["docc"]:
                            continue
                        for (qA, qB), c in self._coefficients(key, st, bkey, kkey).items():
                            for m in st["J"]:
                                w = wid.setdefault((key, m), len(wid))
                                row = C.setdefault((pi, m, qA, qB), {})
                                row[w] = row.get(w, 0.0) + c
                if not C:
                    continue
                sizes = [len(self.ops[f].vectors(bs)) * len(self.ops[f].vectors(ks)) * self.n_w ** len(kinds[f]) * 8
                         for f, (bs, ks) in enumerate(((wA, wAk), (wB, wBk)))]
                f = int(sizes[1] > sizes[0])
                if sizes[f] > self.open_bytes and any(r[0] == "H" for r in structs[keys[0]]["roles"][f]):
                    self._contract_hummr(f, structs, C, wid, (wA, wAk, wB, wBk), Y, sizes[f])
                    continue
                DA = self._held_density(0, kinds[0], (wA, wAk))
                DB = self._density(1, kinds[1], wB, wBk)
                rows = [r for r in C if r[2] in DA and r[3] in DB]
                if not rows:
                    continue
                Wlist = [self._full_weights(k, structs[k], m) for k, m in wid]
                if len(Wlist) * Wlist[0].nbytes <= self.chunk_bytes:
                    Wlist = np.stack(Wlist)
                by_paths = {}
                for r in rows:
                    by_paths.setdefault(r[2:], []).append(r)
                for (qA, qB), rs in by_paths.items():
                    self._contract(DA[qA], DB[qB], rs, C, Wlist, Y)
            for (pi, m), Yp in Y.items():
                bkey, kkey = pairs[pi]
                nA, nAk = Yp.shape[0], Yp.shape[1]
                nB, nBk = Yp.shape[2], Yp.shape[3]
                Yp = Yp.transpose(0, 2, 1, 3).reshape(nA * nB, nAk * nBk)
                out[m] += groups[bkey] @ Yp @ groups[kkey].T
        return out
    def _contract(self, DA, DB, rows, C, Wlist, Y):
        """Y[(pair, m)] (a, A, b, B) += sum_x DA[a, A, xA] DB[b, B, xB] W_eff[row](xA, xB) for the rows of one path pair"""
        nA, nAk = DA.shape[:2]; nB, nBk = DB.shape[:2]
        XA, XB = int(np.prod(DA.shape[2:])), int(np.prod(DB.shape[2:]))
        DA, DB = DA.reshape(nA * nAk, XA), DB.reshape(nB * nBk, XB)
        u, v = nA * nAk, nB * nBk
        # (rows per chunk: the effective tensors and the intermediates of the chunk within chunk_bytes)
        step = max(1, int(self.chunk_bytes // (8 * (XA * XB + max(XA * v, XB * u) + u * v))))
        for c0 in range(0, len(rows), step):
            rs = rows[c0:c0 + step]
            R = len(rs)
            if isinstance(Wlist, np.ndarray):
                Cm = np.zeros((R, len(Wlist)))
                for i, r in enumerate(rs):
                    for w, c in C[r].items():
                        Cm[i, w] = c
                Weff = Cm @ Wlist
            else:
                # (the stacked W would exceed chunk_bytes)
                Weff = np.zeros((R, XA * XB))
                for i, r in enumerate(rs):
                    for w, c in C[r].items():
                        Weff[i] += c * Wlist[w]
            Weff = Weff.reshape(R, XA, XB)
            if XA * XB * v + XA * u * v <= XA * XB * u + XB * u * v:
                T = (Weff.reshape(R * XA, XB) @ DB.T).reshape(R, XA, v)          # (R, xA, bB)
                E = np.einsum("ux,rxv->ruv", DA, T, optimize=True)
            else:
                T = (Weff.transpose(0, 2, 1).reshape(R * XB, XA) @ DA.T).reshape(R, XB, u)
                E = np.einsum("vy,ryu->ruv", DB, T, optimize=True)
            for i, r in enumerate(rs):
                key = r[:2]
                e = E[i].reshape(nA, nAk, nB, nBk)
                Y[key] = Y[key] + e if key in Y else e.copy()
    def _contract_hummr(self, f, structs, C, wid, quad, Y, path_bytes):
        """Y[(pair, m)] += sum_w C[row, w] E_w: E_w of the structure (and order) w with the H slots of fragment f
        contracted inside hummr (_contracted_density, cached) and the other fragment's layout density open"""
        g = 1 - f
        secs = ((quad[0], quad[1]), (quad[2], quad[3]))
        terms = {}                                  # (w, path f, path g) -> [(row, coefficient)]
        for r, row in C.items():
            for w, c in row.items():
                terms.setdefault((w, r[2 + f], r[2 + g]), []).append((r[:2], c))
        S_letters = {"i0": "i", "i1": "k"}
        wlist = list(wid)
        kinds_f = structs[wlist[0][0]]["kinds"][f]
        # the open density of f (numpy route) if it is available, or if it has several structures per path and fits
        # where it is kept (all paths: the held densities for A, the cache for B; otherwise it is rebuilt for every
        # transition of the other fragment, Be2 cc-pVDZ: 1.3 GB ccaa four times, 39 s instead of 18 s for the weak
        # orders 0 and 1); otherwise the hummr contractions per structure
        dense = (f == 0 and kinds_f in self._held) or (f, kinds_f) + secs[f] in self._dens.data
        if not dense and path_bytes <= self.dense_bytes:
            n_paths = len({qf for _, qf, _ in terms})
            fits = n_paths * path_bytes <= (4 * self.chunk_bytes if f == 0 else self._dens.max_bytes)
            dense = len({(w, qf) for w, qf, _ in terms}) >= (DENSE_MIN_STRUCTURES if fits else DENSE_MIN_REBUILT) * n_paths
        # (dense: the open density of f, fetched once for all structures of this call, all of the same leg kinds)
        Df = self._held_density(f, kinds_f, secs[f]) if dense else None
        Dperm = {}          # (path, H slots of f) -> the open density with the H slots last, contiguous (bra ket S, H)
        # (sorted by the path and the H slots of f, so only one such copy is held at a time)
        H_of = lambda w: tuple(r[0] for r in structs[wlist[w][0]]["roles"][f])
        for (w, qf, qg), rows in sorted(terms.items(), key=lambda t: (repr(t[0][1]), H_of(t[0][0]))):
            key, m = wlist[w]
            st = structs[key]
            kinds, roles = st["kinds"], st["roles"]
            Dg = self._density(g, kinds[g], *secs[g]).get(qg)
            if Dg is None:
                continue
            J = st["J"][m]
            if f == 1:
                n_HA = sum(1 for r in roles[0] if r[0] == "H")
                J = J.transpose(list(range(n_HA, J.ndim)) + list(range(n_HA)))
            if dense:
                # (the open density of f, contracted over its H slots with numpy: F as _contracted_density)
                def make(J=J, roles=roles, qf=qf):
                    D = Df.get(qf)
                    if D is None:
                        return None
                    H_axes = tuple(2 + j for j, r in enumerate(roles[f]) if r[0] == "H")
                    if (qf, H_axes) not in Dperm:
                        Dperm.clear()
                        rest = [a for a in range(D.ndim) if a not in H_axes]
                        Dperm[qf, H_axes] = np.ascontiguousarray(D.transpose(rest + list(H_axes)))
                    Dp = Dperm[qf, H_axes]
                    n_H = len(H_axes)
                    out_shape = Dp.shape[:Dp.ndim - n_H] + J.shape[n_H:]
                    X_H = int(np.prod(Dp.shape[Dp.ndim - n_H:]))
                    return (Dp.reshape(-1, X_H) @ J.reshape(X_H, -1)).reshape(out_shape)
                F = self._fcache.get((f, key, m, qf, secs[f]), make)
            else:
                F = self._fcache.get((f, key, m, qf, secs[f]), lambda: _contracted_density(
                    self.ops[f], kinds[f], roles[f], J, qf, secs[f][0], secs[f][1], self.n_w))
            if F is None:
                continue
            sf, sg = ("aA", "bB") if f == 0 else ("bB", "aA")
            spec_F = sf + "".join(S_letters[r[1]] for r in roles[f] if r[0] == "S") \
                        + "".join(r[1] for r in roles[g] if r[0] == "H")
            spec_g = sg + "".join(S_letters[r[1]] if r[0] == "S" else r[1] for r in roles[g])
            S_used = [S_letters[r[1]] for r in roles[f] if r[0] == "S"]
            E = np.einsum(",".join([spec_F, spec_g] + S_used) + "->aAbB", F, Dg, *([self.sw] * len(S_used)),
                          optimize=True)
            for rk, c in rows:
                Y[rk] = Y[rk] + c * E if rk in Y else c * E
    def _held_density(self, f, kinds, sec):
        """the open density of fragment f, held for the current transition of A (f = 0, within memory_budget / 8 in
        total), otherwise from the cache"""
        if f == 0:
            if kinds not in self._held:
                D = self._density(f, kinds, *sec)
                if sum(ByteLRU._size(v) for v in self._held.values()) + ByteLRU._size(D) <= 4 * self.chunk_bytes:
                    self._held[kinds] = D
                return D
            return self._held[kinds]
        return self._density(f, kinds, *sec)
