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

# Spin-tensor algebra of the spin-adapted fragment densities.
#
# Every density type is a normal-ordered operator string with n_c creators and n_a annihilators.  Its
# spin-adapted channels are coupled spherical tensors, each defined by a coupling path.  This module
# provides their components in terms of the spin components of the single operators ("legs"), which is
# all that is needed to derive reversal (adjoint) relations, diagram spin weights and frozen-core
# contractions numerically instead of by hand.
#
# Legs (proper spherical tensors, doubled projections m2 = 2m):
#     c_{+1/2} = c_alpha,  c_{-1/2} = c_beta,     at_{+1/2} = a_beta,  at_{-1/2} = -a_alpha
#
# Lowering types (n_a >= n_c), the convention of hummr (see python_xr_todo.md of the hummr side):
#     layout:  (c_1, ..., c_n, a_1, ..., a_m), the blocks are the ca pairs (c_i, a_i), i = 1..n, followed
#              by the single a_{n+1} ... a_m;  the first restr_ca blocks are rank 0 (operator-local pairs)
#     blocks:  B^k = -sqrt(2) [c x at]^k  (B^0 = E = sum_s c_s a_s),  single at
#     coupling left-nested [[[B_1 x B_2]^{K_2} x B_3]^{K_3} ...]^K,  path = ((2k_1, ...), (2K_1, ...)),
#     the operator being sum_m C_path(m) c_1 ... c_n at_1 ... at_m (c's moved left, no contractions)
#
# Raising types (n_c > n_a) are defined as the adjoints of the lowering type with n_c and n_a swapped,
#     R^K_q(path) = (-1)^(K+q) (T^K_{-q}(path))^dagger,
# with the layout reversed (the orbital index of slot j of R is the one of slot N-1-j of T) and labeled
# by the path of T.  With this definition c is the partner of at (c_m = (-1)^(1/2+m) at_{-m}^dagger), and R is
# again a proper spherical tensor.
#
# Components are dicts {tuple of m2 of the slots in layout order: coefficient}.

import itertools
from functools import lru_cache

import numpy as np
from sympy import Rational
from sympy.physics.quantum.cg import CG


@lru_cache(maxsize=None)
def cg2(j1, m1, j2, m2, J, M):
    """Clebsch-Gordan coefficient <j1 m1 j2 m2|J M> with all arguments doubled (integers)"""
    if abs(m1) > j1 or abs(m2) > j2 or abs(M) > J or m1 + m2 != M:
        return 0.0
    if (j1 + m1) % 2 or (j2 + m2) % 2 or (J + M) % 2:
        return 0.0
    if not (abs(j1 - j2) <= J <= j1 + j2) or (j1 + j2 + J) % 2:
        return 0.0
    return float(CG(Rational(j1, 2), Rational(m1, 2), Rational(j2, 2), Rational(m2, 2), Rational(J, 2), Rational(M, 2)).doit())

def is_raising(n_c, n_a):
    return n_c > n_a

@lru_cache(maxsize=None)
def lowering_paths(n_c, n_a, rank2, restr):
    """all coupling paths ((2k_i), (2K_i)) of a lowering type (n_a >= n_c) with total rank rank2/2"""
    if n_c > n_a:
        raise ValueError("lowering_paths requires n_a >= n_c")
    out = []
    def rec(i, k2s, K2s):
        if i == n_a:
            if K2s and K2s[-1] == rank2:
                out.append((tuple(k2s), tuple(K2s)))
            return
        ranks = ([0] if i < restr else [0, 2]) if i < n_c else [1]
        for k2 in ranks:
            if i == 0:
                rec(1, [k2], [k2])
                continue
            prev = K2s[-1]
            for K2 in range(abs(prev - k2), prev + k2 + 1, 2):
                rec(i + 1, k2s + [k2], K2s + [K2])
    if n_a == 0:
        if rank2 == 0:
            out.append(((), ()))
        return tuple(out)
    rec(0, [], [])
    return tuple(out)

def paths(n_c, n_a, rank2, restr):
    """channel paths of a type: lowering types their own, raising types those of their lowering partner"""
    if is_raising(n_c, n_a):
        return lowering_paths(n_a, n_c, rank2, restr)
    return lowering_paths(n_c, n_a, rank2, restr)

def path_rank2(path):
    return path[1][-1] if path[1] else 0

def _block_comps(is_ca, k2, q2):
    """{(m2 of the block legs): coefficient};  ca block: -sqrt(2) [c x at]^k,  else single at"""
    if is_ca:
        return {(mc, q2 - mc): -np.sqrt(2) * cg2(1, mc, 1, q2 - mc, k2, q2) for mc in (1, -1) if abs(q2 - mc) == 1}
    return {(q2,): 1.0} if abs(q2) == 1 and k2 == 1 else {}

@lru_cache(maxsize=None)
def _lowering_components(n_c, n_a, path, Q2):
    """components of a lowering tensor, legs in layout order (c_1..c_n, a_1..a_m)"""
    k2s, K2s = path
    if n_a == 0:
        return {(): 1.0} if Q2 == 0 else {}
    def comps(i, Qi):
        if i == 0:
            return _block_comps(0 < n_c, k2s[0], Qi)
        out = {}
        for q2 in range(-k2s[i], k2s[i] + 1, 2):
            Qp = Qi - q2
            if abs(Qp) > K2s[i - 1]:
                continue
            c = cg2(K2s[i - 1], Qp, k2s[i], q2, K2s[i], Qi)
            if abs(c) < 1e-14:
                continue
            for ml, cl in comps(i - 1, Qp).items():
                for mr, cr in _block_comps(i < n_c, k2s[i], q2).items():
                    out[ml + mr] = out.get(ml + mr, 0.0) + c * cl * cr
        return out
    block_order = comps(len(k2s) - 1, Q2)    # legs in block order c_1, at_1, c_2, at_2, ..., extra at
    # reorder legs into the layout order
    legs_c = [2 * i for i in range(n_c)]
    legs_a = [2 * i + 1 if i < n_c else n_c + i for i in range(n_a)]
    order = legs_c + legs_a
    return {tuple(m[j] for j in order): c for m, c in block_order.items() if abs(c) > 1e-14}

def _adjoint_leg(kind, m2):
    """(phase, kind, m2) of the adjoint of a leg:  c_m^+ = (-1)^(1/2+m) at_{-m},  at_m^+ = (-1)^(1/2-m) c_{-m}"""
    if kind == "c":
        return (-1) ** ((1 + m2) // 2), "a", -m2
    return (-1) ** ((1 - m2) // 2), "c", -m2

@lru_cache(maxsize=None)
def components(n_c, n_a, path, Q2):
    """components of the tensor T^K_Q(path) of the type (n_c, n_a), legs in its layout order"""
    if not is_raising(n_c, n_a):
        return _lowering_components(n_c, n_a, path, Q2)
    # R^K_Q = (-1)^(K+Q) (T^K_{-Q})^dagger of the partner (n_a, n_c), legs reversed and adjoint
    K2 = path_rank2(path)
    sign = (-1) ** ((K2 + Q2) // 2)
    kinds = ["c"] * n_a + ["a"] * n_c       # layout kinds of the partner
    out = {}
    for m, c in _lowering_components(n_a, n_c, path, -Q2).items():
        phase, new = 1, []
        for kind, m2 in zip(reversed(kinds), reversed(m)):
            p, _, m2_new = _adjoint_leg(kind, m2)
            phase *= p
            new.append(m2_new)
        out[tuple(new)] = out.get(tuple(new), 0.0) + sign * phase * c
    return out

def layout_kinds(n_c, n_a):
    return ["c"] * n_c + ["a"] * n_a

@lru_cache(maxsize=None)
def layout_pairs(n_c, n_a, restr):
    """
    slot structure of the layout: (local pairs [(c_slot, a_slot)] (rank-0 blocks), free c slots, free a slots),
    slots numbered in layout order (c slots 0..n_c-1, a slots n_c..n_c+n_a-1)
    """
    if not is_raising(n_c, n_a):
        local = [(i, n_c + i) for i in range(restr)]
    else:
        # reversed layout of the partner (n_a, n_c): partner slot s -> slot N-1-s
        N = n_c + n_a
        local = sorted((N - 1 - (n_a + i), N - 1 - i) for i in range(restr))    # (partner a_i -> c, partner c_i -> a)
    used = set(x for pair in local for x in pair)
    free_c = [s for s in range(n_c) if s not in used]
    free_a = [s for s in range(n_c, n_c + n_a) if s not in used]
    return tuple(local), tuple(free_c), tuple(free_a)

def decompose(target, n_c, n_a, restr, Q2, rank2s=None, tol=1e-10):
    """
    expand an operator of the type (n_c, n_a), given by its components target (legs in layout order, all with
    total projection Q2), in the tensors T^K_Q2(path) of the type with restr rank-0 local pairs.
    Returns {path: coefficient}; raises if the target is not representable.
    """
    max_rank2 = n_c + n_a
    rank2s = range(abs(Q2) % 2, max_rank2 + 1, 2) if rank2s is None else rank2s
    basis = [(p, components(n_c, n_a, p, Q2)) for K2 in rank2s if K2 >= abs(Q2) for p in paths(n_c, n_a, K2, restr)]
    keys = sorted(set(target) | set(k for _, comp in basis for k in comp))
    if not keys:
        return {}
    index = {k: i for i, k in enumerate(keys)}
    M = np.zeros((len(keys), len(basis)))
    for j, (_, comp) in enumerate(basis):
        for k, c in comp.items():
            M[index[k], j] = c
    o = np.zeros(len(keys))
    for k, c in target.items():
        o[index[k]] = c
    if len(basis) == 0:
        if np.linalg.norm(o) > tol:
            raise RuntimeError("operator not representable (no tensors of this type/projection)")
        return {}
    w, *_ = np.linalg.lstsq(M, o, rcond=None)
    if np.linalg.norm(M @ w - o) > tol * max(1, np.linalg.norm(o)):
        raise RuntimeError(f"operator not representable in the tensors of type {(n_c, n_a)} with restr {restr}")
    return {p: float(x) for (p, _), x in zip(basis, w) if abs(x) > tol}

def permute_components(comps, perm, kinds):
    """
    components of the same operator with its legs permuted: new slot j is old slot perm[j] (perm must keep c
    slots before a slots); the sign of the permutation of the c's and of the a's is included
    """
    sign = _parity([p for p in perm if kinds[p] == "c"]) * _parity([p for p in perm if kinds[p] == "a"])
    return {tuple(m[p] for p in perm): sign * c for m, c in comps.items()}

def _parity(seq):
    seq = list(seq)
    sign = 1
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            if seq[i] > seq[j]:
                sign = -sign
    return sign

@lru_cache(maxsize=None)
def reversal(n_c, n_a, restr, path, q2):
    """
    Adjoint relation used to obtain the densities of the type X = (n_c, n_a) from those of its adjoint type
    Y = (n_a, n_c):
        X_q(path)^dagger = sum_path' coef * Y_{-q}(path')  with Y's legs taken from X's slots perm
    i.e. Y slot j carries the orbital index of X slot perm[j].
    Returns (perm, {path': coef}).
    """
    kinds_x = layout_kinds(n_c, n_a)
    N = n_c + n_a
    # adjoint: legs reversed (Y slot j <- X slot N-1-j) and adjoint
    rev = {}
    for m, c in components(n_c, n_a, path, q2).items():
        phase, new = 1, []
        for slot in reversed(range(N)):
            p, _, m2 = _adjoint_leg(kinds_x[slot], m[slot])
            phase *= p
            new.append(m2)
        rev[tuple(new)] = rev.get(tuple(new), 0.0) + phase * c
    rev_slots = list(reversed(range(N)))      # Y-ordered list of X slots
    # bring the local pairs of the reversed operator into Y's layout
    local_x, _, _ = layout_pairs(n_c, n_a, restr)
    local_y, free_c_y, free_a_y = layout_pairs(n_a, n_c, restr)
    pos = {x: j for j, x in enumerate(rev_slots)}     # X slot -> position in the reversed operator
    local_rev = sorted((pos[a], pos[c]) for c, a in local_x)   # the X a slot becomes a c (and vice versa)
    used = set(x for pair in local_rev for x in pair)
    free_c_rev = [j for j in range(n_a) if j not in used]
    free_a_rev = [j for j in range(n_a, N) if j not in used]
    perm_rev = [None] * N                     # Y slot -> position in the reversed operator
    for (cy, ay), (cr, ar) in zip(local_y, local_rev):
        perm_rev[cy], perm_rev[ay] = cr, ar
    for y, r in zip(free_c_y, free_c_rev):
        perm_rev[y] = r
    for y, r in zip(free_a_y, free_a_rev):
        perm_rev[y] = r
    target = permute_components(rev, perm_rev, layout_kinds(n_a, n_c))
    coefs = decompose(target, n_a, n_c, restr, -q2, rank2s=[path_rank2(path)])
    perm = tuple(rev_slots[r] for r in perm_rev)
    return perm, coefs
