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

# Embedding of active-space densities (frozen core) into the full fragment MO space.
#
# All states have the n_core core orbitals (the first MOs) doubly occupied.  For a density element with some
# slots on core orbitals, the core operators are moved to the right of the active ones (a sign, since core and
# active operators anticommute and the core part has as many c's as a's) and evaluated on the closed-shell core,
#
#     <I| O |J> = sign * <I_act| O_act |J_act> <core| O_core |core>,
#
# which leaves an operator of a lower type on the active orbitals.  Since the core is a singlet, the rank and
# projection of the tensor are preserved, so the CG coefficients of the CG convention cancel and the embedded
# density is a linear combination of the active densities of the lower type (with the overlap <I|J> of the
# active parts, if all slots are on core orbitals).  The coefficients are derived numerically from the tensor
# components of spin_tensors.py.

import itertools
from functools import lru_cache

import numpy as np

from spin_tensors import components, paths, layout_pairs, layout_kinds, decompose, path_rank2


@lru_cache(maxsize=None)
def _core_operators(n_core):
    """Jordan-Wigner c/a of the core spin orbitals (2K + 0/1 for alpha/beta) and the doubly occupied core state"""
    n_so = 2 * n_core
    I, Z, lower = np.eye(2), np.diag([1., -1.]), np.array([[0., 1.], [0., 0.]])
    a = []
    for i in range(n_so):
        op = np.array([[1.]])
        for j in range(n_so):
            op = np.kron(op, Z if j < i else (lower if j == i else I))
        a.append(op)
    full = np.zeros(2 ** n_so)
    full[-1] = 1.0                            # all occupied (|1>  is the second basis state of each factor)
    return a, full

def _core_value(n_core, kinds, orbitals, m2s):
    """<core| op_1 ... op_n |core> for core operators (kind, orbital, m2 of the leg)"""
    a, full = _core_operators(n_core)
    v = full
    coef = 1.0
    for kind, K, m2 in reversed(list(zip(kinds, orbitals, m2s))):
        if kind == "c":                       # c_{+1/2} = c_alpha, c_{-1/2} = c_beta
            v = a[2 * K + (0 if m2 > 0 else 1)].T @ v
        else:                                 # at_{+1/2} = a_beta, at_{-1/2} = -a_alpha
            v = a[2 * K + (1 if m2 > 0 else 0)] @ v
            coef *= 1.0 if m2 > 0 else -1.0
    return coef * (full @ v)

def _parity(seq):
    sign, seq = 1, list(seq)
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            if seq[i] > seq[j]:
                sign = -sign
    return sign

@lru_cache(maxsize=None)
def core_terms(n_c, n_a, restr, path, q2, core_slots, core_orbitals, n_core):
    """
    Contribution of the core assignment (core_slots: slots of the layout on core orbitals core_orbitals) to the
    density of type (n_c, n_a), channel (restr, path).  Returns None (vanishes) or
        (type', restr', active_slots, {path': coef})
    where the active density of type' = (n_c', n_a') (legs in its layout order taken from the slots active_slots
    of the original layout, in that order) contributes with coef.
    """
    kinds = layout_kinds(n_c, n_a)
    N = n_c + n_a
    active = [s for s in range(N) if s not in core_slots]
    core_c = [s for s in core_slots if kinds[s] == "c"]
    core_a = [s for s in core_slots if kinds[s] == "a"]
    if len(core_c) != len(core_a):
        return None
    n_c_act, n_a_act = n_c - len(core_c), n_a - len(core_a)
    # sign of moving the core operators to the right (active order kept, core c's before core a's)
    sign = _parity(active + core_c + core_a)
    # canonical layout of the active type: surviving local pairs first, then the free c's and a's in order
    local, _, _ = layout_pairs(n_c, n_a, restr)
    surviving = [(c, a) for c, a in local if c in active and a in active]
    restr_act = len(surviving)
    local_act, free_c_act, free_a_act = layout_pairs(n_c_act, n_a_act, restr_act)
    used = set(x for pair in surviving for x in pair)
    free_c = [s for s in active if kinds[s] == "c" and s not in used]
    free_a = [s for s in active if kinds[s] == "a" and s not in used]
    slot_of = [None] * (n_c_act + n_a_act)       # active layout slot -> original slot
    for (ca, aa), (c, a) in zip(local_act, surviving):
        slot_of[ca], slot_of[aa] = c, a
    for x, s in zip(free_c_act, free_c):
        slot_of[x] = s
    for x, s in zip(free_a_act, free_a):
        slot_of[x] = s
    # sign of reordering the active operators into the active layout
    sign *= _parity([s for s in slot_of if kinds[s] == "c"]) * _parity([s for s in slot_of if kinds[s] == "a"])
    target = {}
    for m, c in components(n_c, n_a, path, q2).items():
        core_kinds = [kinds[s] for s in core_c + core_a]
        orbs = [core_orbitals[core_slots.index(s)] for s in core_c + core_a]
        val = _core_value(n_core, core_kinds, orbs, [m[s] for s in core_c + core_a])
        if abs(val) < 1e-14:
            continue
        key = tuple(m[s] for s in slot_of)
        target[key] = target.get(key, 0.0) + sign * c * val
    target = {k: v for k, v in target.items() if abs(v) > 1e-14}
    if not target:
        return None
    coefs = decompose(target, n_c_act, n_a_act, restr_act, q2, rank2s=[path_rank2(path)])
    if not coefs:
        return None
    return (n_c_act, n_a_act), restr_act, tuple(slot_of), coefs

def embed(dens, n_core, n_mo, overlaps=None, consume=False):
    """
    dens[name][(chg_bra, chg_ket)][(mult_bra, mult_ket, rank2, restr_ca, path)] = ndarray(bra, ket, active orbitals...)
    in the CG convention (lowering and self-adjoint types as returned by hummr).  Returns the densities over all
    n_mo MOs (core orbitals 0..n_core-1 doubly occupied).  overlaps[(chg, mult)] = <bra|ket> of the active parts
    (identity if not given, which requires the same bra and ket states).
    consume: every active density is removed from dens right after its embedding (to keep the peak memory down).
    This is possible, since the types are processed with decreasing number of operators and the core terms of a
    type only require active densities of types with (at least two) operators less.
    """
    if n_core == 0:
        return dens
    out = {}
    for name in sorted(dens, key=lambda name: -len(name)):
        by_chg = dens[name]
        n_c, n_a = name.count("c"), name.count("a")
        N = n_c + n_a
        for charge_pair, by_channel in by_chg.items():
            for channel in list(by_channel):
                x = by_channel[channel]
                mult_bra, mult_ket, rank2, restr, path = channel
                x = np.asarray(x)
                n_act = x.shape[2]
                q2 = mult_bra - mult_ket
                y = np.zeros(x.shape[:2] + (n_mo,) * N, dtype=x.dtype)
                y[(slice(None), slice(None)) + (slice(n_core, n_core + n_act),) * N] = x
                for n_core_slots in range(2, N + 1, 2):
                    for core_slots in itertools.combinations(range(N), n_core_slots):
                        for core_orbitals in itertools.product(range(n_core), repeat=n_core_slots):
                            term = core_terms(n_c, n_a, restr, path, q2, core_slots, core_orbitals, n_core)
                            if term is None:
                                continue
                            (n_c_act, n_a_act), restr_act, slot_of, coefs = term
                            name_act = "c" * n_c_act + "a" * n_a_act
                            acc = None
                            for path_act, coef in coefs.items():
                                if name_act == "":
                                    if charge_pair[0] != charge_pair[1] or mult_bra != mult_ket:
                                        continue
                                    if overlaps is not None:
                                        rho = np.asarray(overlaps[(charge_pair[0], mult_bra)])
                                    elif x.shape[0] == x.shape[1]:
                                        rho = np.eye(x.shape[0])
                                    else:
                                        raise ValueError("frozen-core embedding of different bra and ket state sets requires their overlaps")
                                else:
                                    key = (mult_bra, mult_ket, rank2, restr_act, path_act)
                                    try:
                                        rho = np.asarray(dens[name_act][charge_pair][key])
                                    except KeyError:
                                        raise KeyError(f"frozen-core embedding of {name} {charge_pair} {channel} requires the "
                                                       f"active density {name_act} {key}, which has not been computed")
                                acc = coef * rho if acc is None else acc + coef * rho
                            if acc is None:
                                continue
                            # place it: original slot slot_of[j] carries the active index j, core slots their orbital
                            index = [None] * N
                            for s, K in zip(core_slots, core_orbitals):
                                index[s] = K
                            # active axes of acc in the active layout order -> original slots
                            perm = np.argsort(slot_of)    # original-slot order of the active axes
                            acc = np.transpose(acc, (0, 1) + tuple(2 + int(j) for j in perm))
                            for s in sorted(slot_of):
                                index[s] = slice(n_core, n_core + n_act)
                            y[(slice(None), slice(None)) + tuple(index)] += acc
                out.setdefault(name, {}).setdefault(charge_pair, {})[channel] = y
                if consume:
                    del by_channel[channel], x
    # in the original order of the types
    return {name: out[name] for name in dens if name in out}
