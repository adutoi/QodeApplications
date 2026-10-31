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

# Pre-contractions of the densities with an orbital-pair compressed overlap (get_ints with pair_thresh).
#
# Every pre-contraction with S (labels "..._Sp#" or "..._S#p") contracts one density index of its own fragment m with
# an inter-fragment block, the other index stays free (orbitals of the other fragment n).  With the factors
# S_mn = L R^T (get_ints.pair_factors, K pairs) the density is contracted with the factor of its own side only
# (precontract.precontract with the factors as integrals), i.e. the free axis has K instead of n_orb entries (the cost
# of hummr's streamed kernel is proportional to the number of free tensor entries), and the K axis is expanded with
# the other factor afterwards, lazily (LAZY_EXPANSION) as a tensornet product, so that the contractions of the diagrams
# run over the K pairs as well (as for the explicit S factors, get_ints.factorized_wrapper).  The result equals the
# pre-contraction with the compressed S.  Be2 cc-pVDZ, 4.0 A (5 of 14 pairs above 0.02): XR[2] build 36 s -> 27 s
# (S2^[4] 4.0 s -> 1.9 s, 22 s / 1.6 s with 0.1), the uncompressed one takes 34 - 43 s; the remaining time is mostly
# the pre-contractions with V (not compressed) and hummr's kernel.

import numpy
from qode.util.dynamic_array import dynamic_array, wrap, cached
from qode.math.tensornet     import raw, tl_tensor
from qode.util               import timer
import precontract as precontract_module
from precontract import precontract, cached_if_small
from get_ints import tensorly_wrapper

# the K axis is expanded lazily, i.e. the pre-contraction is returned as the product of the factorized intermediate and
# the other factor, which the contractions of the diagrams order by the tensornet heuristic (False: expanded directly)
LAZY_EXPANSION = True

def _factor_integrals(S, S_factors, side):
    """
    integrals for precontract: the fragment-diagonal blocks of S, and for the inter-fragment block (a, b) the factor
    of the contracted side, L_ab (n_a x K) for side "first" (labels "..._Sp#": first index contracted) or R_ab^T
    (K x n_b) for side "second" (labels "..._S#p")
    """
    def rule(a, b):
        if a == b:
            return raw(S[a, b])
        L, R = S_factors[a, b]
        return L if side == "first" else numpy.ascontiguousarray(R.T)
    return wrap(dynamic_array(rule, S.ranges), [cached, tensorly_wrapper(timer())])

class _expanded(object):
    """the pre-contractions of one label and fragment pair with the factor of the contracted side, the free K axis
    (the last one) expanded with the factor of the other side"""
    def __init__(self, factorized, other_factor, owner, fragment):
        self.factorized, self.other_factor = factorized, other_factor
        self.storage = dynamic_array(cached_if_small(self._rule, owner, fragment), factorized.ranges)
    def _rule(self, *channel):
        result = self.factorized[channel]
        if result is None:
            return None
        if not LAZY_EXPANSION:
            return tl_tensor.init(numpy.tensordot(numpy.asarray(raw(result)), self.other_factor, axes=([-1], [1])))
        # the expansion as a (lazy) product, so that the contraction of the diagram can run over the K pairs as well
        result = tl_tensor.init(numpy.asarray(raw(result)))
        n = len(result.shape)
        return result(*(list(range(n - 1)) + ["k"])) @ tl_tensor.init(self.other_factor)(n - 1, "k")
    def __getitem__(self, channel):
        return self.storage[channel]

def precontract_S(densities, symm_ints, timings):
    """precontract(densities, symm_ints.S, timings), with the pair factors of a compressed S (symm_ints.S_factors,
    get_ints with pair_thresh) if available"""
    S_factors = getattr(symm_ints, "S_factors", None)
    if S_factors is None:
        return precontract(densities, symm_ints.S, timings)
    return pair_precontract(densities, symm_ints.S, S_factors, timings)

class pair_precontract(object):
    """drop-in replacement of precontract.precontract(densities, S, timings) for an S with pair factors"""
    def __init__(self, densities, S, S_factors, timings):
        self.n_frag = len(densities)
        self.S_factors = S_factors
        self.owner = object()
        self.dense = precontract(densities, S, timings)
        self.factorized = {side: precontract(densities, _factor_integrals(S, S_factors, side), timings) for side in ("first", "second")}
        self.storage = {}
    def __getitem__(self, label):
        if label not in self.storage:
            int_idxstr = label.split("_")[1][1:]
            if len(int_idxstr) == 2 and int_idxstr.count("#") == 1:
                side = "first" if int_idxstr[1] == "#" else "second"
                self.storage[label] = dynamic_array(cached(lambda m, n, label=label, side=side: self._rule(label, side, m, n)),
                                                    [range(self.n_frag)] * 2)
            else:
                self.storage[label] = self.dense[label]
        return self.storage[label]
    def _rule(self, label, side, m, n):
        # m: fragment of the density (contracted index), n: the one of the free index
        if m == n:
            return self.dense[label][m, n]
        if side == "first":
            _, other = self.S_factors[m, n]      # S_mn = L R^T, contracted with L, expanded with R
        else:
            other, _ = self.S_factors[n, m]      # S_nm = L R^T, contracted with R^T, expanded with L
        return _expanded(self.factorized[side][label][m, n], other, self.owner, m)
    def release(self, fragment, transition):
        """drops the large intermediates of fragment for the transition ((chg_i, mult_i), (chg_j, mult_j))"""
        precontract_module.release_large(self.owner, fragment, transition)
        self.dense.release(fragment, transition)
        for cache in self.factorized.values():
            cache.release(fragment, transition)



# Pre-contractions with a fragment-factorized V (only used for the order-2 terms of the leakage increment, see
# get_xr_result_spin_adapt.get_leakage_metric_H).
#
# Every inter-fragment block of V is split as the matrix (axes on the density's fragment) x (axes on the other
# fragment) and factorized by a singular value decomposition, keeping the singular values above v_thresh (as the
# orbital pairs of S, get_ints.pair_factors), V = sum_k A_k B_k.  The density is contracted with A only, i.e. the free
# axes on the other fragment (n_orb or n_orb^2 entries) are replaced by the rank axis, and B is attached lazily (as
# for S).  Blocks with all indices on the density's fragment are contracted as before, and so are the blocks whose
# kept rank exceeds V_FACTOR_MAX_FRACTION of the full one (no saving, e.g. Be2 cc-pVTZ at 3 A with v_thresh = 1e-4:
# rank 30 of 30 for the blocks with one index on the other fragment).
V_FACTOR_MAX_FRACTION = 0.5

import diagrammatic_expansion
from qode.math.tensornet       import evaluate
from diagrams.build_diagram    import resolve_density
from hummr_dens.streamed       import StreamedDensityType

def _split_factors(array, own_axes, thresh):
    """
    (A, B) with array = sum_k A[k, own axes] B[k, other axes] (truncated singular value decomposition of the matrix
    own axes x other axes, singular values above thresh, at least one); A and B carry sqrt(sigma) each
    """
    other_axes = [a for a in range(array.ndim) if a not in own_axes]
    own_shape = [array.shape[a] for a in own_axes]
    other_shape = [array.shape[a] for a in other_axes]
    M = numpy.transpose(array, list(own_axes) + other_axes).reshape(int(numpy.prod(own_shape)), int(numpy.prod(other_shape)))
    U, s, Vt = numpy.linalg.svd(M, full_matrices=False)
    k = max(1, int(numpy.sum(s > thresh)))
    root = numpy.sqrt(s[:k])
    A = (U[:, :k] * root).T.reshape([k] + own_shape)
    B = (Vt[:k].T * root).T.reshape([k] + other_shape)
    return A, B

# Folding of an S factor into a pre-contraction with V (see diagrams/fold_pairs.py): with S_mn = L R^T (pair factors)
# and a free density slot x of the pre-contraction, which the diagram only contracts with the explicit S, the factor of
# the slot's side is contracted into the pre-contraction as well, i.e. the density is contracted with the outer product
# V' (x) factor (V' the own-fragment factor of V, see above, or the complete V block), and the slot's axis carries the K
# pairs instead of the n_orb orbitals.  The work of hummr's kernel per density element grows by K, but the scattering
# into the n_orb^2 rows of two free slots, which dominates for small K, shrinks: for ccaaa with two free slots the
# kernel is 1.6 - 2.2 times faster in Be2 cc-pVDZ (K = 5, n_orb = 14), but 10 - 15 times slower in cc-pVTZ (K = 19,
# n_orb = 30, the folded tensor of 3.7 GB built pair by pair), for cccaaa it is 15 % slower, and folding both slots is
# 2 - 3 times slower.  So only one slot of the types in diagrams.fold_pairs.FOLD_TYPES is folded, and only if
# K FOLD_MAX_RATIO <= n_orb and the folded tensor (|V'| n_orb K entries) does not exceed fold_max_bytes() (one kernel
# call; FOLD_MAX_FRACTION of the cache of the large intermediates, which apply_memory_budget sets to a quarter of the
# budget, at least FOLD_MIN_BYTES); otherwise the unfolded pre-contraction is contracted with the factor afterwards.
FOLD_MAX_RATIO = 2.5
FOLD_MIN_BYTES = 64 * 2**20
FOLD_MAX_FRACTION = 1 / 16

def fold_max_bytes():
    return max(FOLD_MIN_BYTES, int(FOLD_MAX_FRACTION * precontract_module.LARGE_CACHE_BYTES))

class V_factorized_precontract(object):
    """drop-in replacement of precontract.precontract(densities, integrals, timings) (T, U, V), with the
    pre-contractions with the inter-fragment blocks of V done with their fragment factors (singular values above
    v_thresh), and pre-contractions with an S factor folded in (folded, see diagrams/fold_pairs.py)"""
    def __init__(self, densities, integrals, timings, v_thresh):
        self.densities, self.V, self.timings, self.v_thresh = densities, integrals.V, timings, v_thresh
        self.n_frag = len(densities)
        self.base = precontract(densities, integrals, timings)
        self.owner = object()
        self.factors, self.storage, self.folds = {}, {}, {}
        self.ranks = {}    # (blocks, own axes) -> kept / full rank (information)
    def __getitem__(self, label):
        if label not in self.storage:
            if label.split("_")[1][0] == "V":
                self.storage[label] = dynamic_array(cached(lambda *frags, label=label: self._rule(label, frags)),
                                                    [range(self.n_frag)] * label.count("#"))
            else:
                self.storage[label] = self.base[label]
        return self.storage[label]
    def _factors(self, blocks, own_axes):
        key = (blocks, own_axes)
        if key not in self.factors:
            array = numpy.asarray(raw(self.V[blocks]))
            self.factors[key] = _split_factors(array, own_axes, self.v_thresh)
            n_own = int(numpy.prod([array.shape[a] for a in own_axes]))
            self.ranks[key] = (self.factors[key][0].shape[0], min(n_own, array.size // n_own))
        return self.factors[key]
    def _setup(self, label, frags):
        """the index bookkeeping of precontract.precontract: density type, density slot labels (int: free), integral
        labels and blocks, the density's fragment and the axes of the integral block on it"""
        rho_label, int_label = label.split("_")
        rho_type, rho_idxstr = rho_label.split("#")
        free_idx, block_idx = 2, 0
        rho_indices = []
        for idx in rho_idxstr:
            if idx == "X":
                rho_indices.append(free_idx)
                free_idx += 1
            else:
                rho_indices.append(idx)
        int_indices, blocks = [], []
        for idx in int_label[1:]:
            if idx == "#":
                block_idx += 1
                int_indices.append(free_idx)
                blocks.append(frags[block_idx])
                free_idx += 1
            else:
                int_indices.append(idx)
                blocks.append(frags[0])
        m = frags[0]
        own_axes = tuple(a for a, b in enumerate(blocks) if b == m)
        return rho_type, rho_indices, int_indices, tuple(blocks), m, own_axes, free_idx
    def _operands(self, label, frags):
        """(own operand (array, labels), other operand (tl_tensor, labels) or None) of the V block: the fragment
        factors for an inter-fragment block (v_thresh), otherwise the complete block"""
        rho_type, rho_indices, int_indices, blocks, m, own_axes, free_idx = self._setup(label, frags)
        if self.v_thresh is None or len(own_axes) == len(blocks):
            return (numpy.asarray(raw(self.V[blocks])), list(int_indices)), None
        A, B = self._factors(blocks, own_axes)
        kept, full = self.ranks[(blocks, own_axes)]
        if kept > V_FACTOR_MAX_FRACTION * full:
            return (numpy.asarray(raw(self.V[blocks])), list(int_indices)), None
        # the rank axis: the label after the free density slots (the free labels of the other fragment's V axes are
        # not on the own operand; with them skipped the density contraction would have a gap in its free labels)
        rank = 2 + sum(1 for l in rho_indices if isinstance(l, int))
        own_op = (numpy.ascontiguousarray(numpy.moveaxis(A, 0, -1)), [int_indices[a] for a in own_axes] + [rank])
        other_axes = [a for a in range(len(blocks)) if a not in own_axes]
        other_op = (tl_tensor.init(numpy.ascontiguousarray(numpy.moveaxis(B, 0, -1))), [int_indices[a] for a in other_axes] + ["k"])
        return own_op, other_op
    def _contract_rule(self, label, frags, rho_indices, own_labels, operand_fn, other_op):
        """dynamic_array over the channels of the pre-contraction of the density with the operand(s) of operand_fn
        (run(operand) -> raw result, see _run), with other_op (lazy product over the rank axis "k", the last of
        own_labels) attached"""
        rho_type, _, _, _, m, _, _ = self._setup(label, frags)
        Dchg = rho_type.count("a") - rho_type.count("c")
        def contract(chg_i, chg_j, *channel):
            if chg_i - chg_j != Dchg:
                return None
            self.timings.start()
            try:
                result = operand_fn(lambda operand, rho_idx: self._run(label, m, rho_type, rho_idx, operand, chg_i, chg_j, channel))
            except KeyError as err:
                raise RuntimeError(f"missing density {rho_type} for charges {(chg_i, chg_j)} and channel {channel[:5]}: {err}")
            self.timings.record(label)
            result = tl_tensor.init(numpy.asarray(result))
            if other_op is None:
                return result
            labels = sorted(l for l in [0, 1] + rho_indices + own_labels[:-1] if isinstance(l, int))
            return result(*(labels + ["k"])) @ other_op[0](*other_op[1])
        base = self.base[label][frags if len(frags) > 1 else frags[0]]
        return dynamic_array(cached_if_small(contract, self.owner, m), base.ranges)
    def _run(self, label, m, rho_type, rho_indices, operand, chg_i, chg_j, channel):
        """the density of fragment m contracted with one operand (array, labels); axes: the integer labels sorted"""
        densities_m = self.densities[m]
        if isinstance(densities_m[rho_type], StreamedDensityType):
            positions, sign = dict(channel[5])[label]
            canonical = [rho_indices[p] for p in positions]
            result = densities_m[rho_type].provider.contract(rho_type, (chg_i, chg_j), tuple(channel[:5]), canonical, [operand])
            if sign != 1:
                result *= sign
            return result
        rho = resolve_density(densities_m[rho_type][chg_i, chg_j], channel, label)
        return raw(evaluate(rho(*([0, 1] + rho_indices)) @ tl_tensor.init(operand[0])(*operand[1])))
    def _rule(self, label, frags):
        rho_type, rho_indices, int_indices, blocks, m, own_axes, free_idx = self._setup(label, frags)
        own_op, other_op = self._operands(label, frags)
        if other_op is None:
            return self.base[label][frags if len(frags) > 1 else frags[0]]    # complete block (fragment-local or full rank)
        return self._contract_rule(label, frags, rho_indices, own_op[1], lambda run: run(own_op, rho_indices), other_op)
    def folded(self, label, frags, slot, factor_key, factor):
        """
        dynamic_array over the channels of the pre-contraction of label (fragments frags), with its free density slot
        number slot (counting the X of the label) contracted with factor (n_orb x K) as well, i.e. the axis of the
        slot carries the K entries of the factor; factor_key identifies the factor (cache)
        """
        key = (label, frags, slot, factor_key)
        if key not in self.folds:
            rho_type, rho_indices, int_indices, blocks, m, own_axes, free_idx = self._setup(label, frags)
            own_op, other_op = self._operands(label, frags)
            x = [l for l in rho_indices if isinstance(l, int)][slot]
            folded_rho = ["f" if l == x else l for l in rho_indices]
            out = sorted(l for l in [0, 1] + rho_indices + own_op[1] if isinstance(l, int))
            axis = out.index(x)
            n_x, K = factor.shape
            fold = K * FOLD_MAX_RATIO <= n_x and own_op[0].size * n_x * K * 8 <= fold_max_bytes()
            def operand_fn(run):
                if not fold:
                    result = numpy.asarray(run(own_op, rho_indices))
                    return numpy.moveaxis(numpy.tensordot(result, factor, axes=([axis], [0])), -1, axis)
                return run((numpy.multiply.outer(own_op[0], factor), own_op[1] + ["f", x]), folded_rho)
            self.folds[key] = self._contract_rule(label, frags, rho_indices, own_op[1], operand_fn, other_op)
        return self.folds[key]
    def release(self, fragment, transition):
        precontract_module.release_large(self.owner, fragment, transition)
        self.base.release(fragment, transition)

def V_factorized_blocks(densities, integrals, diagrams, contract_cache, timings, precon_timings, target_multiplicity, v_thresh):
    """diagrammatic_expansion.blocks with the pre-contractions with V done by V_factorized_precontract"""
    blocks = diagrammatic_expansion.blocks(densities=densities, integrals=integrals, diagrams=diagrams, contract_cache=contract_cache,
                                           timings=timings, precon_timings=precon_timings, target_multiplicity=target_multiplicity)
    blocks._supersys_info.contract_cache.general = V_factorized_precontract(densities, integrals, precon_timings, v_thresh)
    return blocks



# Pre-contractions with free density slots in the basis of the orbital pairs (used by diagrams/pair_slots.py).
#
# With S_01 = L R^T (pair factors), a free density slot whose label only meets an overlap factor in the diagram (an
# explicit s, or the free index of a pre-contraction with S) can carry the K pairs instead of the orbitals: the slot of
# fragment m is contracted with F_m (F_0 = L, F_1 = R), the explicit s becomes the identity or the remaining factor,
# and the pre-contraction with S keeps its pair axis instead of being expanded.  The large density types are
# transformed by hummr on these slots (the others with the identity) and then contracted with the integrals
# (StreamedDensities.contract_transformed), if at least half of their slots are in the pair basis (otherwise hummr's
# transformation builds most of the density over the orbitals: Be2 cc-pVDZ, 1 of 5 slots, slower than the kernel by
# far) and the transformed density fits (transform_fits); otherwise the pre-contraction is evaluated as before and
# contracted with the factors afterwards (for one slot of the types of fold_pairs.FOLD_TYPES with few pairs, folded
# into the kernel).

def _parse_label(label, frags):
    """(rho_type, rho_indices, int_type, int_indices, blocks): the index bookkeeping of precontract.precontract for
    the pre-contraction label (one density, one integral of type S, T or V) of the fragments frags"""
    rho_label, int_label = label.split("_")
    rho_type, rho_idxstr = rho_label.split("#")
    free_idx, block_idx = 2, 0
    rho_indices = []
    for idx in rho_idxstr:
        if idx == "X":
            rho_indices.append(free_idx)
            free_idx += 1
        else:
            rho_indices.append(idx)
    int_indices, blocks = [], []
    for idx in int_label[1:]:
        if idx == "#":
            block_idx += 1
            int_indices.append(free_idx)
            blocks.append(frags[block_idx])
            free_idx += 1
        else:
            int_indices.append(idx)
            blocks.append(frags[0])
    return rho_type, rho_indices, int_label[0], int_indices, tuple(blocks)

class pair_slot_precontract(object):
    """
    get(label, frags, slots, pair_s): dynamic_array over the channels of the pre-contraction label of the fragments
    frags with the free density slots number slots (counting the X of the label) in the pair basis, and for a
    pre-contraction with S its free index in the pair basis as well (pair_s, i.e. not expanded with the other factor)
    general, rho_S: the pre-contraction caches of the blocks (fallback, factorized V of V_factorized_precontract)
    """
    def __init__(self, densities, integrals, S_factors, general, rho_S):
        self.densities, self.integrals, self.general, self.rho_S = densities, integrals, general, rho_S
        L, R = S_factors[0, 1]
        self.factor = {0: numpy.ascontiguousarray(L), 1: numpy.ascontiguousarray(R)}
        self.owner, self.storage = object(), {}
        self.timings = timer()
        self.calls = {"transformed": 0, "fallback": 0}
    def get(self, label, frags, slots, pair_s):
        key = (label, frags, slots, pair_s)
        if key not in self.storage:
            self.storage[key] = self._rule(label, frags, slots, pair_s)
        return self.storage[key]
    def _fallback_array(self, label, frags, pair_s):
        """the pre-contraction as before (channel array), for pair_s with the pair axis of S not expanded"""
        if label.split("_")[1][0] != "S":
            return self.general[label][frags if len(frags) > 1 else frags[0]]
        if pair_s:
            int_idxstr = label.split("_")[1][1:]
            side = "first" if int_idxstr[1] == "#" else "second"
            return self.rho_S.factorized[side][label][frags[0], frags[1]]
        return self.rho_S[label][frags]
    def _rule(self, label, frags, slots, pair_s):
        rho_type, rho_indices, int_type, int_indices, blocks = _parse_label(label, frags)
        m = frags[0]
        F_m = self.factor[m]
        free = [l for l in rho_indices if isinstance(l, int)]
        folded = [free[j] for j in slots]
        out = sorted(l for l in [0, 1] + rho_indices + int_indices if isinstance(l, int))
        # the slot matrices {density slot label: (F, output label)} and the integral operand
        matrices = {x: (F_m, x) for x in folded}
        # other: (tensor, labels, label of the result axis it is attached over) for a lazily attached factor
        other, res_axes, operands = None, out, []
        if int_type == "S":
            # S_ab(i, j) = sum_k F_a(i, k) F_b(j, k): the contracted index u on the density, the free one t
            free_pos = [p for p, l in enumerate(int_indices) if isinstance(l, int)]
            if len(free_pos) == 1 and blocks[0] != blocks[1]:
                t, u = int_indices[free_pos[0]], int_indices[1 - free_pos[0]]
                matrices[u] = (self.factor[blocks[1 - free_pos[0]]], t)
                if not pair_s:
                    other = (tl_tensor.init(self.factor[blocks[free_pos[0]]]), [t, "k"], t)
            else:
                matrices = None
        elif int_type in ("T", "V"):
            if int_type == "V" and hasattr(self.general, "_operands"):
                own, other_op = self.general._operands(label, frags)    # (factorized V, see V_factorized_precontract)
            else:
                own, other_op = (numpy.asarray(raw(getattr(self.integrals, int_type)[blocks])), list(int_indices)), None
            if other_op is not None:
                res_axes = sorted(l for l in [0, 1] + rho_indices + own[1] if isinstance(l, int))
                other = (other_op[0], other_op[1], own[1][-1])
            operands = [own]
        else:
            matrices = None
        fallback = self._fallback_array(label, frags, pair_s)
        Dchg = rho_type.count("a") - rho_type.count("c")
        from diagrams.fold_pairs import FOLD_TYPES
        kernel_fold = None
        if (len(folded) == 1 and int_type == "V" and hasattr(self.general, "folded") and rho_type in FOLD_TYPES
                and F_m.shape[1] * FOLD_MAX_RATIO <= F_m.shape[0]):
            kernel_fold = self.general.folded(label, frags, slots[0], ("pair", m), F_m)
        def transform_after(result):
            """the factors contracted into the folded axes of a result over the orbitals"""
            result = numpy.asarray(raw(evaluate(result) if not isinstance(result, numpy.ndarray) else result))
            for x in folded:
                axis = out.index(x)
                result = numpy.moveaxis(numpy.tensordot(result, F_m, axes=([axis], [0])), -1, axis)
            return result
        def contract(chg_i, chg_j, *channel):
            if chg_i - chg_j != Dchg:
                return None
            densities_m = self.densities[m]
            rho = densities_m[rho_type]
            self.timings.start()
            if matrices is not None and isinstance(rho, StreamedDensityType) and 2 * len(matrices) >= len(rho_type):
                provider = rho.provider
                positions, sign = dict(channel[5])[label]
                canonical = [rho_indices[p] for p in positions]
                dims = [matrices[l][0].shape[1] if l in matrices else provider.n_act for l in canonical]
                if provider.transform_fits(rho_type, (chg_i, chg_j), tuple(channel[:5]), dims):
                    mats = {l: F for l, (F, _) in matrices.items()}
                    renamed = {l: new for l, (_, new) in matrices.items()}
                    result = provider.contract_transformed(rho_type, (chg_i, chg_j), tuple(channel[:5]), canonical,
                                                           {l: (mats[l], renamed[l]) for l in mats}, operands)
                    if sign != 1:
                        result *= sign
                    self.calls["transformed"] += 1
                    return self._finish(result, res_axes, other)
            elif matrices is not None and not isinstance(rho, StreamedDensityType):
                try:
                    rho_c = resolve_density(rho[chg_i, chg_j], channel, label)
                except KeyError:
                    return None
                labels = [f"pair{l}" if l in matrices else l for l in rho_indices]
                expr = rho_c(*([0, 1] + labels))
                for op, op_labels in operands:
                    expr = expr @ tl_tensor.init(op)(*op_labels)
                for l, (F, new) in matrices.items():
                    expr = expr @ tl_tensor.init(F)(f"pair{l}", new)
                self.calls["transformed"] += 1
                return self._finish(numpy.asarray(raw(evaluate(expr))), res_axes, other)
            # fallback: the pre-contraction as before, then the factors (one slot of the fold types with few pairs:
            # folded into hummr's kernel, see V_factorized_precontract.folded)
            self.calls["fallback"] += 1
            if kernel_fold is not None:
                return kernel_fold[(chg_i, chg_j) + tuple(channel)]
            base = fallback[(chg_i, chg_j) + tuple(channel)]
            if base is None:
                return None
            return tl_tensor.init(transform_after(base)) if folded else base
        return dynamic_array(cached_if_small(contract, self.owner, m), getattr(fallback, "storage", fallback).ranges)
    def _finish(self, result, axes, other):
        """the result (axes: its labels) as a tensor, with the other factor (of S or of the factorized V) attached
        lazily over the pair or rank axis"""
        result = tl_tensor.init(numpy.asarray(result))
        if other is None:
            return result
        factor, factor_labels, over = other
        return result(*["k" if l == over else l for l in axes]) @ factor(*factor_labels)
