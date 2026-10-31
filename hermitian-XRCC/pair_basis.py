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

# The S terms (product-state overlap S2) of a dimer with a pair-compressed S in the basis of the orbital pairs.
#
# With the pair factors of the inter-fragment overlap (get_ints with pair_thresh), S_01 = L R^T and S_10 = R L^T
# (L: n_0 x K, R: n_1 x K, get_ints.pair_factors), every density slot of fragment 0 in an S diagram is contracted with
# L and every one of fragment 1 with R, since the S diagrams only contain inter-fragment overlaps.  The diagrams can
# therefore be evaluated with the densities transformed into the K pairs on every slot (rho~_0 = rho_0 x L ... x L,
# rho~_1 = rho_1 x R ... x R) and S~_01 = S~_10 = 1 (K x K), exactly.  Each density channel is transformed once
# (the large, streamed types by hummr directly, for all channels of a sector pair at once and without forming the
# density over all orbitals, hummr_dens/streamed.py _transformed), instead of being contracted with one S in every diagram,
# which leaves (n_orb)^3 K intermediates in the order-4 diagrams (Be2 cc-pVTZ, 20 x 20 states, K = 11: 0.9 GB per
# ccaa channel and 3 - 4 s per call, the S2^[4] build did not finish in 10 minutes).

import numpy
from collections import OrderedDict
from collections.abc import Mapping
from qode.util               import timer
from qode.util.dynamic_array import dynamic_array, wrap, cached
from qode.math.tensornet     import tl_tensor, evaluate
from get_ints                import tensorly_wrapper
from hummr_dens.streamed     import StreamedDensityType, contract_labelled

def _is_type(key):
    return isinstance(key, str) and len(key) > 0 and set(key) <= {"c", "a"}

def _slot_labels(n):
    return [f"slot{i}" for i in range(n)]

class _Channels(Mapping):
    """the channels of one charge pair of an explicit density type, transformed on access"""
    def __init__(self, by_channel, factor, n_slots):
        self.by_channel, self.factor, self.n_slots = by_channel, factor, n_slots
        self.memo = {}
    def __getitem__(self, channel):
        if channel not in self.memo:
            rho = self.by_channel[channel]
            slots = _slot_labels(self.n_slots)
            expr = rho(*([0, 1] + slots))
            for i, l in enumerate(slots):
                expr = expr @ self.factor(l, 2 + i)
            self.memo[channel] = evaluate(expr)
        return self.memo[channel]
    def __contains__(self, channel):
        return channel in self.by_channel
    def __iter__(self):
        return iter(self.by_channel)
    def __len__(self):
        return len(self.by_channel)

class _ChargePairs(Mapping):
    def __init__(self, by_chg, factor, n_slots):
        self.by_chg, self.factor, self.n_slots = by_chg, factor, n_slots
        self.memo = {}
    def __getitem__(self, chg_pair):
        if chg_pair not in self.memo:
            self.memo[chg_pair] = _Channels(self.by_chg[chg_pair], self.factor, self.n_slots)
        return self.memo[chg_pair]
    def __contains__(self, chg_pair):
        return chg_pair in self.by_chg
    def __iter__(self):
        return iter(self.by_chg)
    def __len__(self):
        return len(self.by_chg)

class _PairProvider(object):
    """
    the contractions of the streamed types (provider.contract) with the densities in the pair basis, which are kept
    in an LRU cache of at most MEMO_FRACTION of the provider's cache (they grow as K^4 for four operators: Be2 cc-pVTZ,
    20 x 20 states, 0.2 GB per ccaa channel for K = 16 at 2.6 A)
    """
    MEMO_FRACTION = 0.5
    def __init__(self, provider, factor):
        self.provider, self.factor = provider, factor
        self.memo, self.memo_bytes = OrderedDict(), 0
        budget = getattr(provider, "memory_budget", None)
        self.memo_limit = None if budget is None else self.MEMO_FRACTION * budget
    def contract(self, name, chg_pair, channel, rho_labels, operands):
        key = (name, tuple(chg_pair), tuple(channel))
        if key in self.memo:
            self.memo.move_to_end(key)
        else:
            slots = _slot_labels(len(name))
            self.memo[key] = self.provider.contract(name, chg_pair, channel, slots,
                                                    [(self.factor, [l, 2 + i]) for i, l in enumerate(slots)])
            self.memo_bytes += self.memo[key].nbytes
            while self.memo_limit is not None and self.memo_bytes > self.memo_limit and len(self.memo) > 1:
                self.memo_bytes -= self.memo.popitem(last=False)[1].nbytes
        labels = [0, 1] + list(rho_labels) + [l for _, ls in operands for l in ls]
        out = sorted({l for l in labels if isinstance(l, int)})
        return contract_labelled([(self.memo[key], [0, 1] + list(rho_labels))] + list(operands), out)

def pair_basis_densities(densities, factors):
    """the densities of every fragment (build_frag_densities) with all slots transformed with factors[m] (n_mo x K)"""
    out = []
    for dens_m, factor in zip(densities, factors):
        factor_t = tl_tensor.init(numpy.ascontiguousarray(factor))
        providers, new = {}, {}
        for key, value in dens_m.items():
            if isinstance(value, StreamedDensityType):
                if id(value.provider) not in providers:
                    providers[id(value.provider)] = _PairProvider(value.provider, numpy.ascontiguousarray(factor))
                new[key] = StreamedDensityType(providers[id(value.provider)], key)
            elif _is_type(key):
                new[key] = _ChargePairs(value, factor_t, len(key))
            else:
                new[key] = value
        out.append(new)
    return out

def has_pair_basis(densities, symm_ints):
    return getattr(symm_ints, "S_factors", None) is not None and len(densities) == 2

def pair_basis_S(densities, symm_ints):
    """(densities, integrals S~) of the S diagrams of a dimer in the pair basis of symm_ints.S_factors
    (has_pair_basis)"""
    L, R = symm_ints.S_factors[0, 1]
    identity = numpy.eye(L.shape[1])
    def block(i, j):
        if i == j:
            raise ValueError("pair basis: the S diagrams contain no fragment-diagonal overlap")
        return identity
    S = wrap(dynamic_array(block, symm_ints.S.ranges), [cached, tensorly_wrapper(timer())])
    dens = pair_basis_densities(densities, [L, R])
    return dens, S
