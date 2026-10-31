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

# The overlap-connected free density slots of the diagrams with integrals (ST, SU, SV) in the basis of the orbital
# pairs, without changing the diagram code.
#
# With the pair-compressed S of a dimer (S_01 = L R^T, S_10 = R L^T, F_0 = L, F_1 = R), every overlap factor connects
# two indices through the K pairs.  A label is put into the pair basis if it connects a free density slot of a
# pre-contraction (with S, T or V) with
#     - an explicit overlap s_ab(y, x): the explicit s becomes the identity (both of its labels in the pair basis) or the
#       remaining factor (F_a(y, k) or F_b(x, k)), or
#     - the free index of a pre-contraction with an inter-fragment S, which then keeps its pair axis instead of being
#       expanded with the other factor.
# In s01s01v1100, e.g., the pre-contraction ccaa0XXsr_V11rs(i0,j0,t,v,p,q) then has K x K instead of n_orb x n_orb
# entries on t and v, and so do the following contractions.  The pre-contractions are evaluated by
# precontract_pairs.pair_slot_precontract.  The plan of a diagram is found once by recording its accesses (plan).

import re
import numpy
from qode.math.tensornet import tl_tensor
from .fold_pairs import _Recorder

# the default of the setting supersys_info.pair_slots (get_leakage_metric_H, leakage_pair_slots): a number r uses the
# pair basis if r K <= n_orb (many pairs: slower, Be2 cc-pVTZ K = 19 of 30, but lower peak memory), True always, False
# never (diagrams as written, with the single-slot fold of fold_pairs.py)
PAIR_SLOTS = 2.5

_plans = {}

def _access_kind(attr, n_args):
    """
    [(kind, slot) for the arguments] of an access: "E" explicit s, "P" free density slot (slot number), "Q" free
    index of an inter-fragment S pre-contraction, None otherwise (state indices, integral indices, bare densities)
    """
    if re.fullmatch(r"s\d\d", attr):
        return [("E", None)] * 2 if attr[1] != attr[2] else [None] * n_args
    match = re.match(r"^([ca]+)(\d)([^_]*)_([STVU])(.*)$", attr)
    if match is None:
        return [None] * n_args
    rho_type, frag, idxstr, int_type, int_rest = match.groups()
    kinds = [None, None]
    n_free = idxstr.count("X")
    kinds += [("P", j) if int_type in ("S", "T", "V") else None for j in range(n_free)]
    int_frags = [c for c in int_rest if c.isdigit()]
    kinds += [("Q", None) if int_type == "S" and f != frag else None for f in int_frags]
    return kinds + [None] * (n_args - len(kinds))

def plan(contraction):
    """
    None or {access index: spec} for the diagram function contraction, with the accesses recorded once:
    spec ("P", attr, args, slots, pair_s) for a pre-contraction, ("E", attr, args, y_paired, x_paired) for an
    explicit s
    """
    if contraction in _plans:
        return _plans[contraction]
    result = None
    try:
        X = _Recorder()
        contraction(X)
        accesses = X.accesses
    except Exception:
        accesses = []
    uses = {}
    kinds = []
    for a, (attr, args) in enumerate(accesses):
        kinds.append(_access_kind(attr, len(args)))
        for pos, label in enumerate(args):
            if isinstance(label, str):
                uses.setdefault(label, []).append((a, pos))
    paired = set()
    for label, where in uses.items():
        if len(where) != 2:
            continue
        k = sorted(kinds[a][pos][0] if kinds[a][pos] is not None else "-" for a, pos in where)
        if k in (["E", "P"], ["P", "Q"]):
            paired.add(label)
    if paired:
        result = {}
        for a, (attr, args) in enumerate(accesses):
            ks = kinds[a]
            if re.fullmatch(r"s\d\d", attr) and ks[0] is not None:
                y, x = args
                if y in paired or x in paired:
                    result[a] = ("E", attr, args, y in paired, x in paired)
            elif any(k is not None and k[0] in ("P", "Q") for k in ks):
                slots = tuple(k[1] for k, l in zip(ks, args) if k is not None and k[0] == "P" and l in paired)
                pair_s = any(k is not None and k[0] == "Q" and l in paired for k, l in zip(ks, args))
                if slots or pair_s:
                    result[a] = ("P", attr, args, slots, pair_s)
    _plans[contraction] = result
    return result

class _PairCache(object):
    """the pre-contractions of pair_slot_precontract for given slots (for build_diagram._precontract_array)"""
    def __init__(self, pairs, slots, pair_s):
        self.pairs, self.slots, self.pair_s = pairs, slots, pair_s
    def __getitem__(self, label):
        pairs, slots, pair_s = self.pairs, self.slots, self.pair_s
        class by_frags(object):
            def __getitem__(self, frags):
                return pairs.get(label, frags if isinstance(frags, tuple) else (frags,), slots, pair_s)
        return by_frags()

class _PairProxy(object):
    """X with the accesses of the plan dispatched by their arguments"""
    def __init__(self, X, specs):
        self._X, self._specs = X, specs    # {attr: {args: callable returning the tensor}}
    def __getattr__(self, attr):
        if attr in ("_X", "_specs"):
            raise AttributeError(attr)
        if attr not in self._specs:
            return getattr(self._X, attr)
        by_args, X = self._specs[attr], self._X
        def access(*args):
            if args in by_args:
                return by_args[args]()(*args)
            return getattr(X, attr)(*args)
        return access

def pair_proxy(X, contraction, supersys_info):
    """X, or a proxy with the overlap-connected free density slots in the pair basis (see above)"""
    setting = getattr(supersys_info, "pair_slots", PAIR_SLOTS)
    if setting is False or X is None:
        return X
    cache = supersys_info.contract_cache
    S_factors = getattr(cache.rho_S, "S_factors", None)
    if S_factors is None or len(supersys_info.densities) != 2:
        return X
    if setting is not True:
        n_orb, K = S_factors[0, 1][0].shape
        if setting * K > n_orb:
            return X
    the_plan = plan(contraction)
    if not the_plan:
        return X
    from .build_diagram import _precontract_array
    from precontract_pairs import pair_slot_precontract
    pairs = getattr(cache, "pair_slots", None)
    if pairs is None or pairs.general is not cache.general:
        pairs = cache.pair_slots = pair_slot_precontract(supersys_info.densities, supersys_info.integrals, S_factors,
                                                          cache.general, cache.rho_S)
    subsystem = [frag for frag, _ in X._subsys_chgs]
    specs = {}
    for spec in the_plan.values():
        kind, attr, args = spec[:3]
        if kind == "E":
            y_paired, x_paired = spec[3:]
            a, b = (subsystem[int(c)] for c in attr[1:])
            F_a, F_b = pairs.factor[a], pairs.factor[b]
            if y_paired and x_paired:
                tensor = tl_tensor.init(numpy.eye(F_a.shape[1]))
            elif x_paired:
                tensor = tl_tensor.init(F_a)                 # s_ab(y, x) = F_a(y, k_x)
            else:
                tensor = tl_tensor.init(F_b)                 # s_ab(y, x) = F_b(x, k_y), as (x, y)
                specs.setdefault(attr, {})[args] = (lambda t=tensor: (lambda y, x: t(x, y)))
                continue
            specs.setdefault(attr, {})[args] = (lambda t=tensor: t)
        else:
            slots, pair_s = spec[3:]
            frag_indices = tuple(int(c) for c in filter(str.isdigit, attr))
            template = re.sub("\\d", "#", attr)
            def make(template=template, frag_indices=frag_indices, slots=slots, pair_s=pair_s):
                return _precontract_array(_PairCache(pairs, slots, pair_s), template, X._subsys_chgs,
                                          template.count("#"), X._n_frag)[frag_indices]
            specs.setdefault(attr, {})[args] = make
    return _PairProxy(X, specs)
