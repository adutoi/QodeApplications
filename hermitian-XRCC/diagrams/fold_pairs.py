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

# Folding of an explicit S factor into a pre-contraction with V, without changing the diagram code.  Superseded by
# pair_slots.py (all overlap-connected slots in the pair basis), which uses the fold below only as the evaluation of
# a pre-contraction with one such slot (pair_slot_precontract); fold_proxy only applies where pair_slots.py does not.
#
# With the pair-compressed S (get_ints with pair_thresh, S_ab = L R^T with K pairs), a diagram such as
#
#     X.cca0tXX_St1(i0,j0,v,r,u) @ X.ccaaa1pqXXs_Vpq0s(i1,j1,w,u,r) @ X.s01(v,w)
#
# contracts the free density slot w of the pre-contraction with V only with the explicit s01(v,w) = L(v,k) R(w,k).
# The pre-contraction is then evaluated with R(w,k) folded in (precontract_pairs.V_factorized_precontract.folded,
# the axis of w carries the K pairs), and s01(v,w) is replaced by L(v,k), with k under the label w, so that the
# diagram code computes the same contraction.  The plan of a diagram (which access is folded over which slot) is found
# once by recording its accesses (fold_plan).  Only one slot of the density types in FOLD_TYPES is folded, and only
# for few pairs (FOLD_MAX_RATIO), see precontract_pairs.py for the timings.  Requires the V_factorized_precontract as
# the general pre-contraction cache (precontract_pairs.V_factorized_blocks) and the pair factors of S
# (precontract_pairs.pair_precontract).

import re
from qode.math.tensornet import tl_tensor

FOLD = True                # False: no folding
# density types whose pre-contractions with V get one S factor folded in: ccaaa and its adjoint cccaa (which hummr's
# kernel evaluates as ccaaa, hummr_dens/streamed.py), not cccaaa (slower, see precontract_pairs.py)
FOLD_TYPES = ("ccaaa", "cccaa")

_plans = {}

class _Dummy(object):
    def __matmul__(self, other):   return self
    def __rmatmul__(self, other):  return self
    def _raw(self):                return 0.0

class _Recorder(object):
    """stands in for X (build_diagram.frag_resolve) and records the accesses (attribute, arguments)"""
    def __init__(self):
        self.accesses = []
    def __getattr__(self, attr):
        if attr == "accesses":
            raise AttributeError(attr)
        if attr.startswith("n_j") or attr in ("Dchg", "Dmult", "n_frag"):
            return 0
        if attr == "n_states":
            return ((1, 1), (1, 1), (1, 1))
        def call(*args):
            self.accesses.append((attr, args))
            return _Dummy()
        return call

def _rho_part(attr):
    """(density type, fragment digit, index string) of a pre-contraction attribute, e.g. ccaaa1pqXXs_Vpq0s"""
    match = re.match(r"^([ca]+)(\d)([^_]*)_", attr)
    return match.groups() if match else None

def fold_plan(contraction):
    """
    None or (attribute of the pre-contraction with V, slot number among its free density slots, attribute of the
    explicit S, position (0, 1) of the slot's label in the arguments of the S) for the diagram function contraction
    """
    if contraction in _plans:
        return _plans[contraction]
    plan = None
    try:
        X = _Recorder()
        contraction(X)
        accesses = X.accesses
    except Exception:
        accesses = []
    for attr, args in accesses:
        rho = _rho_part(attr)
        if rho is None or "_V" not in attr or rho[0] not in FOLD_TYPES:
            continue
        n_free = rho[2].count("X")
        if n_free < 2:
            continue
        for slot, x in enumerate(args[2:2 + n_free]):
            uses = [(a, ar) for a, ar in accesses if x in ar]
            s_uses = [(a, ar) for a, ar in uses if re.fullmatch(r"s\d\d", a)]
            if len(uses) == 2 and len(s_uses) == 1:
                s_attr, s_args = s_uses[0]
                plan = (attr, slot, s_attr, s_args.index(x))
                break
        if plan is not None:
            break
    _plans[contraction] = plan
    return plan

class _FoldedCache(object):
    """the pre-contractions of one label with the S factor folded into its slot (for build_diagram._precontract_array)"""
    def __init__(self, general, slot, factor_of):
        self.general, self.slot, self.factor_of = general, slot, factor_of
    def __getitem__(self, label):
        general, slot, factor_of = self.general, self.slot, self.factor_of
        class by_frags(object):
            def __getitem__(self, frags):
                frags = frags if isinstance(frags, tuple) else (frags,)
                factor_key, factor = factor_of(frags[0])
                return general.folded(label, frags, slot, factor_key, factor)
        return by_frags()

class _FoldProxy(object):
    """X with the folded pre-contraction and the remaining factor of the explicit S"""
    def __init__(self, X, overrides):
        self._X, self._overrides = X, overrides
    def __getattr__(self, attr):
        if attr in ("_X", "_overrides"):
            raise AttributeError(attr)
        if attr in self._overrides:
            return self._overrides[attr]
        return getattr(self._X, attr)

def fold_proxy(X, contraction, supersys_info):
    """X, or a proxy with an S factor folded into one pre-contraction with V (see above), if possible"""
    if not FOLD or X is None:
        return X
    general = supersys_info.contract_cache.general
    S_factors = getattr(supersys_info.contract_cache.rho_S, "S_factors", None)
    if S_factors is None or not hasattr(general, "folded"):
        return X
    plan = fold_plan(contraction)
    if plan is None:
        return X
    from .build_diagram import _precontract_array
    p_attr, slot, s_attr, s_pos = plan
    subsystem = [frag for frag, _ in X._subsys_chgs]
    a, b = (subsystem[int(c)] for c in s_attr[1:])        # absolute fragments of the explicit S block
    L, R = S_factors[a, b]
    folded_factor, other_factor = (R, L) if s_pos == 1 else (L, R)
    from precontract_pairs import FOLD_MAX_RATIO
    if folded_factor.shape[1] * FOLD_MAX_RATIO > folded_factor.shape[0]:
        return X    # too many pairs for the fold to pay off (see precontract_pairs.py)
    def factor_of(m):
        if m != (b if s_pos == 1 else a):
            raise RuntimeError(f"fold_pairs: the folded slot of {p_attr} is not on the fragment of the S factor")
        return (a, b, s_pos), folded_factor
    frag_indices = tuple(int(c) for c in filter(str.isdigit, p_attr))
    template = re.sub("\\d", "#", p_attr)
    folded = _precontract_array(_FoldedCache(general, slot, factor_of), template, X._subsys_chgs, template.count("#"),
                                X._n_frag)[frag_indices]
    other = tl_tensor.init(other_factor)
    # s(y, x) -> the other factor, its pair axis under the label x of the folded slot
    s_fn = (lambda y, x: other(y, x)) if s_pos == 1 else (lambda x, z: other(z, x))
    return _FoldProxy(X, {p_attr: folded, s_attr: s_fn})
