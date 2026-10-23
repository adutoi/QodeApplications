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
Streamed evaluation of the contractions of the large density types (4 and more operators, e.g. ccaa) with the
integrals (the pre-contractions of hermitian-XRCC/precontract.py), without keeping these densities.

A contraction of a density of the Python convention (CG convention, embedded into the full MO space with the frozen
core, reversed partners for the raising / mult_bra > mult_ket channels) is resolved into contractions of the active
densities as hummr returns them (lowering and self-adjoint types, stretched elements):
    - reversed channels via the adjoint type (dens_util.reversed_densities_spin_adapt),
    - the frozen core via the active part plus the core terms (frozen_core.embed), contracted instead of placed,
      which require the lower active densities,
    - the active densities of the large types are built by hummr for one bra x ket sector pair at a time (all
      channels) and kept in an LRU cache up to memory_budget / 2 bytes (the other half for the density engines hummr keeps); the small types (a, aa, ca, caa) are kept.
The active contractions of the large types are done by hummr directly (CIManager::contractGeneralDensitySector,
only the nonzero blocks per intermediate CSF block, no density is formed) if there is a single tensor and no index
repeats; otherwise (or with kernel=False) hummr builds the channels explicitly (stage 1). With a memory budget, the
contractions with outputs about as large as the density use the cached explicit densities as well (_prefer_cached).
"""

from __future__ import annotations

import itertools
from collections import OrderedDict

import numpy as np
import opt_einsum as oe

from dens_util import adjoint_name
from spin_tensors import reversal, cg2

STREAMED_MIN_OPS = 4    # types with at least this many operators are streamed, the others kept explicitly


class StreamedDensityType:
    """stands in for the densities of one large type in the density dictionary of a fragment (see precontract.py)"""
    def __init__(self, provider, name):
        self.provider, self.name = provider, name


def _letters():
    return iter("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

def contract_labelled(operands, out_labels):
    """einsum over [(array, labels)] with arbitrary hashable labels, output in the order of out_labels"""
    mapping, letters = {}, _letters()
    def letter(label):
        if label not in mapping:
            mapping[label] = next(letters)
        return mapping[label]
    subscripts = ",".join("".join(letter(l) for l in labels) for _, labels in operands)
    subscripts += "->" + "".join(letter(l) for l in out_labels)
    return oe.contract(subscripts, *[array for array, _ in operands], optimize="greedy")


class StreamedDensities:
    """
    Provider of the contractions of the large density types of one fragment for one set of bra and ket states.
    fragment:      the HummrXRFragment (hummr engine, frozen core)
    bra/ket:       states[chg][mult] = list of CI vectors
    active_small:  {swapped: active densities of the small types} (CG convention, active space, as hummr returns
                   them), for the bra/ket states (swapped False) and the ket/bra states (swapped True)
    overlaps:      {swapped: {(chg, mult): <bra|ket>}} for the core terms (None: identity, same bra and ket states)
    """
    def __init__(self, fragment, bra, ket, xr_level, active_small, overlaps=None, memory_budget=None, kernel=True):
        from hummr_python import HummrForXRMain
        from hummr_dens.pybind_interface import to_cg_convention, cg_factor
        self._to_cg, self._cg_factor = to_cg_convention, cg_factor
        self.kernel = kernel
        self.calls = {"kernel": 0, "explicit": 0}    # active contractions of the large types
        self.fragment, self.bra, self.ket = fragment, bra, ket
        self.n_core, self.n_act = fragment.n_core, fragment.n_act
        self.n_mo = fragment.orbitals.shape[1]
        self.active_small, self.overlaps = active_small, overlaps
        # the budget is shared by the explicit densities cached here and the density engines hummr keeps between
        # the calls (ket sides, block operators, normal-ordering rest densities; without a budget hummr's default)
        self.memory_budget = None if memory_budget is None else memory_budget / 2
        if memory_budget is not None:
            fragment._engine.set_density_cache_limit(int(memory_budget / 2))
        self._cache, self._cache_bytes = OrderedDict(), 0
        self.native = set()    # names of the types hummr computes (lowering and self-adjoint) up to xr_level
        for level in range(xr_level + 1):
            for name, n_c, n_a, restr in HummrForXRMain.xr_density_strings(level):
                self.native.add(name)

    def streamed_types(self):
        """the large types and their adjoints"""
        large = {name for name in self.native if len(name) >= STREAMED_MIN_OPS}
        return large | {adjoint_name(name) for name in large}

    # ------------------------------------------------------------------------------------------------------------
    # public: one contraction of a density of the Python convention
    # ------------------------------------------------------------------------------------------------------------

    def contract(self, name, chg_pair, channel, rho_labels, operands):
        """
        sum over the contracted labels of rho[name][chg_pair][channel](0, 1, *rho_labels) times the operands
        [(array, labels)], with rho_labels in the canonical slot order of the density (views already applied).
        Output axes: the integer labels in increasing order (0 and 1 the bra and ket states).
        """
        labels = [0, 1] + list(rho_labels) + [l for _, ls in operands for l in ls]
        out = sorted({l for l in labels if isinstance(l, int)})
        return self._full(name, tuple(chg_pair), tuple(channel), (0, 1), list(rho_labels), operands, out, False)

    # ------------------------------------------------------------------------------------------------------------
    # reversed channels
    # ------------------------------------------------------------------------------------------------------------

    def _is_native_channel(self, name, chg_pair, channel):
        if name not in self.native:
            return False
        # self-adjoint types are computed for mult_bra <= mult_ket (equal charges), see CIManager
        return not (adjoint_name(name) == name and chg_pair[0] == chg_pair[1] and channel[0] > channel[1])

    def _full(self, name, chg_pair, channel, state_labels, orb_labels, operands, out, swapped):
        if self._is_native_channel(name, chg_pair, channel):
            return self._embedded(name, chg_pair, channel, state_labels, orb_labels, operands, out, swapped)
        # X = name from its adjoint Y: rho^X_BA(P) = sum_P' c(P,P') ratio rho^Y_AB(P') with Y's slot j carrying the
        # index of X's slot perm[j] (dens_util.reversed_densities_spin_adapt)
        mult_x_bra, mult_x_ket, rank2, restr, path_x = channel
        name_y = adjoint_name(name)
        mult_a, mult_b = mult_x_ket, mult_x_bra                 # Y's bra and ket multiplicities
        sa2, sb2 = mult_a - 1, mult_b - 1
        q2 = sb2 - sa2
        denom = cg2(sa2, sa2, rank2, q2, sb2, sb2)
        if abs(denom) < 1e-12:
            raise KeyError(f"no reversed density {name} {chg_pair} {channel}")
        ratio = cg2(sb2, sb2, rank2, -q2, sa2, sa2) / denom
        perm, coefs = reversal(name.count("c"), name.count("a"), restr, path_x, q2)
        y_labels = [orb_labels[p] for p in perm]
        y_states = (state_labels[1], state_labels[0])
        result = 0
        for path_y, coef in coefs.items():
            result = result + coef * ratio * self._embedded(name_y, (chg_pair[1], chg_pair[0]),
                                                           (mult_a, mult_b, rank2, restr, path_y), y_states,
                                                           y_labels, operands, out, not swapped)
        return result

    # ------------------------------------------------------------------------------------------------------------
    # frozen core
    # ------------------------------------------------------------------------------------------------------------

    def _embedded(self, name, chg_pair, channel, state_labels, orb_labels, operands, out, swapped):
        """contraction of a native density embedded into the full MO space (frozen_core.embed, contracted)"""
        if self.n_core == 0:
            return self._active(name, chg_pair, channel, state_labels, orb_labels, operands, out, swapped)
        from frozen_core import core_terms
        n_c, n_a = name.count("c"), name.count("a")
        N = n_c + n_a
        mult_bra, mult_ket, rank2, restr, path = channel
        result = np.zeros(self._out_shape(chg_pair, channel, state_labels, orb_labels, operands, out, swapped))
        # active part
        self._add_term(result, name, chg_pair, channel, state_labels, orb_labels, operands, out, swapped,
                       ["act"] * N, list(range(N)))
        # core terms: core slots on core orbitals, the remaining slots carry a lower active density
        q2 = mult_bra - mult_ket
        for n_core_slots in range(2, N + 1, 2):
            for core_slots in itertools.combinations(range(N), n_core_slots):
                for core_orbitals in itertools.product(range(self.n_core), repeat=n_core_slots):
                    term = core_terms(n_c, n_a, restr, path, q2, core_slots, core_orbitals, self.n_core)
                    if term is None:
                        continue
                    (n_c_act, n_a_act), restr_act, slot_of, coefs = term
                    assign = ["act"] * N
                    for s, K in zip(core_slots, core_orbitals):
                        assign[s] = K
                    name_act = "c" * n_c_act + "a" * n_a_act
                    for path_act, coef in coefs.items():
                        if name_act == "":
                            if chg_pair[0] != chg_pair[1] or mult_bra != mult_ket:
                                continue
                            key = ("overlap", mult_bra)
                        else:
                            key = (mult_bra, mult_ket, rank2, restr_act, path_act)
                        self._add_term(result, name_act, chg_pair, key, state_labels, orb_labels, operands, out,
                                       swapped, assign, list(slot_of), coef)
        return result

    def _out_shape(self, chg_pair, channel, state_labels, orb_labels, operands, out, swapped):
        bra, ket = (self.ket, self.bra) if swapped else (self.bra, self.ket)
        sizes = {state_labels[0]: len(bra[chg_pair[0]][channel[0]]), state_labels[1]: len(ket[chg_pair[1]][channel[1]])}
        for l in orb_labels:
            sizes[l] = self.n_mo
        for array, labels in operands:
            for l, n in zip(labels, array.shape):
                sizes.setdefault(l, n)
        return tuple(sizes[l] for l in out)

    def _add_term(self, result, name_act, chg_pair, key, state_labels, orb_labels, operands, out, swapped, assign,
                  slot_of, coef=1.0):
        """
        result += coef * contribution of the active density name_act (key: channel, ("overlap", mult) for the
        bra-ket overlap of the core terms without any active operator), whose
        slot j sits on the original slot slot_of[j]; assign[s] = "act" (active orbitals) or the core orbital
        """
        active = slice(self.n_core, self.n_core + self.n_act)
        on_slot = {l: s for s, l in enumerate(orb_labels)}
        # integrals restricted to the orbitals of the density slots they are contracted with
        ops = []
        for array, labels in operands:
            index, kept = [], []
            for l in labels:
                if not isinstance(l, int) and l in on_slot:
                    a = assign[on_slot[l]]
                    index.append(active if a == "act" else a)
                    if a == "act":
                        kept.append(l)
                else:
                    index.append(slice(None))
                    kept.append(l)
            ops.append((array[tuple(index)], kept))
        # placement into the output
        out_index, out_eff = [], []
        for l in out:
            if l in on_slot:
                a = assign[on_slot[l]]
                out_index.append(active if a == "act" else a)
                if a == "act":
                    out_eff.append(l)
            else:
                out_index.append(slice(None))
                out_eff.append(l)
        act_labels = [orb_labels[s] for s in slot_of]
        result[tuple(out_index)] += coef * self._active(name_act, chg_pair, key, state_labels, act_labels, ops,
                                                        out_eff, swapped)

    # ------------------------------------------------------------------------------------------------------------
    # active densities
    # ------------------------------------------------------------------------------------------------------------

    def _active(self, name, chg_pair, channel, state_labels, orb_labels, operands, out, swapped):
        if name == "":
            mult = channel[1]
            if self.overlaps is None:
                ket = self.bra if swapped else self.ket
                rho = np.eye(len(ket[chg_pair[1]][mult]))
            else:
                rho = self.overlaps[swapped][(chg_pair[0], mult)]
        elif len(name) < STREAMED_MIN_OPS:
            rho = self.active_small[swapped][name][chg_pair][channel]
        else:
            if self.kernel and not self._prefer_cached(orb_labels, operands):
                result = self._kernel(name, chg_pair, channel, state_labels, orb_labels, operands, out, swapped)
                if result is not None:
                    self.calls["kernel"] += 1
                    return result
            self.calls["explicit"] += 1
            rho = self._native(name, channel[3], (chg_pair[0], channel[0]), (chg_pair[1], channel[1]), swapped)[channel]
        return contract_labelled([(rho, list(state_labels) + list(orb_labels))] + list(operands), out)

    def _prefer_cached(self, orb_labels, operands):
        """
        With a memory budget, contractions whose output is about as large as the density (e.g. two free density
        indices and two free axes of the tensor, the 4-index intermediates of XR[2]) use the cached explicit
        densities (_native), which then serve all such contractions of the sector pair, instead of rebuilding the
        density in every call (as hummr does for them, see contractChannel).
        """
        if self.memory_budget is None or len(operands) != 1:
            return False
        tensor, t_labels = operands[0]
        n_act = self.fragment.n_act
        free_t = [axis for axis, label in enumerate(t_labels) if label not in orb_labels]
        out_size = n_act ** sum(isinstance(l, int) for l in orb_labels) * int(np.prod([tensor.shape[a] for a in free_t]))
        return out_size * n_act >= n_act ** len(orb_labels)

    def _kernel(self, name, chg_pair, channel, state_labels, orb_labels, operands, out, swapped):
        """the active contraction done by hummr directly, None if the pattern is not supported"""
        if len(operands) != 1:
            return None
        tensor, t_labels = operands[0]
        t_labels = list(t_labels)
        letters = [l for l in orb_labels if not isinstance(l, int)]
        if len(set(letters)) != len(letters) or len(set(t_labels)) != len(t_labels):
            return None                                       # diagonal (repeated index)
        if any(not isinstance(l, int) and l not in orb_labels for l in t_labels):
            return None                                       # trace within the tensor
        if any(not isinstance(l, int) and l not in t_labels for l in orb_labels):
            return None                                       # density index summed without the tensor
        slot_axis = [t_labels.index(l) if not isinstance(l, int) else -1 for l in orb_labels]
        cg = self._cg_factor(channel)
        if abs(cg) < 1e-12:
            raise KeyError(f"the channel {channel} of {name} vanishes")
        bra, ket = (self.ket, self.bra) if swapped else (self.bra, self.ket)
        bra_sector, ket_sector = (chg_pair[0], channel[0]), (chg_pair[1], channel[1])
        try:
            bra_vecs, ket_vecs = bra[bra_sector[0]][bra_sector[1]], ket[ket_sector[0]][ket_sector[1]]
        except KeyError:
            raise KeyError(f"no states for the density {name} between {bra_sector} and {ket_sector}")
        result = self.fragment.contract_density_sector(name.count("c"), name.count("a"), channel[3], bra_sector,
                                                       ket_sector, bra_vecs, ket_vecs, channel, slot_axis, tensor)
        result = np.asarray(result)
        result *= 1.0 / cg                   # CG convention (in place, the array is new)
        labels = list(state_labels) + [l for l in orb_labels if isinstance(l, int)] \
                 + [l for a, l in enumerate(t_labels) if a not in slot_axis]
        return np.transpose(result, [labels.index(l) for l in out])

    def _native(self, name, restr, bra_sector, ket_sector, swapped):
        """active densities (CG convention) of all channels of one sector pair, built by hummr (LRU cache)"""
        key = (name, restr, bra_sector, ket_sector, swapped)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        bra, ket = (self.ket, self.bra) if swapped else (self.bra, self.ket)
        try:
            bra_vecs, ket_vecs = bra[bra_sector[0]][bra_sector[1]], ket[ket_sector[0]][ket_sector[1]]
        except KeyError:
            raise KeyError(f"no states for the density {name} between {bra_sector} and {ket_sector}")
        data = self.fragment.density_sector(name.count("c"), name.count("a"), restr, bra_sector, ket_sector,
                                            bra_vecs, ket_vecs)
        data = {channel: self._to_cg(name, channel, np.asarray(array)) for channel, array in data.items()}
        n_bytes = sum(array.nbytes for array in data.values())
        self._cache[key] = data
        self._cache_bytes += n_bytes
        while self.memory_budget is not None and self._cache_bytes > self.memory_budget and len(self._cache) > 1:
            _, evicted = self._cache.popitem(last=False)
            self._cache_bytes -= sum(array.nbytes for array in evicted.values())
        return data
