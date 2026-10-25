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
The XR[0] Hamiltonian of a dimer with the states of one fragment (A) replaced by its complete (selected) CSF space,
the states of the other fragment (B) fixed (the embedded problem of the Davidson sweeps, see the todo file, part D):

    H^eff_{pb,p'b'} = <p|H_A|p'> d_bb' + d_pp' (H_B)_bb' + sum_t sum_I <p|O^A_{t,I}|p'> K^t_bb'[I]

Its action on trial vectors x[p, b] (A's CSFs in the local sector order x B's local states) is evaluated with hummr's
open contractions (CIManager::contractGeneralDensityOpen). The effective tensors K are extracted from the existing
diagram code: every XR[0] diagram term contains exactly one piece of A, whose density is replaced by a probe
(ProbeProvider), which resolves it into hummr's native active densities like the streamed provider (reversed
channels, frozen core) and returns the integral side with A's active slot values on A's bra-state axis, so that the
diagram itself contracts it with B's piece.
"""

import itertools
import numpy as np

from qode.util import struct, timer
from qode.math.tensornet import raw
from hummr_dens.streamed import StreamedDensities, StreamedDensityType, contract_labelled
import diagram_lists as D
from diagrams import ST_diagrams, SU_diagrams, SV_diagrams
from diagrams.build_diagram import frag_resolve
from diagrams.coupl_fac import diagram_couplings
from precontract import precontract
from get_xr_result_spin_adapt import local_sectors
from XR_term import _multiplicity_block_is_zero


class ProbeProvider(StreamedDensities):
    """
    Stands in for all densities of fragment A: a contraction of A's density (Python convention) with operands is
    resolved into the native active densities (terms) and returns, per term, the operands with the term's active
    slot values on the bra-state axis (label 0, concatenated over the terms; the ket-state axis has length 1).
    The terms of the last calls are in self.calls: [(name, chg_pair, channel, swapped, n_slots, offset), ...].
    """
    uncached = True    # precontract must not cache the results (the terms are recorded per call)

    def __init__(self, fragment, xr_level):
        super().__init__(fragment, {}, {}, xr_level, active_small=None)
        self.calls = []

    def contract(self, name, chg_pair, channel, rho_labels, operands):
        self._mode, self._terms = "count", []
        super().contract(name, chg_pair, channel, rho_labels, operands)
        offsets, self._P = [], 0
        for term in self._terms:
            offsets.append(self._P)
            self._P += self.n_act ** term[4]
        self._mode, self._next, self._offsets = "compute", 0, offsets
        result = super().contract(name, chg_pair, channel, rho_labels, operands)
        self.calls.append([term + (offset,) for term, offset in zip(self._terms, offsets)])
        return result

    def _out_shape(self, chg_pair, channel, state_labels, orb_labels, operands, out, swapped):
        sizes = {0: self._P if self._mode == "compute" else 1, 1: 1}
        for l in orb_labels:
            sizes[l] = self.n_mo
        for array, labels in operands:
            for l, n in zip(labels, array.shape):
                sizes.setdefault(l, n)
        return tuple(sizes[l] for l in out)

    def _active(self, name, chg_pair, channel, state_labels, orb_labels, operands, out, swapped):
        # (density slots that are outputs are restricted to the active orbitals here, see _add_term)
        sizes = {0: self._P if self._mode == "compute" else 1, 1: 1}
        for l in orb_labels:
            sizes[l] = self.n_act
        for array, labels in operands:
            for l, n in zip(labels, array.shape):
                sizes.setdefault(l, n)
        shape = tuple(sizes[l] for l in out)
        if self._mode == "count":
            self._terms.append((name, tuple(chg_pair), tuple(channel), swapped, len(orb_labels)))
            return np.zeros(shape)
        t, self._next = self._next, self._next + 1
        offset, k = self._offsets[t], len(orb_labels)
        # identities put the active slot values on the pseudo axes ("p", j)
        identity = np.eye(self.n_act)
        pseudo = [("p", j) for j in range(k)]
        others = [l for l in out if l not in (0, 1)]
        operands = list(operands) + [(identity, [pseudo[j], orb_labels[j]]) for j in range(k)]
        block = contract_labelled(operands, pseudo + others) if operands else np.ones(())
        block = np.reshape(block, (self.n_act ** k,) + block.shape[k:])
        result = np.zeros(shape)
        index = [slice(None)] * len(out)
        index[out.index(0)] = slice(offset, offset + self.n_act ** k)
        index[out.index(1)] = 0
        # the remaining axes of the result are the others in the order of out
        result[tuple(index)] = block
        return result


def probe_densities(fragment, real, xr_level):
    """density dictionary of fragment A for the probe: all types by the probe, the bookkeeping of the real one"""
    from dens_util import adjoint_name
    probe = ProbeProvider(fragment, xr_level)
    names = set(probe.native) | {adjoint_name(name) for name in probe.native}
    dens = {name: StreamedDensityType(probe, name) for name in names}
    for key in ("n_states", "n_states_bra", "overlap", "n_elec"):
        dens[key] = real[key]
    dens["rectangular"] = False
    return dens, probe


def _diagram_sets(ints, frag_order):
    """(diagram catalog, integrals, labels) of the XR[0] Hamiltonian (get_xr_H, xr_order 0)"""
    symm, bior, _ = ints
    if frag_order == 1:
        return [(ST_diagrams, struct(S=symm.S, T=symm.T), D.ST1[0]), (SU_diagrams, struct(S=symm.S, U=symm.U), D.SU1[0]),
                (SV_diagrams, struct(S=symm.S, V=symm.V), D.SV1[0])]
    return [(ST_diagrams, struct(S=symm.S, T=bior.T), D.ST2[0]), (SU_diagrams, struct(S=symm.S, U=bior.U), D.SU2[0]),
            (SV_diagrams, struct(S=symm.S, V=bior.V), D.SV2[0])]


class EmbeddedHamiltonian:
    """
    H^eff of fragment a (0 or 1 of the dimer) for the fixed states of the other fragment: dens are the densities of
    both fragments for the current states (as for get_xr_H, the ones of fragment a only provide the sectors and
    bookkeeping), fragment the HummrXRFragment of a, H1_other the monomer Hamiltonian of the other fragment
    (from get_xr_H). Trial vectors: x[p, b] with p over a's CSFs of its sectors (local order, see local_sectors) and b
    over the other fragment's local states.
    """
    def __init__(self, a, ints, dens, fragment, monomer_charges, H1_other, target_multiplicity=1, xr_level=0):
        self.a, self.b = a, 1 - a
        self.target_multiplicity = target_multiplicity
        self.fragment, self.monomer_charges = fragment, monomer_charges
        self.H1_other = H1_other
        layout = fragment.csf_layout()
        self.sectors_a = [(chg, mult) for chg, mult, _ in local_sectors(dens[a]['n_states'], monomer_charges[a])]
        self.n_csf = {sec: sum(dim for _, _, dim in layout[sec[0]][sec[1]]) for sec in self.sectors_a}
        self.rows, offset = {}, 0
        for sec in self.sectors_a:
            self.rows[sec] = slice(offset, offset + self.n_csf[sec])
            offset += self.n_csf[sec]
        self.dim_a = offset
        self.cols = {(chg, mult): sl for chg, mult, sl in local_sectors(dens[self.b]['n_states'], monomer_charges[self.b])}
        self.dim_b = max(sl.stop for sl in self.cols.values())
        self.timings = timer()
        probe_dens, self.probe = probe_densities(fragment, dens[a], xr_level)
        both = [None, None]
        both[a], both[self.b] = probe_dens, dens[self.b]
        self.monomer, self.dimer = {}, {}    # effective tensors, see _collect
        for frag_order in (1, 2):
            for catalog, integrals, labels in _diagram_sets(ints, frag_order):
                info = struct(densities=both, integrals=integrals, timings=self.timings,
                              target_multiplicity=target_multiplicity,
                              contract_cache=struct(rho_S=precontract(both, getattr(integrals, "S", integrals), self.timings),
                                                    general=precontract(both, integrals, self.timings)))
                self._collect(info, catalog, labels, frag_order)

    def _collect(self, info, catalog, labels, frag_order):
        """K tensors of all diagram terms: monomer[(term, sector)] = T[slots], dimer[(term, a_bra, a_ket, b_ket)] =
        T[slots, b (all of the other fragment's local states), b' (its ket sector)]"""
        a, b = self.a, self.b
        if frag_order == 1:
            subsystems = [((a,), [((sec,), (sec,))]) for sec in self.sectors_a]
        else:
            pairs = []
            for a_bra, a_ket in itertools.product(self.sectors_a, repeat=2):
                for b_bra, b_ket in itertools.product(self.cols, repeat=2):
                    t = [None, None]
                    t[a], t[b] = (a_bra, a_ket), (b_bra, b_ket)
                    # the blocks XR_term evaluates: total charge conserved, coupling to the target multiplicity
                    if a_bra[0] + b_bra[0] != a_ket[0] + b_ket[0]:
                        continue
                    if _multiplicity_block_is_zero(t, self.target_multiplicity):
                        continue
                    pairs.append(t)
            subsystems = [((0, 1), pairs)]
        for subsystem, transitions in subsystems:
            for trans in transitions:
                if frag_order == 1:
                    trans = [(trans[0][0], trans[1][0])]
                subsys_chgs = tuple((frag, ((t[0][0], t[1][0]), (t[0][1], t[1][1]))) for frag, t in zip(subsystem, trans))
                for label in labels:
                    diagram = catalog.catalog[frag_order][label]
                    if diagram.permutations is None:
                        continue
                    for phase, permutation in diagram.permutations:
                        X = frag_resolve(info, subsys_chgs, permutation)
                        if not all(X.Dchg[m] == Dchg for m, Dchg in enumerate(diagram.Dchgs)):
                            continue
                        couplings = diagram_couplings(info, X, label, len(subsys_chgs))
                        m_a = permutation.index(subsystem.index(a))
                        for prefactor, channels in couplings:
                            Xk = X if channels is None else X.with_channels(channels)
                            self.probe.calls = []
                            if label == "u100":
                                # special diagram (density of the permuted fragment 0, overlap of fragment 1)
                                res = np.asarray(raw(Xk.ca0pq_U1pq(0, 1)))
                                if m_a == 1:    # fragment a only enters by its overlap <p|x>
                                    self._add_overlap(phase * prefactor * res, trans)
                                    continue
                                value = np.einsum("ij,kl->ikjl", res, np.asarray(Xk.overlap[1]))
                            else:
                                value = diagram.contraction(Xk)
                            if value is None or (isinstance(value, list) and len(value) == 0):
                                continue
                            if len(self.probe.calls) != 1:
                                raise RuntimeError(f"diagram {label}: expected one density of fragment {a}, "
                                                   f"found {len(self.probe.calls)}")
                            value = phase * prefactor * np.asarray(value)
                            self._add(value, frag_order, m_a, trans, self.probe.calls[0])

    def _add(self, value, frag_order, m_a, trans, terms):
        n = self.fragment.n_act
        if frag_order == 1:
            # axes (A bra = pseudo, A ket = 1)
            sec = trans[0][0]
            for name, chg_pair, channel, swapped, k, offset in terms:
                T = value[offset:offset + n ** k, 0].reshape((n,) * k)
                key = ((name, chg_pair, channel, swapped), sec)
                self.monomer[key] = self.monomer.get(key, 0) + T
            return
        # axes (i of permuted fragment 0, i of 1, j of 0, j of 1)
        a_bra, a_ket = trans[self.a]
        b_bra, b_ket = trans[self.b]
        m_b = 1 - m_a
        value = np.transpose(value, (m_a, 2 + m_a, m_b, 2 + m_b))[:, 0]    # (pseudo, b, b')
        for name, chg_pair, channel, swapped, k, offset in terms:
            T = value[offset:offset + n ** k].reshape((n,) * k + value.shape[1:])
            key = ((name, chg_pair, channel, swapped), a_bra, a_ket, b_ket)
            if key not in self.dimer:
                self.dimer[key] = np.zeros((n,) * k + (self.dim_b, T.shape[-1]))
            self.dimer[key][..., self.cols[b_bra], :] += T

    def _add_overlap(self, value, trans):
        """dimer term in which fragment a only enters by its overlap: value[b, b'] (the other fragment's piece)"""
        a_bra, a_ket = trans[self.a]
        b_bra, b_ket = trans[self.b]
        if a_bra != a_ket:
            return
        key = (("", (a_bra[0], a_bra[0]), ("overlap", a_bra[1]), False), a_bra, a_ket, b_ket)
        if key not in self.dimer:
            self.dimer[key] = np.zeros((self.dim_b, value.shape[1]))
        self.dimer[key][self.cols[b_bra], :] += value

    def _open(self, term, trial, T, diagonal, transpose=False):
        """sigma of one native term: open bra (native bra = A's bra CSFs) or open ket (swapped: native ket), for
        transpose the other side (trial vectors on A's bra CSFs, sigma on its ket CSFs)"""
        name, chg_pair, channel, swapped = term
        if name == "":
            raise ValueError("overlap terms are handled separately")
        n_c, n_a = name.count("c"), name.count("a")
        path = list(channel[4][0]) + list(channel[4][1])
        sigma = self.fragment._engine.contract_density_open(n_c, n_a, channel[3], chg_pair[0], channel[0], chg_pair[1],
                                                            channel[1], list(trial), swapped == transpose, channel[2], path,
                                                            diagonal, np.ascontiguousarray(T))
        return np.asarray(sigma) / self.probe._cg_factor(channel)

    def _csf_diagonal(self, term, sec, T):
        """sum_slots <p|O|p>[slots] T[slots, g] of one native term for all CSFs p of sector sec: (CSF, g)"""
        name, chg_pair, channel, swapped = term
        if chg_pair[0] != chg_pair[1] or channel[0] != channel[1] or (chg_pair[0], channel[0]) != sec:
            raise ValueError(f"term {term} is not diagonal in sector {sec}")
        # <p|O^+|p> = <p|O|p> (real), the swapped (adjoint) terms need no special treatment
        n_c, n_a = name.count("c"), name.count("a")
        path = list(channel[4][0]) + list(channel[4][1])
        diag = self.fragment._engine.contract_density_csf_diagonal(n_c, n_a, channel[3], chg_pair[0], channel[0],
                                                                   channel[2], path, np.ascontiguousarray(T))
        return np.asarray(diag) / self.probe._cg_factor(channel)

    def diagonal(self):
        """the diagonal of H^eff: D[p, b] = H^eff_{pb,pb} (for the Davidson preconditioner and the CIPSI denominator)"""
        if hasattr(self, "_diagonal"):
            return self._diagonal
        D = np.zeros((self.dim_a, self.dim_b)) + np.real(np.diag(np.asarray(self.H1_other)))[None, :]
        for (term, sec), T in self.monomer.items():
            rows = self.rows[sec]
            if term[0] == "":
                D[rows] += float(T)
                continue
            D[rows] += self._csf_diagonal(term, sec, np.asarray(T)[..., None])
        for (term, a_bra, a_ket, b_ket), T in self.dimer.items():
            if a_bra != a_ket:
                continue
            cols = self.cols[b_ket]
            j = np.arange(cols.stop - cols.start)
            T_diag = np.asarray(T)[..., cols.start + j, j]    # (slots, b) with b = b' in b_ket
            if term[0] == "":
                D[self.rows[a_bra], cols] += T_diag[None, :]
                continue
            D[self.rows[a_bra], cols] += self._csf_diagonal(term, a_bra, np.ascontiguousarray(T_diag))
        self._diagonal = D
        return D

    def dimer_terms(self, x, transpose=False):
        """the contributions of the individual dimer terms to H^eff x (transpose: (H^eff)^T x), for the density-matrix
        perturbation"""
        out = []
        for (term, a_bra, a_ket, b_ket), T in self.dimer.items():
            y = np.zeros((self.dim_a, self.dim_b))
            self._add_dimer_term(y, x, term, a_bra, a_ket, b_ket, T, transpose)
            out.append(y)
        return out

    def _add_dimer_term(self, y, x, term, a_bra, a_ket, b_ket, T, transpose):
        """y += (one dimer term of H^eff) x, or its transpose"""
        cols = self.cols[b_ket]
        if not transpose:
            if term[0] == "":
                y[self.rows[a_bra]] += x[self.rows[a_ket]][:, cols] @ T.T
                return
            trial = [x[self.rows[a_ket], j] for j in range(cols.start, cols.stop)]
            y[self.rows[a_bra]] += self._open(term, trial, T, False)
            return
        # (H^eff)^T: from A's bra CSFs and all of the other fragment's states to A's ket CSFs and the states of b_ket
        if term[0] == "":
            y[self.rows[a_ket], cols] += x[self.rows[a_bra]] @ T
            return
        trial = [x[self.rows[a_bra], j] for j in range(self.dim_b)]
        y[self.rows[a_ket], cols] += self._open(term, trial, np.swapaxes(T, -1, -2), False, transpose=True)

    def sigma(self, x):
        """H^eff x for x[p, b] (A's CSFs x the other fragment's local states)"""
        sigma = x @ np.asarray(self.H1_other).T
        for (term, sec), T in self.monomer.items():
            rows = self.rows[sec]
            trial = [x[rows, j] for j in range(self.dim_b)]
            if term[0] == "":
                sigma[rows] += float(T) * x[rows]
                continue
            sigma[rows] += self._open(term, trial, T, True)
        for (term, a_bra, a_ket, b_ket), T in self.dimer.items():
            self._add_dimer_term(sigma, x, term, a_bra, a_ket, b_ket, T, False)
        return sigma

    def sigma_T(self, x):
        """(H^eff)^T x for x[p, b], e.g. for the left eigenvectors (H^eff is not symmetric at XR[0])"""
        sigma = x @ np.asarray(self.H1_other)
        for (term, sec), T in self.monomer.items():
            rows = self.rows[sec]
            if term[0] == "":
                sigma[rows] += float(T) * x[rows]
                continue
            trial = [x[rows, j] for j in range(self.dim_b)]
            sigma[rows] += self._open(term, trial, T, True, transpose=True)
        for (term, a_bra, a_ket, b_ket), T in self.dimer.items():
            self._add_dimer_term(sigma, x, term, a_bra, a_ket, b_ket, T, True)
        return sigma

    def states_to_csf(self, states_a, d):
        """x[p, b] of the product-basis vector d[a, b] (a over A's local states) for A's states (CI vectors per sector)"""
        x = np.zeros((self.dim_a, self.dim_b))
        offset = 0
        for chg, mult, sl in local_sectors({c: {m: len(v) for m, v in bm.items()} for c, bm in states_a.items()},
                                           self.monomer_charges[self.a]):
            C = np.array(states_a[chg][mult], dtype=float).T    # (CSF, state)
            x[self.rows[(chg, mult)]] += C @ d[sl]
        return x
