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

# Spin-adapted counterpart of get_xr_result.py.
#
# The fragment state spaces are organized in sectors (charge, multiplicity) and the local
# basis of every fragment is ordered as
#
#     charge (in the order of monomer_charges) -> multiplicity (ascending) -> state
#
# H1 and H2 returned from get_xr_H are given in this local ordering, where H2 is in the
# tensor-product ordering (fragment-0 index major) expected by excitonic.fci.

import numpy
import qode.util
from qode.util.dynamic_array import dynamic_array, wrap, cached
from qode.math.tensornet import raw
import qode.math
from qode.util import struct, timer
import excitonic
import diagrammatic_expansion   # defines information structure for housing results of diagram evaluations
import XR_term                  # knows how to use ^this information to pack a matrix for use in XR model
from diagrams import S_diagrams, ST_diagrams, SU_diagrams, SV_diagrams    # definitions of diagrams needed for S and SH
import os
import precontract as precontract_module
from precontract import precontract
from precontract_pairs import precontract_S, V_factorized_blocks
from pair_basis import has_pair_basis, pair_basis_S
import diagram_lists as D

# default byte budget of the cached intermediates of one XR Hamiltonian build (see resolve_memory_budget)
DEFAULT_MEMORY_BUDGET = 8e9

# overlap checks (see check_overlap): warn if the lowest eigenvalue of the product-state overlap S2^[n] in the target
# block is below S_WARN_THRESH (nearly singular or indefinite, XR[n] then has spurious eigenvalues), or deviates by more
# than S_CONV_THRESH (relative) from the one of the highest order of the check (overlap expansion not converged at n)
S_WARN_THRESH = 0.05
S_CONV_THRESH = 0.2



def drop_empty_sectors(states):
    """
    Remove (charge, multiplicity) sectors without any states (and charges without any
    multiplicity), because they can neither be handed to the density builder nor to the
    Hamiltonian build.
    """
    ret = {}
    for chg, mult_dict in states.items():
        non_empty = {mult: vecs for mult, vecs in mult_dict.items() if len(vecs) > 0}
        if non_empty:
            ret[chg] = non_empty
    return ret

def count_sectors(states):
    return sorted((chg, mult) for chg, mult_dict in states.items() for mult in mult_dict)

def local_sectors(n_states, monomer_charges):
    """
    Ordered list of (charge, multiplicity, slice) for the local basis of one fragment.
    This is the single place defining the local ordering used for H1, H2 and the eigenvectors.
    """
    ret = []
    offset = 0
    for chg in monomer_charges:
        if chg not in n_states:
            continue
        for mult in sorted(n_states[chg]):
            n = n_states[chg][mult]
            ret.append((chg, mult, slice(offset, offset + n)))
            offset += n
    return ret

def estimate_xr_memory(n_mo, n_act, xr_level, states, n_frag=2):
    """
    Rough memory (bytes) of one XR Hamiltonian build besides the budgeted caches, for n_frag fragments like the one
    with the states (states[chg][mult] = vectors), n_mo orbitals and n_act active orbitals:
        - the integrals of the supersystem (a few tensors of (n_frag n_mo)^4),
        - the cached pre-contracted intermediates with at most 3 orbital indices (per fragment ~ n_labels x the
          bra-ket state pairs of the sector pairs x n_act^3; n_labels grows with the XR order),
        - the transient 4-index intermediates of XR[2] (a few dozen of the largest sector pair x n_act^4),
        - a fixed overhead (Python, libraries, hummr's CI spaces).
    The coefficients are calibrated on all-electron Be2 cc-pVTZ XR[2] (n_act = 30, 3 states per sector); this is
    only meant for choosing the cache budget.
    """
    n_states = [len(vecs) for by_mult in states.values() for vecs in by_mult.values() if len(vecs) > 0]
    pairs = sum(n_states) ** 2                       # bra-ket state pairs over all sector pairs (upper bound)
    largest_pair = max(n_states) ** 2 if n_states else 0
    n_labels = {0: 3, 1: 6}.get(xr_level, 9)
    integrals = 5 * (n_frag * n_mo) ** 4
    small = n_frag * n_labels * pairs * n_act ** 3
    transient = 35 * largest_pair * n_act ** 4 if xr_level >= 2 else 35 * largest_pair * n_act ** 3
    return 8 * (integrals + small + transient) + 1e9

def _meminfo():
    """(MemTotal, MemAvailable) in bytes and the resident memory of this process"""
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, value = line.split(":")
            info[key] = int(value.split()[0]) * 1024
    with open("/proc/self/status") as f:
        rss = next(int(line.split()[1]) * 1024 for line in f if line.startswith("VmRSS"))
    return info["MemTotal"], info["MemAvailable"], rss

def resolve_memory_budget(memory_budget, n_mo, n_act, xr_level, states, n_frag=2):
    """
    The byte budget of the cached intermediates of one XR Hamiltonian build:
        - a number: used as given,
        - None: DEFAULT_MEMORY_BUDGET,
        - "auto" or ("auto", node_memory[, fraction]): as large as possible, such that the estimated memory of the
          build (estimate_xr_memory) plus the budget fill fraction (default 0.85) of the node memory (default
          MemTotal, minus what other processes use; a given node_memory is used as it is).
    """
    if memory_budget is None:
        return DEFAULT_MEMORY_BUDGET
    if memory_budget == "auto" or (isinstance(memory_budget, tuple) and memory_budget[0] == "auto"):
        spec = memory_budget if isinstance(memory_budget, tuple) else ("auto",)
        fraction = spec[2] if len(spec) > 2 else 0.85
        total, available, rss = _meminfo()
        if len(spec) > 1 and spec[1] is not None:
            node_memory, others = spec[1], 0
        else:
            node_memory, others = total, max(total - available - rss, 0)
        estimate = estimate_xr_memory(n_mo, n_act, xr_level, states, n_frag)
        return max(fraction * node_memory - others - estimate, 0.0)
    return float(memory_budget)

def apply_memory_budget(memory_budget, n_frag=2):
    """
    Splits the budget B of one XR Hamiltonian build: B/4 for the large pre-contracted intermediates (precontract),
    B/n_frag for the density provider of every fragment, which keeps half of it as explicit densities and gives the
    other half to hummr's density engines (see hummr_dens/streamed.py). For a dimer of one molecule (one shared
    hummr instance) this is B/4 + 2 x B/4 + B/4. Returns the budget of one density provider.
    """
    precontract_module.set_large_cache_limit(memory_budget / 4)
    return memory_budget / n_frag

def build_frag_densities(dens_builder, frag_name, states, xr_level, n_elec_ref, bra_states=None, streamed=False,
                         memory_budget=None):
    """
    states[chg][mult] = list of CSF vectors
    bra_states:         optional different bra states (same structure, e.g. single CSFs for the state gradients),
                        the densities are then <bra_states|...|states> and the Hamiltonian rectangular.  For
                        xr_level >= 1 the overlap (S) diagrams are additionally built between the bra states
                        (rho['bra_bra'], only the types with at most 2 operators), since H2 = S2^-1 S2H2 requires
                        S2 in the bra space.  This is exact if the bra states span the ket states (per sector,
                        e.g. all single CSFs as bra), which get_xr_H checks.
    streamed:           the large density types (4 and more operators) are not built, their contractions with the
                        integrals are evaluated on demand (hummr_dens/streamed.py)
    memory_budget:      bytes for the cached intermediates of the XR Hamiltonian build (None: DEFAULT_MEMORY_BUDGET,
                        "auto": as large as the node memory allows, see resolve_memory_budget)
    Returns the density dictionary of one fragment, including the bookkeeping entries
    required by diagrammatic_expansion.
    """
    states = drop_empty_sectors(states)
    options = {}
    if hasattr(dens_builder, "engine"):    # hummr backend (the reference backend of the tests has no such options)
        fragment = dens_builder.engine.fragments[frag_name]
        memory_budget = apply_memory_budget(resolve_memory_budget(memory_budget, fragment.orbitals.shape[1], fragment.n_act,
                                                                  xr_level, states))
        options = dict(streamed=streamed, memory_budget=memory_budget)
    if bra_states is None:
        rho = dens_builder.build_densities(frag_name, xr_level, states, states, **options)
        bra_states = states
    else:
        bra_states = drop_empty_sectors(bra_states)
        if count_sectors(bra_states) != count_sectors(states):
            raise ValueError("bra and ket states need to have the same (chg, mult) sectors")
        rho = dens_builder.build_densities(frag_name, xr_level, bra_states, states, **options)
    rho['n_states'] = {chg: {mult: len(vecs) for mult, vecs in mult_dict.items()} for chg, mult_dict in states.items()}
    rho['n_states_bra'] = {chg: {mult: len(vecs) for mult, vecs in mult_dict.items()} for chg, mult_dict in bra_states.items()}
    # <bra|ket> per sector, which replaces the Kronecker delta of spectator fragments (e.g. in u100)
    rho['overlap'] = {chg: {mult: (numpy.eye(len(vecs)) if bra_states is states else
                                   numpy.array(bra_states[chg][mult], dtype=float) @ numpy.array(vecs, dtype=float).T)
                            for mult, vecs in mult_dict.items()} for chg, mult_dict in states.items()}
    rho['n_elec'] = {chg: n_elec_ref - chg for chg in states}
    rho['rectangular'] = bra_states is not states
    if rho['rectangular'] and xr_level >= 1:
        rho['unspanned_sectors'] = []    # checked by get_xr_H
        for chg, mult_dict in states.items():
            for mult, vecs in mult_dict.items():
                B, K = numpy.array(bra_states[chg][mult], dtype=float), numpy.array(vecs, dtype=float)
                residual = K - numpy.linalg.lstsq(B.T, K.T, rcond=None)[0].T @ B
                if numpy.linalg.norm(residual) > 1e-8 * max(1.0, numpy.linalg.norm(K)):
                    rho['unspanned_sectors'].append((chg, mult))
        bra_bra = dens_builder.build_densities(frag_name, 0, bra_states, bra_states, max_ops=2)
        bra_bra['n_states'] = bra_bra['n_states_bra'] = rho['n_states_bra']
        bra_bra['overlap'] = {chg: {mult: numpy.array(vecs, dtype=float) @ numpy.array(vecs, dtype=float).T
                                    for mult, vecs in mult_dict.items()} for chg, mult_dict in bra_states.items()}
        bra_bra['n_elec'] = rho['n_elec']
        bra_bra['rectangular'] = False
        rho['bra_bra'] = bra_bra
    return rho

def target_block(dens, dimer_sectors, target_charge, target_multiplicity, key='n_states'):
    """mask over the dimer-sector ordering of XR_term.dimer_matrix: the product states with the total charge
    target_charge (None: all), whose multiplicities can couple to target_multiplicity"""
    return numpy.concatenate([
        numpy.full(dens[0][key][chg0][mult0] * dens[1][key][chg1][mult1],
                   (target_charge is None or chg0 + chg1 == target_charge) and multiplicities_couple(mult0, mult1, target_multiplicity))
        for (chg0, mult0), (chg1, mult1) in dimer_sectors])

def dimer_sector_list(dens, monomer_charges):
    """the sector pairs ((chg0, mult0), (chg1, mult1)) of the dimer-sector ordering of XR_term.dimer_matrix"""
    frag_sectors = [local_sectors(rho['n_states'], monomer_charges[m]) for m, rho in enumerate(dens)]
    return [((chg0, mult0), (chg1, mult1)) for chg0, mult0, _ in frag_sectors[0] for chg1, mult1, _ in frag_sectors[1]]

def dimer_labels(n_states, monomer_charges):
    """for every product state of the dimer-sector ordering of XR_term.dimer_matrix (row-major within every sector
    pair) the local state indices (i0, i1) of the two fragments in the ordering of local_sectors (n_states[frag]:
    {chg: {mult: number of states}})"""
    frag_sectors = [local_sectors(n_states[m], monomer_charges[m]) for m in range(2)]
    return numpy.concatenate([numpy.stack(numpy.meshgrid(numpy.arange(sl0.start, sl0.stop), numpy.arange(sl1.start, sl1.stop),
                                                         indexing="ij"), axis=-1).reshape(-1, 2)
                              for _, _, sl0 in frag_sectors[0] for _, _, sl1 in frag_sectors[1]])

def S_term_blocks(dens, symm_ints, timings, precon_timings, target_multiplicity):
    """the blocks of the S diagrams (diagrammatic_expansion.blocks), in the basis of the orbital pairs for a
    pair-compressed S of a dimer (pair_basis.py)"""
    if has_pair_basis(dens, symm_ints):
        dens_k, S_k = pair_basis_S(dens, symm_ints)
        return diagrammatic_expansion.blocks(densities=dens_k, integrals=S_k, diagrams=S_diagrams,
                                             contract_cache=precontract(dens_k, S_k, precon_timings), timings=timings,
                                             precon_timings=precon_timings, target_multiplicity=target_multiplicity)
    return diagrammatic_expansion.blocks(densities=dens, integrals=symm_ints.S, diagrams=S_diagrams,
                                         contract_cache=precontract_S(dens, symm_ints, precon_timings), timings=timings,
                                         precon_timings=precon_timings, target_multiplicity=target_multiplicity)

def overlap_terms(ints, dens, monomer_charges, target_multiplicity, max_order, order_ints=None, terms=None):
    """
    {k: the order-k term of the product-state overlap S2} (k = 0: the identity) for k = 0 ... max_order, in the full
    dimer-sector ordering of XR_term.dimer_matrix (all sector pairs).  order_ints: {k: ints} the integrals of the
    order-k term (get_ints, e.g. with pair_thresh), the remaining orders with ints.  terms: a dict of terms already
    computed for the same densities and order_ints, which are reused; the missing ones are added to it (so that the
    caller can share them between the overlap truncation, check_overlap and the metric of the XR orders).
    The orders 3 and 4 require the densities of xr_level 1 and 2 (build_frag_densities).  Since the terms are
    symmetric, only the blocks with bra <= ket are evaluated (XR_term.dimer_matrix, symmetric).
    """
    if any(rho.get('rectangular', False) for rho in dens):
        raise ValueError("the overlap matrices require square densities (bra states = ket states)")
    if max_order > 4:
        raise NotImplementedError("the S expansion is implemented up to order 4")
    order_ints = dict(order_ints or {})
    terms = {} if terms is None else terms
    missing = [k for k in range(max_order + 1) if k not in terms]
    if not missing:
        return terms
    t = timer()
    dimer_sectors = dimer_sector_list(dens, monomer_charges)
    S_blocks = {}
    for k in missing:
        symm_ints = order_ints.get(k, ints)[0]
        if id(symm_ints) not in S_blocks:
            S_blocks[id(symm_ints)] = S_term_blocks(dens, symm_ints, t, t, target_multiplicity)
        diagrams = {0: D.S0[0]} if k == 0 else {2: D.S2[k]}
        terms[k] = XR_term.dimer_matrix(S_blocks[id(symm_ints)], diagrams, (0, 1), dimer_sectors, t, symmetric=True)
    return terms

def overlap_matrices(ints, dens, monomer_charges, target_multiplicity, target_charge=0, orders=(1, 2, 3), order_ints=None,
                     terms=None):
    """
    ({n: S2^[n]}, labels): the product-state overlap S2 with the S expansion up to order n for the requested orders, in
    the target block (target_block), and for every row the local state indices (i0, i1) of the two fragments in the
    ordering of local_sectors.  The orders 3 and 4 require the densities of xr_level 1 and 2 (build_frag_densities).
    order_ints, terms: see overlap_terms (the order-k terms are taken from / added to terms).
    Cost (Be2, Davidson-XR[0] states, streamed): S2^[1] + S2^[2] about 1/3 of the XR[0] Hamiltonian build, S2^[3] about
    as much as the latter (factor 0.5 in 6-31G, 1.4 in cc-pVDZ, growing with the number of orbitals), S2^[4] 10 to 35
    times more (without the pair compression of the orders 3 and 4).  For a subset of the (orthonormal) fragment states,
    S2^[n] is the principal submatrix of the rows whose local states are both kept.
    """
    terms = overlap_terms(ints, dens, monomer_charges, target_multiplicity, max(orders), order_ints, terms)
    labels = dimer_labels([rho['n_states'] for rho in dens], monomer_charges)
    mask = target_block(dens, dimer_sector_list(dens, monomer_charges), target_charge, target_multiplicity)
    matrices, S2 = {}, None
    for n in range(max(orders) + 1):
        term = terms[n][numpy.ix_(mask, mask)]
        S2 = term if S2 is None else S2 + term
        if n in orders:
            matrices[n] = 0.5 * (S2 + S2.T)
    return matrices, labels[mask]

def overlap_spectra(ints, dens, monomer_charges, target_multiplicity, target_charge=0, max_order=3, order_ints=None, terms=None):
    """
    {n: (lowest, largest eigenvalue)} of the product-state overlap S2 with the S expansion up to order n = 1 ...
    max_order, in the target block (see overlap_matrices)
    """
    matrices, _ = overlap_matrices(ints, dens, monomer_charges, target_multiplicity, target_charge, range(1, max_order + 1),
                                   order_ints, terms)
    spectra = {}
    for n, S2 in matrices.items():
        eigvals = numpy.linalg.eigvalsh(S2)
        spectra[n] = (eigvals[0], eigvals[-1])
    return spectra

def check_overlap(ints, dens, monomer_charges, target_multiplicity, target_charge=0, max_order=3, xr_orders=(1, 2),
                  order_ints=None, terms=None):
    """
    Prints the lowest and largest eigenvalues of the product-state overlap S2^[n], n = 1 ... max_order (overlap_spectra)
    and warns for the orders xr_orders (XR[n] with n >= 1), whose S2^[n] is nearly singular or indefinite, or whose
    lowest eigenvalue is not converged (S_WARN_THRESH, S_CONV_THRESH, relative to the one of max_order), and if the
    product basis itself is nearly linearly dependent (lowest eigenvalue of max_order).  order_ints, terms: see
    overlap_terms.  Returns the spectra.
    """
    spectra = overlap_spectra(ints, dens, monomer_charges, target_multiplicity, target_charge, max_order, order_ints, terms)
    print(f"overlap of the product states (target block): lowest / largest eigenvalue of S2 with the S expansion up to order n")
    for n, (low, high) in spectra.items():
        print(f"    n = {n}: {low:9.5f} / {high:9.5f}")
    ref = spectra[max_order][0]
    if ref < S_WARN_THRESH:
        print(f"WARNING: the product basis is nearly linearly dependent or the overlap expansion is not converged at order "
              f"{max_order} (lowest eigenvalue {ref:.5f})")
    for n in xr_orders:
        if n < 1 or n > max_order:
            continue
        low = spectra[n][0]
        if low < S_WARN_THRESH:
            print(f"WARNING: XR[{n}]: S2^[{n}] is {'indefinite' if low < 0 else 'nearly singular'} (lowest eigenvalue {low:.5f}), "
                  f"XR[{n}] {'has' if low < 0 else 'may have'} spurious eigenvalues")
        elif n < max_order and abs(low - ref) > S_CONV_THRESH * abs(ref):
            print(f"WARNING: XR[{n}]: the lowest eigenvalue of S2^[{n}] ({low:.5f}) deviates by {abs(low - ref) / abs(ref):.0%} "
                  f"from the one at order {max_order} ({ref:.5f}), the overlap expansion is not converged at XR[{n}]")
    return spectra

def overlap_truncation(S2, labels, weights, sector_slices, min_eigval, max_weight):
    """
    The fragment states to remove, such that the lowest eigenvalue of the product-state overlap S2 (target block, with
    the local state indices labels of its rows, overlap_matrices) is at least min_eigval: the states are removed in the
    order of increasing weights (weights[frag], local ordering), as long as their weight is below max_weight and their
    sector keeps at least one state.  Since S2 of a subset of the states is a principal submatrix, its lowest eigenvalue
    does not decrease with every removal (Cauchy interlacing), so the smallest number of removals is found by bisection.
    Returns (keep[frag] boolean masks over the local states, lowest eigenvalue, number of removed states).
    """
    sector_of = [numpy.concatenate([numpy.full(sl.stop - sl.start, k) for k, sl in enumerate(slices)]) for slices in sector_slices]
    left = [numpy.array([sl.stop - sl.start for sl in slices]) for slices in sector_slices]
    candidates = []
    for w, frag, i in sorted((w, frag, i) for frag in range(2) for i, w in enumerate(weights[frag]) if w < max_weight):
        if left[frag][sector_of[frag][i]] > 1:
            left[frag][sector_of[frag][i]] -= 1
            candidates.append((frag, i))
    def keep_masks(k):
        keep = [numpy.ones(len(w), dtype=bool) for w in weights]
        for frag, i in candidates[:k]:
            keep[frag][i] = False
        return keep
    def lowest(k):
        keep = keep_masks(k)
        rows = keep[0][labels[:, 0]] & keep[1][labels[:, 1]]
        return numpy.linalg.eigvalsh(S2[numpy.ix_(rows, rows)])[0]
    lo, hi = 0, len(candidates)
    if lowest(lo) >= min_eigval:
        hi = lo
    elif lowest(hi) >= min_eigval:
        while hi - lo > 1:    # lowest(lo) < min_eigval <= lowest(hi)
            mid = (lo + hi) // 2
            lo, hi = (mid, hi) if lowest(mid) < min_eigval else (lo, mid)
    return keep_masks(hi), lowest(hi), hi

def _s2h2(blocks, xr_order, dimer_sectors, matrix_timer):
    """S2H2 of XR[1]/[2] in the dimer-sector ordering of XR_term.dimer_matrix, including the monomer terms (blocks: the
    ST/SU/SV blocks of get_xr_H with the symmetric, biorthogonal and V_diff integrals)"""
    def M(blocks, terms):
        return XR_term.dimer_matrix(blocks, terms, (0,1), dimer_sectors, matrix_timer)
    if xr_order == 1:
        return M(blocks.ST_symm, {1: D.ST1[0], 2: D.ST2[0]}) + M(blocks.SU_symm, {1: D.SU1[0], 2: D.SU2[0]}) \
             + M(blocks.ST_bior, {2: D.ST2[1]}) + M(blocks.SU_bior, {2: D.SU2[1]}) \
             + M(blocks.SV_diff, {1: D.SV1[0], 2: D.SV2[0]}) + M(blocks.SV_bior, {2: D.SV2[1]})
    elif xr_order == 2:
        return M(blocks.ST_symm, {1: D.ST1[0], 2: D.ST2[0] + D.ST2[1]}) + M(blocks.SU_symm, {1: D.SU1[0], 2: D.SU2[0] + D.SU2[1]}) \
             + M(blocks.ST_bior, {2: D.ST2[2]}) + M(blocks.SU_bior, {2: D.SU2[2]}) \
             + M(blocks.SV_symm, {1: D.SV1[0], 2: D.SV2[0]}) + M(blocks.SV_diff, {2: D.SV2[1]}) + M(blocks.SV_bior, {2: D.SV2[2]})
    raise NotImplementedError(f"S2H2 is implemented for the xr orders 1 and 2, not {xr_order}")

def _product_ordering(H2blocked, dens, frag_sectors, dimer_sectors):
    """
    H2 reordered from the dimer-sector ordering of XR_term.dimer_matrix, i.e.
        for sector pair: for state0: for state1
    into the tensor-product ordering of the local fragment bases (see local_sectors)
    (rows refer to the bra states, columns to the ket states, which differ for rectangular builds)
    """
    rho0, rho1 = dens
    def product_mapping(key):
        dimer_index = {}
        idx = 0
        for (chg0, mult0), (chg1, mult1) in dimer_sectors:
            for state0 in range(rho0[key][chg0][mult0]):
                for state1 in range(rho1[key][chg1][mult1]):
                    dimer_index[(chg0, mult0, state0, chg1, mult1, state1)] = idx
                    idx += 1
        frag_states = [[(chg, mult, state) for chg, mult in frag_sectors[m] for state in range(rho[key][chg][mult])] for m, rho in enumerate(dens)]
        return [dimer_index[key0 + key1] for key0 in frag_states[0] for key1 in frag_states[1]]
    return H2blocked[numpy.ix_(product_mapping('n_states_bra'), product_mapping('n_states'))]

def get_xr_H(ints, dens, xr_order, monomer_charges, target_multiplicity, target_charge=None):
    """
    target_charge: if given, XR[1]/[2] warn if the lowest eigenvalue of the inverted S2 in the block of this total
                   charge (and the multiplicities coupling to target_multiplicity) is below S_WARN_THRESH
    """
    diag_timer = timer()
    precon_timer = timer()
    matrix_timer = timer()

    symm_ints, bior_ints, nuc_rep = ints

    # The engines that build the terms
    BeN_rho = dens
    contract_cache = precontract_S(BeN_rho, symm_ints, precon_timer)
    ST_blocks_symm = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=struct(S=symm_ints.S, T=symm_ints.T), diagrams=ST_diagrams, contract_cache=contract_cache, timings=diag_timer, precon_timings=precon_timer, target_multiplicity=target_multiplicity)
    SU_blocks_symm = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=struct(S=symm_ints.S, U=symm_ints.U), diagrams=SU_diagrams, contract_cache=contract_cache, timings=diag_timer, precon_timings=precon_timer, target_multiplicity=target_multiplicity)
    SV_blocks_symm = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=struct(S=symm_ints.S, V=symm_ints.V), diagrams=SV_diagrams, contract_cache=contract_cache, timings=diag_timer, precon_timings=precon_timer, target_multiplicity=target_multiplicity)
    ST_blocks_bior = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=struct(S=symm_ints.S, T=bior_ints.T), diagrams=ST_diagrams, contract_cache=contract_cache, timings=diag_timer, precon_timings=precon_timer, target_multiplicity=target_multiplicity)
    SU_blocks_bior = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=struct(S=symm_ints.S, U=bior_ints.U), diagrams=SU_diagrams, contract_cache=contract_cache, timings=diag_timer, precon_timings=precon_timer, target_multiplicity=target_multiplicity)
    SV_blocks_bior = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=struct(S=symm_ints.S, V=bior_ints.V), diagrams=SV_diagrams, contract_cache=contract_cache, timings=diag_timer, precon_timings=precon_timer, target_multiplicity=target_multiplicity)

    # sectors under consideration (empty sectors have already been dropped from the densities)
    frag_sectors = [[(chg, mult) for chg, mult, _ in local_sectors(rho['n_states'], monomer_charges[m])] for m, rho in enumerate(BeN_rho)]

    # the sector pairs, which cannot couple to target_multiplicity, are kept here on purpose,
    # since the tensor-product basis of excitonic.fci requires them. They just do not couple.
    dimer_sectors = [(sec0, sec1) for sec0 in frag_sectors[0] for sec1 in frag_sectors[1]]

    #########
    # Build
    #########

    if xr_order == 0:
        # XR[0] is the zeroth order of S H with the biorthogonal Hamiltonian, so the monomer terms are built with the
        # biorthogonal integrals as well (as the dimer terms); the symmetric ones would add the fragment-local part
        # of the first-order correction (h_pq - h^p_q) without its dimer counterpart (Be2 6-31G 4.5 A: -3.6 mEh)
        # Note on the bookkeeping: only H1 x 1 + 1 x H1 + H2 enters for a dimer.  In the paper (arXiv:2409.07628,
        # eqs. newexcitonic2b/c, S0HmonomerB, S0Hdimer) H1 is the isolated monomer (own nuclei, symmetric integrals,
        # as for XR[1]/[2] below) and H2 contains the fragment-local terms of the dimer with biorthogonal minus
        # symmetric integrals.  Here these fragment-local dimer terms are kept in H1 instead (biorthogonal H1, no
        # subtraction in H2), which gives the same dimer Hamiltonian, but H1 is then not the isolated-monomer
        # Hamiltonian (unlike for XR[1]/[2]).  For more than two fragments this has to follow the paper (one
        # fragment-local correction per dimer, with that dimer's biorthogonal integrals).  The embedded Davidson
        # (embedded.py, H1_other) relies on the biorthogonal H1 here.
        H1 = []
        for m in [0,1]:
            H1 += [  XR_term.monomer_matrix(ST_blocks_bior, {1: D.ST1[0]}, m, frag_sectors[m], matrix_timer) \
                   + XR_term.monomer_matrix(SU_blocks_bior, {1: D.SU1[0]}, m, frag_sectors[m], matrix_timer) \
                   + XR_term.monomer_matrix(SV_blocks_bior, {1: D.SV1[0]}, m, frag_sectors[m], matrix_timer) ]

        H2blocked  = XR_term.dimer_matrix(ST_blocks_bior, {2: D.ST2[0]}, (0,1), dimer_sectors, matrix_timer)
        H2blocked += XR_term.dimer_matrix(SU_blocks_bior, {2: D.SU2[0]}, (0,1), dimer_sectors, matrix_timer)
        H2blocked += XR_term.dimer_matrix(SV_blocks_bior, {2: D.SV2[0]}, (0,1), dimer_sectors, matrix_timer)
    elif xr_order in (1, 2):
        # as in get_xr_result.get_xr_H:  H2 = S2^-1 S2H2 - (monomer terms), with the S expansion up to xr_order
        # S2 is inverted in the bra space, i.e. for rectangular builds (bra != ket states) it is built from the
        # densities between the bra states (see build_frag_densities)
        if any(rho.get('rectangular', False) and 'bra_bra' not in rho for rho in BeN_rho):
            raise ValueError("rectangular XR[1]/[2] Hamiltonians require the bra-bra densities (build_frag_densities with xr_level >= 1)")
        for rho in BeN_rho:
            if rho.get('unspanned_sectors'):
                raise ValueError(f"the bra states do not span the ket states of the sectors (chg, mult) {rho['unspanned_sectors']}, "
                                 "which the rectangular XR[1]/[2] Hamiltonian requires (S2 is inverted in the bra space)")
        S_rho = [rho['bra_bra'] if rho.get('rectangular', False) else rho for rho in BeN_rho]
        S_cache = contract_cache if all(a is b for a, b in zip(S_rho, BeN_rho)) else precontract_S(S_rho, symm_ints, precon_timer)
        S_blocks       = diagrammatic_expansion.blocks(densities=S_rho, integrals=symm_ints.S, diagrams=S_diagrams, contract_cache=S_cache, timings=diag_timer, precon_timings=precon_timer, target_multiplicity=target_multiplicity)
        SV_blocks_diff = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=struct(S=symm_ints.S, V=bior_ints.V_diff), diagrams=SV_diagrams, contract_cache=contract_cache, timings=diag_timer, precon_timings=precon_timer, target_multiplicity=target_multiplicity)

        # H1: the isolated monomers (symmetric integrals, arXiv:2409.07628 eq. S0HmonomerB), subtracted again from H2
        # below; this differs from the bookkeeping of XR[0] above, see the note there
        H1 = []
        for m in [0,1]:
            H1 += [  XR_term.monomer_matrix(ST_blocks_symm, {1: D.ST1[0]}, m, frag_sectors[m], matrix_timer) \
                   + XR_term.monomer_matrix(SU_blocks_symm, {1: D.SU1[0]}, m, frag_sectors[m], matrix_timer) \
                   + XR_term.monomer_matrix(SV_blocks_symm, {1: D.SV1[0]}, m, frag_sectors[m], matrix_timer) ]

        S_diagrams_order = D.S2[1] if xr_order == 1 else D.S2[1] + D.S2[2]
        S2 = XR_term.dimer_matrix(S_blocks, {0: D.S0[0], 2: S_diagrams_order}, (0,1), dimer_sectors, matrix_timer)

        S2H2 = _s2h2(struct(ST_symm=ST_blocks_symm, SU_symm=SU_blocks_symm, SV_symm=SV_blocks_symm, ST_bior=ST_blocks_bior,
                            SU_bior=SU_blocks_bior, SV_bior=SV_blocks_bior, SV_diff=SV_blocks_diff), xr_order, dimer_sectors, matrix_timer)

        # The rows/columns of the sector pairs, which cannot couple to target_multiplicity, vanish (they are only
        # kept for the tensor-product basis), so S2 is inverted on the coupling ones (identity elsewhere).
        coupling = numpy.concatenate([
            numpy.full(S_rho[0]['n_states'][chg0][mult0] * S_rho[1]['n_states'][chg1][mult1],
                       multiplicities_couple(mult0, mult1, target_multiplicity))
            for (chg0, mult0), (chg1, mult1) in dimer_sectors])
        if target_charge is not None:
            block = target_block(S_rho, dimer_sectors, target_charge, target_multiplicity)
            S2_target = S2[numpy.ix_(block, block)]
            low = numpy.linalg.eigvalsh(0.5 * (S2_target + S2_target.T))[0]
            if low < S_WARN_THRESH:
                print(f"WARNING: XR[{xr_order}]: S2^[{xr_order}] is {'indefinite' if low < 0 else 'nearly singular'} "
                      f"(lowest eigenvalue {low:.5f} in the target block), XR[{xr_order}] {'has' if low < 0 else 'may have'} spurious eigenvalues")
        S2 = S2 + numpy.diag(~coupling * 1.0)
        S2inv = qode.math.precise_numpy_inverse(S2)

        H2blocked  = S2inv @ S2H2
        H2blocked -= XR_term.dimer_matrix(ST_blocks_symm, {1: D.ST1[0]}, (0,1), dimer_sectors, matrix_timer)
        H2blocked -= XR_term.dimer_matrix(SU_blocks_symm, {1: D.SU1[0]}, (0,1), dimer_sectors, matrix_timer)
        H2blocked -= XR_term.dimer_matrix(SV_blocks_symm, {1: D.SV1[0]}, (0,1), dimer_sectors, matrix_timer)
    else:
        raise NotImplementedError(f"xr order {xr_order} is not implemented for the spin-adapted build")

    H2 = _product_ordering(H2blocked, BeN_rho, frag_sectors, dimer_sectors)

    return H1, H2

def multiplicities_couple(mult1, mult2, target_mult):
    """True if two fragment multiplicities can couple to target_mult (always for spin orbitals, i.e. None)"""
    if target_mult is None or mult1 is None or mult2 is None:
        return True
    return abs(mult1 - mult2) + 1 <= target_mult <= mult1 + mult2 - 1 and (mult1 + mult2 + target_mult) % 2 == 1

def product_basis_mask(dens, monomer_charges, target_charge, target_multiplicity):
    """
    Boolean mask over the tensor-product basis (ordering of get_xr_H), which selects the product
    states with total charge target_charge and fragment multiplicities that can couple to
    target_multiplicity. The remaining product states are spurious: their spin-coupled states do
    not exist for this target, but they still appear in the tensor-product basis with their bare
    monomer energies (and may even be lower than the target state for non-singlet targets).
    """
    frag_states = [[(chg, mult) for chg, mult, sl in local_sectors(rho['n_states'], monomer_charges[m]) for _ in range(sl.stop - sl.start)]
                   for m, rho in enumerate(dens)]
    return numpy.array([chg0 + chg1 == target_charge and multiplicities_couple(mult0, mult1, target_multiplicity)
                        for chg0, mult0 in frag_states[0] for chg1, mult1 in frag_states[1]])

def get_xr_states_from_H(H1, H2, target_state=0, basis_mask=None):
    out = struct(log=qode.util.textlog(echo=True))
    diagonalizer_timer = timer()
    E, T = excitonic.fci((H1,[[None,H2],[None,None]]), out, diagonalizer_timer, target_state=target_state, basis_mask=basis_mask)
    out.log("\nTotal Excitonic Energy (without nuc_rep) = ", E)
    return E, T

def xr_approximated(xr_order, leakage_ints=None, metric_ints=None, leakage_v_thresh=None):
    """True if one of the approximations (pair-compressed integrals of the leakage increments or of the overlap
    terms, factorized V of the order-2 increment) applies to XR[xr_order], see get_xr_states"""
    return xr_order >= 1 and (any(k <= xr_order for k in (leakage_ints or {})) or any(k <= xr_order for k in (metric_ints or {}))
                              or (xr_order >= 2 and (leakage_v_thresh or {}).get(2) is not None))

def get_xr_states(ints, dens, xr_order, monomer_charges, target_multiplicity, target_state=0, target_charge=0,
                  leakage_ints=None, metric_ints=None, leakage_v_thresh=None, metric_terms=None, leakage_pair_slots=2.5):
    """
    XR[xr_order] energies and states.  If one of the approximations leakage_ints (orders k <= xr_order), metric_ints
    (orders k <= xr_order) or leakage_v_thresh (xr_order 2) applies (xr_approximated), XR[n] is evaluated in the
    equivalent leakage form H0 + (S2^[n])^-1 L^[n] (get_leakage_metric_H with metric_order = leakage_order = n, which
    is identical to get_xr_H for exact integrals), otherwise with get_xr_H.  metric_terms, leakage_pair_slots: see
    get_leakage_metric_H.
    """
    if xr_approximated(xr_order, leakage_ints, metric_ints, leakage_v_thresh):
        return get_leakage_metric_states(ints, dens, monomer_charges, target_multiplicity, target_state=target_state,
                                         target_charge=target_charge, metric_order=xr_order, leakage_order=xr_order,
                                         leakage_ints=leakage_ints, metric_ints=metric_ints, leakage_v_thresh=leakage_v_thresh,
                                         metric_terms=metric_terms, leakage_pair_slots=leakage_pair_slots)
    H1, H2 = get_xr_H(ints, dens, xr_order, monomer_charges, target_multiplicity, target_charge=target_charge)
    mask = product_basis_mask(dens, monomer_charges, target_charge, target_multiplicity)
    return get_xr_states_from_H(H1, H2, target_state=target_state, basis_mask=mask)

def _int_difference(A, B, two_electron=False):
    """the integrals A - B (blocked as A, wrapped as in get_ints)"""
    from get_ints import tensorly_wrapper, tensorly_wrapper2
    def rule(*indices):
        return numpy.asarray(raw(A[indices])) - numpy.asarray(raw(B[indices]))
    return wrap(dynamic_array(rule, A.ranges), [cached, (tensorly_wrapper2 if two_electron else tensorly_wrapper)(timer())])

def get_leakage_metric_H(ints, dens, monomer_charges, target_multiplicity, target_charge=None, metric_order=4,
                         leakage_ints=None, metric_ints=None, leakage_v_thresh=None, leakage_order=2, metric_terms=None,
                         leakage_pair_slots=2.5):
    """
    H1, H2 (as get_xr_H) of the leakage-metric ansatz

        H = H0 + (S2^[metric_order])^-1 L^[leakage_order],    L^[n] = S2H2^[n] - S2^[n] H0

    For metric_order = leakage_order = n this is XR[n] (n = 1, 2) in the leakage form (get_xr_states uses it for the
    approximated XR orders); the leakage-metric ansatz proper is leakage_order 2 with metric_order 3 or 4.

    with the exact (biorthogonal) XR[0] Hamiltonian H0 (H1 is the one of XR[0]).  Since L^[2] contains -S2^[2] H0, this
    is S2^-1 [S2H2^[2] + sum_{k=3..n} S2_(k) H0], i.e. the S orders 3 ... n of S H are resolved with the exact
    Hamiltonian of the product space (exact for a complete fragment state space); only the leakage of these orders out
    of the product space is neglected.  (Be2 cc-pVDZ with metric_order 4: within +22 to +40 uEh of FCI at 3.0 - 4.0 A
    in the complete optimized state space, XR[2] +86 uEh at 4.0 A.)
    leakage_ints: {k: ints} the integrals (get_ints, e.g. with pair_thresh) of the order-k increment
                  L^[k] - L^[k-1] (k = 1, 2), the remaining orders with ints.  Be2 cc-pVDZ: a pair-compressed order 2
                  (pair_thresh 0.02 / 0.1) is within 10 / 50 uEh of the exact one, a compressed order 1 fails
                  (1e2 to 1e3 uEh, the small pairs couple to the large core and nuclear-attraction integrals)
    metric_ints:  {k: ints} the integrals of the order-k term of S2^[metric_order] (k = 1 ... metric_order), the
                  remaining orders with ints.  Be2 cc-pVDZ: compressed orders 3 and 4 (pair_thresh 0.1) change the
                  energy by less than 8 uEh, except for a nearly singular metric (complete state space at 3.0 A,
                  lowest eigenvalue 0.02: +50 uEh)
    leakage_v_thresh: {2: threshold}: the order-2 diagrams of the order-2 increment (the S2V terms of order 2) are
                  evaluated with the inter-fragment blocks of V factorized into (axes on one fragment) x (axes on the
                  other fragment), keeping the singular values above the threshold (precontract_pairs.
                  V_factorized_precontract); all other terms with the complete V
    leakage_pair_slots: the overlap-connected free density slots of the leakage diagrams in the basis of the pairs of
                  the pair-compressed overlap (diagrams/pair_slots.py, exact): a number r if r K <= n_orb, True always,
                  False never (faster for few pairs, lower peak memory in any case, see run_xr)
    (leakage_ints, metric_ints and leakage_v_thresh entries of orders above leakage_order / metric_order are ignored.)
    metric_terms: {k: the order-k term of S2} already computed with the integrals metric_ints.get(k, ints) for these
                  densities (overlap_terms), which are used instead of evaluating them again
    Requires the densities of xr_level 2 (xr_level 1 for leakage_order = metric_order = 1, xr_level 1 for
    metric_order 3; square, bra states = ket states).
    """
    if any(rho.get('rectangular', False) for rho in dens):
        raise ValueError("the leakage-metric Hamiltonian requires square densities (bra states = ket states)")
    if leakage_order not in (1, 2):
        raise ValueError(f"leakage_order has to be 1 or 2, not {leakage_order}")
    if not leakage_order <= metric_order <= 4:
        raise ValueError(f"metric_order has to be in {leakage_order} ... 4, not {metric_order}")
    leakage_ints = {k: i for k, i in (leakage_ints or {}).items() if k <= leakage_order}
    metric_ints = {k: i for k, i in (metric_ints or {}).items() if k <= metric_order}
    if not set(leakage_ints) <= {1, 2}:
        raise ValueError(f"leakage_ints: the leakage is implemented for the orders 1 and 2, not {sorted(leakage_ints)}")
    leakage_v_thresh = {k: v for k, v in (leakage_v_thresh or {}).items() if k <= leakage_order}
    if not set(leakage_v_thresh) <= {2}:
        raise ValueError(f"leakage_v_thresh: the factorized V is implemented for the order 2, not {sorted(leakage_v_thresh)}")
    if not set(metric_ints) <= set(range(1, metric_order + 1)):
        raise ValueError(f"metric_ints: orders {sorted(metric_ints)} outside of 1 ... {metric_order}")
    diag_timer = timer()
    precon_timer = timer()
    matrix_timer = timer()

    frag_sectors = [[(chg, mult) for chg, mult, _ in local_sectors(rho['n_states'], monomer_charges[m])] for m, rho in enumerate(dens)]
    dimer_sectors = [(sec0, sec1) for sec0 in frag_sectors[0] for sec1 in frag_sectors[1]]
    def M(blocks, terms):
        return XR_term.dimer_matrix(blocks, terms, (0,1), dimer_sectors, matrix_timer)

    # the blocks of every set of integrals (compressed ones only enter through their orders), and their terms
    engines, memo = {}, {}
    def engine(i):
        if id(i) not in engines:
            symm_ints, bior_ints, _ = i
            S = symm_ints.S
            contract_cache = precontract_S(dens, symm_ints, precon_timer)
            def B(integrals, diagrams):
                blocks = diagrammatic_expansion.blocks(densities=dens, integrals=integrals, diagrams=diagrams, contract_cache=contract_cache,
                                                       timings=diag_timer, precon_timings=precon_timer, target_multiplicity=target_multiplicity)
                blocks._supersys_info.pair_slots = leakage_pair_slots    # (diagrams/pair_slots.py)
                return blocks
            engines[id(i)] = struct(B=B, contract_cache=contract_cache,
                                    S=S_term_blocks(dens, symm_ints, diag_timer, precon_timer, target_multiplicity),
                                    ST_symm=B(struct(S=S, T=symm_ints.T), ST_diagrams), SU_symm=B(struct(S=S, U=symm_ints.U), SU_diagrams),
                                    SV_symm=B(struct(S=S, V=symm_ints.V), SV_diagrams), ST_bior=B(struct(S=S, T=bior_ints.T), ST_diagrams),
                                    SU_bior=B(struct(S=S, U=bior_ints.U), SU_diagrams), SV_bior=B(struct(S=S, V=bior_ints.V), SV_diagrams),
                                    SV_diff=B(struct(S=S, V=bior_ints.V_diff), SV_diagrams))
        return engines[id(i)]
    def term(i, name, f):
        if (id(i), name) not in memo:
            memo[id(i), name] = f()
        return memo[id(i), name]
    def S_order(i, k):
        """the order-k term of S2 (k = 0: the identity; symmetric, see overlap_terms)"""
        return term(i, ("S", k), lambda: XR_term.dimer_matrix(engine(i).S, {0: D.S0[0]} if k == 0 else {2: D.S2[k]}, (0,1),
                                                              dimer_sectors, matrix_timer, symmetric=True))
    for k, matrix in (metric_terms or {}).items():
        if k <= metric_order:
            memo[id(metric_ints.get(k, ints)), ("S", k)] = matrix
    def H0(i, monomer=True):
        """the XR[0] Hamiltonian (dimer-sector ordering) with or without the monomer terms"""
        e = engine(i)
        def f():
            terms = {1: D.ST1[0], 2: D.ST2[0]} if monomer else {2: D.ST2[0]}
            H = M(e.ST_bior, terms) + M(e.SU_bior, {k: {1: D.SU1, 2: D.SU2}[k][0] for k in terms})
            return H + M(e.SV_bior, {k: {1: D.SV1, 2: D.SV2}[k][0] for k in terms})
        return term(i, ("H0", monomer), f)
    def L1(i):
        return term(i, "L1", lambda: _s2h2(engine(i), 1, dimer_sectors, matrix_timer) - (S_order(i, 0) + S_order(i, 1)) @ H0(i))
    def dL2(i):
        """L^[2] - L^[1]: only the terms of S2H2 that change from XR[1] to XR[2] (_s2h2), i.e. the order-2 diagrams and
        the orders 0 and 1 with the integrals that change between them (as differences), and S2_(2) H0"""
        def f():
            symm_ints, bior_ints, _ = i
            e, S = engine(i), symm_ints.S
            dT = _int_difference(symm_ints.T, bior_ints.T)
            dU = _int_difference(symm_ints.U, bior_ints.U)
            dV0 = _int_difference(symm_ints.V, bior_ints.V_diff, two_electron=True)
            dV1 = _int_difference(bior_ints.V_diff, bior_ints.V, two_electron=True)
            if leakage_v_thresh.get(2) is None:
                SV_bior = e.SV_bior
            else:
                SV_bior = V_factorized_blocks(dens, struct(S=S, V=bior_ints.V), SV_diagrams, e.contract_cache, diag_timer, precon_timer,
                                              target_multiplicity, leakage_v_thresh[2])
                SV_bior._supersys_info.pair_slots = leakage_pair_slots
            dS2H2 = M(e.B(struct(S=S, T=dT), ST_diagrams), {2: D.ST2[1]}) + M(e.ST_bior, {2: D.ST2[2]}) \
                  + M(e.B(struct(S=S, U=dU), SU_diagrams), {2: D.SU2[1]}) + M(e.SU_bior, {2: D.SU2[2]}) \
                  + M(e.B(struct(S=S, V=dV0), SV_diagrams), {1: D.SV1[0], 2: D.SV2[0]}) \
                  + M(e.B(struct(S=S, V=dV1), SV_diagrams), {2: D.SV2[1]}) + M(SV_bior, {2: D.SV2[2]})
            return dS2H2 - S_order(i, 2) @ H0(i)
        return term(i, "dL2", f)

    leakage = L1(leakage_ints.get(1, ints))
    if leakage_order == 2:
        leakage = leakage + dL2(leakage_ints.get(2, ints))
    metric = sum(S_order(metric_ints.get(k, ints), k) for k in range(metric_order + 1))
    metric = 0.5 * (metric + metric.T)

    # (as for XR[1]/[2] in get_xr_H, the metric is inverted on the sector pairs, which can couple to target_multiplicity)
    coupling = numpy.concatenate([numpy.full(dens[0]['n_states'][chg0][mult0] * dens[1]['n_states'][chg1][mult1],
                                             multiplicities_couple(mult0, mult1, target_multiplicity))
                                  for (chg0, mult0), (chg1, mult1) in dimer_sectors])
    if target_charge is not None:
        block = target_block(dens, dimer_sectors, target_charge, target_multiplicity)
        low = numpy.linalg.eigvalsh(metric[numpy.ix_(block, block)])[0]
        if low < S_WARN_THRESH:
            print(f"WARNING: {'leakage metric' if metric_order > leakage_order else f'XR[{metric_order}] (leakage form)'}: S2^[{metric_order}] is {'indefinite' if low < 0 else 'nearly singular'} "
                  f"(lowest eigenvalue {low:.5f} in the target block), the energy {'has' if low < 0 else 'may have'} spurious eigenvalues")
    metric = metric + numpy.diag(~coupling * 1.0)

    e = engine(ints)
    H1 = [  XR_term.monomer_matrix(e.ST_bior, {1: D.ST1[0]}, m, frag_sectors[m], matrix_timer)
          + XR_term.monomer_matrix(e.SU_bior, {1: D.SU1[0]}, m, frag_sectors[m], matrix_timer)
          + XR_term.monomer_matrix(e.SV_bior, {1: D.SV1[0]}, m, frag_sectors[m], matrix_timer) for m in [0,1]]
    H2blocked = H0(ints, monomer=False) + numpy.linalg.solve(metric, leakage)
    return H1, _product_ordering(H2blocked, dens, frag_sectors, dimer_sectors)

def get_leakage_metric_states(ints, dens, monomer_charges, target_multiplicity, target_state=0, target_charge=0, metric_order=4,
                              leakage_ints=None, metric_ints=None, leakage_v_thresh=None, leakage_order=2, metric_terms=None,
                              leakage_pair_slots=2.5):
    """get_xr_states for the leakage-metric Hamiltonian (get_leakage_metric_H)"""
    H1, H2 = get_leakage_metric_H(ints, dens, monomer_charges, target_multiplicity, target_charge=target_charge,
                                  metric_order=metric_order, leakage_ints=leakage_ints, metric_ints=metric_ints,
                                  leakage_v_thresh=leakage_v_thresh, leakage_order=leakage_order, metric_terms=metric_terms,
                                  leakage_pair_slots=leakage_pair_slots)
    mask = product_basis_mask(dens, monomer_charges, target_charge, target_multiplicity)
    return get_xr_states_from_H(H1, H2, target_state=target_state, basis_mask=mask)
