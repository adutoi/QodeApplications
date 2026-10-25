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
import qode.math
from qode.util import struct, timer
import excitonic
import diagrammatic_expansion   # defines information structure for housing results of diagram evaluations
import XR_term                  # knows how to use ^this information to pack a matrix for use in XR model
from diagrams import S_diagrams, ST_diagrams, SU_diagrams, SV_diagrams    # definitions of diagrams needed for S and SH
import os
import precontract as precontract_module
from precontract import precontract
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

def overlap_spectra(ints, dens, monomer_charges, target_multiplicity, target_charge=0, max_order=3):
    """
    {n: (lowest, largest eigenvalue)} of the product-state overlap S2 with the S expansion up to order n = 1 ...
    max_order, in the target block (target_block).  The orders 3 and 4 require the densities of xr_level 1 and 2
    (build_frag_densities).  Cost (Be2, Davidson-XR[0] states, streamed): S2^[1] + S2^[2] about 1/3 of the XR[0]
    Hamiltonian build, S2^[3] about as much as the latter (factor 0.5 in 6-31G, 1.4 in cc-pVDZ, growing with the
    number of orbitals), S2^[4] 10 to 35 times more.
    """
    if any(rho.get('rectangular', False) for rho in dens):
        raise ValueError("the overlap spectra require square densities (bra states = ket states)")
    if max_order > 4:
        raise NotImplementedError("the S expansion is implemented up to order 4")
    symm_ints = ints[0]
    t = timer()
    S_blocks = diagrammatic_expansion.blocks(densities=dens, integrals=symm_ints.S, diagrams=S_diagrams,
                                             contract_cache=precontract(dens, symm_ints.S, t), timings=t, precon_timings=t,
                                             target_multiplicity=target_multiplicity)
    frag_sectors = [[(chg, mult) for chg, mult, _ in local_sectors(rho['n_states'], monomer_charges[m])] for m, rho in enumerate(dens)]
    dimer_sectors = [(sec0, sec1) for sec0 in frag_sectors[0] for sec1 in frag_sectors[1]]
    mask = target_block(dens, dimer_sectors, target_charge, target_multiplicity)
    spectra, S2 = {}, None
    for n in range(1, max_order + 1):
        terms = {0: D.S0[0], 2: D.S2[1]} if n == 1 else {2: D.S2[n]}    # (the identity with the first order)
        term = XR_term.dimer_matrix(S_blocks, terms, (0, 1), dimer_sectors, t)[numpy.ix_(mask, mask)]
        S2 = term if S2 is None else S2 + term
        eigvals = numpy.linalg.eigvalsh(0.5 * (S2 + S2.T))
        spectra[n] = (eigvals[0], eigvals[-1])
    return spectra

def check_overlap(ints, dens, monomer_charges, target_multiplicity, target_charge=0, max_order=3, xr_orders=(1, 2)):
    """
    Prints the lowest and largest eigenvalues of the product-state overlap S2^[n], n = 1 ... max_order (overlap_spectra)
    and warns for the orders xr_orders (XR[n] with n >= 1), whose S2^[n] is nearly singular or indefinite, or whose
    lowest eigenvalue is not converged (S_WARN_THRESH, S_CONV_THRESH, relative to the one of max_order), and if the
    product basis itself is nearly linearly dependent (lowest eigenvalue of max_order).  Returns the spectra.
    """
    spectra = overlap_spectra(ints, dens, monomer_charges, target_multiplicity, target_charge, max_order)
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
    contract_cache = precontract(BeN_rho, symm_ints.S, precon_timer)
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
        H1 = []
        for m in [0,1]:
            H1 += [  XR_term.monomer_matrix(ST_blocks_symm, {1: D.ST1[0]}, m, frag_sectors[m], matrix_timer) \
                   + XR_term.monomer_matrix(SU_blocks_symm, {1: D.SU1[0]}, m, frag_sectors[m], matrix_timer) \
                   + XR_term.monomer_matrix(SV_blocks_symm, {1: D.SV1[0]}, m, frag_sectors[m], matrix_timer) ]

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
        S_cache = contract_cache if all(a is b for a, b in zip(S_rho, BeN_rho)) else precontract(S_rho, symm_ints.S, precon_timer)
        S_blocks       = diagrammatic_expansion.blocks(densities=S_rho, integrals=symm_ints.S, diagrams=S_diagrams, contract_cache=S_cache, timings=diag_timer, precon_timings=precon_timer, target_multiplicity=target_multiplicity)
        SV_blocks_diff = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=struct(S=symm_ints.S, V=bior_ints.V_diff), diagrams=SV_diagrams, contract_cache=contract_cache, timings=diag_timer, precon_timings=precon_timer, target_multiplicity=target_multiplicity)

        H1 = []
        for m in [0,1]:
            H1 += [  XR_term.monomer_matrix(ST_blocks_symm, {1: D.ST1[0]}, m, frag_sectors[m], matrix_timer) \
                   + XR_term.monomer_matrix(SU_blocks_symm, {1: D.SU1[0]}, m, frag_sectors[m], matrix_timer) \
                   + XR_term.monomer_matrix(SV_blocks_symm, {1: D.SV1[0]}, m, frag_sectors[m], matrix_timer) ]

        S_diagrams_order = D.S2[1] if xr_order == 1 else D.S2[1] + D.S2[2]
        S2 = XR_term.dimer_matrix(S_blocks, {0: D.S0[0], 2: S_diagrams_order}, (0,1), dimer_sectors, matrix_timer)

        if xr_order == 1:
            S2H2  = XR_term.dimer_matrix(ST_blocks_symm, {1: D.ST1[0], 2: D.ST2[0]}, (0,1), dimer_sectors, matrix_timer)
            S2H2 += XR_term.dimer_matrix(SU_blocks_symm, {1: D.SU1[0], 2: D.SU2[0]}, (0,1), dimer_sectors, matrix_timer)
            S2H2 += XR_term.dimer_matrix(ST_blocks_bior, {2: D.ST2[1]}, (0,1), dimer_sectors, matrix_timer)
            S2H2 += XR_term.dimer_matrix(SU_blocks_bior, {2: D.SU2[1]}, (0,1), dimer_sectors, matrix_timer)
            S2H2 += XR_term.dimer_matrix(SV_blocks_diff, {1: D.SV1[0], 2: D.SV2[0]}, (0,1), dimer_sectors, matrix_timer)
            S2H2 += XR_term.dimer_matrix(SV_blocks_bior, {2: D.SV2[1]}, (0,1), dimer_sectors, matrix_timer)
        else:
            S2H2  = XR_term.dimer_matrix(ST_blocks_symm, {1: D.ST1[0], 2: D.ST2[0] + D.ST2[1]}, (0,1), dimer_sectors, matrix_timer)
            S2H2 += XR_term.dimer_matrix(SU_blocks_symm, {1: D.SU1[0], 2: D.SU2[0] + D.SU2[1]}, (0,1), dimer_sectors, matrix_timer)
            S2H2 += XR_term.dimer_matrix(ST_blocks_bior, {2: D.ST2[2]}, (0,1), dimer_sectors, matrix_timer)
            S2H2 += XR_term.dimer_matrix(SU_blocks_bior, {2: D.SU2[2]}, (0,1), dimer_sectors, matrix_timer)
            S2H2 += XR_term.dimer_matrix(SV_blocks_symm, {1: D.SV1[0], 2: D.SV2[0]}, (0,1), dimer_sectors, matrix_timer)
            S2H2 += XR_term.dimer_matrix(SV_blocks_diff, {2: D.SV2[1]}, (0,1), dimer_sectors, matrix_timer)
            S2H2 += XR_term.dimer_matrix(SV_blocks_bior, {2: D.SV2[2]}, (0,1), dimer_sectors, matrix_timer)

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

    # Reorder H2 from the dimer-sector ordering of XR_term.dimer_matrix, i.e.
    #     for sector pair: for state0: for state1
    # into the tensor-product ordering of the local fragment bases (see local_sectors).
    # (rows refer to the bra states, columns to the ket states, which differ for rectangular builds)
    rho0, rho1 = BeN_rho
    def product_mapping(key):
        dimer_index = {}
        idx = 0
        for (chg0, mult0), (chg1, mult1) in dimer_sectors:
            for state0 in range(rho0[key][chg0][mult0]):
                for state1 in range(rho1[key][chg1][mult1]):
                    dimer_index[(chg0, mult0, state0, chg1, mult1, state1)] = idx
                    idx += 1
        frag_states = [[(chg, mult, state) for chg, mult in frag_sectors[m] for state in range(rho[key][chg][mult])] for m, rho in enumerate(BeN_rho)]
        return [dimer_index[key0 + key1] for key0 in frag_states[0] for key1 in frag_states[1]]

    H2 = H2blocked[numpy.ix_(product_mapping('n_states_bra'), product_mapping('n_states'))]

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

def get_xr_states(ints, dens, xr_order, monomer_charges, target_multiplicity, target_state=0, target_charge=0):
    H1, H2 = get_xr_H(ints, dens, xr_order, monomer_charges, target_multiplicity, target_charge=target_charge)
    mask = product_basis_mask(dens, monomer_charges, target_charge, target_multiplicity)
    return get_xr_states_from_H(H1, H2, target_state=target_state, basis_mask=mask)
