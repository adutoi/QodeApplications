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
from precontract import precontract
import diagram_lists as D



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

def build_frag_densities(dens_builder, frag_name, states, xr_level, n_elec_ref, bra_states=None):
    """
    states[chg][mult] = list of CSF vectors
    bra_states:         optional different bra states (same structure, e.g. single CSFs for the state gradients),
                        the densities are then <bra_states|...|states> and the Hamiltonian rectangular
    Returns the density dictionary of one fragment, including the bookkeeping entries
    required by diagrammatic_expansion.
    """
    states = drop_empty_sectors(states)
    if bra_states is None:
        rho = dens_builder.build_densities(frag_name, xr_level, states, states)
        bra_states = states
    else:
        bra_states = drop_empty_sectors(bra_states)
        if count_sectors(bra_states) != count_sectors(states):
            raise ValueError("bra and ket states need to have the same (chg, mult) sectors")
        rho = dens_builder.build_densities(frag_name, xr_level, bra_states, states)
    rho['n_states'] = {chg: {mult: len(vecs) for mult, vecs in mult_dict.items()} for chg, mult_dict in states.items()}
    rho['n_states_bra'] = {chg: {mult: len(vecs) for mult, vecs in mult_dict.items()} for chg, mult_dict in bra_states.items()}
    # <bra|ket> per sector, which replaces the Kronecker delta of spectator fragments (e.g. in u100)
    rho['overlap'] = {chg: {mult: (numpy.eye(len(vecs)) if bra_states is states else
                                   numpy.array(bra_states[chg][mult], dtype=float) @ numpy.array(vecs, dtype=float).T)
                            for mult, vecs in mult_dict.items()} for chg, mult_dict in states.items()}
    rho['n_elec'] = {chg: n_elec_ref - chg for chg in states}
    return rho

def get_xr_H(ints, dens, xr_order, monomer_charges, target_multiplicity):
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
        S_blocks       = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=symm_ints.S, diagrams=S_diagrams, contract_cache=contract_cache, timings=diag_timer, precon_timings=precon_timer, target_multiplicity=target_multiplicity)
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
            numpy.full(BeN_rho[0]['n_states'][chg0][mult0] * BeN_rho[1]['n_states'][chg1][mult1],
                       multiplicities_couple(mult0, mult1, target_multiplicity))
            for (chg0, mult0), (chg1, mult1) in dimer_sectors])
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
    H1, H2 = get_xr_H(ints, dens, xr_order, monomer_charges, target_multiplicity)
    mask = product_basis_mask(dens, monomer_charges, target_charge, target_multiplicity)
    return get_xr_states_from_H(H1, H2, target_state=target_state, basis_mask=mask)
