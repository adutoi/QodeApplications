#    (C) Copyright 2023, 2025 Anthony D. Dutoi and Marco Bauer
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

import numpy
from qode.util import recursive_looper, compound_range

def _can_couple_to_target(multiplicities, target_mult):
    """
    Return True if the fragment multiplicities can couple to target_mult.

    Multiplicities are 2S+1, so for two fragments m1 and m2 the allowed
    total multiplicities are

        |m1-m2|+1, |m1-m2|+3, ..., m1+m2-1.

    For more than two fragments, couple them successively.

    If any multiplicity is None, the calculation is unrestricted and
    this spin-adapted selection rule is not applied.
    """

    if target_mult is None:
        return True

    if any(mult is None for mult in multiplicities):
        return True

    possible = {multiplicities[0]}

    for mult in multiplicities[1:]:
        new_possible = set()

        for m in possible:
            lo = abs(m - mult) + 1
            hi = m + mult - 1

            for total in range(lo, hi + 1, 2):
                new_possible.add(total)

        possible = new_possible

        if not possible:
            return False

    return target_mult in possible


def _multiplicity_block_is_zero(subsys_transitions, target_mult):
    """
    Return True if the bra/ket multiplicity sectors cannot contribute
    to a scalar Hamiltonian matrix element in the requested total-spin
    sector.
    """

    bra_mults = [
        transition[0][1]
        for transition in subsys_transitions
    ]

    ket_mults = [
        transition[1][1]
        for transition in subsys_transitions
    ]

    return (
        not _can_couple_to_target(bra_mults, target_mult)
        or
        not _can_couple_to_target(ket_mults, target_mult)
    )

def _evaluate_block(result, op_blocks, frag_order, active_diagrams, subsys_indices, subsys_transitions, timings, bra_det=False, ket_det=False):
    # Can handle (sub)systems with any number of fragments using diagrams of any fragment order.
    # So can use for trimer_matrix, etc.
    # !!! Phases for some fragment_order < subsystem_size have not yet been coded in (easy)!
    full = slice(None)
    def ascending(array):
        ascending = True
        if len(array)>1:
            v = array[0]
            for a in array[1:]:
                if a<=v:  ascending = False
                v = a
        return ascending
    #
    rho = op_blocks.densities
    m = subsys_indices    # alias makes code more readable
    n_frag = len(m)
    n_states_i = [
        rho[m[x]]['n_states_bra'][chg_i][mult_i]
        if mult_i is not None
        else rho[m[x]]['n_states_bra'][chg_i]
        for x,((chg_i,mult_i), _) in enumerate(subsys_transitions)
    ]

    n_states_j = [
        rho[m[x]]['n_states'][chg_j][mult_j]
        if mult_j is not None
        else rho[m[x]]['n_states'][chg_j]
        for x,(_, (chg_j,mult_j)) in enumerate(subsys_transitions)
    ]
    loops = [(m_,range(n_frag)) for m_ in range(frag_order)]
    def kernel(*frags):
        nonlocal result
        if ascending(frags):    # only loop over unique groups of size frag_order
            other_charges_match = True
            frags_chgs_i, frags_chgs_j = 0, 0
            for x,((chg_i,mult_i),(chg_j,mult_j)) in enumerate(subsys_transitions):
                if x not in frags:
                    if chg_i!=chg_j:  other_charges_match = False
                    if mult_i!=mult_j:other_charges_match = False
                else:
                    frags_chgs_i += chg_i
                    frags_chgs_j += chg_j
                # TODO: make similar test for multiplicities, but that is not quite that simple.
                # One could of course check for all possible multiplicity combinations with the
                # environment, but one probably gets all the contributions already from having the
                # spectator fragments in their ground state, along with their corresponding mult,
                # since the fragments are expected to be weakly coupled and charge transfer as well
                # as spin-flips are only induced by excitations, originating from correlations, which
                # are only part of the active (non-spectator) fragments.
                # TODO: the following charges test only tests, whether the charges are symmetric, which
                # is not general, because also frags of differing frags_chgs can couple!!!
            if other_charges_match and frags_chgs_i==frags_chgs_j:
                # TODO: Once the basis is not required to be the entire tensor product basis
                # anymore, the non-zero sectors can simply be filtered out earlier.
                if _multiplicity_block_is_zero(
                    subsys_transitions,
                    op_blocks.target_multiplicity,
                ):
                    return
                block = None
                for diagram in active_diagrams:
                    timings.start()
                    #diagram_block = op_blocks[tuple(m[x] for x in frags)][tuple(subsys_transitions[x] for x in frags)][diagram]
                    subsys_trans_frags = [subsys_transitions[x] for x in frags]
                    subsys_chgs = tuple((chg_i, chg_j) for ((chg_i,_),(chg_j,_)) in subsys_trans_frags)
                    subsys_mults = tuple((mult_i, mult_j) for ((_,mult_i),(_,mult_j)) in subsys_trans_frags)
                    
                    diagram_block = op_blocks[tuple(m[x] for x in frags)][subsys_chgs][subsys_mults][diagram]
                    timings.record("block evaluation")
                    if diagram_block is not None:
                        if block is None:
                            if bra_det:
                                block = numpy.zeros(n_states_i)    # make ndarray
                            elif ket_det:
                                block = numpy.zeros(n_states_j)    # make ndarray
                            else:
                                block = numpy.zeros(n_states_i+n_states_j)    # concatenate lists and make ndarray
                        # !!! phases not yet implemented should be handled here.
                        #count = 0
                        for J in compound_range([range(n) for n in n_states_j], inactive=frags):    # with frags of interest inactive, i vs j does not matter here
                            for frag in frags:  J[frag] = full
                            if bra_det or ket_det:
                                indices = tuple(J)
                            else:
                                indices = tuple(J+J)    # concatenate J with itself for "diagonal" element (wrt specified indices)
                            #try:
                            #if block[indices].shape != diagram_block.shape and len(diagram_block.shape) < 4:
                            #    print(block[indices].shape, diagram_block.shape)
                            #    diagram_block = diagram_block.T
                            #    print("diagram block has been transposed...this is just a test...why does it require transposing after all??????")
                            #try:
                            block[indices] += diagram_block
                            #    print("worked")
                            #except:
                            #    print(indices, block[indices], diagram_block)
                            #except IndexError:
                            #    print(indices)
                            #    count += 1
                            #    if count >= 130:
                            #        raise IndexError("blablabla")
                if block is not None:
                    if bra_det:
                        block = block.reshape(numpy.prod(n_states_i))
                    elif ket_det:
                        block = block.reshape(numpy.prod(n_states_j))
                    else:
                        block = block.reshape(numpy.prod(n_states_i), numpy.prod(n_states_j))
                    result += block
    recursive_looper(loops, kernel)

def monomer_matrix(
    op_blocks,
    active_diagrams,
    subsys_index,
    monomer_sectors,
    timings,
):
    rho = op_blocks.densities[subsys_index]

    dim_bra = sum(
        rho['n_states_bra'][chg][mult]
        for chg, mult in monomer_sectors
    )

    dim_ket = sum(
        rho['n_states'][chg][mult]
        for chg, mult in monomer_sectors
    )

    Matrix = numpy.zeros((dim_bra, dim_ket))

    Ibeg = 0

    for chg_i, mult_i in monomer_sectors:

        Iend = Ibeg + rho['n_states_bra'][chg_i][mult_i]

        Jbeg = 0

        for chg_j, mult_j in monomer_sectors:

            Jend = Jbeg + rho['n_states'][chg_j][mult_j]

            result = Matrix[Ibeg:Iend, Jbeg:Jend]

            subsys_transitions = [
                (
                    (chg_i, mult_i),
                    (chg_j, mult_j),
                )
            ]

            for frag_order in active_diagrams:

                _evaluate_block(
                    result,
                    op_blocks,
                    frag_order,
                    active_diagrams[frag_order],
                    (subsys_index,),
                    subsys_transitions,
                    timings,
                )

            #if numpy.linalg.norm(result) > 1e-6:
            #    print(subsys_index, subsys_transitions, result)

            Jbeg = Jend

        Ibeg = Iend

    return Matrix


def dimer_matrix(
    op_blocks,
    active_diagrams,
    subsys_indices,
    dimer_sectors,
    timings,
    bra_det=False,
    ket_det=False,
):
    # This code is restricted specifically to dimer (sub)systems
    rho1, rho2 = (
        op_blocks.densities[m]
        for m in subsys_indices
    )

    #
    # The ordering above is the matrix ordering.
    #
    bra_offsets = {}
    ket_offsets = {}

    Ibeg = 0
    Jbeg = 0

    for (chg1, mult1), (chg2, mult2) in dimer_sectors:

        key = (chg1, chg2, mult1, mult2)

        dim_bra = (
            rho1['n_states_bra'][chg1][mult1]
            * rho2['n_states_bra'][chg2][mult2]
        )

        dim_ket = (
            rho1['n_states'][chg1][mult1]
            * rho2['n_states'][chg2][mult2]
        )

        bra_offsets[key] = Ibeg
        ket_offsets[key] = Jbeg

        Ibeg += dim_bra
        Jbeg += dim_ket

    Matrix = numpy.zeros((Ibeg, Jbeg))

    #
    # Fill every allowed bra/ket sector.
    #
    for (bra_chg1, bra_mult1), (bra_chg2, bra_mult2) in dimer_sectors:

        bra_key = (
            bra_chg1,
            bra_chg2,
            bra_mult1,
            bra_mult2,
        )

        Ibeg = bra_offsets[bra_key]

        bra_dim = (
            rho1['n_states_bra'][bra_chg1][bra_mult1]
            * rho2['n_states_bra'][bra_chg2][bra_mult2]
        )

        Iend = Ibeg + bra_dim

        for (ket_chg1, ket_mult1), (ket_chg2, ket_mult2) in dimer_sectors:

            ket_key = (
                ket_chg1,
                ket_chg2,
                ket_mult1,
                ket_mult2,
            )

            Jbeg = ket_offsets[ket_key]

            ket_dim = (
                rho1['n_states'][ket_chg1][ket_mult1]
                * rho2['n_states'][ket_chg2][ket_mult2]
            )

            Jend = Jbeg + ket_dim

            result = Matrix[Ibeg:Iend, Jbeg:Jend]

            subsys_transitions = [
                (
                    (bra_chg1, bra_mult1),
                    (ket_chg1, ket_mult1),
                ),
                (
                    (bra_chg2, bra_mult2),
                    (ket_chg2, ket_mult2),
                )
            ]

            for frag_order in active_diagrams:

                _evaluate_block(
                    result,
                    op_blocks,
                    frag_order,
                    active_diagrams[frag_order],
                    subsys_indices,
                    subsys_transitions,
                    timings,
                    bra_det=bra_det,
                    ket_det=ket_det,
                )

    return Matrix
