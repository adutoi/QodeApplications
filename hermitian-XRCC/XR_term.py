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

def _evaluate_block(result, op_blocks, frag_order, active_diagrams, subsys_indices, subsys_charges, timings, bra_det=False, ket_det=False):
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
    n_states_i = [rho[m[x]]['n_states_bra'][chg_i] for x,(chg_i,_) in enumerate(subsys_charges)]
    n_states_j = [rho[m[x]]['n_states'][chg_j] for x,(_,chg_j) in enumerate(subsys_charges)]
    loops = [(m_,range(n_frag)) for m_ in range(frag_order)]
    def kernel(*frags):
        nonlocal result
        if ascending(frags):    # only loop over unique groups of size frag_order
            other_charges_match = True
            frags_chgs_i, frags_chgs_j = 0, 0
            for x,(chg_i,chg_j) in enumerate(subsys_charges):
                if x not in frags:
                    if chg_i!=chg_j:  other_charges_match = False
                else:
                    frags_chgs_i += chg_i
                    frags_chgs_j += chg_j
            if other_charges_match and frags_chgs_i==frags_chgs_j:
                block = None
                for diagram in active_diagrams:
                    timings.start()
                    diagram_block = op_blocks[tuple(m[x] for x in frags)][tuple(subsys_charges[x] for x in frags)][diagram]
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
                            block[indices] += diagram_block
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

def monomer_matrix(op_blocks, active_diagrams, subsys_index, charge_blocks, timings):
    # This code is restricted specifically to monomer (sub)systems
    rho = op_blocks.densities[subsys_index]
    dim_bra = sum(rho['n_states_bra'][chg] for chg in charge_blocks)
    dim_ket = sum(rho['n_states'][chg] for chg in charge_blocks)
    Matrix = numpy.zeros((dim_bra,dim_ket))
    #
    Ibeg = 0
    for chg_i in charge_blocks:
        Iend = Ibeg + rho['n_states_bra'][chg_i]
        Jbeg = 0
        for chg_j in charge_blocks:
            Jend = Jbeg + rho['n_states'][chg_j]
            result = Matrix[Ibeg:Iend,Jbeg:Jend]
            subsys_charges = [(chg_i,chg_j)]
            for frag_order in active_diagrams:
                _evaluate_block(result, op_blocks, frag_order, active_diagrams[frag_order], (subsys_index,), subsys_charges, timings)
            Jbeg = Jend
        Ibeg = Iend
    return Matrix

def dimer_matrix(op_blocks, active_diagrams, subsys_indices, charge_blocks, timings, bra_det=False, ket_det=False):
    # This code is restricted specifically to dimer (sub)systems
    rho1, rho2 = (op_blocks.densities[m] for m in subsys_indices)
    dim_bra = sum(rho1['n_states_bra'][chg1]*rho2['n_states_bra'][chg2] for chg1,chg2 in charge_blocks)
    dim_ket = sum(rho1['n_states'][chg1]*rho2['n_states'][chg2] for chg1,chg2 in charge_blocks)
    if bra_det and not ket_det:
        Matrix = numpy.zeros(dim_bra)
        #
        Ibeg = 0
        for chg_i1,chg_i2 in charge_blocks:
            Iend = Ibeg + rho1['n_states_bra'][chg_i1]*rho2['n_states_bra'][chg_i2]
            for chg_j1,chg_j2 in charge_blocks:
                result = Matrix[Ibeg:Iend]
                subsys_charges = [(chg_i1,chg_j1),(chg_i2,chg_j2)]
                for frag_order in active_diagrams:
                    _evaluate_block(result, op_blocks, frag_order, active_diagrams[frag_order], subsys_indices, subsys_charges, timings, bra_det=bra_det)
            Ibeg = Iend
    elif ket_det and not bra_det:
        Matrix = numpy.zeros((dim_ket))
        #
        #Ibeg = 0
        for chg_i1,chg_i2 in charge_blocks:
            #Iend = Ibeg + rho1['n_states_bra'][chg_i1]*rho2['n_states_bra'][chg_i2]
            Jbeg = 0
            for chg_j1,chg_j2 in charge_blocks:
                Jend = Jbeg + rho1['n_states'][chg_j1]*rho2['n_states'][chg_j2]
                result = Matrix[Jbeg:Jend]
                subsys_charges = [(chg_i1,chg_j1),(chg_i2,chg_j2)]
                for frag_order in active_diagrams:
                    _evaluate_block(result, op_blocks, frag_order, active_diagrams[frag_order], subsys_indices, subsys_charges, timings, ket_det=ket_det)
                Jbeg = Jend
            #Ibeg = Iend
    else:
        #dim_ket = sum(rho1['n_states'][chg1]*rho2['n_states'][chg2] for chg1,chg2 in charge_blocks)
        Matrix = numpy.zeros((dim_bra,dim_ket))
        #
        Ibeg = 0
        for chg_i1,chg_i2 in charge_blocks:
            Iend = Ibeg + rho1['n_states_bra'][chg_i1]*rho2['n_states_bra'][chg_i2]
            Jbeg = 0
            for chg_j1,chg_j2 in charge_blocks:
                Jend = Jbeg + rho1['n_states'][chg_j1]*rho2['n_states'][chg_j2]
                result = Matrix[Ibeg:Iend,Jbeg:Jend]
                subsys_charges = [(chg_i1,chg_j1),(chg_i2,chg_j2)]
                for frag_order in active_diagrams:
                    _evaluate_block(result, op_blocks, frag_order, active_diagrams[frag_order], subsys_indices, subsys_charges, timings)
                Jbeg = Jend
            Ibeg = Iend
    return Matrix
