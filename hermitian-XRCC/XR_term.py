#    (C) Copyright 2023 Anthony D. Dutoi
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

def _evaluate_block(result, op_blocks, frag_order, active_diagrams, subsys_indices, subsys_charges):
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
    n_states_i = [rho[m[x]]['n_states'][chg_i] for x,(chg_i,_) in enumerate(subsys_charges)]
    n_states_j = [rho[m[x]]['n_states'][chg_j] for x,(_,chg_j) in enumerate(subsys_charges)]
    loops = [(m_,range(n_frag)) for m_ in range(frag_order)]
    def kernel(*frags):
        nonlocal result
        if ascending(frags):    # only loop over unique groups of size frag_order
            other_charges_match = True
            for x,(chg_i,chg_j) in enumerate(subsys_charges):
                if x not in frags and chg_i!=chg_j:
                    other_charges_match = False
            if other_charges_match:
                block = None
                for diagram in active_diagrams:
                    diagram_block = op_blocks[tuple(m[x] for x in frags)][tuple(subsys_charges[x] for x in frags)][diagram]
                    if diagram_block is not None:
                        if block is None:
                            block = numpy.zeros(n_states_i+n_states_j)    # concatenate lists and make ndarray
                        # !!! phases not yet implemented should be handled here.
                        for I in compound_range([range(n) for n in n_states_i], inactive=frags):    # with frags of interest inactive, i vs j does not matter here
                            for frag in frags:  I[frag] = full
                            indices = tuple(I+I)    # concatenate I with itself for "diagonal" element (wrt specified indices)
                            block[indices] += diagram_block
                if block is not None:
                    block = block.reshape(numpy.prod(n_states_i), numpy.prod(n_states_j))
                    result += block
    recursive_looper(loops, kernel)

def monomer_matrix(op_blocks, active_diagrams, subsys_index, charge_blocks):
    # This code is restricted specifically to monomer (sub)systems
    rho = op_blocks.densities[subsys_index]
    dim = sum(rho['n_states'][chg] for chg in charge_blocks)
    Matrix = numpy.zeros((dim,dim))
    #
    Ibeg = 0
    for chg_i in charge_blocks:
        Iend = Ibeg + rho['n_states'][chg_i]
        Jbeg = 0
        for chg_j in charge_blocks:
            Jend = Jbeg + rho['n_states'][chg_j]
            result = Matrix[Ibeg:Iend,Jbeg:Jend]
            subsys_charges = [(chg_i,chg_j)]
            for frag_order in active_diagrams:
                _evaluate_block(result, op_blocks, frag_order, active_diagrams[frag_order], (subsys_index,), subsys_charges)
            Jbeg = Jend
        Ibeg = Iend
    return Matrix

def dimer_matrix(op_blocks, active_diagrams, subsys_indices, charge_blocks):
    # This code is restricted specifically to dimer (sub)systems
    rho1, rho2 = (op_blocks.densities[m] for m in subsys_indices)
    dim = sum(rho1['n_states'][chg1]*rho2['n_states'][chg2] for chg1,chg2 in charge_blocks)
    Matrix = numpy.zeros((dim,dim))
    #
    Ibeg = 0
    for chg_i1,chg_i2 in charge_blocks:
        Iend = Ibeg + rho1['n_states'][chg_i1]*rho2['n_states'][chg_i2]
        Jbeg = 0
        for chg_j1,chg_j2 in charge_blocks:
            Jend = Jbeg + rho1['n_states'][chg_j1]*rho2['n_states'][chg_j2]
            result = Matrix[Ibeg:Iend,Jbeg:Jend]
            subsys_charges = [(chg_i1,chg_j1),(chg_i2,chg_j2)]
            for frag_order in active_diagrams:
                _evaluate_block(result, op_blocks, frag_order, active_diagrams[frag_order], subsys_indices, subsys_charges)
            Jbeg = Jend
        Ibeg = Iend
    return Matrix
