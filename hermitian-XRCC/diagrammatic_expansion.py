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

from qode.util import struct
from precontract import precontract

##########
# This function does the heavy lifting.  Called by instances of innermost class below.
##########

# This function should be moved to build_diagram.py because it will be more understandble in that context (assuming it even survives the next overhaul)
def _build_block(diagram_term, permutation):
    frag_order = len(permutation)
    if frag_order==0:
        result = diagram_term()
    else:
        reorder = [m for m in permutation] + [frag_order+m for m in permutation]
        result = diagram_term().transpose(reorder)
    return result

##########
# These classes build something ('blocks' class) that looks like a nested array with lazy evaluation.
# It is specific to the kinds of fragment-divided diagrams in the XR model, which need density matrices to be computed,
#  but it is agnostic to the integrals and diagram implementation.  So should work for S, SH, etc.
##########

class _charges(object):
    def __init__(self, supersys_info, subsystem, charges, diagrams, bra_det):
        self._supersys_info = supersys_info
        self._subsystem = subsystem
        self._charges = charges
        self._diagrams = diagrams
        self._results = {}
        self._bra_det = bra_det
    def __getitem__(self, label):
        if label not in self._results:
            frag_order = len(self._subsystem)
            try:
                terms = self._diagrams.catalog[frag_order][label](self._supersys_info, tuple(zip(self._subsystem, self._charges)))
            except:
                raise NotImplementedError("diagram \'{}\' not implemented for {} bodies".format(label, frag_order))
            else:
                self._results[label] = None
                for term_permutation in terms:
                    if term_permutation is not None:
                        term, permutation = term_permutation
                        result = _build_block(term, permutation)    # This should have a bra_det flag, but not label!
                        if self._bra_det and len(result) == 0:
                            continue
                        if self._results[label] is None:  self._results[label]  = result
                        else:                             self._results[label] += result
        return self._results[label]

class _subsystem(object):
    def __init__(self, supersys_info, subsystem, diagrams, bra_det):
        self._supersys_info = supersys_info
        self._subsystem = subsystem
        self._diagrams = diagrams
        self._items = {}
        self._bra_det = bra_det
    def __getitem__(self, charges):
        if charges is None:  charges = tuple()    # just to make top-level syntax prettier
        charges = tuple(charges)                  # dict index must be hashable
        if charges not in self._items:
            self._items[charges] = _charges(self._supersys_info, self._subsystem, charges, self._diagrams, self._bra_det)
        return self._items[charges]

class blocks(object):
    def __init__(self, densities, integrals, diagrams, contract_cache, timings, precon_timings, bra_det=False):
        contract_cache = struct(rho_S=contract_cache, general=precontract(densities, integrals, precon_timings))
        self._supersys_info = struct(densities=densities, integrals=integrals, contract_cache=contract_cache, timings=timings)
        self._diagrams = diagrams
        self._items = {}
        self.densities = self._supersys_info.densities    # "public" member providing access to system definition
        self._bra_det = bra_det
    def __getitem__(self, subsystem):
        if subsystem is None:  subsystem = tuple()        # just to make top-level syntax prettier
        subsystem = tuple(subsystem)                      # dict index must be hashable
        if subsystem not in self._items:
            self._items[subsystem] = _subsystem(self._supersys_info, subsystem, self._diagrams, self._bra_det)
        return self._items[subsystem]
