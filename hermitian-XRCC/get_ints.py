#    (C) Copyright 2018, 2019, 2023 Anthony D. Dutoi
# 
#    This file is part of Qode.
# 
#    Qode is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
# 
#    Qode is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
# 
#    You should have received a copy of the GNU General Public License
#    along with Qode.  If not, see <http://www.gnu.org/licenses/>.
#
import numpy
from qode.util.PyC import Double
from qode.atoms.integrals.fragments import AO_integrals, semiMO_integrals, spin_orb_integrals, Nuc_repulsion

def Double_array(rule):
    def wrap_it(*indices):
        return numpy.array(rule(*indices), dtype=Double.numpy)
    return wrap_it

def get_ints(fragments):
    # More needs to be done regarding the basis to prevent mismatches with the fragment states
    AO_ints     = AO_integrals(fragments)
    SemiMO_ints = semiMO_integrals(AO_ints, [frag.basis.MOcoeffs for frag in fragments], cache=True)     # Cache because multiple calls to each block during biorthogonalization

    SemiMO_spin_ints = spin_orb_integrals(SemiMO_ints, rule_wrappers=[Double_array])     # no need to cache because each block only called once by contraction code
    return SemiMO_spin_ints, Nuc_repulsion(fragments)
