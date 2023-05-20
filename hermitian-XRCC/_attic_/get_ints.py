#    (C) Copyright 2018, 2019, 2023 Anthony D. Dutoi
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
import tensorly
from qode.atoms.integrals.fragments import AO_integrals, semiMO_integrals, spin_orb_integrals, Nuc_repulsion

def tensorly_wrapper(rule):
    def wrap_it(*indices):
        return tensorly.tensor(rule(*indices))
    return wrap_it

def get_ints(fragments):
    AO_ints     = AO_integrals(fragments)
    SemiMO_ints = semiMO_integrals(AO_ints, [frag.basis.MOcoeffs for frag in fragments], cache=True)
    SemiMO_spin_ints = spin_orb_integrals(SemiMO_ints, rule_wrappers=[tensorly_wrapper], cache=True)
    return SemiMO_spin_ints, Nuc_repulsion(fragments)
