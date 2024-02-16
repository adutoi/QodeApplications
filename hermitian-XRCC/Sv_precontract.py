#    (C) Copyright 2024 Anthony D. Dutoi
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
from qode.util.dynamic_array import dynamic_array, cached
from qode.math.tensornet     import evaluate



def precontract(densities, integrals):
    p,q,r,s = "pqrs"
    precontractions = {}

    def ccaaaMpqXrs_Vpqrs(m):
        V = integrals.V[m,m,m,m]
        densities_m = densities[m]
        n_states = densities_m["n_states"]
        def ccaaaMpqXrs_Vpqrs_m(chg_i,chg_j):
            def ccaaaMpqXrs_Vpqrs_m_charges(i,j):
                ccaaa = densities_m["ccaaa"][chg_i,chg_j][i,j]
                return evaluate(ccaaa(p,q,0,r,s) @ V(p,q,r,s))
            if chg_i-chg_j==+1:
                return dynamic_array(cached(ccaaaMpqXrs_Vpqrs_m_charges), [range(n_states[chg_i]), range(n_states[chg_j])])
            else:
                return None
        return dynamic_array(cached(ccaaaMpqXrs_Vpqrs_m), [n_states.keys()]*2)
    precontractions["ccaaa#pqXrs_Vpqrs"] = dynamic_array(cached(ccaaaMpqXrs_Vpqrs), [range(len(densities))])

    return precontractions
