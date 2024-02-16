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
from qode.util.dynamic_array import dynamic_array
from qode.math.tensornet     import evaluate

p,q,r,s = "pqrs"



def ccaaaMpqXrs_Vpqrs(densities, integrals):
    result = {}
    for m,densities_m in enumerate(densities):
        V = integrals.V[m,m,m,m]
        result[m] = {}
        for chg_bra,n_states_bra in densities_m["n_states"].items():
            for chg_ket,n_states_ket in densities_m["n_states"].items():
                if chg_bra-chg_ket==+1:
                    result[m][chg_bra,chg_ket] = {}
                    for i in range(n_states_bra):
                        for j in range(n_states_ket):
                            ccaaa = densities_m["ccaaa"][chg_bra,chg_ket][i,j]
                            result[m][chg_bra,chg_ket][i,j] = evaluate(ccaaa(p,q,0,r,s) @ V(p,q,r,s))
    return result



def precontract(densities, integrals):
    precontractions = {}
    precontractions["ccaaa{}pqXrs_Vpqrs"] = ccaaaMpqXrs_Vpqrs(densities, integrals)
    return precontractions
