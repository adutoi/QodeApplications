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
from qode.math.tensornet import evaluate

def precontract(contract_cache, contract_label, densities, integrals, int_indices, rho_label, rho_indices):
    rhos = [densities_m[rho_label] for densities_m in densities]
    for m,rhos_m in rhos:
        ints_m = integrals[(m,)*len(int_indices)]
        if m not in contract_cache:  contract_cache[m] = {}
        contract_cache_m = contract_cache[m]
        if contract_label not in contract_cache_m:
            contract_cache_m[contract_label] = {}
            for charge in rhos_m:
                contract_cache_m[contract_label][charge] = []
                for rho_row in rhos_m[charge]:
                    row = []
                    for rho in rho_row:
                        row += [evaluate(ints_m(*ints_indices) @ rho(*rho_indices))]
                    contract_cache_m[contract_label][charge] += [row]
