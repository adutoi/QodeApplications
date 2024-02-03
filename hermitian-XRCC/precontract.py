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

p,q,r,s = "pqrs"

def _contract(rho, V, new_label, old_label, V_indices, rho_indices):
        rho[new_label] = {}
        for charges in rho[old_label]:
            rho[new_label][charges] = []
            for row_in in rho[old_label][charges]:
                row_out = []
                for old_rho in row_in:
                    row_out += [evaluate(V(*V_indices) @ old_rho(*rho_indices))]
                rho[new_label][charges] += [row_out]

def precontract(fragments, integrals):
    for m,rho in enumerate(fragments):
        V = integrals.V[m,m,m,m]
        _contract(rho, V, "Va", "ccaaa", (p,q,r,s), (p,q,0,r,s))
        _contract(rho, V, "cV", "cccaa", (p,q,r,s), (p,q,0,s,r))
