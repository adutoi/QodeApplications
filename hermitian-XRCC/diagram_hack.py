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

# This is meant as a minimally invasive hack that accomplishes the functionality needed 
# by the state_solver stuff while preserving the original structure of the code.
#     1. The state_solver code needs the ability to contract the ket indices with eachother (?- these belong to different fragments!)
#     2. The structure of the code should be such that the diagrams are completely blind to any permutations on the outside, and only need information from "X"
#        (which is a goal that it was notably just shy of after the last overhaul)
# This is not a completely general solution and it should be replaced by something more philosophically sound once the dust settles on what we need.

def state_indices(contract_last):
    i0, i1, j0, j1 = 0, 1, 2, 3    # Just enumerate the free (uncontracted remaining) indices of a two-fragment transition density tensor
    if contract_last:              # But if we want to contract the last two indices, we can give tensornet these values instead.
        j0, j1 = "z", "z"
    return i0, i1, j0, j1

def no_result(X, contract_last):
    (i0s,j0s),(i1s,j1s) = X.n_states[0], X.n_states[1]
    return (contract_last and j0s!=j1s)    # encapsulates the trigger to return [] in case last indices cannot be contracted due to mismatched lengths
