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

def permutations(indexable):
    if len(indexable)<2:
        return {0: [list(indexable)], 1:[]}    # only every happens if initial indexable has 0 or 1 items in it
    elif len(indexable)==2:
        return {
            0: [[indexable[0], indexable[1]]],
            1: [[indexable[1], indexable[0]]]
        }
    else:
        indexable_permutations = {0:[], 1:[]}
        p, q = 0, 1
        for i,item in enumerate(indexable):
            remainder = indexable[:i] + indexable[i+1:]
            remainder_permutations = permutations(remainder)
            indexable_permutations[p] += [[item]+P for P in remainder_permutations[0]]
            indexable_permutations[q] += [[item]+P for P in remainder_permutations[1]]
            p, q = q, p
        return indexable_permutations

def are_permutations(A, B):
    result = False, None
    B_permutations = permutations(B)
    for parity in [0, 1]:
        if list(A) in B_permutations[parity]:
            result = True, parity
    return result
