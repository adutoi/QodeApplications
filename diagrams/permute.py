#    (C) Copyright 2024, 2025 Anthony D. Dutoi
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

# Return a list of list, where the internal lists are permutations of the original indexable input.
def permutations(indexable, _prefix=None):
    indexable = list(indexable)
    _prefix = [] if _prefix is None else _prefix
    if len(indexable)<=1:
        return [_prefix + indexable]
    else:
        indexable_permutations = []
        for i in range(len(indexable)):
            local_copy = list(indexable)
            p = local_copy.pop(i)    # gets the i-th element and removes it from local_copy
            indexable_permutations += permutations(local_copy, _prefix=_prefix+[p])
        return indexable_permutations

# Return a list of dicts, where the dict entries can be interpreted as key->value substitutions necessary
# to generate permutations of the original indexable input.
def permutation_subs(indexable):
    return [dict(zip(indexable,permutation)) for permutation in permutations(indexable)]

# Returns a dict with two keys, 0 and 1, each containing a list of lists, where the internal lists are permutations
# of the original indexable input.  Those in the list mapped to zero have even parity with respect to the input,
# and those mapped to 1 have odd parity. 
def permutations_by_parity(indexable):
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
            remainder_permutations = permutations_by_parity(remainder)
            indexable_permutations[p] += [[item]+P for P in remainder_permutations[0]]
            indexable_permutations[q] += [[item]+P for P in remainder_permutations[1]]
            p, q = q, p
        return indexable_permutations

# Returns 0 if A and B are not permutations of each other and the sign (+/- 1) of the permutation otherwise
def are_permutations(A, B):
    result = 0
    B_permutations = permutations_by_parity(B)
    for parity in [0, 1]:
        if list(A) in B_permutations[parity]:
            result = (-1)**parity
    return result
