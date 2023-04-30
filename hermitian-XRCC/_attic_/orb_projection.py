#    (C) Copyright 2023 Marco Bauer
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

# Here we perform the orbital rotation into the projected basis of the other
# fragment

import tensorly as tl
from tendot import tendot
#from cache import cached_member_function

# build the transformation matrix
# f0.T 0   00 01  f0 0
# 0  f1.T  10 11  0  f1 = U.T @ tensor @ U
# return transformation matrix as separated blocks

def transformation_mat(s01, s10, thresh=1e-5):
    # fragment dimers only!!!
    s00_proj = s01 @ s10
    s11_proj = s10 @ s01
    # this is probably a little faster with an actual eigenvalue decomposition function,
    # but it produced the same result
    s00_proj_eigs = tl.decomposition.parafac(s00_proj, len(s00_proj[0]), normalize_factors=True)
    s11_proj_eigs = tl.decomposition.parafac(s11_proj, len(s11_proj[0]), normalize_factors=True)
    # 0 and 1 spaces can be different after treshholding
    # Note, that U0 and U1 are the transpose, of what you would expect them to be!!!
    U0 = tl.tensor([vec for i, vec in enumerate(s00_proj_eigs[1][1].T) if s00_proj_eigs[0][i] > thresh])  # non negligible orbs on frag 0
    U1 = tl.tensor([vec for i, vec in enumerate(s11_proj_eigs[1][1].T) if s11_proj_eigs[0][i] > thresh])  # non negligible orbs on frag 1
    U0_0, U0_1 = tl.shape(U0)
    U1_0, U1_1 = tl.shape(U1)
    full = tl.zeros((U0_0 + U1_0, U0_1 + U1_1))
    full = tl.index_update(full, tl.index[:U0_0, :U0_1], U0)
    full = tl.index_update(full, tl.index[U0_0:, U0_1:], U1)
    return U0, U1, full

def orb_proj(U, ten):
    # fragment dimers only!!!
    # Due to definition of U0 and U1, the transformation looks like
    # U @ ten @ U.T
    ten_dim = tl.ndim(ten)
    # one can also write this recursive
    if ten_dim == 1:
        # p -> i
        return tendot(U, ten, axes=([1], [0]))  # ip p -> i
    elif ten_dim == 2:
        # pr -> ij
        partial = tendot(U, ten, axes=([1], [0]))  # ip pr -> ir
        return tendot(partial, U, axes=([1], [1]))  # ir jr -> ij
    elif ten_dim == 3:
        # pqr -> ijk
        partial = tendot(U, ten, axes=([1], [1]))  # jq pqr -> jpr
        partial = tendot(U, partial, axes=([1], [1]))  # ip jpr -> ijr
        return tendot(partial, U, axes=([2], [1]))  # ijr kr -> ijk
    elif ten_dim == 4:
        # pqrs -> ijkl
        partial = tendot(U, ten, axes=([1], [2]))  # kr pqrs -> kpqs
        partial = tendot(U, partial, axes=([1], [2]))  # jq kpqs -> jkps
        partial = tendot(U, partial, axes=([1], [2]))  # ip jkps -> ijks
        return tendot(partial, U, axes=([3], [1]))  # ijks ls -> ijkl
    elif ten_dim == 5:
        partial = tendot(U, ten, axes=([1], [3]))
        partial = tendot(U, partial, axes=([1], [3]))
        partial = tendot(U, partial, axes=([1], [3]))
        partial = tendot(U, partial, axes=([1], [3]))
        return tendot(partial, U, axes=([4], [1]))
    else:
        raise NotImplementedError(f"orb_proj has not been implemented for tensors of dimension {ten_dim}")

# Caching this is great for the runtime, but might lead to memory issues if the fragment number
# is high!
# Only cache this, if U and dens are available from small e.g. strings and also cached
#@cached_member_function
def orb_proj_density(U, density):
    # we loop over state indices in explicit loops anyway, so no need to make the whole
    # tensor a tl.tensor object
    return [[orb_proj(U, tl.tensor(density[i][j])) for j in range(len(density[i]))] for i in range(len(density))]

def orb_proj_ints(U0, U1, key, int):
    U_map = {0:U0, 1:U1}
    if len(key) == 2:
        partial = tendot(U_map[key[0]], int, axes=([1], [0]))
        return tendot(partial, U_map[key[1]], axes=([1], [1]))
    elif len(key) == 4:
        partial = tendot(U_map[key[0]], int, axes=([1], [2]))
        partial = tendot(U_map[key[1]], partial, axes=([1], [2]))
        partial = tendot(U_map[key[2]], partial, axes=([1], [2]))
        return tendot(partial, U_map[key[3]], axes=([3], [1]))
    else:
        raise NotImplementedError(f"orb_proj_ints cannot handle keys of len {len(key)}")


