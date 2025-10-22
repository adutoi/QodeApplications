#    (C) Copyright 2023, 2024, 2025 Anthony D. Dutoi and Marco Bauer
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
from qode.math.tensornet import raw

p, q, r, s, t, u, v, w = "pqrstuvw"    # some contraction indices for easier reading



def v0000(X):
    i0, j0 = 0, 1
    return 1 * raw(
        #  X.ccaa0(i0,j0,p,q,s,r)
        #@ X.v0000(p,q,r,s)
          X.ccaa0pqsr_Vpqrs(i0,j0)
        )
