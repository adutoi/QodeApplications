#    (C) Copyright 2023 Anthony D. Dutoi
# 
#    This file is part of Qode.
# 
#    Qode is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
# 
#    Qode is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
# 
#    You should have received a copy of the GNU General Public License
#    along with Qode.  If not, see <http://www.gnu.org/licenses/>.
#

import tensorly as tl

# Maybe try this with tensorly.tenalg.tensordot with
# tensorly.tenalg.set_backend('einsum'), which only does the dispatching different(?),
# so it works, even though the actual backend is set to e.g. pytorch,

# Keeping the backend agnostic, while also feeding tensordot either full or
# decomposed tensors, requires us to write a wrapper! Tensorly can deal with
# decomposed tensors as well as being backend agnostic.

def tendot(a, b, axes):
    #if a == decomp and b != decomp:
    #    _
    #elif b is decomp and a != decomp:
    #    return tendot(b, a, axes=axes.T)
    #elif a == decomp and b == decomp:
    #    _
    #else:
    return tl.tensordot(a, b, axes=axes)






