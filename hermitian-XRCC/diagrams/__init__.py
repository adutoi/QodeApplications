#    (C) Copyright 2025 Anthony D. Dutoi
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

##########
# Here are the implementations of the actual diagrams.
# They must take the arguments (X, i0,i1,...,j0,j1,...) where and instance of frag_resolve (see build_diagram),
# which provides all of the input tensors and/or intermediate contractions.  These functions then return a scalar
# which is the evaluated diagram.  Don't forget to update the "catalog" dictionary at the end.
##########

##########
# A dictionary catalog.  The string association lets users specify active diagrams at the top level
# for each kind of integral (symmetric, biorthogonal, ...).  The first argument to build_diagram() is
# one of the functions found above, which performs the relevant contraction for the given diagram for
# a fixed permutation (adjusted externally).
##########

from . import S_diagrams
from . import ST_diagrams
from . import SU_diagrams
from . import SV_diagrams
