#    (C) Copyright 2025 Anthony D. Dutoi
# 
#    This file is part of QodeApplications.
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
import tensorly
from qode.util.PyC import Double
import qode.math.tensornet as tensornet

# The problem we are trying to solve here is as follows:
# 1. Very often we need to deal with "primitive" tensor data from elsewhere (integrals from other 
#    packages, densities computed by C code) and/or do some standard linear algebra operations
#    on it (inversion, SVD, CPD, etc.) that are not implemented in tensornet (and may never be 
#    because we would need to somehow standardize the interfaces to all of the backend functions
#    that perform these tasks ... it is just not what tensornet was built for).
# 2. We need all of the abstracted methods that deal with tensornet objects to be given tensors with
#    equivalent backends and data standards, but without having to remember and recode that each
#    time, and, equivalently, leave a path the completely replace the backend without finding all
#    such hardcoded instantiations.
# So this file creates a standard interface consisting of two paired functions init() and raw(),
# which put "primitive"/concrete data (numpy, pytorch, tensorly) into the standard tensornet form 
# encapsulation (with the standard backend, with the standard data format) and also extracts it
# (executing the defined contractions and sums as necessary).
#
# There will eventually be several options living here, with only one pair active at any time
# (probably just commenting out inactive code unless something more dynamic is ever necessary)
# The idea is that the code should *run* with any of these options, though one might see
# large performance differences, also depending on options (e.g., backends to the backends).
#
# For future notes, maybe we should speak of abstact (tensornet), concrete (tensorly), and
# backend (numpy, pytorch).

def init(raw_tensor):
    return tensornet.tl_tensor.init(tensorly.tensor(raw_tensor, dtype=Double.tensorly))

def raw(tensor):
    return tensornet.raw(tensor)
