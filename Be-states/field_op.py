#    (C) Copyright 2023 Anthony D. Dutoi
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
import numpy
from qode.util.PyC import import_C, Double, BigInt

# Import the C module in a python wrapper for external aesthetics and to avoid having compile
# flags in multiple places (which points to a weakness in PyC that changing these does not
# force a recompile and defining them inconsistently will silently just use the first one
# imported.

field_op = import_C("field_op", flags="-O3 -lm -fopenmp")
field_op.orbs_per_configint.return_type(int)
field_op.bisect_search.return_type(int)

num_bits = field_op.orbs_per_configint()

opPsi_1e   = field_op.opPsi_1e
opPsi_2e   = field_op.opPsi_2e


class packed_configs(object):
    def __init__(self, configs):
        self.size = 1
        self.length = len(configs)
        self.array = numpy.array(configs, dtype=BigInt.numpy)
    def __len__(self):
        return self.length

def find_index(config, configs):
    C_config = numpy.array([config], dtype=BigInt.numpy)
    return field_op.bisect_search(C_config, configs.array, configs.size, 0, len(configs)-1)
