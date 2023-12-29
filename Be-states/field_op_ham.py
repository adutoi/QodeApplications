#    (C) Copyright 2018, 2019, 2023 Yuhong Liu and Anthony Dutoi
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
from qode.util.PyC import Double
import field_op



class Hamiltonian(object):
    def __init__(self, h, V, thresh=1e-10, n_threads=1):
        self.h = h
        self.V = V
        self.thresh = thresh
        self.n_threads = n_threads
    def set_n_threads(self, n_threads):
        self.n_threads = n_threads
    def __call__(self, Psi, configs):
        HPsi = numpy.zeros(len(configs), dtype=Double.numpy)
        field_op.opPsi_1e(HPsi, Psi, self.h, configs, self.thresh, self.n_threads)
        field_op.opPsi_2e(HPsi, Psi, self.V, configs, self.thresh, self.n_threads)
        return HPsi
