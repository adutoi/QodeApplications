#    (C) Copyright 2019 Anthony D. Dutoi
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

import numpy

class empty(object):  pass     # Basically just a dictionary class


basis_set = """\
CUSTOM
cartesian
Be     0
S   6   1.00
      0.1264585690e+04       0.1944757590e-02
      0.1899368060e+03       0.1483505200e-01
      0.4315908900e+02       0.7209054629e-01
      0.1209866270e+02       0.2371541500e+00
      0.3806323220e+01       0.4691986519e+00
      0.1272890300e+01       0.3565202279e+00
S    3   1.00
      0.3196463098e+01      -0.1126487285e+00       0.5598019980e-01
      0.7478133038e+00      -0.2295064079e+00       0.2615506110e+00
      0.2199663302e+00       0.1186916764e+01       0.7939723389e+00
S    1   1.00
      0.8230990070e-01       0.1000000000e+01       0.1000000000e+01
****
"""


class monomer_data(object):
    def __init__(self, displacement):
        self.atoms = [("Be",displacement)]
        self.n_elec = 4
        self.basis = empty()
        self.basis.n_spatial_orb = 3
        self.basis.AOcode = basis_set
        self.basis.MOcoeffs = numpy.load("Be_C.npy")[:self.basis.n_spatial_orb,:self.basis.n_spatial_orb]	# top-left block because rest of code is for spin-restriced orbitals
