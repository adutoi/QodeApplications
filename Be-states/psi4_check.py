#    (C) Copyright 2018, 2023 Anthony D. Dutoi
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

import psi4

def print_HF_energy(geometry, basis_string):
    molecule = psi4.geometry(geometry)
    molecule.update_geometry()
    psi4.set_output_file("output.dat")
    psi4.set_options({"scf_type":"pk", "PRINT_MOS":"True"})
    print("Psi4 HF energy       = ", psi4.energy("SCF/{}".format(basis_string), molecule=molecule))

