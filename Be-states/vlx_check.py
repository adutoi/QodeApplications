#    (C) Copyright 2025 Marco Bauer
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

import veloxchem as vlx

def print_HF_energy(geometry, basis_string):
    Be = vlx.Molecule.read_str(geometry)
    basis = vlx.MolecularBasis.read(Be, basis_string)
    scfdrv = vlx.ScfRestrictedDriver()
    hf_results = scfdrv.compute(Be, basis)
    print("Psi4 HF energy       = ", hf_results['scf_energy'])
