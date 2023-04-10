#    (C) Copyright 2018 Anthony D. Dutoi
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

# Usage (within a Psi4 conda environment):  python main.py <distance>

import sys
import numpy
import psi4
import qode.atoms.integrals.spatial_to_spin as spatial_to_spin
import qode.atoms.integrals.external_engines.psi4_ints as integrals
from   qode.many_body.self_consistent_field.fermionic import RHF_RoothanHall_Nonorthogonal

def MO_transform(H, V, C):
    H = C.T @ H @ C
    for _ in range(4):  V = numpy.tensordot(V, C, axes=([0],[0]))       # cycle through the tensor axes (this assumes everything is real)
    return H, V



basis = """\
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





# Normal AO SCF of Be atom
n_elec_1 = 4
Be_1 = """\
Be
"""
S_1, T_1, U_1, V_1, X_1 = integrals.AO_ints(Be_1, basis)
H_1 = T_1 + U_1
_, _, C_1 = RHF_RoothanHall_Nonorthogonal(n_elec_1, (S_1,H_1,V_1), thresh=1e-12)
H_1_MO, V_1_MO = MO_transform(H_1, V_1, C_1)

# Set up dimer
n_elec_2 = 8
Be_2 = """\
Be
Be  1  {distance:f}
""".format(distance=float(sys.argv[1]))
S_2, T_2, U_2, V_2, X_2 = integrals.AO_ints(Be_2, basis)
H_2 = T_2 + U_2
Enuc_2 = X_2.mol.nuclear_repulsion_energy()

# Psi4 energy of dimer for reference
psi4.set_output_file("output.dat")
psi4.set_options({"PRINT_MOS":"True", "E_CONVERGENCE":1e-12, "SCF_TYPE":"PK", "DIIS":"False"})
print("Psi4 Be2 HF energy = ", psi4.energy("SCF", molecule=X_2.mol))

# Normal AO SCF Be dimer
energy_2, _, C_2 = RHF_RoothanHall_Nonorthogonal(n_elec_2, (S_2,H_2,V_2), thresh=1e-12)
print("As computed here   = ", energy_2 + Enuc_2)
H_2_MO, V_2_MO = MO_transform(H_2, V_2, C_2)


# Put everything in terms of spin orbitals
C_1    = spatial_to_spin.one_electron_blocked(C_1)
H_1_MO = spatial_to_spin.one_electron_blocked(H_1_MO)
V_1_MO = spatial_to_spin.two_electron_blocked(V_1_MO)
S_2    = spatial_to_spin.one_electron_blocked(S_2)
C_2    = spatial_to_spin.one_electron_blocked(C_2)
H_2_MO = spatial_to_spin.one_electron_blocked(H_2_MO)
V_2_MO = spatial_to_spin.two_electron_blocked(V_2_MO)

# Dump to disk
numpy.save(    "data/Be_C.npy", C_1)
numpy.save(    "data/Be_h.npy", H_1_MO)
numpy.save(    "data/Be_V.npy", V_1_MO)
numpy.save("data/Be2_{}_S.npy".format(sys.argv[1]), S_2)
numpy.save("data/Be2_{}_C.npy".format(sys.argv[1]), C_2)
numpy.save("data/Be2_{}_h.npy".format(sys.argv[1]), H_2_MO)
numpy.save("data/Be2_{}_V.npy".format(sys.argv[1]), V_2_MO)
