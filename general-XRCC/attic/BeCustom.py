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
from build_density_tensors import build_density_tensors

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
        self.basis.MOcoeffs = numpy.load("atomic_states_new/data/Be_C.npy")[:self.basis.n_spatial_orb,:self.basis.n_spatial_orb]	# top-left block because rest of code is for spin-restriced orbitals
    def load_states(self, states_location, Vints, n_states=("all","all","all","all","all"), ref_state=(0,0)):
        # Load descriptions of the fragment many-electron states.  Then process them (using build_density_tensors) to arrive at
        # quantities suitable for contracting with the integrals.  The small dictionaries hold symbolic representations of fragment
	# electron configurations and their associated numerical coefficients in the fragment wavefunctions.
        self.n_elec_ref = 4	# The number of electrons in the reference state of the monomer ("cation (+1)" and "anion (-1)" and technically interpreted relative to the reference, not zero, as would be the chemical definition)
        nCat2, nCat, nNeut, nAn, nAn2 = n_states
        neutral = empty()
        neutral.coeffs  = numpy.load("atomic_states_new/H/4-6-4/{}/Z_2e.npy".format(states_location)).T   #
        if nNeut != "all":  neutral.coeffs = neutral.coeffs[:int(nNeut),:]
        neutral.configs = numpy.load("atomic_states_new/data/configs_2e-stable.npy")                           #

        cation = empty()                                                                                   # In a supersystem, any given
        cation.coeffs   = numpy.load("atomic_states_new/H/4-6-4/{}/Z_1e.npy".format(states_location)).T   # fragment may be found as neutral
        if nCat  != "all":  cation.coeffs  =  cation.coeffs[:int(nCat),:]
        cation.configs  = numpy.load("atomic_states_new/data/configs_1e-stable.npy")                           # or with an extra or missing electron
        anion = empty()                                                                                    # (anionic or cationic, respectively).
        anion.coeffs    = numpy.load("atomic_states_new/H/4-6-4/{}/Z_3e.npy".format(states_location)).T   #
        if nAn   != "all":  anion.coeffs   =   anion.coeffs[:int(nAn),:]
        anion.configs   = numpy.load("atomic_states_new/data/configs_3e-stable.npy")                           #

        cation2 = empty()                                                                                   # In a supersystem, any given
        cation2.coeffs   = numpy.load("atomic_states_new/H/4-6-4/{}/Z_0e.npy".format(states_location)).T   # fragment may be found as neutral
        if nCat2  != "all":  cation2.coeffs  =  cation2.coeffs[:int(nCat2),:]
        cation2.configs  = numpy.load("atomic_states_new/data/configs_0e-stable.npy")                           # or with an extra or missing electron
        anion2 = empty()                                                                                    # (anionic or cationic, respectively).
        anion2.coeffs    = numpy.load("atomic_states_new/H/4-6-4/{}/Z_4e.npy".format(states_location)).T   #
        if nAn2   != "all":  anion2.coeffs   =   anion2.coeffs[:int(nAn2),:]
        anion2.configs   = numpy.load("atomic_states_new/data/configs_4e-stable.npy")                           #

        states = {+2:cation2, +1:cation, 0:neutral, -1:anion, -2:anion2}    # Collect together all the states for the different allowed (relative) charges
        #states = {+1:cation, 0:neutral, -1:anion}    # Collect together all the states for the different allowed (relative) charges
        self.rho, mem_use = build_density_tensors(states, self.basis.n_spatial_orb, Vints, n_core=1)  # Compute the density tensors (n_core is the number of uncorrelated/frozen orbitals)
        ref_chg, ref_idx = ref_state
        self.state_indices = [(ref_chg,ref_idx)]                                                       # List of all charge and state indices, reference state needs to be first, but otherwise irrelevant order
        for i in range(states[ref_chg].coeffs.shape[0]):
            if   i!=ref_idx:  self.state_indices += [(ref_chg,i)]
        for chg in states:
            if chg!=ref_chg:  self.state_indices += [(chg,i) for i in range(states[chg].coeffs.shape[0])]
        print("rho computed, storing {} floating point numbers".format(mem_use))
