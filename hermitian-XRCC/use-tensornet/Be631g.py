#    (C) Copyright 2019 Anthony D. Dutoi
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

import sys
import pickle
import numpy
from build_density_tensors import build_density_tensors

class empty(object):  pass     # Basically just a dictionary class



def _load_states(digit, n_states, states_location):
    def _is_not_zero(n_states):
        if n_states == "all":  return True
        n_states = int(n_states)
        return (n_states!=0)
    def _is_not_all(n_states):
        if n_states == "all":  return False
        else:                  return True
    states_file  = "states/16-115-550/{}/Z_{}e.npy".format(states_location, digit)
    configs_file = "integrals/configs_{}e-stable.npy".format(digit)
    data = None
    if _is_not_zero(n_states):
        try:
            coeffs = numpy.load(states_file).T
        except:
            if _is_not_all(n_states):  raise ValueError("asked for non-zero number of states to be used from non-existant file")   # asking for all from non-existant file is fine, just returns None in the end
        else:
            data = empty()
            data.coeffs  = coeffs
            data.configs = numpy.load(configs_file)
            if _is_not_all(n_states):
                n_states = int(n_states)
                data.coeffs = data.coeffs[:n_states,:]
    return data



states_location = "{}/4.5".format(sys.argv[1])
n_states = ("all","all","all","all","all")
ref_state = (0,0)

Be = empty()

Be.atoms = [("Be",[0,0,0])]

Be.n_elec_ref = 4	# The number of electrons in the reference state of the monomer ("cation (+1)" and "anion (-1)" and technically interpreted relative to the reference, not zero, as would be the chemical definition)

Be.basis = empty()
Be.basis.n_spatial_orb = 9
Be.basis.AOcode = "6-31G"
Be.basis.MOcoeffs = numpy.load("integrals/Be_C.npy")[:Be.basis.n_spatial_orb,:Be.basis.n_spatial_orb]	# top-left block because rest of code is for spin-restriced orbitals
Be.basis.core = [0]	# indices of spatial MOs to freeze

# Load descriptions of the fragment many-electron states.  Then process them (using build_density_tensors) to arrive at
# quantities suitable for contracting with the integrals.  In a supersystem, any given fragment may be found as neutral
# or with an extra or missing electron (anionic or cationic, respectively).  The data structures inside the states dictionary
# hold symbolic representations of fragment electron configurations and their associated numerical coefficients in the fragment wavefunctions.

states = {}
for (chg,val_e),n_st in zip([(+2,0), (+1,1), (0,2), (-1,3), (-2,4)], n_states):
    data = _load_states(val_e, n_st, states_location)
    if data is not None:  states[chg] = data		# Collect together all the states for the different allowed (relative) charges

Be.rho, mem_use = build_density_tensors(states, Be.basis.n_spatial_orb, n_core=len(Be.basis.core))  # Compute the density tensors (n_core is the number of uncorrelated/frozen orbitals)

ref_chg, ref_idx = ref_state
Be.state_indices = [(ref_chg,ref_idx)]                                                       # List of all charge and state indices, reference state needs to be first, but otherwise irrelevant order
for i in range(states[ref_chg].coeffs.shape[0]):
    if   i!=ref_idx:  Be.state_indices += [(ref_chg,i)]
for chg in states:
    if chg!=ref_chg:  Be.state_indices += [(chg,i) for i in range(states[chg].coeffs.shape[0])]



print("rho computed, storing {} floating point numbers".format(mem_use))
pickle.dump(Be, open("Be631g.pkl", "wb"))
