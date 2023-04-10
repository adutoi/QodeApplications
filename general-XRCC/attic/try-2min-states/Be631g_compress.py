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

def _parse(n_states):
    if isinstance(n_states,int):  n_states = [n_states]
    else:                         n_states = [int(n) for n in n_states.split(":")]
    if len(n_states)==1:  return slice(0,n_states[0]), n_states[0]
    else:                 return slice(n_states[0],n_states[1]), n_states[1]-n_states[0]

def _is_not_zero(n_states):
    if n_states == "all":  return True
    _, num = _parse(n_states)
    return (num!=0)

def _is_not_all(n_states):
    if n_states == "all":  return False
    else:                  return True

def _load_states(digit, n_states, states_location):
    states_file  = "compress/{}/z_{}e.npy".format(states_location, digit)
    configs_file = "compress/configs/{}e.npy".format(digit)
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
                states_slice, _ = _parse(n_states)
                data.coeffs = data.coeffs[states_slice,:]
    return data




class monomer_data(object):
    def __init__(self, displacement, core):
        self.atoms = [("Be",displacement)]
        self.basis = empty()
        self.basis.n_spatial_orb = 9
        self.basis.AOcode = "6-31G"
        self.basis.MOcoeffs = numpy.load("compress/Be_C.npy")[:self.basis.n_spatial_orb,:self.basis.n_spatial_orb]	# top-left block because rest of code is for spin-restriced orbitals
        self.basis.core = core	# indices of spatial MOs to freeze
    def load_states(self, states_location, Vints, n_states):
        # Load descriptions of the fragment many-electron states.  Then process them (using build_density_tensors) to arrive at
        # quantities suitable for contracting with the integrals.  In a supersystem, any given fragment may be found as neutral
        # or with an extra or missing electron (anionic or cationic, respectively).  The data structures inside the states dictionary
        # hold symbolic representations of fragment electron configurations and their associated numerical coefficients in the fragment wavefunctions.

        self.n_elec_ref = 4	# The number of electrons in the reference state of the monomer ("cation (+1)" and "anion (-1)" and technically interpreted relative to the reference, not zero, as would be the chemical definition)

        states = {}
        for (chg,val_e),n_st in zip([(+2,0), (+1,1), (0,2), (-1,3), (-2,4)], n_states):
            data = _load_states(val_e, n_st, states_location)
            if data is not None:  states[chg] = data		# Collect together all the states for the different allowed (relative) charges

        self.rho, mem_use = build_density_tensors(states, self.basis.n_spatial_orb, Vints, n_core=1)  # Compute the density tensors (n_core is the number of uncorrelated/frozen orbitals)

        self.state_indices = []
        for chg in states:
            self.state_indices += [(chg,i) for i in range(states[chg].coeffs.shape[0])]

        print("rho computed, storing {} floating point numbers".format(mem_use))
