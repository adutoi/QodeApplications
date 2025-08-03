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
import numpy
from build_density_tensors import build_density_tensors
#from build_adc_density_tensors import build_adc_density_tensors, get_psi4_Ca
import pickle





class empty(object):  pass     # Basically just a dictionary class

def _is_not_zero(n_states):
    if n_states == "all":  return True
    n_states = int(n_states)
    return (n_states!=0)

def _is_not_all(n_states):
    if n_states == "all":  return False
    else:                  return True

def _load_states(digit, n_states, states_location):
    states_file  = "atomic_states/states/16-115-550/{}/Z_{}e.npy".format(states_location, digit)
    configs_file = "atomic_states/integrals/configs_{}e-stable.npy".format(digit)
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




class monomer_data(object):
    def __init__(self, displacement):
        self.atoms = [("Be",displacement)]
        self.basis = empty()
        self.basis.n_spatial_orb = 9
        self.basis.AOcode = "6-31G"
        #self.basis.MOcoeffs, self.basis.n_orb_map = get_psi4_Ca()
        #print(self.basis.n_orb_map)
        #self.basis.MOcoeffs = pickle.load(open("Mo_coeffs_ref.pkl", mode="rb"))
        #pickle.dump(self.basis.MOcoeffs, open("Mo_coeffs_ref.pkl", mode="wb"))
        self.basis.MOcoeffs = numpy.load("atomic_states/integrals/Be_C.npy")[:self.basis.n_spatial_orb,:self.basis.n_spatial_orb]	# top-left block because rest of code is for spin-restriced orbitals
        #print(numpy.asarray(self.basis.MOcoeffs) + numpy.load("atomic_states/integrals/Be_C.npy")[:self.basis.n_spatial_orb,:self.basis.n_spatial_orb])
        self.basis.core = [0]	# indices of spatial MOs to freeze
        if not hasattr(self.basis, "n_orb_map"):
            self.basis.n_orb_map = {}
    def load_states(self, states_location, n_states, ref_state=(0,0), coeff_trans_mat=None):
        # Load descriptions of the fragment many-electron states.  Then process them (using build_density_tensors) to arrive at
        # quantities suitable for contracting with the integrals.  In a supersystem, any given fragment may be found as neutral
        # or with an extra or missing electron (anionic or cationic, respectively).  The data structures inside the states dictionary
        # hold symbolic representations of fragment electron configurations and their associated numerical coefficients in the fragment wavefunctions.

        self.n_elec_ref = 4	# The number of electrons in the reference state of the monomer ("cation (+1)" and "anion (-1)" and technically interpreted relative to the reference, not zero, as would be the chemical definition)

        states = {}
        for (chg,val_e),n_st in zip([(+2,0), (+1,1), (0,2), (-1,3), (-2,4)], n_states):
            data = _load_states(val_e, n_st, states_location)
            if data is not None:  states[chg] = data		# Collect together all the states for the different allowed (relative) charges

        self.rho, mem_use = build_density_tensors(states, self.basis.n_spatial_orb, n_core=len(self.basis.core), coeff_trans_mat=coeff_trans_mat)  # Compute the density tensors (n_core is the number of uncorrelated/frozen orbitals)
        #self.rho = build_adc_density_tensors(states)
        #self.rho = pickle.load(open("dumped_densities.pkl", mode="rb"))
        #pickle.dump(self.rho, open("dumped_densities.pkl", mode="wb"))

        ref_chg, ref_idx = ref_state
        self.state_indices = [(ref_chg,ref_idx)]                                                       # List of all charge and state indices, reference state needs to be first, but otherwise irrelevant order
        for i in range(states[ref_chg].coeffs.shape[0]):
            if   i!=ref_idx:  self.state_indices += [(ref_chg,i)]
        for chg in states:
            if chg!=ref_chg:  self.state_indices += [(chg,i) for i in range(states[chg].coeffs.shape[0])]

        #print("rho computed, storing {} floating point numbers".format(mem_use))
        print("rho computed")
