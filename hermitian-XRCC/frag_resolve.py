#    (C) Copyright 2024 Anthony D. Dutoi and Marco Bauer
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
import re    # regular expressions
from qode.util.dynamic_array import dynamic_array



#    densities and integrals are the full arrays for the supersystem
#    subsystem is always in acending order, naming the fragment indices of interest here
#    charges gives the bra and ket charges (as a 2-tuple) for each such respective fragment in the subsystem
#    permutation gives a potential reordering of the fragments in the subsystem
class frag_resolve(object):
    def __init__(self, supersys_info, subsys_chgs, permutation=(0,)):
        self._storage = {}
	#
        self._supersys_info = supersys_info
        subsys_chgs = list(subsys_chgs)    # zip has no len
        self._n_frag = len(subsys_chgs)
        self.P = 0 if permutation==(0,1) else 1    # needs to be generalized for n>=3.
        # Some diagrams need to know the number of e- in the ket for the combined "latter" frags of the un(!)permuted subsystem
        n_i = 0
        label = "".join(str(i) for i in range(self._n_frag))
        for m,(frag_idx,(chg_i,_)) in reversed(list(enumerate(subsys_chgs))):    # before permutation
            n_i += self._supersys_info.densities[frag_idx]['n_elec'][chg_i]
            self._storage["n_i"+label[m:]] = n_i%2    # explicitly label which are included in the latter frags (to store all possibilities)
        # rearrange fragments to given permutation
        self._subsys_chgs = [subsys_chgs[m] for m in permutation]
        # dynamically allocated, cached "virtual" arrays for the target information
        self._storage["Dchg#"] =   _Dchg_array(self._subsys_chgs, self._n_frag)
        self._storage["s##"]   = _subsys_array(self._supersys_info.integrals.S, self._subsys_chgs, 2, self._n_frag)
        try:
            self._storage["t##"]   = _subsys_array(self._supersys_info.integrals.T, self._subsys_chgs, 2, self._n_frag)
        except:
            pass
        try:
            self._storage["u#_##"] = _subsys_array(self._supersys_info.integrals.U, self._subsys_chgs, 3, self._n_frag)
        except:
            pass
        try:
            self._storage["v####"] = _subsys_array(self._supersys_info.integrals.V, self._subsys_chgs, 4, self._n_frag)
        except:
            pass
        self._densities        = _subsys_array(self._supersys_info.densities,   self._subsys_chgs, 1, self._n_frag)
    def __getattr__(self, attr):
        if attr[:3]=="n_i":
            return self._storage[attr]
        else:
            frag_indices = tuple(int(i) for i in filter(lambda c: c.isdigit(), attr))    # extract the digits from the string (heaven forbid >=9-fragment subsystem)
            label_template = re.sub("\d", "#", attr)                                     # replace digits with hashes to anonymize the label
            label_template = label_template.replace("U#_", "U#")    # underscore throws off later decisions/splits
            if label_template not in self._storage:    # Then it is a density or a precontraction, and ...
                if "_S" in label_template:             # ... it is a precontraction with S specifically, or ...
                    frag_count = label_template.count("#")
                    self._storage[label_template] = _precontract_array(self._supersys_info.contract_cache.rho_S,    label_template, self._subsys_chgs, frag_count, self._n_frag)
                elif "_" in label_template:            # ... it is a general precontraction, or ...
                    frag_count = label_template.count("#")
                    self._storage[label_template] = _precontract_array(self._supersys_info.contract_cache.general, label_template, self._subsys_chgs, frag_count, self._n_frag)
                else:                                  # ... otherwise must be a single-fragment density
                    self._storage[label_template] = _density_array(self._densities, label_template[:-1], self._subsys_chgs, self._n_frag)
            return self._storage[label_template][frag_indices]



def _subsys_array(array, subsys_chgs, n_indices, n_frag):
    subsystem, _ = zip(*subsys_chgs)    # "unzip" subsystem indices from their charges
    def _rule(*indices):
        absolute_indices = tuple(subsystem[index] for index in indices)
        if len(absolute_indices)==1:  absolute_indices = absolute_indices[0]
        return array[absolute_indices]
    return dynamic_array(_rule, [range(n_frag)]*n_indices)

def _Dchg_array(subsys_chgs, n_frag):
    _, charges = zip(*subsys_chgs)    # "unzip" subsystem indices from their charges
    def _rule(*indices):
        index = indices[0]
        chg_i, chg_j = charges[index]
        return chg_i - chg_j
    return dynamic_array(_rule, [range(n_frag)])

def _density_array(densities, label, subsys_chgs, n_frag):
    _, charges = zip(*subsys_chgs)    # "unzip" subsystem indices from their charges
    def _rule(*indices):
        index = indices[0]
        try:
            rho = densities[index][label][charges[index]]    # charges[index] is the bra and ket charge
        except KeyError:
            rho = None    # eventually return an object whose __getitem__ member reports exactly what is missing (in case access is attempted)
        return rho
    return dynamic_array(_rule, [range(n_frag)])

def _precontract_array(contract_cache, label, subsys_chgs, n_indices, n_frag):
    subsystem, charges = zip(*subsys_chgs)    # "unzip" subsystem indices from their charges
    precon = contract_cache[label]
    n_densities = len(label.split("_")) - 1
    def _rule(*indices):
        rho_charges = tuple(charges[m] for m in indices[:n_densities])
        indices = tuple(subsystem[m] for m in indices)
        if len(indices)==1:  indices = indices[0]
        contraction = precon[indices]
        try:
            for braket_charge in rho_charges:
                contraction = contraction[braket_charge]
        except KeyError:
            contraction = None    # eventually return an object whose __getitem__ member reports exactly what is missing (in case access is attempted)
        return contraction
    return dynamic_array(_rule, [range(n_frag)]*n_indices)
