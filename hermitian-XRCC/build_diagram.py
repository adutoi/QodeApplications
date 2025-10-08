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

####
# In this file is all the book keeping stuff for getting the right arrays to the diagram contraction
# codes without burdening the user there with too many details (just focusing on some fragments that
# are always numbered from zero onward (but which could be any (permutation of) fragments), and using
# a lightweight syntax.  The backbone is dynamic_array, which could also be called lazy_array.  It
# generates its elements (which in this case are themselves tensors) on demand and caches the ones it
# has generated.  Meanwhile there is an encapsulating structure which interpolates between the relative
# (temporary, dummy, anonymized) indices and the absolute indices that identify specific fragments,
# and it generates its dot-referenced attributes dynanically, but dissecting their names, saving some
# characters over using [""] for dict entries.  it also synthesizes several arrays that (must) live
# at different scopes of the code, making everything look like it is in one place.  Oh, and it keeps
# track of execution times.
###



# This function informs a primitive contraction of the charge cases for which it is valid and the permutations
# of the anonymized fragments for which it should be executed.
def build_diagram(contraction, Dchgs=(0,), permutations=((0,),)):
    # The returned function further requires information about the actual fragments composing the subsystem
    # being computed, and their charges.
    def get_permuted_diagrams(supersys_info, subsys_chgs):
        label = contraction.__name__
        # This function is used immediately below.  It feeds permuted input tensors to the contraction and
        # returns the final objective function that does the contraction once informed of the fragment states.
        def permuted_diagram(X):
            def do_contraction(**args):
                supersys_info.timings.start()
                result = contraction(X, **args)
                supersys_info.timings.record(label)
                return result
            return do_contraction
        # build list of fully qualified diagram contraction functions for permutations where charge criteria met
        if permutations is None:    # handles 0-mer/identity case
            permuted_diagrams = [(permuted_diagram(None), tuple())]
        else:
            permuted_diagrams = []
            for permutation in permutations:
                X = frag_resolve(supersys_info, subsys_chgs, permutation)    # see below
                if all(X.Dchg[m]==Dchg for m,Dchg in enumerate(Dchgs)):
                    permuted_diagrams += [(permuted_diagram(X), permutation)]
                else:
                    permuted_diagrams += [None]
        return permuted_diagrams
    return get_permuted_diagrams



# This class's job is to look up and return information about specific fragments, in specific charge states, in a
# specific permuation, mapping relative (temporary, dummy, anonymized) indices to absolute ones.  It can provide
# blocks of integrals, arrays of density tensors (that take state indices), and cached contractions of such.  It can
# also provide distilled information about numbers of electrons and charge changes.  It provides a very condensed 
# syntax by parsing the string of the attribute requested (avoiding a lot of [""] patterns) and forwarding the 
# request to a dynamic_array that maps indices and computes some distilled quantities.  The dynamic_array rules
# are found after this class definition.
#  supersys_info contains attributes for the full array of integrals (fragment blocked) and density tensors
#  subsys_chgs   is an array of 2-tuples giving the absolute indices of the fragments in the subsystem (always in 
#                ascending order) and their bra and ket charges (also a 2-tuple), relative to the reference state.
#  permutation   gives the ordering of these fragments as they are to be used in the computation qualified by this
#                information structure.
class frag_resolve(object):
    def __init__(self, supersys_info, subsys_chgs, permutation):
        self._storage = {}
	#
        self._supersys_info = supersys_info
        self._n_frag = len(subsys_chgs)
        self.P = 1 if permutation==(1,0) else 0    # needs to be generalized for n>2.
        # Some diagrams need to know the number of e- in the ket for the combined "latter" frags of the un(!)permuted subsystem
        n_i = 0
        label = "".join(str(i) for i in range(self._n_frag))
        for m,(frag_idx,(chg_i,_)) in reversed(list(enumerate(subsys_chgs))):    # before permutation
            n_i += self._supersys_info.densities[frag_idx]['n_elec'][chg_i]
            self._storage["n_i"+label[m:]] = n_i%2    # explicitly label which are included in the latter frags (to store all possibilities)
        # rearrange fragments to given permutation
        self._subsys_chgs = [subsys_chgs[m] for m in permutation]
        # dynamically allocated, cached "virtual" arrays for the target information.  Not all integrals provided (or provided differently
        # but if not provided, then it is not needed and can be skipped . . . a little dirty but works
        self._storage["Dchg"] =   _Dchg_array(self._subsys_chgs, self._n_frag)
        self._storage["n_states"]     = _n_states_array(self._supersys_info.densities, self._subsys_chgs, self._n_frag)
        try:
            self._storage["s##"]   = _subsys_array(self._supersys_info.integrals.S, self._subsys_chgs, 2, self._n_frag)    # one of these ...
        except:
            self._storage["s##"]   = _subsys_array(self._supersys_info.integrals,   self._subsys_chgs, 2, self._n_frag)    # ... needs to work
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
        if attr[:3]=="n_i" or attr=="Dchg" or attr=="n_states":
            return self._storage[attr]
        #elif attr == "ket_coeffs":
        #    return _density_array(self._densities, label_template[:-1], self._subsys_chgs, self._n_frag)
        else:
            frag_indices = tuple(int(i) for i in filter(lambda c: c.isdigit(), attr))    # extract the digits from the string (heaven forbid >=9-fragment subsystem)
            label_template = re.sub("\\d", "#", attr)                                     # replace digits with hashes to anonymize the label
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



# The things below are wrapper functions to make the dynamic_array objects, encapsulating the rules used as element generators.

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

def _n_states_array(densities, subsys_chgs, n_frag):
    subsystem, charges = zip(*subsys_chgs)    # "unzip" subsystem indices from their charges
    def _rule(*indices):
        index = indices[0]
        n_states = densities[subsystem[index]]['n_states']
        n_states_bra = densities[subsystem[index]]['n_states_bra']
        chg_i, chg_j = charges[index]
        return n_states_bra[chg_i], n_states[chg_j]
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
