#    (C) Copyright 2024, 2025 Anthony D. Dutoi
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

import copy
import itertools
from qode.util import struct
from qode.math.permute import permutation_subs
from primitives import index, h_int, v_int, s_int, r_int, delta, c_op, a_op, scalar_sum

# this syntax because operations performed by classes, but not in place
def frag_sorted(obj):         return obj._frag_sorted()
def frag_factorized(obj):     return obj._frag_factorized()
def frag_permuted(obj,perm):  return obj._frag_permuted(perm)
def simplified(obj):          return obj._simplified()
def condense_perm(obj):       return obj._condense_perm()
def ct_ordered(obj):          return obj._ct_ordered()
def mult_by(obj, x):          return obj._mult_by(x)



# A container to hold a group of integrals that are to be contracted with eachother and/or the indices of field operators
class integral_product(object):
    def __init__(self, integrals):
        try:    # test for copy
            new_integrals = integrals._integrals
        except AttributeError:   # new from list/iterable
            self._integrals = list(integrals)
            self._publication_ordered = False
            self._abbreviated = False
            self._code = False
        else:   # continue with copy
            other = integrals
            self._integrals = list(new_integrals)
            self._publication_ordered = other._publication_ordered
            self._abbreviated = other._abbreviated
            self._code = other._code
    def _integrals_by_type(self):
        return (
            [integral for integral in self._integrals if integral.kind in ("h","v","s")],
            [integral for integral in self._integrals if integral.kind in ("h","v")],
            [integral for integral in self._integrals if integral.kind=="s"],
            [integral for integral in self._integrals if integral.kind=="d"],
            [integral for integral in self._integrals if integral.kind=="r"],
        )
    def permute_frags(self, perm):    # permute fragment indices in place
        _1, _2, _3, d_ints, r_ints = self._integrals_by_type()
        for rho in d_ints+r_ints:    # do only density integrals because otherwise will do repeated indices twice
            rho.substitute_frag(perm)
    def moint_indices(self):   # return creation and annihilation indices on all MO ints (not densities!), with sigma ints after h or v ints
        _1, hv_ints, s_ints, _2, _3 = self._integrals_by_type()
        c_indices = list(itertools.chain.from_iterable([integral.c_indices for integral in hv_ints+s_ints]))
        a_indices = list(itertools.chain.from_iterable([integral.a_indices for integral in hv_ints+s_ints]))
        return c_indices, a_indices
    def dens_indices(self):      # return all indices on all densities, creation before annhilation for each fragment, in order
        _1, _2, _3, _4, r_ints = self._integrals_by_type()
        dens_ints = {}   # integral objects sorted by fragment (to be populated)
        for integral in r_ints:
            frag = integral.fragment()
            if frag not in dens_ints:
                dens_ints[frag] = []
            dens_ints[frag] += [integral]
        indices = []    # flatten the above structure and store only indices
        for frag in sorted(dens_ints):
            for integral in dens_ints[frag]:
                indices += integral.c_indices + integral.a_indices
        return indices
    def compare_sign_to(self, other):   # sign of permutation if integrals are (reordering of) the same types with the same indices (else 0)
        result = 0
        if len(self._integrals)==len(other._integrals):
            result = 1
            for self_integral in self._integrals:
                sign = 0
                for other_integral in other._integrals:                  # works because ...
                    s = self_integral.compare_sign_to(other_integral)    # ... neither list ...
                    if s!=0:                                             # ... should contain ...
                        sign = s                                         # ... duplicates
                result *= sign
        return result
    def append(self, integrals):
        self._integrals += integrals
    def rho_notation(self):   # toggle overall notation style for rho integrals (global choice stored at integrals level; ignored for non-density integrals)
        for integral in self._integrals:  integral.rho_notation()
    def abbreviated(self):    # toggle abbreviated notation for indices (store toggle also at integral_product and integral level)
        for integral in self._integrals:  integral.abbreviated()
        self._abbreviated = not self._abbreviated
    def code(self):           # toggle code representation (store toggle at integral_product and integral level)
        for integral in self._integrals:  integral.code()
        self._code = not self._code
    def publication_ordered(self):   # toggle integral ordering (stored only at integral_product level)
        self._publication_ordered = not self._publication_ordered
    def ct_character(self):
        _1, _2, _3, _4, r_ints = self._integrals_by_type()
        return tuple(rho.ct_character() for rho in r_ints)
    def abbrev_hack(self):
        _1, hv_ints, s_ints, _2, _3 = self._integrals_by_type()
        return "".join(integral.abbrev_hack() for integral in s_ints+hv_ints)
    def int_type(self):
        hvs_ints, hv_int, s_ints, d_ints, r_ints = self._integrals_by_type()
        symbol = "S"
        if len(hv_int)==1:
            symbol += hv_int[0].symbols()[1][0]
        return symbol, len(s_ints)
    def __str__(self):
        # this all seems a little hectic.  how to clean up?
        hvs_ints, hv_int, s_ints, d_ints, r_ints = self._integrals_by_type()
        if self._code:
            string = "#  " + "\n        #@ ".join(str(integral) for integral in d_ints+r_ints+hvs_ints)
            #
            r_ints_precontract = {r_int.fragment():None for r_int in r_ints}

            if len(hv_int)==1:
                hv_indices = hv_int[0].frag_indices()
                idx = max(set(hv_indices), key=hv_indices.count)    # finds the most frequently occuring index in hv_indices
                r_ints_precontract[idx] = hv_int[0]

            for frag in r_ints_precontract:
                if r_ints_precontract[frag] is None:
                    for i,s_int in enumerate(s_ints):
                        if frag in s_int.frag_indices():
                            r_ints_precontract[frag] = s_int
                            del(s_ints[i])
                            break

            all_integrals = [str(d_int) for d_int in d_ints]
            for r_int in r_ints:
                intA = r_int
                if r_ints_precontract[r_int.fragment()] is not None:
                    intB = r_ints_precontract[r_int.fragment()]
                    #all_integrals += [str(intA) + " @ " + str(intB)]
                    symbolA, fragsA, lettersA = intA.code_components()
                    symbolB, fragsB, lettersB = intB.code_components()
                    #fragsA = fragsA[0]
                    #fragsB = "".join(fragsB)
                    #lettersA = ",".join(lettersA)
                    #lettersB = ",".join(lettersB)
                    #all_integrals += [f"X.{symbolA[0]}{fragsA}(i{fragsA},j{fragsA},{lettersA}) @ X.{symbolB[0]}{fragsB}({lettersB})"]
                    free_indices = [f"(i{fragsA[0]},j{fragsA[0]}"]
                    precontract = f"X.{symbolA[1]}{fragsA[0]}"
                    for p in lettersA:
                        if p in lettersB:
                            precontract  += p
                        else:
                            precontract  += "X"
                            free_indices += [p]
                    precontract += f"_{symbolB[1]}"
                    for p,f in zip(lettersB,fragsB):
                        if p in lettersA:
                            precontract  += p
                        else:
                            precontract  += f"{f}"
                            free_indices += [p]
                    precontract += ",".join(free_indices) + ")"
                    all_integrals += [precontract]
                else:
                    all_integrals += [str(intA)]
            all_integrals += [str(s_int) for s_int in s_ints]

            string += "\n          " + "\n        @ ".join(all_integrals)
            #
            return string
        else:
            enclose = "{}"
            deltas = ""
            connect = " ~ "
            if self._abbreviated:
                integrals = hvs_ints
                enclose = "\\langle {} \\rangle"
                deltas  = "{}".format(connect.join(str(integral) for integral in d_ints))
            elif self._publication_ordered:
                integrals  = d_ints + r_ints + hvs_ints
            else:
                integrals = self._integrals
            return deltas + enclose.format(connect.join(str(integral) for integral in integrals))



# a container to hold a string of operators, perhaps as a matrix element
class operator_string(object):
    def __init__(self, ops):
        try:    # test for copy
            new_ops = ops._ops
        except AttributeError:   # new from dict or list
            try:
                self._ops = dict(ops)
            except TypeError:
                self._ops = {None: list(ops)}
            self._braket = False
        else:   # continue with copy
            self._ops = dict(new_ops)
            self._braket = ops._braket
    def _flatten_ops(self):
        return list(itertools.chain.from_iterable([self._ops[key] for key in sorted(self._ops.keys())]))
    def _frag_sorted(self):
        perm = 0
        sorted_ops = {}
        for op in self._flatten_ops():
            frag = op.idx.fragment
            if frag not in sorted_ops:
                sorted_ops[frag] = []
            sorted_ops[frag] += [op]
            for other_frag,other_ops in sorted_ops.items():
                if other_frag>frag:
                    perm += len(other_ops)
        return perm, operator_string(sorted_ops)
    def _frag_factorized(self):
        frags = self._braket
        frag_pows = []
        integrals = []
        #
        ### i_2 convention ... might have bug in integrals list increment that orders things wrongly
        #for frag in sorted(frags):
        #    raw_string = []
        #    if frag in self._ops:
        #        raw_string = self._ops[frag]        # a list of operators ...
        #        integrals = integrals + [r_int(raw_string)]    # ... now expressed as a rho integral within a list
        #    else:
        #        raw_string = []
        #        integrals = integrals + [delta(frag)] 
        #    for sub_frags in frag_pows:    # builds a list of lists containing the ...
        #        sub_frags += [frag]        # ... identities of all *subsequent* fragments (assuming ascending order) ...
        #    if len(raw_string)%2:          # ... if there is an odd number of ...
        #        frag_pows += [[]]          # ... operators on this one
        #
        ### j_1 convention
        for frag in reversed(sorted(frags)):
            raw_string = []
            if frag in self._ops:
                raw_string = self._ops[frag]        # a list of operators ...
                integrals = [r_int(raw_string)] + integrals    # ... now expressed as a rho integral within a list
            else:
                raw_string = []
                integrals = [delta(frag)] + integrals
            for sub_frags in frag_pows:    # builds a list of lists containing the ...
                sub_frags += [frag]        # ... identities of all *prior* fragments (assuming ascending order) ...
            if len(raw_string)%2:          # ... if there is an odd number of ...
                frag_pows += [[]]          # ... operators on this one
        #
        frag_pows = list(itertools.chain.from_iterable(frag_pows))                            # flatten the list and keep only the identities ...
        frag_pows_simplified = [frag for frag in set(frag_pows) if frag_pows.count(frag)%2]   # ... of those that occur an odd number of times
        return frag_pows_simplified, integrals                                                # simplification would also happen implicitly later, but want explicit logic recorded here.
    def as_braket(self, frags):   # toggle option to express as expectation value
        self._braket = frags
    def __len__(self):
        return len(self._flatten_ops())
    def __eq__(self, other):
        try:
            result = (self._ops == other._ops)   # this works because equality test defined for field operators
        except:
            result = False
        return result
    def __str__(self):
        bra, ket = "", ""
        if self._braket is not False:    # other option is tuple of frags
            bra = "{{}}^{{{bra}}}\\langle".format(bra=" ".join(f"i_{{{n}}}" for n in self._braket))
            ket =     "\\rangle_{{{ket}}}".format(ket=" ".join(f"j_{{{n}}}" for n in self._braket))
        ops = " ".join(str(op) for op in self._flatten_ops())
        return f"{bra}{ops}{ket}"



# Represents a single contraction of sigma & molecular integrals with field operators, in either raw or matrix element form
# (upon provision of states).  So a list of integrals and operators must be provided
class diagram(object):
    def __init__(self, arg1, arg2=None, scalar=None, perm_list=None, _frags=None):
        if arg2 is None:    # if not copying, first two arguments are mandatory
            other = arg1
            integrals = other._integrals
            op_string = other._op_string
            scalar    = other._scalar    if scalar    is None else scalar       # allows copying with different scalar ...
            perm_list = other._perm_list if perm_list is None else perm_list    # ... or perm_list
            self._publication_ordered = other._publication_ordered
            self._abbreviated         = other._abbreviated
            self._code                = other._code
            self._frags               = other._frags
        else:
            integrals, op_string = arg1, arg2
            scalar    = 1  if scalar    is None else scalar
            perm_list = [] if perm_list is None else perm_list
            self._publication_ordered = False
            self._abbreviated         = False
            self._code                = False
            self._frags               = _frags
        self._integrals = integral_product(integrals)                       # promote ...
        self._op_string = operator_string(op_string)                        # ... types and ...
        self._scalar    = scalar_sum(scalar)                                # ... force ...
        self._perm_list = [(sign,dict(perm)) for sign,perm in perm_list]    # ... copies
    @staticmethod
    def from_integrals(integrals):    # instantiate with operator string implied from integrals (operators from each subsequent integral nested inside those from prior)
        c_indices, a_indices = integral_product(integrals).moint_indices()    # the indices in each list are in order of appearance
        all_indices = c_indices + a_indices
        for i,p in enumerate(all_indices):
            if p in all_indices[i+1:]:
                raise ValueError("illegal combination of integrals with duplicate indices")
        c_ops = [c_op(p) for p in          c_indices ]
        a_ops = [a_op(p) for p in reversed(a_indices)]
        return diagram(integrals, operator_string(c_ops + a_ops))
    def _frag_sorted(self):
        permute, op_string = frag_sorted(self._op_string)
        scalar = scalar_sum(self._scalar)
        scalar.perm_mult(permute)
        return diagram(self._integrals, op_string, scalar, _frags=self._frags)
    def _frag_factorized(self):
        integrals, scalar = integral_product(self._integrals), scalar_sum(self._scalar)    # copies, so ok to modify
        frag_pows, ints_factorized = frag_factorized(self._op_string)
        scalar.frag_phase_mult(frag_pows)
        integrals.append(ints_factorized)
        return diagram(integrals, [], scalar, _frags=self._frags)    # new from modified copies
    def _frag_permuted(self, perm):    # return a copy with fragment indices permuted
        new_term = copy.deepcopy(self)
        new_term._integrals.permute_frags(perm)
        return new_term
    def _mult_by(self, x):
        new_term = copy.deepcopy(self)
        new_term._scalar.mult_by(x)
        return new_term
    @staticmethod
    def compare_signs(term1, term2):    # sign of permutation if contraction of integrals and operators are otherwise the same (else 0) [scale and phase handled externally]
        # Assumes operator strings sorted and factorized before reduction to densities (otherwise false negatives); integrals may be permuted.
        # Since full contraction is implied, can redefine letters for indices to see equality, and since integrals
        # and operators reference the same index objects, just change letters therein (deepcopy preserves structure).
        if len(term1._op_string)>0 or len(term2._op_string)>0:
            raise RuntimeError("can only compare terms that have been reduced to products of MO integrals and single-fragment densities")
        term1_copy, term2_copy = copy.deepcopy(term1), copy.deepcopy(term2)    # because we will mess with indices
        for term in [term1_copy, term2_copy]:    # for each of the two copied terms, standardize the letters used for density indices
            for i,p in enumerate(term._integrals.dens_indices()):    # ordered by fragment, then creation before annihilation
                p.letter = "abcdefghijklmnopqrstuvwzyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789`~!@#$%^&*()-_=+[{]}|;:,<.>/?"[i]    # can be crazy because never printed. need more, use and actual list of strings?
        return term1_copy._integrals.compare_sign_to(term2_copy._integrals)
    def ct_character(self):
        return self._integrals.ct_character()
    def scalar(self):    # a copy of the scalar for external processing
        return scalar_sum(self._scalar)
    def name(self):
        return self._integrals.abbrev_hack()
    def rho_notation(self):           # toggle ...
        self._integrals.rho_notation()
    def as_braket(self, frags):       # toggle ...
        self._frags = frags
        self._op_string.as_braket(frags)
    def abbreviated(self):            # toggle ...
        self._abbreviated = not self._abbreviated
        self._integrals.abbreviated()
        self._scalar.abbreviated()
    def code(self):                   # toggle ...
        self._code = not self._code
        self._integrals.code()
        self._scalar.code()
    def publication_ordered(self):    # toggle ...
        self._publication_ordered = not self._publication_ordered
        self._integrals.publication_ordered()
        self._scalar.publication_ordered()
    def catalog_entry(self):
        if len(self._op_string)>0:
            raise RuntimeError("can only output code for terms that have been reduced to products of MO integrals and single-fragment densities")
        def format_charges(the_tuple):
            extra = "," if (len(the_tuple)==1) else ""
            return "({}),".format(",".join(f"{value:+d}" if (value!=0) else "0" for value in the_tuple) + extra)
        name  = self.name()
        name1 = f"\"{name}\":"
        name2 = f"{name},"
        Dchgs = format_charges(self._integrals.ct_character())
        permutations = []
        for sign,perm in self._perm_list:    # sign is stored as a character
            phase = f"{sign}1"
            permutations += [(phase, tuple(P for P in perm.values()))]
        permutations = [f"({phase},{perm})" for phase,perm in permutations]
        permutations = ",".join(perm.replace(" ", "") for perm in permutations)
        permutations = f"[{permutations}]"
        return f"    {name1:15s}  build_diagram({name2:13s} Dchgs={Dchgs:8s} permutations={permutations}),"
    def int_type(self):
        return self._integrals.int_type()
    def __str__(self):
        if self._code:
            if len(self._op_string)>0:
                raise RuntimeError("can only output code for terms that have been reduced to products of MO integrals and single-fragment densities")
            if len(self._frags)>1:
                string = f"def {self.name()}(X, contract_last=False):\n" \
                          "    if no_result(X, contract_last):  return []\n" \
                          "    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3\n"
            else:
                string = f"def {self.name()}(X):\n" \
                          "    i0, j0 = 0, 1\n"
            string += f"    return {self._scalar} * raw(\n" \
                      f"        {self._integrals}\n" \
                       "        )\n"
        else:
            string = f"~{self._op_string}" if len(self._op_string)>0 else ""
            if self._publication_ordered:
                string = f"{self._scalar}{string}{self._integrals}"
            else:
                string = f"{self._integrals}{string}{self._scalar}"
            for sign,perm in self._perm_list[1:]:    # sign is stored as a character ... first permutation already accounted for
                perms = " \\\\ ".join(f"{p}\\rightarrow{q}" for p,q in perm.items())
                string += f"~{sign}~P_{{\\substack{{ {perms} }} }}"
        return string



class diagram_sum(object):
    def __init__(self, terms, _frags=None):
        self._terms = list(terms)
        self._code = False
        self._frags = _frags
    def _frag_sorted(self):
        return diagram_sum([frag_sorted(term) for term in self._terms], self._frags)
    def _frag_factorized(self):
        return diagram_sum([frag_factorized(term) for term in self._terms], self._frags)
    def _simplified(self):    # collects together terms that are identical or related by a phase factor
        same = []    # eventually a list of groups (tuples of indices) of equivalent terms (ordered and hashable), but with duplicate groups
        for term_1 in self._terms:
            same_1 = []    # everything that is the same as the current term_1, including itself
            for i,term_2 in enumerate(self._terms):
                if diagram.compare_signs(term_1, term_2)!=0:
                    same_1 += [i]
            same += [tuple(sorted(same_1))]
        new_terms = []
        for group in sorted(set(same)):    # loop without duplicates; sorting just for aesthetics
            term = self._terms[group[0]]    # only need one example from each group to build a new term ...
            scalar = scalar_sum(0)          # ... but will need to scale to account for (anti) equivalences
            for i in group:    # build up scalar by looping over all terms in the group
                other_term = self._terms[i]    # could be same as term
                other_scalar = other_term.scalar()
                other_scalar.perm_mult(0 if diagram.compare_signs(term, other_term)==+1 else 1)
                scalar.increment(other_scalar)    # automatically simplified after each increment
            new_terms += [diagram(term, scalar=scalar)]
        return diagram_sum(new_terms, self._frags)
    def _condense_perm(self):    # identify and condense terms that differ only by a permutation of fragments
        perms = permutation_subs(self._frags)
        new_terms = []    # the new terms, where each stores information about fragment permuations
        exclude = []      # indices of terms to exclude from later iterations because they were taken care of in previous iterations
        for i,term_i in enumerate(self._terms):
            if i not in exclude:
                perm_list = [("+",perms[0])]    # information about permutations of term_i that occur
                for j,term_j in list(enumerate(self._terms))[i+1:]:    # slice after enumeration so absolute indexing is correct
                    if j not in exclude:
                        for perm in perms:    # it will only give nonzero under one permutation, maximum
                            perm_sign = diagram.compare_signs(term_i, frag_permuted(term_j,perm))
                            if perm_sign!=0:    # we have found a permutation under which (scaled) contraction with the integrals is identical
                                scalar_sign = term_i.scalar().compare_sign_to(term_j.scalar())
                                if   perm_sign*scalar_sign==+1:  sign = "+"
                                elif perm_sign*scalar_sign==-1:  sign = "-"
                                else:  raise RuntimeError("unexpected scaling of terms (generalize this code block if not erroneous)")
                                perm_list += [(sign, perm)]    # list of permutations and associated signs for this term
                                exclude += [j]                 # exclude terms identified as permutations of prior terms
                new_terms += [diagram(term_i, perm_list=perm_list)]
        return diagram_sum(new_terms, self._frags)
    def _ct_ordered(self):
        return diagram_sum(sorted([diagram(term) for term in self._terms], reverse=True, key=lambda x: sorted(x.ct_character())), self._frags)    # sorting the CT character gives the lowest overall first without fragment bias
    def _mult_by(self, x):
        return diagram_sum([mult_by(term,x) for term in self._terms], self._frags)
    def rho_notation(self):           # toggle ...
        for term in self._terms:  term.rho_notation()
        return self
    def as_braket(self, frags):              # toggle ...
        self._frags = frags
        for term in self._terms:  term.as_braket(frags)
        return self
    def abbreviated(self):            # toggle ...
        for term in self._terms:  term.abbreviated()
        return self
    def code(self):                   # toggle ...
        self._code = not self._code
        for term in self._terms:  term.code()
        return self
    def publication_ordered(self):    # toggle ...
        for term in self._terms:  term.publication_ordered()
        return self
    def __str__(self):
        if self._code:
            int_type, order = self._terms[0].int_type()
            connect = "\n"
            string  =  f"{int_type}{len(self._frags)}[{order}] = [\n          " + ", ".join(f"\"{term.name()}\"" for term in self._terms) + ",\n         ]\n\n"
        else:
            connect = "\\\\\n&+~"
            string = ""
        string += connect.join(str(term) for term in self._terms)
        if not self._code and len(self._terms)>1:
            string = " &~~~~~" + string + "~ "
        if self._code:
            string += f"\ncatalog[{len(self._frags)}] = {{\n"
            string += connect.join(term.catalog_entry() for term in self._terms)
            string +=  "\n}"
        return string
