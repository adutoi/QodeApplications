#    (C) Copyright 2024 Anthony D. Dutoi
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
import copy
import itertools
from qode.util import struct, as_tuple

_letters = "pqrstuvwxyzabcdefghijklmno"



class index(struct):
    def __init__(self, letter, fragment):
        struct.__init__(self, letter=letter, fragment=fragment)
    def __str__(self):
        return f"{self.letter}_{self.fragment}"



class sigmaError(ValueError):
    def __init__(self, message=None):
        if message is None:
            self.message = "diagonal block of sigma assumed to be zero"
        ValueError.__init__(self, self.message)

class integral_type(object):
    def __init__(self, c_indices, a_indices=None):
        if a_indices is None:
            other = c_indices
            self.c_indices = list(other.c_indices)
            self.a_indices = list(other.a_indices)
        else:
            self.c_indices = list(c_indices)
            self.a_indices = list(a_indices)
    def rho_notation(self):
        try:
            on_off = self._rho_notation
        except AttributeError:
            pass
        else:
            self._rho_notation = not on_off
    def _labels(self):
        return (
            ",".join(str(p) for p in self.c_indices),
            ",".join(str(p) for p in self.a_indices)
        )
    def __eq__(self, other):
        try:
            result = ((self.kind, self.c_indices, self.a_indices) == (other.kind, other.c_indices, other.a_indices))
        except:
            result = False
        return result

class h_int(integral_type):
    kind = "h"
    def __init__(self, p, q=None):
        if q is None:
            integral_type.__init__(self, p)
        else:
            integral_type.__init__(self, [p], [q])
    def __str__(self):
        c, a = self._labels()
        return f"h^{{{c}}}_{{{a}}}"

class s_int(h_int):
    kind = "s"
    def __init__(self, p, q=None):
        h_int.__init__(self, p, q)
        if self.c_indices[0].fragment==self.a_indices[0].fragment:
            raise sigmaError()
    def __str__(self):
        c, a = self._labels()
        return f"\\sigma_{{{c} {a}}}"

class v_int(integral_type):
    kind = "v"
    def __init__(self, p, q=None, r=None, s=None):
        if (q is None) and (r is None) and (s is None):
            integral_type.__init__(self, p)
        else:
            integral_type.__init__(self, [p,q], [r,s])
    def __str__(self):
        c, a = self._labels()
        return f"v^{{{c}}}_{{{a}}}"

class d_int(integral_type):
    kind = "d"
    def __init__(self, c_indices, a_indices=None):
        if a_indices is None:
            integral_type.__init__(self, c_indices)
        else:
            integral_type.__init__(self, c_indices, a_indices)
        self._rho_notation = False    # publically accessible.  only affects printing
    def fragment(self):
        try:
            frag = self.c_indices[0].fragment    # these should all be the same ...
        except IndexError:
            frag = self.a_indices[0].fragment    # ... if they exist at all
        return frag
    def __str__(self):
        if self._rho_notation:    # i and j indices suppressed
            c, a = self._labels()
            return f"\\rho_{{{c}}}^{{{a}}}"
        else:
            frag = self.fragment()
            c = " ".join(str(c_op(p)) for p in self.c_indices)
            a = " ".join(str(a_op(p)) for p in self.a_indices)
            return f"{{}}^{{i_{{ {frag} }} }}\\langle {c} {a} \\rangle_{{j_{{ {frag} }} }}"
    @staticmethod
    def from_ops(op_list):
        c_indices = [op.idx for op in op_list if op.kind=="c"]
        a_indices = [op.idx for op in op_list if op.kind=="a"]
        all_indices = c_indices + a_indices
        frag = all_indices[0].fragment
        if any(p.fragment!=frag for p in all_indices):
            raise ValueError("density tensor indices should all belong to the same fragment")
        return d_int(c_indices, a_indices)

# make this behave like a list.  only need two of the for indices functions
class integral_list(object):
    def __init__(self, integrals):
        try:
            new_integrals = integrals._integrals
        except AttributeError:
            self._integrals = list(integrals)
        else:
            self._integrals = list(new_integrals)
    def moint_c_indices(self):
        return list(itertools.chain.from_iterable([reversed(integral.c_indices) for integral in self._integrals if integral.kind!="d"]))    # exemption on kind ...
    def moint_a_indices(self):
        return list(itertools.chain.from_iterable([reversed(integral.a_indices) for integral in self._integrals if integral.kind!="d"]))    # ... is just reflection of ...
    def dens_c_indices(self):
        return list(itertools.chain.from_iterable([integral.c_indices for integral in self._integrals if integral.kind=="d"]))              # ... the limited purposes ...
    def dens_a_indices(self):
        return list(itertools.chain.from_iterable([integral.a_indices for integral in self._integrals if integral.kind=="d"]))              # ... of these functions
    def fragments(self):
        return {p.fragment for p in self.moint_c_indices()} | {p.fragment for p in self.moint_a_indices()}
    def raw_list(self):
        return list(self._integrals)
    def rho_notation(self):
        for integral in self._integrals:  integral.rho_notation()
    def __eq__(self, other):
        # This will eventually need to be generalized in the following way:
        # We should not just test if one integral is equal to another but also if it is equal and opposite.
        # A more general function on v_int and d_int should return +1, 0, -1 for this, taking into account permutations.
        # A similar function should be implemented for integral_list, and __eq__ for both should be deprecated.
        # Then, assuming again that lists are the same length with no duplicates, we run over one list and ask
        # for its comparison to each member of the other.  If any integral fails to find a match, then the integral_list
        # function also returns 0, otherwise, it returns the product of all the +1/-1 match values.
        # The __eq__ function here is only used in the diagram_term equality, which should be similarly generalized.
        try:
            result = True
            if len(self._integrals)==len(other._integrals):
                for integral in self._integrals:            # works because lists should not contain duplicates ...
                    if integral not in other._integrals:    # ... and integrals have equality defined (so works with "in" operator)
                        result = False
            else:
                result = False
        except:
            result = False
        return result
    def __str__(self):
        return " ~ ".join(str(integral) for integral in self._integrals)



# this syntax because operations not performed in place
def frag_sorted(obj):      return obj._frag_sorted()
def frag_factorized(obj):  return obj._frag_factorized()    # rename to exp_val and take brackets off of sorted and unsorted ones? (brackets/exp_val as earlier option?
def simplified(obj):       return obj._simplified()



class field_op(object):
    def __init__(self, idx):
        self.idx = idx
    def __eq__(self, other):
        try:
            result = ((self.kind, self.idx) == (other.kind, other.idx))
        except:
            result = False
        return result
    def __str__(self):
        return self._symbol.format(p=self.idx)

class c_op(field_op):
    _symbol = "\\hat{{c}}_{{{p}}}"
    kind = "c"
    __init__ = field_op.__init__

class a_op(field_op):
    _symbol = "\\hat{{a}}^{{{p}}}"
    kind = "a"
    __init__ = field_op.__init__

class operator_string(object):
    def __init__(self, ops):
        try:
            new_ops = ops._ops
        except AttributeError:
            try:
                self._ops = dict(ops)
            except TypeError:
                self._ops = {None: list(ops)}
        else:
            self._ops = dict(new_ops)
    def _flatten_ops(self):
        return itertools.chain.from_iterable([self._ops[key] for key in sorted(self._ops.keys())])
    def fragments(self):
        return set(op.idx.fragment for op in self._flatten_ops())
    def _frag_sorted(self):
        perm = 0
        sorted_ops = {frag:[] for frag in self.fragments()}
        for op in self._flatten_ops():
            frag = op.idx.fragment
            sorted_ops[frag] += [op]
            for other_frag,other_ops in sorted_ops.items():
                if other_frag>frag:
                    perm += len(other_ops)
        return perm, operator_string(sorted_ops)
    def _frag_factorized(self):
        frags = sorted(self._ops.keys())    # aesthetic only.  this sorting will affect the order of printing of product
        if frags[0] is None:
            raise RuntimeError("cannot factorize operator string that has not been explicitly sorted")
        phases = []
        integrals = []
        for frag in frags:
            raw_string = self._ops[frag]
            integrals += [d_int.from_ops(raw_string)]
            for phase in phases:
                phase.frags += [frag]
            phases += [struct(n_ops=len(raw_string), frags=[])]
        return phases, integrals
    def __eq__(self, other):
        try:
            result = (self._ops == other._ops)
        except:
            result = False
        return result
    def __str__(self):
        bra = " ".join(f"i_{{{n}}}" for n in self.fragments())
        ket = " ".join(f"j_{{{n}}}" for n in self.fragments())
        ops = " ".join(str(op) for op in self._flatten_ops())
        return f"{{}}^{{{bra}}}\\langle{ops}\\rangle_{{{ket}}}"



# op_strings does not need to be a list any more -> op_string
class diagram_term(object):
    def __init__(self, integrals, op_strings=None, scalar_sum=None):
        if op_strings is None:    # allows copying with different scalar_sum
            other = integrals
            integrals  = other._integrals
            op_strings = other._op_strings
        def simplify_scalars(scalar_sum):    # just an encapsulation of some sublogic
            new_scalar_sum = []
            for scalar in scalar_sum:
                const_pow = scalar.const_pow % 2
                frag_pows = tuple(frag for frag in sorted(set(scalar.frag_pows)) if scalar.frag_pows.count(frag)%2)    # need to sort for later equality checks
                new_scalar_sum += [struct(const_pow=const_pow, frag_pows=frag_pows, scale=scalar.scale)]
            return new_scalar_sum
        if scalar_sum is None:
            scalar_sum = [struct(const_pow=0, frag_pows=tuple(), scale=1)]    # -1 should be raised to the power of the constant plus the number of electrons in the ket for each fragment listed, all times the scale
        self._scalar_sum = simplify_scalars(scalar_sum)               # simplifies __str__ function, and deep-copy-making side-effect is desired
        self._integrals = integral_list(integrals)
        self._op_strings = [operator_string(op_string) for op_string in op_strings]
    @staticmethod
    def are_multiples(term1, term2):
        # Since Einstein summation implied, can rearrange letters for indices to see equality.
        # Since integrals and operators both point to same indices (ie, same object id) can just change letters therein,
        #  and fortunately deepcopy preserves the "connectivity".
        # Assumes operator strings having been sorted and factorized (otherwise could get false negatives),
        #  but test on integrals equality allows for permutations of scalar factors
        # Only tests integrals and operators; scale and phase handled upon combination
        #print("# # #")
        if term1._op_strings or term2._op_strings:
            raise RuntimeError("can only compare terms that have been reduced to products of MO integrals and single-fragment densities")
        term1_copy, term2_copy = copy.deepcopy(term1), copy.deepcopy(term2)
        #print("before", term1_copy)
        #print("before", term2_copy)
        for term in [term1_copy, term2_copy]:
            letter_idx = 0
            indices = term._integrals.dens_c_indices() + term._integrals.dens_a_indices()
            for p in indices:
                p.letter = _letters[letter_idx]
                letter_idx += 1
        #print("after", term1_copy)
        #print("after", term2_copy)
        #print(term1_copy._integrals == term2_copy._integrals)
        return (term1_copy._integrals == term2_copy._integrals)    # only care about multiples, do not test the scalars
    def are_frag_perm_multiples(term1, term2, perm):    # perm needs to be a dict bc frags may not be contiguous nor start at zero
        #print("#####")
        #print("term1", term1)
        #print("term2", term2)
        term2_copy = copy.deepcopy(term2)
        integrals = []
        for integral in term2_copy._integrals.raw_list():
            if integral.kind=="d":
                integrals += [integral.fragment()]
            else:
                integrals += [integral]
        for integral in term2_copy._integrals.raw_list():
            if integral.kind=="d":
                integrals[integrals.index(perm[integral.fragment()])] = integral
        integrals = integral_list(integrals)
        #print(integrals)
        indices = integrals.dens_c_indices() + integrals.dens_a_indices()
        for p in indices:
            p.fragment = perm[p.fragment]
        #print(integrals)
        term2_copy._integrals = integrals
        #print("term2_copy", term2_copy)
        return diagram_term.are_multiples(term1, term2_copy)
    def rho_notation(self):
        self._integrals.rho_notation()
    def scalar_sum(self):
        return [struct(scalar) for scalar in self._scalar_sum]    # do not return mutalbe original
    def _frag_sorted(self):
        permute = 0
        op_strings = []
        for op_string in self._op_strings:
            perm, op_sorted = frag_sorted(op_string)
            permute += perm
            op_strings += [op_sorted]
        scalar_sum = [struct(scalar) for scalar in self._scalar_sum]
        for scalar in scalar_sum:
            scalar.const_pow += permute
        return diagram_term(self._integrals, op_strings, scalar_sum)
    def _frag_factorized(self):
        all_frag_phases = []
        integrals = self._integrals.raw_list()    # a copy, so ok to modify
        for op_string in self._op_strings:
            frag_phases, ints_factorized = frag_factorized(op_string)
            all_frag_phases += [frag_phases]
            integrals += ints_factorized
        scalar_sum = [struct(scalar) for scalar in self._scalar_sum]    # these copies are shallow ...
        for scalar in scalar_sum:
            scalar.frag_pows = list(scalar.frag_pows)                   # ... so copy lists for modification
            for frag_phases in all_frag_phases:
                for frag_phase in frag_phases:
                    if frag_phase.n_ops%2:
                        scalar.frag_pows = scalar.frag_pows + frag_phase.frags    # cannot use += because tuples immutable
        return diagram_term(integral_list(integrals), [], scalar_sum)
    def __str__(self):
        def scalar_sum_str(scalar_sum):    # just an encapsulation of some sublogic
            substrings = []
            for scalar in scalar_sum:
                subsubstrings = []
                if scalar.scale!=1:
                    subsubstrings += [f"{scalar.scale}"]
                if scalar.const_pow or len(scalar.frag_pows)>0:
                    powers = [f"n_{{i_{frag}}}" for frag in scalar.frag_pows]
                    if scalar.const_pow:  powers += ["1"]
                    power = "+".join(powers)
                    subsubstrings += [f"(-1)^{{{power}}}"]
                substring = "\\cdot".join(subsubstrings)
                if not substring:  substring = "1"
                substrings += [substring]
            string = "+".join(substrings)
            if len(substrings)>1:
                string = f"\\big[{string}\\big]"
            if string=="1":
                string = ""
            else:
                string = f"\\times {string}"
            return string
        string = f"{self._integrals}"
        for op_string in self._op_strings:
            string += f"~{op_string}"
        string += scalar_sum_str(self._scalar_sum)
        return string
    @staticmethod
    def from_integrals(integrals):
        integrals = integral_list(integrals)
        c_indices = list(integrals.moint_c_indices())    # these lists are ordered ...
        a_indices = list(integrals.moint_a_indices())    # .... for this purpose
        all_indices = c_indices + a_indices
        for i,p in enumerate(all_indices):
            for q in all_indices[i+1:]:
                if p==q:
                    raise ValueError("illegal combination of integrals with duplicate indices")
        c_ops = [c_op(p) for p in          c_indices ]
        a_ops = [a_op(p) for p in reversed(a_indices)]
        return diagram_term(integrals, [operator_string(c_ops + a_ops)])

class term_list(object):
    def __init__(self, terms):
        self._terms = list(terms)
    def rho_notation(self):
        for term in self._terms:  term.rho_notation()
    def _frag_sorted(self):
        return term_list([frag_sorted(term) for term in self._terms])
    def _frag_factorized(self):
        return term_list([frag_factorized(term) for term in self._terms])
    def _simplified(self):
        same = []
        for term_1 in self._terms:
            same_1 = []    # everything that is the same as the current term_1, including itself
            for i,term_2 in enumerate(self._terms):
                if diagram_term.are_multiples(term_1, term_2):
                    same_1 += [i]
            same += [tuple(sorted(same_1))]    # eventually contains all groups of equivalent terms (ordered and hashable), but with duplicate groups
        same = set(same)    # remove duplicates
        new_terms = []
        for group in sorted(same):    # sorting just for aesthetics
            term = self._terms[group[0]]
            scalar_sum = list(itertools.chain.from_iterable(self._terms[i].scalar_sum() for i in group))

            same_pows = []
            for scalar_1 in scalar_sum:
                same_pows_1 = []    # everything that has the same pows as the current scalar_1, including itself
                for i,scalar_2 in enumerate(scalar_sum):
                    if scalar_1.frag_pows==scalar_2.frag_pows:
                        same_pows_1 += [i]
                same_pows += [tuple(sorted(same_pows_1))]    # eventually contains all groups of equivalent powers (ordered and hashable), but with duplicate groups
            same_pows = set(same_pows)    # remove duplicates
            combined_scalar_sum = []
            for pow_group in sorted(same_pows):   # sorting is redundant?
                frag_pows = scalar_sum[pow_group[0]].frag_pows
                raw_scalar = 0
                for i in pow_group:
                    raw_scalar += scalar_sum[i].scale * (-1)**scalar_sum[i].const_pow
                combined_scalar_sum += [struct(const_pow=int(raw_scalar<0), frag_pows=frag_pows, scale=abs(raw_scalar))]

            new_terms += [diagram_term(term, scalar_sum=combined_scalar_sum)]
        return term_list(new_terms)
    def __str__(self):
        return " &~" + "\\\\\n+&~".join(str(term) for term in self._terms)



frag_range = lambda n: range(1, n+1)

def combine(old, new):
    if old==[1]:  return       [new]
    else:         return old + [new]

def s_diagram(S_order, n_frags, letters=_letters):
    letter_idx = 0
    prototerms = [[1]]
    for o in range(S_order):
        new_prototerms = []
        for p_frag in frag_range(n_frags):
            for q_frag in frag_range(n_frags):
                p = index(letters[letter_idx],   p_frag)
                q = index(letters[letter_idx+1], q_frag)
                try:
                    factor = s_int(p, q)
                except sigmaError:
                    pass
                else:
                    new_prototerms += [factor]
        prototerms = [combine(old,new) for old,new in itertools.product(prototerms, new_prototerms)]
        letter_idx += 2
    return term_list([diagram_term.from_integrals(prototerm) for prototerm in prototerms])

def h_diagram(S_order, n_frags):
    prototerms = []
    for p_frag in frag_range(n_frags):
        for q_frag in frag_range(n_frags):
            p = index("p", p_frag)
            q = index("q", q_frag)
            prototerms += [[h_int(p, q)]]
    return gen_diagram(S_order, n_frags, prototerms, letters=_letters[4:])

def v_diagram(S_order, n_frags):
    prototerms = []
    for p_frag in frag_range(n_frags):
        for q_frag in frag_range(n_frags):
            for r_frag in frag_range(n_frags):
                for s_frag in frag_range(n_frags):
                    p = index("p", p_frag)
                    q = index("q", q_frag)
                    r = index("r", r_frag)
                    s = index("s", s_frag)
                    prototerms += [[v_int(p, q, r, s)]]
    return gen_diagram(S_order, n_frags, prototerms, letters=_letters[4:])



if __name__ == "__main__":
    S_order = int(sys.argv[1])
    terms = s_diagram(S_order, n_frags=2)
    print(terms, "\n")
    sorted_terms = frag_sorted(terms)
    print(sorted_terms, "\n")
    factored_terms = frag_factorized(sorted_terms)
    print(factored_terms, "\n")
    factored_terms.rho_notation()
    print(factored_terms, "\n")
    simplified_terms = simplified(factored_terms)
    print(simplified_terms, "\n")
    same = []
    for term_1 in simplified_terms._terms:
        print("###################################")
        same_1 = []    # everything that is the same as the current term_1, including itself
        for i,term_2 in enumerate(simplified_terms._terms):
            if diagram_term.are_frag_perm_multiples(term_1, term_2, {1:2, 2:1}):
                print(i)
                same_1 += [i]
        same += [tuple(sorted(same_1))]    # eventually contains all groups of equivalent terms (ordered and hashable), but with duplicate groups
    same = set(same)    # remove duplicates
    print(same)
