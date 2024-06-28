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
from copy import deepcopy
from itertools import chain
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
    def __eq__(self, other):
        return (self._symbol, self.c_indices, self.a_indices) == (other._symbol, other.c_indices, other.a_indices)
    def __str__(self):
        labels = {
            "bra":  ",".join(str(p) for p in self.c_indices),
            "ket":  ",".join(str(p) for p in self.a_indices)
        }
        return self._symbol.format(**labels)

class s_int(integral_type):
    _symbol = "\\sigma_{{{bra} {ket}}}"
    def __init__(self, p, q=None):
        if p.fragment==q.fragment:  raise sigmaError()
        if q is None:
            integral_type.__init__(self, p)
        else:
            integral_type.__init__(self, [p], [q])

class h_int(integral_type):
    _symbol = "h^{{{bra}}}_{{{ket}}}"
    def __init__(self, p, q=None):
        if q is None:
            integral_type.__init__(self, p)
        else:
            integral_type.__init__(self, [p], [q])

class v_int(integral_type):
    _symbol = "v^{{{bra}}}_{{{ket}}}"
    def __init__(self, p, q=None, r=None, s=None):
        if (q is None) and (r is None) and (s is None):
            integral_type.__init__(self, p)
        else:
            integral_type.__init__(self, [p,q], [r,s])

class d_int(integral_type):
    _symbol = "\\rho_{{{bra}}}^{{{ket}}}"
    def __init__(self, c_indices, a_indices=None):
        if a_indices is None:
            integral_type.__init__(self, c_indices)
        else:
            integral_type.__init__(self, c_indices, a_indices)

class integral_list(object):
    def __init__(self, integrals):
        try:
            new_integrals = integrals._integrals
        except AttributeError:
            self._integrals = list(integrals)
        else:
            self._integrals = list(new_integrals)
    def c_indices(self):
        return chain.from_iterable([reversed(integral.c_indices) for integral in self._integrals])
    def a_indices(self):
        return chain.from_iterable([reversed(integral.a_indices) for integral in self._integrals])
    def fragments(self):
        return {p.fragment for p in self.c_indices()} | {p.fragment for p in self.a_indices()}
    def __eq__(self, other):
        # This will eventually need to be generalized in the following way:
        # We should not just test if one integral is equal to another but also if it is equal and opposite.
        # A more general function on v_int and d_int should return +1, 0, -1 for this, taking into account permutations.
        # A similar function should be implemented for integral_list, and __eq__ for both should be deprecated.
        # Then, assuming again that lists are the same length with no duplicates, we run over one list and ask
        # for its comparison to each member of the other.  If any integral fails to find a match, then the integral_list
        # function also returns 0, otherwise, it returns the product of all the +1/-1 match values.
        # The __eq__ function here is only used in the diagram_term equality, which should be similarly generalized.
        result = True
        if len(self._integrals)==len(other._integrals):
            for integral in self._integrals:            # works because lists should not contain duplicates ...
                if integral not in other._integrals:    # ... and integrals have equality defined (so works with in operator)
                    result = False
        else:
            result = False
        return result
    def __str__(self):
        return "".join(str(integral) for integral in self._integrals)



# this syntax because operations not performed in place
def frag_sorted(obj):      return obj._frag_sorted()
def frag_factorized(obj):  return obj._frag_factorized()
def rho_notation(obj):     return obj._rho_notation()
def simplified(obj):       return obj._simplified()



class field_op(object):
    def __init__(self, idx):
        self.idx = idx
    def __eq__(self, other):
        return (self.kind, self.idx) == (other.kind, other.idx)
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
    def __init__(self, ops, frags=None):    # should frags be set automatically from ops?
        try:
            new_ops = ops._ops
        except AttributeError:
            try:
                self._ops = dict(ops)
            except TypeError:
                self._ops = {None: list(ops)}
            self._frags = set(frags)
            self._dens_tens = None
        else:
            self._ops   = dict(new_ops)
            self._frags = set(ops._frags)
            if ops._dens_tens is None:
                self._dens_tens = None
            else:
                self._dens_tens = d_int(ops._dens_tens)
            if frags is not None:
                raise ValueError("not allowed to reset fragments in copy initialization of operator_string")
    def fragments(self):
        return set(self._frags)
    def c_indices(self):
        return [op.idx for op in self._flatten_ops() if op.kind=="c"]
    def a_indices(self):
        return [op.idx for op in self._flatten_ops() if op.kind=="a"]
    def rho_notation(self):
        if len(self._frags)!=1 or list(self._frags)[0] not in self._ops:
            raise RuntimeError("operator string cannot be put in single-fragment density form")
        self._dens_tens = d_int(self.c_indices(), self.a_indices())
        return self    # could this convenience be mistaken for not having done an in-place modification?
    def _flatten_ops(self):
        return chain.from_iterable([self._ops[key] for key in sorted(self._ops.keys())])
    def _frag_sorted(self):
        perm = 0
        sorted_ops = {frag:[] for frag in self._frags}
        for op in self._flatten_ops():
            frag = op.idx.fragment
            sorted_ops[frag] += [op]
            for other_frag,other_ops in sorted_ops.items():
                if other_frag>frag:
                    perm += len(other_ops)
        return perm, operator_string(sorted_ops, self._frags)
    def _frag_factorized(self):
        frags = sorted(self._ops.keys())
        if frags[0] is None:
            raise RuntimeError("cannot factorize operator string that has not been explicitly sorted")
        phases = []
        op_strings = []
        for frag in frags:
            raw_string = self._ops[frag]
            op_strings += [operator_string({frag: raw_string}, [frag])]
            for phase in phases:
                phase.frags += [frag]
            phases += [struct(n_ops=len(raw_string), frags=[])]
        return phases, op_strings
    def __eq__(self, other):
        return (self._ops, self._frags) == (other._ops, other._frags) 
    def __str__(self):
        if self._dens_tens:
            return str(self._dens_tens)    # i and j indices suppressed
        else:
            bra = " ".join(f"i_{{{n}}}" for n in self._frags)
            ket = " ".join(f"j_{{{n}}}" for n in self._frags)
            ops = " ".join(str(op) for op in self._flatten_ops())
            string = f"{{}}^{{{bra}}}\\langle{ops}\\rangle_{{{ket}}}"
        return string



class diagram_term(object):
    def __init__(self, integrals, op_strings, scalars=None):
        frags = set()
        for op_string in op_strings:
            frags |= op_string.fragments()
        if integrals.fragments()!=frags:
            raise ValueError("integrals and operators do not refer to same fragment set")
        def simplify_scalars(scalars):    # just an encapsulation of some sublogic
            new_scalars = []
            for scalar in scalars:
                const_pow = scalar.const_pow % 2
                frag_pows = tuple(frag for frag in sorted(set(scalar.frag_pows)) if scalar.frag_pows.count(frag)%2)    # need to sort for later equality checks
                new_scalars += [struct(const_pow=const_pow, frag_pows=frag_pows, scale=scalar.scale)]
            return new_scalars
        if scalars is None:
            scalars = [struct(const_pow=0, frag_pows=tuple(), scale=1)]    # -1 should be raised to the power of the constant plus the number of electrons in the ket for each fragment listed, all times the scale
        self._scalars = simplify_scalars(scalars)               # simplifies __str__ function, and deep-copy-making side-effect is desired
        self._integrals = integral_list(integrals)
        self._op_strings = [operator_string(op_string) for op_string in op_strings]
    def scalars(self):
        return [struct(scalar) for scalar in self._scalars]    # do not return mutalbe original
    def _frag_sorted(self):
        permute = 0
        op_strings = []
        for op_string in self._op_strings:
            perm, op_sorted = frag_sorted(op_string)
            permute += perm
            op_strings += [op_sorted]
        scalars = [struct(scalar) for scalar in self._scalars]
        for scalar in scalars:
            scalar.const_pow += permute
        return diagram_term(self._integrals, op_strings, scalars)
    def _frag_factorized(self):
        all_frag_phases = []
        op_strings = []
        for op_string in self._op_strings:
            frag_phases, op_factorized = frag_factorized(op_string)
            all_frag_phases += [frag_phases]
            op_strings += op_factorized    # assuming for now that there is no need to sort this (ie, recursion depth is 1)
        scalars = [struct(scalar) for scalar in self._scalars]    # these copies are shallow ...
        for scalar in scalars:
            scalar.frag_pows = list(scalar.frag_pows)                   # ... so copy lists for modification
            for frag_phases in all_frag_phases:
                for frag_phase in frag_phases:
                    if frag_phase.n_ops%2:
                        scalar.frag_pows = scalar.frag_pows + frag_phase.frags    # cannot use += because tuples immutable
        return diagram_term(self._integrals, op_strings, scalars)
    def _rho_notation(self):
        dens_ops = [operator_string(op_string).rho_notation() for op_string in self._op_strings]
        return diagram_term(self._integrals, dens_ops, self._scalars)
    def __eq__(self, other):
        # Since Einstein summation implied, can rearrange letters for indices to see equality.
        # Since integrals and operators both point to same indices (ie, same object id) can just change letters therein,
        #  and fortunately deepcopy preserves the "connectivity".
        # Assumes operator strings having been sorted and factorized (otherwise could get false negatives),
        #  but test on integrals equality allows for permutations of scalar factors
        # Only tests integrals and operators; scale and phase handled upon combination
        self_copy, other_copy = deepcopy(self), deepcopy(other)
        for term in self_copy, other_copy:
            letter_idx = 0
            for op_string in term._op_strings:
                for p in op_string.c_indices():
                    p.letter = _letters[letter_idx]
                    letter_idx += 1
                for p in op_string.a_indices():
                    p.letter = _letters[letter_idx]
                    letter_idx += 1
        return (self_copy._integrals, self_copy._op_strings) == (other_copy._integrals, other_copy._op_strings)
    def __str__(self):
        def scalars_str(scalars):    # just an encapsulation of some sublogic
            substrings = []
            for scalar in scalars:
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
        string += scalars_str(self._scalars)
        return string
    @staticmethod
    def from_integrals(integrals):
        integrals = integral_list(integrals)
        c_indices = list(integrals.c_indices())    # these lists are ordered ...
        a_indices = list(integrals.a_indices())    # .... for this purpose
        all_indices = c_indices + a_indices
        for i,p in enumerate(all_indices):
            for q in all_indices[i+1:]:
                if p==q:
                    raise ValueError("illegal combination of integrals with duplicate indices")
        c_ops = [c_op(p) for p in reversed(c_indices)]
        a_ops = [a_op(p) for p in          a_indices ]
        return diagram_term(integrals, [operator_string(c_ops + a_ops, integrals.fragments())])

class term_list(object):
    def __init__(self, terms):
        self._terms = list(terms)
    def _frag_sorted(self):
        return term_list([frag_sorted(term) for term in self._terms])
    def _frag_factorized(self):
        return term_list([frag_factorized(term) for term in self._terms])
    def _rho_notation(self):
        return term_list([rho_notation(term) for term in self._terms])
    def _simplified(self):
        # This digs too much into the innards of diagram_term but whatever, it is the top layer and we are almost done with what we want
        # fix abuse
        same = []
        for term_1 in self._terms:
            same_1 = []    # everything that is the same as the current term_1, including itself
            for i,term_2 in enumerate(self._terms):
                if term_1==term_2:    # abuse of ==  (ok for now)
                    same_1 += [i]
            same += [tuple(sorted(same_1))]    # eventually contains all groups of equivalent terms (ordered and hashable), but with duplicate groups
        same = set(same)    # remove duplicates
        new_terms = []
        for group in sorted(same):    # sorting just for aesthetics
            integrals  = self._terms[group[0]]._integrals
            op_strings = self._terms[group[0]]._op_strings
            scalars = list(chain.from_iterable(self._terms[i].scalars() for i in group))

            same_pows = []
            for scalar_1 in scalars:
                same_pows_1 = []    # everything that has the same pows as the current scalar_1, including itself
                for i,scalar_2 in enumerate(scalars):
                    if scalar_1.frag_pows==scalar_2.frag_pows:
                        same_pows_1 += [i]
                same_pows += [tuple(sorted(same_pows_1))]    # eventually contains all groups of equivalent powers (ordered and hashable), but with duplicate groups
            same_pows = set(same_pows)    # remove duplicates
            combined_scalars = []
            for pow_group in sorted(same_pows):   # sorting is redundant?
                frag_pows = scalars[pow_group[0]].frag_pows
                raw_scalar = 0
                for i in pow_group:
                    raw_scalar += scalars[i].scale * (-1)**scalars[i].const_pow
                combined_scalars += [struct(const_pow=int(raw_scalar<0), frag_pows=frag_pows, scale=abs(raw_scalar))]

            new_terms += [diagram_term(integrals, op_strings, combined_scalars)]
        return term_list(new_terms)
    def __str__(self):
        return " &~" + "\\\\\n+&~".join(str(term) for term in self._terms)



frag_range = lambda n: range(1, n+1)

def gen_diagram(S_order, n_frags, prototerms, letters):
    letter_idx = 0
    for o in range(S_order):
        prototerms, old_prototerms = [], prototerms
        for p_frag in frag_range(n_frags):
            for q_frag in frag_range(n_frags):
                p = index(letters[letter_idx],   p_frag)
                q = index(letters[letter_idx+1], q_frag)
                try:
                    factor = s_int(p, q)
                except sigmaError:
                    pass
                else:
                    for prototerm in old_prototerms:
                        if prototerm is None:
                            prototerms += [[factor]]
                        else:
                            prototerms += [[factor] + prototerm]
        letter_idx += 2
    return term_list([diagram_term.from_integrals(prototerm) for prototerm in prototerms])

def s_diagram(S_order, n_frags):
    return gen_diagram(S_order, n_frags, prototerms=[None], letters=_letters)

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
    sorted_terms = frag_sorted(terms)
    factored_terms = frag_factorized(sorted_terms)
    rho_terms = rho_notation(factored_terms)
    print(terms, "\n")
    print(sorted_terms, "\n")
    print(factored_terms, "\n")
    print(rho_terms, "\n")
    simplified_terms = simplified(rho_terms)
    print(simplified_terms, "\n")
