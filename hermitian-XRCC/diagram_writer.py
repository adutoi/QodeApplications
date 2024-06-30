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

import copy
import itertools
from qode.util import struct, as_tuple
from permute import are_permutations

letters = "pqrstuvwxyzabcdefghijklmno"



# this syntax because operations not performed in place
def frag_sorted(obj):      return obj._frag_sorted()
def frag_factorized(obj):  return obj._frag_factorized()
def simplified(obj):       return obj._simplified()
def condense_perm(obj):    return obj._condense_perm()



class index(struct):
    def __init__(self, letter, fragment):
        struct.__init__(self, letter=letter, fragment=fragment, abbreviated=False)
    def __str__(self):
        if self.fragment is None:
            return f"{self.letter}"
        elif self.abbreviated:
            return f"{self.fragment}"
        else:
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
            self._abbreviated = other._abbreviated
            self._code = other._code
        else:
            self.c_indices = list(c_indices)
            self.a_indices = list(a_indices)
            self._abbreviated = False
            self._code = False
    def rho_notation(self):
        try:
            on_off = self._rho_notation
        except AttributeError:
            pass
        else:
            self._rho_notation = not on_off
    def abbreviated(self):
        self._abbreviated = not self._abbreviated
        for index in self.c_indices + self.a_indices:    # direct toggle of indices is unreliable ...
            index.abbreviated = self._abbreviated        # because each will be toggled once for each occurance
    def code(self):
        self._code = not self._code
    def _labels(self):
        if self._code:
            letters = []
            frags = []
            for p in self.c_indices + self.a_indices:
                letters += [p.letter]
                frags   += [str(p.fragment)]
            return (",".join(frags), ",".join(letters))
        else:
            return (
                " ".join(str(p) for p in self.c_indices),
                " ".join(str(p) for p in self.a_indices)
            )
    def compare_sign_to(self, other):
        result = True
        if self.kind==other.kind:
            c_are_perms, c_parity = are_permutations(self.c_indices, other.c_indices)
            a_are_perms, a_parity = are_permutations(self.a_indices, other.a_indices)
            if c_are_perms and a_are_perms:
                return (-1)**(c_parity + a_parity)
            else:
                result = 0
        else:
            result = 0
        return result

class h_int(integral_type):
    kind = "h"
    def __init__(self, p, q=None):
        if q is None:
            integral_type.__init__(self, p)
        else:
            integral_type.__init__(self, [p], [q])
    def __str__(self):
        if self._code:
            frags, letters = self._labels()
            return f"h[{frags}]({letters})"
        else:
            c, a = self._labels()
            return f"h^{{{c}}}_{{{a}}}"

class s_int(h_int):
    kind = "s"
    def __init__(self, p, q=None):
        h_int.__init__(self, p, q)
        if (self.c_indices[0].fragment is not None) and (self.c_indices[0].fragment==self.a_indices[0].fragment):
            raise sigmaError()
    def __str__(self):
        if self._code:
            frags, letters = self._labels()
            return f"s[{frags}]({letters})"
        else:
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
        if self._code:
            frags, letters = self._labels()
            return f"v[{frags}]({letters})"
        else:
            c, a = self._labels()
            return f"v^{{{c}}}_{{{a}}}"

class d_int(integral_type):
    kind = "d"
    def __init__(self, c_indices, a_indices=None):
        if a_indices is None:
            integral_type.__init__(self, c_indices)
        else:
            integral_type.__init__(self, c_indices, a_indices)
        self._rho_notation = False
    def fragment(self):
        try:
            frag = self.c_indices[0].fragment    # these should all be the same ...
        except IndexError:
            frag = self.a_indices[0].fragment    # ... if they exist at all
        return frag
    def __str__(self):
        if self._code:
            ca = "c"*len(self.c_indices) + "a"*len(self.a_indices)
            frags, letters = self._labels()
            frag = frags[0]    # because they should all be the same
            return f"{ca}[{frag}](i{frag},j{frag},{letters})"
        else:
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
            self._publication_ordered = False
            self._abbreviated = False
            self._code = False
        else:
            other = integrals
            self._integrals = list(new_integrals)
            self._publication_ordered = other._publication_ordered
            self._abbreviated = other._abbreviated
            self._code = other._code
    def moint_c_indices(self):
        aesthetically_ordered = [integral for integral in self._integrals if integral.kind in ("h","v")] + [integral for integral in self._integrals if integral.kind=="s"]
        return list(itertools.chain.from_iterable([integral.c_indices for integral in aesthetically_ordered]))    # exemption on kind ...
    def moint_a_indices(self):
        aesthetically_ordered = [integral for integral in self._integrals if integral.kind in ("h","v")] + [integral for integral in self._integrals if integral.kind=="s"]
        return list(itertools.chain.from_iterable([integral.a_indices for integral in aesthetically_ordered]))    # ... is just reflection of ...
    def dens_indices(self):
        dens_ints = {frag:[] for frag in self.fragments()}
        for integral in self._integrals:
            if integral.kind=="d":                                                                                                          # ... the limited purposes of these functions
                dens_ints[integral.fragment()] += [integral]
        indices = []
        for frag in sorted(dens_ints.keys()):
            for integral in dens_ints[frag]:
                indices += integral.c_indices + integral.a_indices
        return indices
    def fragments(self):
        return {p.fragment for p in self.moint_c_indices()} | {p.fragment for p in self.moint_a_indices()}
    def raw_list(self):
        return list(self._integrals)
    def rho_notation(self):
        for integral in self._integrals:  integral.rho_notation()
    def publication_ordered(self):
        self._publication_ordered = not self._publication_ordered
    def abbreviated(self):
        for integral in self._integrals:  integral.abbreviated()
        self._abbreviated = not self._abbreviated
    def code(self):
        for integral in self._integrals:
            integral.code()
        self._code = not self._code
    def compare_sign_to(self, other):
        result = 1
        if len(self._integrals)==len(other._integrals):
            for self_integral in self._integrals:
                sign = 0
                for other_integral in other._integrals:
                    s = self_integral.compare_sign_to(other_integral)
                    if s!=0:        # works because neither list ...
                        sign = s    # ... should contain duplicates
                result *= sign
        else:
            result = 0
        return result
    def __str__(self):
        integrals = self._integrals
        enclose = "{}"
        connect = " ~ "
        if self._abbreviated:
            integrals  = [integral for integral in self._integrals if integral.kind!="d"]
            enclose = "\\langle {} \\rangle"
        elif self._publication_ordered:
            integrals  = [integral for integral in self._integrals if integral.kind=="d"]
            integrals += [integral for integral in self._integrals if integral.kind!="d"]
        if self._code:
            connect = "\n    @ "
        return enclose.format(connect.join(str(integral) for integral in integrals))



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
            self._exp_val = False
        else:
            self._ops = dict(new_ops)
            self._exp_val = ops._exp_val
    def as_exp_val(self):
        self._exp_val = not self._exp_val
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
        bra, ket = "", ""
        if self._exp_val:
            bra = "{{}}^{{{bra}}}\\langle".format(bra=" ".join(f"i_{{{n}}}" for n in self.fragments()))
            ket =     "\\rangle_{{{ket}}}".format(ket=" ".join(f"j_{{{n}}}" for n in self.fragments()))
        ops = " ".join(str(op) for op in self._flatten_ops())
        return f"{bra}{ops}{ket}"



# op_strings does not need to be a list any more -> op_string
class diagram_term(object):
    def __init__(self, integrals, op_strings=None, scalar_sum=None, perm_list=None):
        publication_ordered = False
        abbreviated = False
        code = False
        if op_strings is None:    # if not copying, op_strings is required, so use as switch
            other = integrals
            integrals  = other._integrals
            op_strings = other._op_strings
            if scalar_sum is None:    # allows copying with different scalar_sum ...
                scalar_sum = other._scalar_sum
            if perm_list is None:     # ... or perm_list
                perm_list = other._perm_list
            publication_ordered = other._publication_ordered
            abbreviated = other._abbreviated
            code = other._code
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
        if perm_list is None:
            perm_list = []
        self._perm_list = [(sign,dict(perm)) for sign,perm in perm_list]    # just making a copy
        self._publication_ordered = publication_ordered
        self._abbreviated = abbreviated
        self._code = code
    @staticmethod
    def compare_signs(term1, term2):
        # Since Einstein summation implied, can rearrange letters for indices to see equality.
        # Since integrals and operators both point to same indices (ie, same object id) can just change letters therein,
        #  and fortunately deepcopy preserves the "connectivity".
        # Assumes operator strings having been sorted and factorized (otherwise could get false negatives),
        #  but test on integrals equality allows for permutations of scalar factors
        # Only tests integrals and operators; scale and phase handled upon combination
        if term1._op_strings or term2._op_strings:
            raise RuntimeError("can only compare terms that have been reduced to products of MO integrals and single-fragment densities")
        term1_copy, term2_copy = copy.deepcopy(term1), copy.deepcopy(term2)
        for term in [term1_copy, term2_copy]:
            letter_idx = 0
            for p in term._integrals.dens_indices():    # these are ordere by fragment
                p.letter = letters[letter_idx]
                letter_idx += 1
        return term1_copy._integrals.compare_sign_to(term2_copy._integrals)    # only care about multiples, do not test the scalars
    def compare_frag_perm_signs(term1, term2, perm):    # perm needs to be a dict bc frags may not be contiguous nor start at zero
        term2_copy = copy.deepcopy(term2)
        for p in term2_copy._integrals.dens_indices():
            p.fragment = perm[p.fragment]
        return diagram_term.compare_signs(term1, term2_copy)
    def rho_notation(self):
        self._integrals.rho_notation()
    def as_exp_val(self):
        for op_string in self._op_strings:  op_string.as_exp_val()
    def publication_ordered(self):
        self._integrals.publication_ordered()
        self._publication_ordered = not self._publication_ordered
    def abbreviated(self):
        self._integrals.abbreviated()
        self._abbreviated = not self._abbreviated
    def code(self):
        self._integrals.code()
        self._code = not self._code
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
        def scalar_sum_str(scalar_sum):    # just an encapsulation of some sublogic (does read self._publication_ordered and self._abbreviated)
            if self._abbreviated:
                return ""
            substrings = []
            for scalar in scalar_sum:
                subsubstrings = []
                if scalar.scale!=1:
                    subsubstrings += [f"{scalar.scale}"]
                if scalar.const_pow or len(scalar.frag_pows)>0:
                    if self._code:
                        powers = [f"n_i{frag}" for frag in scalar.frag_pows]
                    else:
                        powers = [f"n_{{i_{frag}}}" for frag in scalar.frag_pows]
                    if scalar.const_pow:  powers += ["1"]
                    power = "+".join(powers)
                    if self._code:
                        subsubstrings += [f"(-1)**({power})"]
                    else:
                        subsubstrings += [f"(-1)^{{{power}}}"]
                mult = "" if self._publication_ordered else "\\cdot"
                if self._code:
                    mult = " * "
                substring = mult.join(subsubstrings)
                if not substring:  substring = "1"
                substrings += [substring]
            string = "+".join(substrings)
            if len(substrings)>1:
                string = f"({string})"
            if self._code:
                string = f"{string} *\n      "
            elif string=="1":
                string = ""
            elif self._publication_ordered:
                string = f"{string}~"
            else:
                string = f"\\times {string}"
            return string
        if self._code and len(self._op_strings)>0:
            raise RuntimeError("can only output code for terms that have been reduced to products of MO integrals and single-fragment densities")
        string = "".join(f"~{op_string}" for op_string in self._op_strings)
        scalar = scalar_sum_str(self._scalar_sum)
        if self._publication_ordered or self._code:
            string = scalar + string + f"{self._integrals}"
        else:
            string = f"{self._integrals}" + string + scalar
        if self._code:
            string += "\n        # + permutations"
        else:
            for sign,perm in self._perm_list:    # sign is stored as a character
                perms = " \\\\ ".join(f"{p}\\rightarrow{q}" for p,q in sorted(perm.items()))
                string += f"~{sign}~P_{{\\substack{{ {perms} }} }}"
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
        self._code = False
    def rho_notation(self):
        for term in self._terms:  term.rho_notation()
        return self
    def as_exp_val(self):
        for term in self._terms:  term.as_exp_val()
        return self
    def publication_ordered(self):
        for term in self._terms:  term.publication_ordered()
        return self
    def abbreviated(self):
        for term in self._terms:  term.abbreviated()
        return self
    def code(self):
        for term in self._terms:  term.code()
        self._code = not self._code
        return self
    def _frag_sorted(self):
        return term_list([frag_sorted(term) for term in self._terms])
    def _frag_factorized(self):
        return term_list([frag_factorized(term) for term in self._terms])
    def _simplified(self):
        same = []
        for term_1 in self._terms:
            same_1 = []    # everything that is the same as the current term_1, including itself
            for i,term_2 in enumerate(self._terms):
                if diagram_term.compare_signs(term_1, term_2)!=0:
                    same_1 += [i]
            same += [tuple(sorted(same_1))]    # eventually contains all groups of equivalent terms (ordered and hashable), but with duplicate groups
        same = set(same)    # remove duplicates
        new_terms = []
        for group in sorted(same):    # sorting just for aesthetics
            term = self._terms[group[0]]
            scalar_sum = []
            for i in group:
                other_term = self._terms[i]    # could be same as term
                parity = 0 if diagram_term.compare_signs(term, other_term)==+1 else 1
                for scalar in other_term.scalar_sum():
                    scalar.const_pow += parity    # copy, so ok to modify
                scalar_sum += [scalar]
            #
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
            #
            new_terms += [diagram_term(term, scalar_sum=combined_scalar_sum)]
        return term_list(new_terms)
    def _condense_perm(self):
        new_terms = []
        exclude = []
        for i,term_i in enumerate(self._terms):
            if i not in exclude:
                perm_list = []
                for j,term_j in list(enumerate(self._terms))[i+1:]:
                    if j not in exclude:
                        for perm in [{1:2, 2:1}]:    # NEED TO GENERALIZE
                            perm_sign = diagram_term.compare_frag_perm_signs(term_i, term_j, perm)
                            if perm_sign!=0:
                                scalar_sum_i = term_i.scalar_sum()
                                scalar_sum_j = term_j.scalar_sum()
                                if len(scalar_sum_i)>1 or len(scalar_sum_j)>1:
                                    raise RuntimeError("cannot handle this yet")
                                scalar_i = scalar_sum_i[0]
                                scalar_j = scalar_sum_j[0]
                                if perm_sign==+1:
                                    sign = "+" if scalar_i.const_pow==scalar_j.const_pow else "-"
                                else:
                                    sign = "-" if scalar_i.const_pow==scalar_j.const_pow else "+"
                                if as_tuple(scalar_i("frag_pows,scale"))!=as_tuple(scalar_j("frag_pows,scale")):
                                    raise RuntimeError("cannot handle this yet")
                                perm_list += [(sign,perm)]
                                exclude += [j]
                new_terms += [diagram_term(term_i, perm_list=perm_list)]
        return term_list(new_terms)
    def __str__(self):
        if self._code:
            connect = "\n+ "
        else:
            connect = "\\\\\n&+~"
        string = connect.join(str(term) for term in self._terms)
        if len(self._terms)>1:
            if not self._code:
                string = " &~~~~~" + string + "~ "
        return string



def combine(old, new):
    if old==[1]:  return       [new]
    else:         return old + [new]

def make_terms(frags, S_order, MOint=None):
    prototerms = [[1]]
    letter_idx = 4 if MOint else 0    # reserve p,q,r,s for MO integral, even with h
    for o in range(S_order):
        new_prototerms = []
        for p_frag in frags:
            for q_frag in frags:
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
    if MOint=="h":
        new_prototerms = []
        for p_frag in frags:
            for q_frag in frags:
                p = index(letters[0], p_frag)
                q = index(letters[1], q_frag)
                new_prototerms += [h_int(p, q)]
        prototerms = [combine(old,new) for old,new in itertools.product(prototerms, new_prototerms)]
    if MOint=="v":
        new_prototerms = []
        for p_frag in frags:
            for q_frag in frags:
                for s_frag in frags:
                    for r_frag in frags:
                        p = index(letters[0], p_frag)
                        q = index(letters[1], q_frag)
                        r = index(letters[2], r_frag)
                        s = index(letters[3], s_frag)
                        new_prototerms += [v_int(p, q, r, s)]
        prototerms = [combine(old,new) for old,new in itertools.product(prototerms, new_prototerms)]
    return term_list([diagram_term.from_integrals(copy.deepcopy(prototerm)) for prototerm in prototerms])    # weird things happen if integral instances shared accross terms
