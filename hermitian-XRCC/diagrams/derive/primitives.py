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

from qode.util import struct
from permute import are_permutations



# Primitive class represent and index.  Instances are shared between integral and operator objects.
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



# Base class for all integrals just holds list of index objects (see above) and has some administrative options.
# Child class defines the variable letter associated with the integral (h, v, sigma, rho) and the latex (super/subscript) conventions
class integral_type(object):
    def __init__(self, arg1, arg2=None, _kind=None):    # _kind only for use by child classes to prevent copying across types with general code below
        if arg2 is None:   # copy
            other = arg1
            if other.kind!=_kind:
                raise ValueError("Attempt to construct integral object from incompatible type")
            self.c_indices = list(other.c_indices)
            self.a_indices = list(other.a_indices)
            self._fragment = None    # not necessarily defined, unless all indices on same fragment
            self._code = other._code
            self._abbreviated = other._abbreviated
        else:              # new from index lists
            c_indices, a_indices = arg1, arg2
            self.c_indices = list(arg1)
            self.a_indices = list(arg2)
            self._fragment = None    # not necessarily defined, unless all indices on same fragment
            self._code = False
            self._abbreviated = False
    def compare_sign_to(self, other):    # sign of permutation if integrals are the same type, and contain the same indices (else 0)
        result = 0
        if self.kind==other.kind and self._fragment==other._fragment:    # second condition only needed for fragment Kroneckers (have no orb indices)
            c_sign = are_permutations(self.c_indices, other.c_indices)
            a_sign = are_permutations(self.a_indices, other.a_indices)
            result = c_sign * a_sign
        return result
    def code(self):            # toggle code representation
        self._code = not self._code
    def abbreviated(self):     # toggle abbreviated notation for indices (do through integral, else unreliable because toggled once for each occurance) 
        self._abbreviated = not self._abbreviated
        for index in self.c_indices+self.a_indices:
            index.abbreviated = self._abbreviated
    def rho_notation(self):    # toggle overall notation style for rho integrals (global choice, but ignored for non-density integrals)
        try:
            self._rho_notation = not self._rho_notation
        except AttributeError:
            pass
    def frag_indices(self):
        return [p.fragment for p in self.c_indices + self.a_indices]
    def code_components(self):
            fragments = [str(p.fragment) for p in self.c_indices + self.a_indices]
            letters   = [p.letter        for p in self.c_indices + self.a_indices]
            return self.symbols(), fragments, letters
    def _code_labels(self):   # return index labels in code style
        return (
            "".join(str(p.fragment) for p in self.c_indices + self.a_indices),
            ",".join(p.letter       for p in self.c_indices + self.a_indices)
        )
    def _tex_labels(self):    # return index labels in tex style
        return (
            " ".join(str(p) for p in self.c_indices),
            " ".join(str(p) for p in self.a_indices)
        )

# 2-index h integral
class h_int(integral_type):
    kind = "h"
    def __init__(self, arg1, arg2=None):
        if arg2 is None:   # copy
            other = arg1
            integral_type.__init__(self, other, _kind=self.kind)
        else:   # new from two indices
            p, q = arg1, arg2
            integral_type.__init__(self, [p], [q])
    def abbrev_hack(self):
        frags, letters = self._code_labels()
        return f"h{frags}"
    def symbols(self):
        return "h", "H"
    def __str__(self):
        if self._code:
            frags, letters = self._code_labels()
            return f"X.h{frags}({letters})"
        else:
            c, a = self._tex_labels()
            return f"h^{{{c}}}_{{{a}}}"

# 2-index sigma integral (indices not allowed to be the same)
class s_int(integral_type):
    kind = "s"
    def __init__(self, arg1, arg2=None):
        if arg2 is None:   # copy
            other = arg1
            integral_type.__init__(self, other, _kind=self.kind)
        else:   # new from two indices
            p, q = arg1, arg2
            if (p.fragment is not None) and (p.fragment==q.fragment):
                raise RuntimeError("sigma integrals should not have equal indices")
            integral_type.__init__(self, [p], [q])
    def abbrev_hack(self):
        frags, letters = self._code_labels()
        return f"s{frags}"
    def symbols(self):
        return "s", "S"
    def __str__(self):
        if self._code:
            frags, letters = self._code_labels()
            return f"X.s{frags}({letters})"
        else:
            c, a = self._tex_labels()
            return f"\\sigma_{{{c} {a}}}"

# 4-index v integral
class v_int(integral_type):
    kind = "v"
    def __init__(self, arg1, arg2=None, arg3=None, arg4=None):
        if (arg2 is None) and (arg3 is None) and (arg4 is None):   # copy
            other = arg1
            integral_type.__init__(self, other, _kind=self.kind)
        else:   # new from four indices
            p, q, r, s, = arg1, arg2, arg3, arg4
            integral_type.__init__(self, [p,q], [r,s])
    def abbrev_hack(self):
        frags, letters = self._code_labels()
        return f"v{frags}"
    def symbols(self):
        return "v", "V"
    def __str__(self):
        if self._code:
            frags, letters = self._code_labels()
            return f"X.v{frags}({letters})"
        else:
            c, a = self._tex_labels()
            return f"v^{{{c}}}_{{{a}}}"

# rho integral with arbitrary indices
class r_int(integral_type):
    kind = "r"   # r for rho, though the notation is variable
    def __init__(self, op_list):    # construction directly from an associated string of operators
        c_indices = [op.idx for op in op_list if op.kind=="c"]
        a_indices = [op.idx for op in op_list if op.kind=="a"]
        integral_type.__init__(self, c_indices, a_indices)
        all_indices = c_indices + a_indices
        self._fragment = all_indices[0].fragment
        if any(p.fragment!=self._fragment for p in all_indices):
            raise ValueError("density tensor indices should all belong to the same fragment")
        self._rho_notation = False
    def substitute_frag(self, perm):    # substitute fragment indices in place according to entry in perm
        for p in self.c_indices:
            p.fragment = perm[p.fragment]    # perm needs to be a dict bc frags may not be contiguous nor start at zero
        for p in self.a_indices:
            p.fragment = perm[p.fragment]    # perm needs to be a dict bc frags may not be contiguous nor start at zero
        self._fragment = perm[self._fragment]
    def fragment(self):   # get the fragment index associated with the density
        return self._fragment
    def ct_character(self):
        return len(self.a_indices) - len(self.c_indices)
    def symbols(self):
        symbol = "c"*len(self.c_indices) + "a"*len(self.a_indices)
        return symbol, symbol
    def __str__(self):
        if self._code:
            ca = "c"*len(self.c_indices) + "a"*len(self.a_indices)
            _, letters = self._code_labels()
            return f"X.{ca}{self._fragment}(i{self._fragment},j{self._fragment},{letters})"
        else:
            if self._rho_notation:    # i and j indices suppressed
                c, a = self._tex_labels()
                return f"\\rho_{{{c}}}^{{{a}}}"
            else:
                c = " ".join(str(c_op(p)) for p in self.c_indices)
                a = " ".join(str(a_op(p)) for p in self.a_indices)
                return f"{{}}^{{i_{{ {self._fragment} }} }}\\langle {c} {a} \\rangle_{{j_{{ {self._fragment} }} }}"

# fragment-local state Kronecker delta
class delta(integral_type):
    kind = "d"   # k for delta
    def __init__(self, fragment):
        integral_type.__init__(self, [], [])
        self._fragment = fragment
        self._rho_notation = False
    def substitute_frag(self, perm):    # substitute fragment indices in place according to entry in perm
        self._fragment = perm[self._fragment]
    def fragment(self):   # get the fragment index associated with the density
        return self._fragment
    def __str__(self):
        if self._code:
            return f"delta(i{self._fragment},j{self._fragment})"
        else:
            if self._rho_notation:    # i and j indices suppressed
                return f"\\delta_{{ i_{{ {self._fragment} }} j_{{ {self._fragment} }}   }}"
            else:
                return f"{{}}^{{i_{{ {self._fragment} }} }}\\langle\\rangle_{{j_{{ {self._fragment} }} }}"



# Base class for creation and annihilation operators just holds the index object and has some administrative functions.
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

# creation
class c_op(field_op):
    _symbol = "\\hat{{c}}_{{{p}}}"
    kind = "c"
    __init__ = field_op.__init__

# annihilation
class a_op(field_op):
    _symbol = "\\hat{{a}}^{{{p}}}"
    kind = "a"
    __init__ = field_op.__init__



# A class that represents a scalar in a purpose-specific way.  As the name hints, internally it may be a sum of other scalars,
# and it can be initialized by a list of such terms, if one looks through the code to see how those are represented.
# In practice, one only needs the default initialization to unity (no argument) and the copy construction.
# Later note: self._scalars[i].scale should be an int or a fractions.Fraction
class scalar_sum(object):
    def __init__(self, arg):
        if arg==0:   # instantiate a single term representing zero (use special int value avoid mutable defaults in signature)
            arg = []
        if arg==1:   # instantiate a single term representing unity (use special int value avoid mutable defaults in signature)
            arg = [struct(scale=1, const_pow=0, frag_pows=tuple())]   # = <scale> * (-1)^(<const_pow> + "sum of numbers of electrons in the bra/ket for fragments <frag_pows>")
        try:                     # copy from other ...
            other = arg
            self._scalars = other._scalars
            self._abbreviated = other._abbreviated
            self._publication_ordered = other._publication_ordered
            self._code = other._code
        except AttributeError:   # or initialize to list of given terms (assumed to be structs of the form listed above)
            scalars = arg
            self._scalars = scalars
            self._abbreviated = False
            self._publication_ordered = False
            self._code = False
        try:
            self._scalars = [struct(scalar) for scalar in self._scalars]    # copy contents because lists and structs are mutable (but not contents of given structs)
            self._simplify()
        except:
            raise TypeError(f"attempt to initialize scalar_sum object from incompatible type: {type(arg)}")
    def _simplify(self):
        self._simplify_scalars()
        if len(self._scalars)>1:    # only test for efficiency reasons
            self._combine_scalars()
    def _simplify_scalars(self):   # in-place simplification/standardization of each term (does not change value represented)
        new_scalars = []
        for scalar in self._scalars:
            const_pow = scalar.const_pow % 2
            frag_pows = tuple(frag for frag in sorted(set(scalar.frag_pows)) if scalar.frag_pows.count(frag)%2)    # need to sort for later equality checks
            new_scalars += [struct(scale=scalar.scale, const_pow=const_pow, frag_pows=frag_pows)]
        self._scalars = new_scalars
    def _combine_scalars(self):
        same_pows = []    # eventually contains all groups having equivalent fragment powers (ordered and hashable), but with duplicate groups
        for scalar_1 in self._scalars:
            same_pows_1 = []    # everything that has the same pows as the current scalar_1, including itself
            for i,scalar_2 in enumerate(self._scalars):
                if scalar_1.frag_pows==scalar_2.frag_pows:
                    same_pows_1 += [i]
            same_pows += [tuple(sorted(same_pows_1))]
        new_scalars = []
        for pow_group in sorted(set(same_pows)):    # loop without duplicates; sort for benefit of later comparisons
            frag_pows = self._scalars[pow_group[0]].frag_pows
            raw_scalar = 0
            for i in pow_group:
                raw_scalar += self._scalars[i].scale * (-1)**self._scalars[i].const_pow
            new_scalars += [struct(const_pow=int(raw_scalar<0), frag_pows=frag_pows, scale=abs(raw_scalar))]
        self._scalars = new_scalars
    def compare_sign_to(self, other):    # if nonzero is returned, that is the factor by which other is multiplied to obtain self (else uncertain of scaling)
        self._simplify()     # in the strictest sense, compare_signs is not a pure function of other now ...
        other._simplify()    # ... but this only changes representation (and to the best choice)
        result = 0           # the result if the expressions do not even have the same number of terms
        if len(self._scalars)==len(other._scalars):
            for i,(scalarS,scalarO) in enumerate(zip(self._scalars, other._scalars)):
                pow_test    = (scalarO.frag_pows == scalarS.frag_pows)    # need to have term-for-term matching frag_pows
                int_test    =  scalarO.scale   %    scalarS.scale         # we can only be sure of equal scalings for each pair if the ratios of terms are integers
                ratio_test  =  scalarO.scale   //   scalarS.scale                  # compute the ratio of terms assuming ...
                ratio_test *=  (-1)**abs(scalarO.const_pow - scalarS.const_pow)    # ... that the above tests pass
                if i==0:                           # for the first term ...
                    result = ratio = ratio_test    # ... tentatively save the ratio as the result and for further checking
                if (not pow_test) or (int_test) or (ratio_test!=ratio):    # if these tests fail for any term (last is always true for i=0) ...
                    result = 0                                                # ... then the overall result is 0
        return result
    def increment(self, other):   # in-place addition of new term (resist urge to try to make this like a real scalar class with + and +=)
        self._scalars += [struct(scalar) for scalar in other._scalars]
        self._simplify()
    def perm_mult(self, permute):   # in-place multiplication by (-1)^<permute>
        for scalar in self._scalars:
            scalar.const_pow += permute
        self._simplify()
    def frag_phase_mult(self, frag_pows):   # in-place multiplication by (-1)^("sum of numbers of electrons in the bra/ket for fragments <frag_pows>")
        for scalar in self._scalars:
            scalar.frag_pows = scalar.frag_pows + tuple(frag_pows)    # cannot use += because tuples immutable
        self._simplify()
    def mult_by(self, x):
        for scalar in self._scalars:
            scalar.scale *= x
    def publication_ordered(self):   # toggle integral ordering (why does this matter here?)
        self._publication_ordered = not self._publication_ordered
    def abbreviated(self):           # toggle abbreviated notation for indices, etc
        self._abbreviated = not self._abbreviated
    def code(self):                  # toggle code representation
        self._code = not self._code
    def __str__(self):
        if self._abbreviated:
            return ""
        terms = []
        for scalar in self._scalars:
            term = "1"     # if no nontrivial factors
            factors = []   # start with no such factors
            if scalar.scale!=1:    # assuming simplified, then scale>0
                if scalar.scale%1:   # it is nontrivial fraction
                    factors += [f"({scalar.scale})"]
                else:                # it evaluates to an integer
                    factors += [f"{scalar.scale}"]
            if len(scalar.frag_pows)>0:
                if self._code:
                    ### n_i1 or n_i2 conventions
                    #powers = "+".join(f"X.n_i{frag}" for frag in scalar.frag_pows)
                    ### n_j1 or n_j2 conventions
                    powers = "+".join(f"X.n_j{frag}" for frag in scalar.frag_pows)
                else:
                    ### n_i1 or n_i2 conventions
                    #powers = "+".join(f"n_{{i_{frag}}}" for frag in scalar.frag_pows)
                    ### n_j1 or n_j2 conventions
                    powers = "+".join(f"n_{{j_{frag}}}" for frag in scalar.frag_pows)
                if scalar.const_pow:  powers += "+1"
                if self._code:  factors += [f"(-1)**({powers})"]
                else:           factors += [f"(-1)^{{{powers}}}"]
                if self._code:                   mult = " * "      # multiplication convention ...
                elif self._publication_ordered:  mult = ""         # ... assuming there are ...
                else:                            mult = "\\cdot"   # ... nontrivial factors
                term = mult.join(factors)    # some factors are available to build the term
            else:
                if factors:           term = f"{factors[0]}"   # there is one nontrivial factor
                if scalar.const_pow:  term = f"-{term}"
            terms += [term]
        string = "+".join(terms)
        if len(terms)==0:
            string = "0"    # hopefully this would be suppressed at a higher level, but at least it is correct here.
        if len(terms)>1:
            string = f"({string})"
        if not self._code:
            if string=="1":
                string = ""
            elif self._publication_ordered:
                string = f"{string}~"
            else:
                string = f"\\times {string}"
        return string
