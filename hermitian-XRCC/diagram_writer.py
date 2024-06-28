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
from itertools import chain
from qode.util import struct, as_tuple



class index(struct):
    def __init__(self, letter, fragment):
        struct.__init__(self, letter=letter, fragment=fragment)
    def __eq__(self, other):
        return as_tuple(self("letter fragment"))==as_tuple(other("letter fragment"))
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
    def __str__(self):
        return "".join(str(integral) for integral in self._integrals)



def frag_sorted(obj):                # this syntax because ...
    return obj._frag_sorted()        # ... not sorted in place

def frag_factorized(obj):            # this syntax because ...
    return obj._frag_factorized()    # ... not factorized in place

def frag_simplified(obj):            # this syntax because ...
    return obj._frag_simplified()    # ... not simplified in place



class field_op(object):
    def __init__(self, index):
        self.index = index   # should protect this from being changed
    def __str__(self):
        return self._symbol.format(p=self.index)

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
            self._dens_op = None
        else:
            self._ops   = dict(new_ops)
            self._frags = set(ops._frags)
            if ops._dens_op is None:
                self._dens_op = None
            else:
                self._dens_op = d_int(ops._dens_op)
            if frags is not None:
                raise ValueError("not allowed to reset fragments in copy initialization of operator_string")
    def as_dens_op(self):
        try:
            ops = self._ops[list(self._frags)[0]]
        except KeyError:
            raise RuntimeError("operator string cannot be put in single-fragment density form")
        if len(self._frags)!=1:
            raise RuntimeError("operator string cannot be put in single-fragment density form")
        c_indices, a_indices = [], []
        for op in ops:
            if op.kind=="c":  c_indices += [op.index]
            if op.kind=="a":  a_indices += [op.index]
        self._dens_op = d_int(c_indices, a_indices)
        return self    # could this convenience be mistaken for not having done an in-place modification?
    def _flatten_ops(self):
        return chain.from_iterable([self._ops[key] for key in sorted(self._ops.keys())])
    def fragments(self):
        return set(self._frags)
    def _frag_sorted(self):
        perm = 0
        sorted_ops = {frag:[] for frag in self._frags}
        for op in self._flatten_ops():
            frag = op.index.fragment
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
    def __str__(self):
        if self._dens_op:
            return str(self._dens_op)    # i and j indices suppressed
        else:
            bra = " ".join(f"i_{{{n}}}" for n in self._frags)
            ket = " ".join(f"j_{{{n}}}" for n in self._frags)
            ops = " ".join(str(op) for op in self._flatten_ops())
            string = f"{{}}^{{{bra}}}\\langle{ops}\\rangle_{{{ket}}}"
        return string



class diagram_term(object):
    def __init__(self, integrals, op_strings, _phase=None):
        frags = set()
        for op_string in op_strings:
            frags |= op_string.fragments()
        if integrals.fragments()!=frags:
            raise ValueError("integrals and operators do not refer to same fragment set")
        if _phase is None:
            _phase = struct(const=0, frags=[])    # -1 should be raised to the power of the constant plus the number of electrons in the ket for each fragment listed
        self._phase = _phase
        self._integrals = integral_list(integrals)
        self._op_strings = [operator_string(op_string) for op_string in op_strings]
    def _frag_sorted(self):
        permute = 0
        op_strings = []
        for op_string in self._op_strings:
            perm, op_sorted = frag_sorted(op_string)
            permute += perm
            op_strings += [op_sorted]
        phase = struct(const=self._phase.const + permute, frags=list(self._phase.frags))
        return diagram_term(self._integrals, op_strings, phase)
    def _frag_factorized(self):
        all_phases = []
        op_strings = []
        for op_string in self._op_strings:
            phases, op_factorized = frag_factorized(op_string)
            all_phases += [phases]
            op_strings += op_factorized    # assuming for now that there is no need to sort this (ie, recursion depth is 1)
        phase_tot = struct(const=self._phase.const, frags=list(self._phase.frags))
        for phases in all_phases:
            for phase in phases:
                if phase.n_ops%2:
                    phase_tot.frags += phase.frags
        return diagram_term(self._integrals, op_strings, phase_tot)
    def _frag_simplified(self):
        dens_ops = [operator_string(op_string).as_dens_op() for op_string in self._op_strings]
        return diagram_term(self._integrals, dens_ops, self._phase)
    def __str__(self):
        sign = "-" if self._phase.const%2 else "+"
        string = f"{sign}~{self._integrals}"
        for op_string in self._op_strings:
            string += f"~{op_string}"
        frag_pows = self._phase.frags
        pows = {frag: frag_pows.count(frag)%2 for frag in sorted(set(frag_pows))}
        if sum(count for frag,count in pows.items())>0:
            power = "".join(f"n_{{i_{frag}}}" for frag,count in pows.items() if count)
            string += f"\\times (-1)^{{{power}}}"
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
    def _frag_simplified(self):
        return term_list([frag_simplified(term) for term in self._terms])
    def __str__(self):
        string = "\\\\\n".join(str(term) for term in self._terms)
        if string[:2]=="+~":  string = "  " + string[2:]
        else:                 string = " -" + string[2:]
        return string



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

_letters = "pqrstuvwxyzabcdefghijklmno"

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
    #print()
    #print(s_diagram(S_order,   n_frags=2))
    #print()
    #print(h_diagram(S_order-1, n_frags=2))
    #print()
    #print(v_diagram(S_order-1, n_frags=2))
    #print()
    terms = s_diagram(S_order, n_frags=2)
    print(terms, "\n")
    terms = frag_sorted(terms)
    print(terms, "\n")
    terms = frag_factorized(terms)
    print(terms, "\n")
    terms = frag_simplified(terms)
    print(terms, "\n")
