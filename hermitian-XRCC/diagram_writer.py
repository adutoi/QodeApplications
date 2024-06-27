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
import itertools
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
    def __init__(self, n_elec, indices):
        if len(indices)!=2*n_elec:
            raise ValueError(f"only {len(indices)} indices given for {n_elec}-electron operator")
        self._n_elec  = n_elec
        self._indices = list(indices)
    def c_op_indices(self):
        return self._indices[:self._n_elec]
    def a_op_indices(self):
        return self._indices[self._n_elec:]
    def __str__(self):
        labels = {
            "bra":  ",".join(str(p) for p in self.c_op_indices()),
            "ket":  ",".join(str(p) for p in self.a_op_indices())
        }
        return self._symbol.format(**labels)

class s_int(integral_type):
    _symbol = "\\sigma_{{{bra} {ket}}}"
    def __init__(self, p, q):
        if p.fragment==q.fragment:  raise sigmaError()
        integral_type.__init__(self, 1, [p,q])

class h_int(integral_type):
    _symbol = "h^{{{bra}}}_{{{ket}}}"
    def __init__(self, p, q):
        integral_type.__init__(self, 1, [p,q])

class v_int(integral_type):
    _symbol = "v^{{{bra}}}_{{{ket}}}"
    def __init__(self, p, q, r, s):
        integral_type.__init__(self, 2, [p,q,r,s])

class integral_list(object):
    def __init__(self, integrals):
        try:
            new_integrals = integrals._integrals
        except AttributeError:
            self._integrals = list(integrals)
        else:
            self._integrals = list(new_integrals)
    def c_op_indices(self):
        return itertools.chain.from_iterable([reversed(integral.c_op_indices()) for integral in self._integrals])
    def a_op_indices(self):
        return itertools.chain.from_iterable([reversed(integral.a_op_indices()) for integral in self._integrals])
    def fragments(self):
        return {p.fragment for p in self.c_op_indices()} | {p.fragment for p in self.a_op_indices()}
    def __str__(self):
        return "".join(str(integral) for integral in self._integrals)



def frag_sorted(obj):                # this syntax because ...
    return obj._frag_sorted()        # ... not sorted in place

def frag_factorized(obj):            # this syntax because ...
    return obj._frag_factorized()    # ... not factorized in place



class field_op(object):
    def __init__(self, index):
        self.index = index   # should protect this from being changed
    def __str__(self):
        return self._symbol.format(p=self.index)

class c_op(field_op):
    _symbol = "\\hat{{c}}_{{{p}}}"
    __init__ = field_op.__init__

class a_op(field_op):
    _symbol = "\\hat{{a}}^{{{p}}}"
    __init__ = field_op.__init__

class operator_string(object):
    def __init__(self, ops, frags=None):
        try:
            new_frags = ops._frags
        except AttributeError:
            try:
                self._ops = dict(ops)
            except TypeError:
                self._ops = {None: list(ops)}
            self._frags = set(frags)
        else:
            self._ops   = dict(ops._ops)
            self._frags = set(new_frags)
            if frags is not None:  raise ValueError("not allowed to reset fragments in copy initialization of operator_string")
    def _flatten_ops(self):
        return itertools.chain.from_iterable([self._ops[key] for key in sorted(self._ops.keys())])
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
            op_strings += [operator_string(raw_string, [frag])]
            for phase in phases:
                phase.frags += [frag]
            phases += [struct(sign=(-1)**len(raw_string), frags=[])]
        return phases, op_strings
    def __str__(self):
        bra = " ".join(f"i_{{{n}}}" for n in self._frags)
        ket = " ".join(f"j_{{{n}}}" for n in self._frags)
        ops = " ".join(str(op) for op in self._flatten_ops())
        return f"{{}}^{{{bra}}}\\langle{ops}\\rangle_{{{ket}}}"



class term(object):
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
        return term(self._integrals, op_strings, phase)
    def _frag_factorized(self):
        phases = []
        op_strings = []
        for op_string in self._op_strings:
            phase, op_factorized = frag_factorized(op_string)
            phases += [phase]
            op_strings += op_factorized    # assuming for now that there is no need to sort this (ie, recursion depth is 1)
        phase_tot = struct(const=self._phase.const, frags=list(self._phase.frags))
        for phase in phases:
            for ph in phase:
                if ph.sign==-1:
                    phase_tot.frags += ph.frags
        return term(self._integrals, op_strings, phase_tot)
    def __str__(self):
        sign = "+" if (-1)**self._phase.const==1 else "-"
        string = f"{sign}~{self._integrals}"
        for op_string in self._op_strings:
            string += f"~{op_string}"
        frags = self._phase.frags
        pows = {frag: frags.count(frag)%2 for frag in sorted(set(frags))}
        if sum(count for frag,count in pows.items())>0:
            power = "".join(f"n_{{i_{frag}}}" for frag,count in pows.items() if count)
            string += f"\\times (-1)^{{{power}}}"
        return string
    @staticmethod
    def from_integrals(integrals):
        integrals = integral_list(integrals)
        c_op_indices = list(integrals.c_op_indices())    # these lists are ordered ...
        a_op_indices = list(integrals.a_op_indices())    # .... for this purpose
        all_indices = c_op_indices + a_op_indices
        for i,p in enumerate(all_indices):
            for q in all_indices[i+1:]:
                if p==q:
                    raise ValueError("illegal combination of integrals with duplicate indices")
        c_ops = [c_op(p) for p in reversed(c_op_indices)]
        a_ops = [a_op(p) for p in          a_op_indices ]
        return term(integrals, [operator_string(c_ops + a_ops, integrals.fragments())])

class term_list(object):
    def __init__(self, terms):
        self._terms = list(terms)
    def _frag_sorted(self):
        return term_list([frag_sorted(t) for t in self._terms])
    def _frag_factorized(self):
        return term_list([frag_factorized(t) for t in self._terms])
    def __str__(self):
        string = "\\\\\n".join(str(t) for t in self._terms)
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
    return term_list([term.from_integrals(prototerm) for prototerm in prototerms])

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
