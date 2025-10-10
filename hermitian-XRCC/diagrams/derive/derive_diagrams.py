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

# Usage: python derive_diagrams.py [no-compile|leave-tex]    # compiles and deletes tex/log files by default
#
# This code and its subsidiary files are very special-purpose, designed only to do the necessary
# operations, in the usual order, to derive the working diagram equations and code for XR<N>[<o>].
# It has only been used for max <N>=2, but should be general for higher fragment orders.  That said, trying
# to use the functions outside the order given below to do the derivation differently could have
# unpredicatable results.  Nevertheless, the apparent modularity is not meant to deceive, but rather to
# make shorter the distance to increasing levels of generalization, if ever useful.  As a side-note, the
# true diagrammatic theory should be explored as a means of checking the results for higher <N> and <o>,
# although it is likely that this procedural algebraic approach will remain valuable for churning out
# working code.
# 
# There are four layers:
#     permute.py
# In here are truly general pure functions (albiet purpose-built) that take care of generating permutaions
# and keeping track of their parity.
#     primitives.py
# In here are definitions of single quantities like indices, integrals and field operators that bear/contain
# those indices, and scalars represented in a special-purpose way to keep track of number-dependent signs.
#     containers.py
# In here are containers for products of integrals, strings of field operators (and special-purpose matrix
# elements thereof), terms that contract integrals with strings of field operators (or their matrix elements),
# and sums of such terms. 
#     derive_diagrams.py
# This is the present file, which must only instantiate the integrals (and their indices) that are used
# to defined diagrams and manipulate sums of such terms using the pre-ordained sequence of operations/functions.
#
# The most important operational principle is that the indices of integrals that are to be contracted with
# indices of field operators are the same objects.  So even as the field operators are reordered, their
# matrix elements factored, and their notation changed to that of density tensors, the indices are passed
# along and the connection between them and the integrals is never lost.
# 
# Some decisions about variable conventions are left "outside" (contained in this file) and available for
# alteration without deeper changes.  Other decisions about things that could be variable are hard-coded
# at a layer below this file.
#     "outside" choices
# 1. The users is has conrol over the labels of the fragments (ie, [0,1,...], [1,2,...], ["A","B",...]).  Any
# object that resolves to a single character under str() may be used. 2. The user may choose which letters are
# used as indices, which are used for which Hamiltonian molecular integrals, which are used for sigma integrals,
# and in what order.  Again, any object that resolves to a single character may be used.  3. Finally, the sums
# of diagrams that occur in the derivations arise from different fragment-index patterns for the the contracted
# integrals; the user can control what order these different patters occur in.
#     "inside" choices
# 1. Within a given diagram/term the orderings in which the contracted hamiltonian integrals, densities, and
# state Kroneckers are displayed at each stage are (currently) hard coded.  There is one function that alters
# the ordering, but only to another hard-coded convention. 2. The so-described j_1 vs i_2 (etc) convention is
# hard-coded (at two locations), but alternatives live commented out in the code.
#
# With respect to the functions that perform the manipulations that build the derivation, they have two calling
# protocols.  Unbound functions of the form xxx(object) leave object unaltered and return a "copy" of the object
# with the desired manipulation having been performed.  Bound functions of the form object.xxx() change the object
# in place, but in some superficial way (ie, a flag is toggled that changes a printing convention).  Neither
# changes the abstract meaning represented by the object, in keeping with the interpretation of the manipulations as
# a derivation.  The unbound functions generally call a private function of the object to do the actual work of
# building the modified copy.
#
# To do:  for purposes of aesthetics, clean up the "printing" options (abbreviated, rho_notation, code, etc).
# It is so messy and hard to follow, culminating in .abbrev_hack() and .catalog_entry().  The mistake was making
# these things internal flags so that I could use str().  Just specify from outside at render time.  
# And now even .as_braket() makes a non-trivial permanent change.



# Called by make_section() below to generate the initial object to be manipulated in the derivation produced there.

import copy
import itertools
from primitives import index, s_int, h_int, v_int
from containers import diagram, diagram_sum

def make_terms(frags, MOint, S_order, letters):
    ### A one-off utility for combining lists interpreted as lists of factors
    def combine(old, new):
        if old==[1]:  return       [new]
        else:         return old + [new]
    ### Build prototerms order-by-order as outer product of previous fragment-label possibilities with those for new sigma.
    prototerms = [[1]]   # each sublist of prototerms will contain S_order number of sigma integrals, but with different fragment labels.
    letter_idx = 4 if MOint else 0    # each order gets two new letters, reserving p,q,r,s for molecular integral if present (even for h)
    for o in range(S_order):
        p, q = letters[letter_idx : letter_idx+2]
        letter_idx += 2
        new_prototerms = []   # list of new fragment-labelings for given index letters
        for p_frag in frags:
            for q_frag in frags:
                if p_frag!=q_frag or (p_frag==None and q_frag==None):
                    new_prototerms += [s_int(index(p, p_frag), index(q, q_frag))]
        prototerms = [combine(old,new) for old,new in itertools.product(prototerms, new_prototerms)]
    ### Extend logic above by additional iteration for molecular integral
    if MOint=="h":
        p, q = letters[0:2]
        new_prototerms = []
        for p_frag in frags:
            for q_frag in frags:
                new_prototerms += [h_int(index(p, p_frag), index(q, q_frag))]
        prototerms = [combine(old,new) for old,new in itertools.product(prototerms, new_prototerms)]
    if MOint=="v":
        p, q, r, s = letters[0:4]
        new_prototerms = []
        for p_frag in frags:
            for q_frag in frags:
                for r_frag in frags:
                    for s_frag in frags:
                        new_prototerms += [v_int(index(p, p_frag), index(q, q_frag), index(r, r_frag), index(s, s_frag))]
        prototerms = [combine(old,new) for old,new in itertools.product(prototerms, new_prototerms)]
    ###
    prototerms = [copy.deepcopy(prototerm) for prototerm in prototerms]    # the iterative product-building above shares integrals and indices across terms, so disconnect
    return diagram_sum([diagram.from_integrals(prototerm) for prototerm in prototerms])



# Generate a latex section for each type of diagram (MOint), with each order (up to max_order) as a subsection,
# for the given fragment indices (can be anything hashable for which str() is defined).

import math
import fractions
from containers import frag_sorted, frag_factorized, ct_ordered, simplified, condense_perm, mult_by

def make_section(MOint, max_order, frags, letters="pqrstuvwxyzabcdefghijklmno"):
    ### Some latex/typesetting utilities
    greek = {1:"Mono", 2:"Di", 3:"Tri", 4:"Tetra", 5:"Penta", 6:"Hexa", 7:"Hepta", 8:"Octa"}
    def make_braket(frags_):
        bra = "\\langle" + " ".join(f"\\psi^{{i_{{{frag}}} }}" for frag in frags_) + "|"
        ket = "|" + " ".join(f"\\psi_{{j_{{{frag}}} }}" for frag in frags_) + "\\rangle"
        def braket_(opr):
            return bra + str(opr) + ket
        return braket_
    section        = lambda label:     f"\\section{{ {label} }}\n\n"
    subsection     = lambda label:     f"\\subsection{{ {label} }}\n\n"
    subsubsection  = lambda label:     f"\\subsubsection{{ {label} }}\n\n"
    equation       = lambda contents:  f"\\begin{{align}}\n{contents}\n\\end{{align}}\n\n"   # used on its own for the starting quantity
    eq_equation    = lambda contents:  equation("=~" + str(contents))   # each subsequent step in its own block with leading equal sign
    code           = lambda contents:  f"\\begin{{lstlisting}}\n{contents}\n\\end{{lstlisting}}\n\n"
    ### Initialize the derivation
    new_section   = section(f"{MOint} diagrams")
    new_functions = f"\n\n\n###         {MOint} diagrams\n"
    MOintOpr = f"\\hat{{{MOint}}}"    # promotion of raw character flag to latex string
    min_order = 0
    if MOint=="S":    # because S already included for all diagram types, and 0th order for S alone is trivial delta
        MOint = None
        MOintOpr = ""
        min_order = 1
    ### Perform the derivation for each fragment and S order
    for N in range(1, 1+len(frags)):
        braket = make_braket(frags[:N])
        prefix = f"{N}-"
        if N in greek:  prefix = greek[N]
        new_section   += subsection(f"{prefix}mer Terms")
        new_functions += f"\n###     {prefix}mer Terms\n"
        max_order_ = 0 if (N==1) else max_order    # because off-diagonal blocks of sigma are zero
        for S_order in range(min_order, max_order_+1):
            new_section   += subsubsection(f"Order {S_order}")
            new_functions += f"\n### Order {S_order}\n\n"
            new_section += equation(f"{S_order}!\\cdot" + braket(f"\\hat{{S}}^{{[{S_order}]}}{MOintOpr}"))   # factorial on LHS to remove parentheses on RHS
            #
            terms = make_terms([None], MOint, S_order, letters)   # generic terms (integral+operator strings) with no fragment indices (therefore only one term)
            new_section += eq_equation(braket(terms))
            #
            terms = make_terms(frags[:N], MOint, S_order, letters)   # terms with all allowed distributions of fragment indices on the integrals/operators
            new_section += eq_equation(braket(terms))
            #
            sorted_terms = frag_sorted(terms).as_braket(frags[:N])
            new_section += eq_equation(sorted_terms)
            #
            factored_terms = frag_factorized(sorted_terms)
            new_section += eq_equation(factored_terms)
            new_section += eq_equation(factored_terms.rho_notation())
            #
            ordered_terms = ct_ordered(factored_terms)
            new_section += eq_equation(ordered_terms)
            #
            simplified_terms = simplified(ordered_terms)
            new_section += eq_equation(simplified_terms)
            #
            condensed_terms = condense_perm(simplified_terms)
            new_section += eq_equation(condensed_terms.publication_ordered())
            new_section += eq_equation(f"{S_order}!\\Big[" + str(condensed_terms.abbreviated()) + "\\Big]")
            #
            code_terms = mult_by(condensed_terms.abbreviated(), fractions.Fraction(1, math.factorial(S_order)))
            new_funtion = str(code_terms.code())
            new_section += code(new_funtion)
            new_functions += new_funtion + "\n"
    ###
    return new_section, new_functions



# SCRIPT MAIN
# template.tex is read from disk to provide header/footer admin, with a special contents flag at the
# location to inject the actual derivation, which is built below.

import sys
import os

# expose this conspicuosly because we change convention between prose and code
frags = (0,1)    # for code
#frags = (1,2)    # for publications

library = ""
derivation = """\
The Einstein summation convention applies for all repeated indices, regardless of whether they are sub/superscript.
A subscript on an index itself indicates a restriction of that index to the specified fragment.
"""
section, functions = make_section("S", 4, frags)
derivation += section
library += functions 
section, functions = make_section("h", 2, frags)
derivation += section
library += functions 
section, functions = make_section("v", 2, frags)
derivation += section
library += functions 

if not "no-compile" in sys.argv:
    template = open("template.tex", "r").read()
    texout   = open("diagrams.tex", "w")
    texout.write(template.replace("%-%-%-%-% CONTENTS %-%-%-%-%", derivation))
    texout.close()
    os.system("pdflatex diagrams; pdflatex diagrams")
    pyout = open("diagrams.py", "w")
    pyout.write(library)
    pyout.close()
    if not "leave-tex" in sys.argv:
        os.system("rm -rf diagrams.aux diagrams.log texput.log diagrams.tex")
