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
import os
from diagram_writer import s_diagram, h_diagram, v_diagram, frag_sorted, frag_factorized, simplified, condense_perm

section     = lambda label:     f"\\section{{ {label} }}\n\n"
equation    = lambda contents:  f"\\begin{{align}}\n{contents}\n\\end{{align}}\n\n" 
eq_equation = lambda contents:  equation("=~" + str(contents))
code        = lambda contents:  f"\\begin{{lstlisting}}\n{contents}\n\\end{{lstlisting}}\n\n"

def take_matrix_element(frags):
    def braket(opr):
        bra = "\\langle" + " ".join(f"\\psi^{{i_{{{frag}}} }}" for frag in frags) + "|"
        ket = "|" + " ".join(f"\\psi_{{j_{{{frag}}} }}" for frag in frags) + "\\rangle"
        return bra + str(opr) + ket
    return braket


n_frags = 2

frags = list(range(1, n_frags+1))
braket = take_matrix_element(frags)
derivation = """\
The Einstein summation convention applies for all repeated indices, regardless of whether they are sub/superscript.
A subscript on an index itself indicates a restriction of that index to the specified fragment.
"""

for S_order in [3]:

    derivation += equation("3!\\cdot" + braket(f"\\hat{{S}}^{{[{S_order}]}}"))

    terms = s_diagram(S_order, [None])
    derivation += eq_equation(braket(terms))

    terms = s_diagram(S_order, frags)
    derivation += eq_equation(braket(terms))

    sorted_terms = frag_sorted(terms).as_exp_val()
    derivation += eq_equation(sorted_terms)

    factored_terms = frag_factorized(sorted_terms)
    derivation += eq_equation(factored_terms)

    derivation += eq_equation(factored_terms.rho_notation())

    simplified_terms = simplified(factored_terms)
    derivation += eq_equation(simplified_terms)

    condensed_terms = condense_perm(simplified_terms)
    derivation += eq_equation(condensed_terms.publication_ordered())

    derivation += eq_equation(condensed_terms.abbreviated())

    condensed_terms.abbreviated().code()
    derivation += code(condensed_terms)


if not "no-compile" in sys.argv:
    template = open("template.tex", "r").read()
    target   = open("diagrams.tex", "w")
    target.write(template.replace("%-%-%-%-% CONTENTS %-%-%-%-%", derivation))
    target.close()
    os.system("pdflatex diagrams; pdflatex diagrams")
    if not "leave-tex" in sys.argv:
        os.system("rm -rf diagrams.aux diagrams.log texput.log diagrams.tex")
