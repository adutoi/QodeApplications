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

import os
from diagram_writer import s_diagram, h_diagram, v_diagram, frag_sorted, frag_factorized, simplified, condense_perm

section  = lambda label:     f"\\section{{ {label} }}\n\n"
equation = lambda contents:  f"\\begin{{align}}\n{contents}\n\\end{{align}}\n\n" 

def braket(opr, frags):
    bra = "\\langle" + " ".join(f"\\psi^{{i_{{{frag}}} }}" for frag in frags) + "|"
    ket = "|" + " ".join(f"\\psi_{{j_{{{frag}}} }}" for frag in frags) + "\\rangle"
    return bra + opr + ket



n_frags = 2

frags = list(range(1, n_frags+1))
derivation = """\
The Einstein summation convention applies for all repeated indices, regardless of whether they are sub/superscript.
A subscript on an index indicates a restriction of that index to the specified fragment.
"""

for S_order in [3]:

    derivation += equation(braket(f"\\hat{{S}}^{{[{S_order}]}}", frags) + "~=")

    terms = s_diagram(S_order, frags)
    derivation += equation(terms)

    sorted_terms = frag_sorted(terms)
    derivation += equation(sorted_terms)

    factored_terms = frag_factorized(sorted_terms)
    derivation += equation(factored_terms)

    factored_terms.rho_notation()
    derivation += equation(factored_terms)

    simplified_terms = simplified(factored_terms)
    derivation += equation(simplified_terms)

    condensed_terms = condense_perm(simplified_terms)
    derivation += equation(condensed_terms)



template = open("template.tex", "r").read()
target   = open("diagrams.tex", "w")
target.write(template.replace("%-%-%-%-% CONTENTS %-%-%-%-%", derivation))
target.close()
os.system("pdflatex diagrams; pdflatex diagrams")
os.system("rm -rf diagrams.aux diagrams.log diagrams.tex")
