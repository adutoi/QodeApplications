#    (C) Copyright 2018 Anthony D. Dutoi
# 
#    This file is part of Qode.
# 
#    Qode is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
# 
#    Qode is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
# 
#    You should have received a copy of the GNU General Public License
#    along with Qode.  If not, see <http://www.gnu.org/licenses/>.
#
import sys

Be_file = \
"""\
Molecule( 'Be',
[(4, (0.0, 0.0, 0.0))],
units='Angstrom')
"""

Be2_template = \
"""\
Molecule( 'Be2_{}ang',
[(4,  ( 0.0, 0.0, 0.0)),
(4,  ( 0.0, 0.0, {}))],
units='Angstrom')
"""


if len(sys.argv)==1:
	f = open("Be.in", 'w')
	f.write(Be_file)
	f.close()
else:
	f = open("Be2_{}ang.in".format(sys.argv[1]), 'w')
	f.write(Be2_template.format(sys.argv[1],sys.argv[1].replace("_",".")))
	f.close()
