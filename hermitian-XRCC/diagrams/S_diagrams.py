#    (C) Copyright 2023, 2024, 2025 Anthony D. Dutoi and Marco Bauer
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
from .build_diagram  import build_diagram
from .S_0mer_0 import *
from .S_2mer_1 import *
from .S_2mer_2 import *
from .S_2mer_3 import *
from .S_2mer_4 import *



catalog = {}

catalog[0] = {
    "identity":      build_diagram(identity,     Dchgs=None,    permutations=None),
}

catalog[2] = {
    "s01":           build_diagram(s01,          Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "s01s10":        build_diagram(s01s10,       Dchgs=(0,0),   permutations=[(0,1)]),
    "s01s01":        build_diagram(s01s01,       Dchgs=(-2,+2), permutations=[(0,1),(1,0)]),
    "s01s01s10":     build_diagram(s01s01s10,    Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "s01s01s01":     build_diagram(s01s01s01,    Dchgs=(-3,+3), permutations=[(0,1),(1,0)]),
    "s01s01s10s10":  build_diagram(s01s01s10s10, Dchgs=(0,0),   permutations=[(0,1)]),
    "s01s01s01s10":  build_diagram(s01s01s01s10, Dchgs=(-2,+2), permutations=[(0,1),(1,0)]),
    "s01s01s01s01":  build_diagram(s01s01s01s01, Dchgs=(-4,+4), permutations=[(0,1),(1,0)]),
}
