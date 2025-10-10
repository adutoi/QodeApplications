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
from .build_diagram import build_diagram
from .SU_1mer_0 import *
from .SU_2mer_0 import *
from .SU_2mer_1 import *
from .SU_2mer_2 import *



catalog = {}

catalog[1] = {
    "u000":          build_diagram(u000,         Dchgs=(0,),    permutations=[(0,)]),
}

catalog[2] = {
    "u100":          build_diagram(u100,         Dchgs=(0,0),   permutations=[(0,1),(1,0)]),
    "u001":          build_diagram(u001,         Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "u101":          build_diagram(u101,         Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "s01u010":       build_diagram(s01u010,      Dchgs=(0,0),   permutations=[(0,1),(1,0)]),
    "s01u000":       build_diagram(s01u000,      Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "s01u011":       build_diagram(s01u011,      Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "s01u001":       build_diagram(s01u001,      Dchgs=(-2,+2), permutations=[(0,1),(1,0)]),
    "s01u110":       build_diagram(s01u110,      Dchgs=(0,0),   permutations=[(0,1),(1,0)]),
    "s01u100":       build_diagram(s01u100,      Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "s01u111":       build_diagram(s01u111,      Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "s01u101":       build_diagram(s01u101,      Dchgs=(-2,+2), permutations=[(0,1),(1,0)]),
    "s01s10u000":    build_diagram(s01s10u000,   Dchgs=(0,0),   permutations=[(0,1),(1,0)]),
    "s01s01u010":    build_diagram(s01s01u010,   Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "s01s10u001":    build_diagram(s01s10u001,   Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "s01s01u000":    build_diagram(s01s01u000,   Dchgs=(-2,+2), permutations=[(0,1),(1,0)]),
    "s01s01u011":    build_diagram(s01s01u011,   Dchgs=(-2,+2), permutations=[(0,1),(1,0)]),
    "s01s01u001":    build_diagram(s01s01u001,   Dchgs=(-3,+3), permutations=[(0,1),(1,0)]),
    "s01s10u100":    build_diagram(s01s10u100,   Dchgs=(0,0),   permutations=[(0,1),(1,0)]),
    "s01s01u110":    build_diagram(s01s01u110,   Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "s01s10u101":    build_diagram(s01s10u101,   Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "s01s01u100":    build_diagram(s01s01u100,   Dchgs=(-2,+2), permutations=[(0,1),(1,0)]),
    "s01s01u111":    build_diagram(s01s01u111,   Dchgs=(-2,+2), permutations=[(0,1),(1,0)]),
    "s01s01u101":    build_diagram(s01s01u101,   Dchgs=(-3,+3), permutations=[(0,1),(1,0)]),
}
