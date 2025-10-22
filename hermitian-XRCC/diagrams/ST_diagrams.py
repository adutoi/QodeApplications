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
from .ST_1mer_0 import *
from .ST_2mer_0 import *
from .ST_2mer_1 import *
from .ST_2mer_2 import *
from .ST_2mer_3 import *
from .ST_2mer_4 import *



catalog = {}

catalog[1] = {
    "t00":           build_diagram(t00,          Dchgs=(0,),    permutations=[(+1,(0,))]),
}

catalog[2] = {
    "t01":           build_diagram(t01,          Dchgs=(-1,+1), permutations=[(+1,(0,1)),(-1,(1,0))]),
    "s01t10":        build_diagram(s01t10,       Dchgs=(0,0),   permutations=[(+1,(0,1)),(+1,(1,0))]),
    "s01t00":        build_diagram(s01t00,       Dchgs=(-1,+1), permutations=[(+1,(0,1)),(-1,(1,0))]),
    "s01t11":        build_diagram(s01t11,       Dchgs=(-1,+1), permutations=[(+1,(0,1)),(-1,(1,0))]),
    "s01t01":        build_diagram(s01t01,       Dchgs=(-2,+2), permutations=[(+1,(0,1)),(+1,(1,0))]),
    "s01s10t00":     build_diagram(s01s10t00,    Dchgs=(0,0),   permutations=[(+1,(0,1)),(+1,(1,0))]),
    "s01s01t10":     build_diagram(s01s01t10,    Dchgs=(-1,+1), permutations=[(+1,(0,1)),(-1,(1,0))]),
    "s01s10t01":     build_diagram(s01s10t01,    Dchgs=(-1,+1), permutations=[(+1,(0,1)),(-1,(1,0))]),
    "s01s01t00":     build_diagram(s01s01t00,    Dchgs=(-2,+2), permutations=[(+1,(0,1)),(+1,(1,0))]),
    "s01s01t11":     build_diagram(s01s01t11,    Dchgs=(-2,+2), permutations=[(+1,(0,1)),(+1,(1,0))]),
    "s01s01t01":     build_diagram(s01s01t01,    Dchgs=(-3,+3), permutations=[(+1,(0,1)),(-1,(1,0))]),
    "s01s01s10t10":  build_diagram(s01s01s10t10, Dchgs=(0,0),   permutations=[(+1,(0,1)),(+1,(1,0))]),
    "s01s01s10t00":  build_diagram(s01s01s10t00, Dchgs=(-1,+1), permutations=[(+1,(0,1)),(-1,(1,0))]),
    "s01s01s10t11":  build_diagram(s01s01s10t11, Dchgs=(-1,+1), permutations=[(+1,(0,1)),(-1,(1,0))]),
    "s01s01s01t10":  build_diagram(s01s01s01t10, Dchgs=(-2,+2), permutations=[(+1,(0,1)),(+1,(1,0))]),
    "s01s01s10t01":  build_diagram(s01s01s10t01, Dchgs=(-2,+2), permutations=[(+1,(0,1)),(+1,(1,0))]),
    "s01s01s01t00":  build_diagram(s01s01s01t00, Dchgs=(-3,+3), permutations=[(+1,(0,1)),(-1,(1,0))]),
    "s01s01s01t11":  build_diagram(s01s01s01t11, Dchgs=(-3,+3), permutations=[(+1,(0,1)),(-1,(1,0))]),
    "s01s01s01t01":  build_diagram(s01s01s01t01, Dchgs=(-4,+4), permutations=[(+1,(0,1)),(+1,(1,0))]),
    "s01s01s10s10t00":  build_diagram(s01s01s10s10t00, Dchgs=(0,0),   permutations=[(+1,(0,1)),(+1,(1,0))]),
    "s01s01s01s10t10":  build_diagram(s01s01s01s10t10, Dchgs=(-1,+1), permutations=[(+1,(0,1)),(-1,(1,0))]),
    "s01s01s10s10t01":  build_diagram(s01s01s10s10t01, Dchgs=(-1,+1), permutations=[(+1,(0,1)),(-1,(1,0))]),
    "s01s01s01s10t00":  build_diagram(s01s01s01s10t00, Dchgs=(-2,+2), permutations=[(+1,(0,1)),(+1,(1,0))]),
    "s01s01s01s10t11":  build_diagram(s01s01s01s10t11, Dchgs=(-2,+2), permutations=[(+1,(0,1)),(+1,(1,0))]),
    "s01s01s01s01t10":  build_diagram(s01s01s01s01t10, Dchgs=(-3,+3), permutations=[(+1,(0,1)),(-1,(1,0))]),
    "s01s01s01s10t01":  build_diagram(s01s01s01s10t01, Dchgs=(-3,+3), permutations=[(+1,(0,1)),(-1,(1,0))]),
    "s01s01s01s01t00":  build_diagram(s01s01s01s01t00, Dchgs=(-4,+4), permutations=[(+1,(0,1)),(+1,(1,0))]),
    "s01s01s01s01t11":  build_diagram(s01s01s01s01t11, Dchgs=(-4,+4), permutations=[(+1,(0,1)),(+1,(1,0))]),
    "s01s01s01s01t01":  build_diagram(s01s01s01s01t01, Dchgs=(-5,+5), permutations=[(+1,(0,1)),(-1,(1,0))]),
}
