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
from .SV_1mer_0 import *
from .SV_2mer_0 import *
from .SV_2mer_1 import *
from .SV_2mer_2 import *



catalog = {}

catalog[1] = {
    "v0000":         build_diagram(v0000,        Dchgs=(0,),    permutations=[(+1,(0,))]),
}

catalog[2] = {
    "v0101":         build_diagram(v0101,        Dchgs=(0,0),   permutations=[(+1,(0,1))]),
    "v0001":         build_diagram(v0001,        Dchgs=(-1,+1), permutations=[(+1,(0,1)),(-1,(1,0))]),
    "v0100":         build_diagram(v0100,        Dchgs=(+1,-1), permutations=[(+1,(0,1)),(-1,(1,0))]),
    "v0011":         build_diagram(v0011,        Dchgs=(-2,+2), permutations=[(+1,(0,1)),(+1,(1,0))]),
    "s01v0100":      build_diagram(s01v0100,     Dchgs=(0,0),   permutations=[(+1,(0,1)),(+1,(1,0))]),
    "s01v1101":      build_diagram(s01v1101,     Dchgs=(0,0),   permutations=[(+1,(0,1)),(+1,(1,0))]),
    "s01v0000":      build_diagram(s01v0000,     Dchgs=(-1,+1), permutations=[(+1,(0,1)),(-1,(1,0))]),
    "s01v0101":      build_diagram(s01v0101,     Dchgs=(-1,+1), permutations=[(+1,(0,1)),(-1,(1,0))]),
    "s01v1100":      build_diagram(s01v1100,     Dchgs=(+1,-1), permutations=[(+1,(0,1)),(-1,(1,0))]),
    "s01v1111":      build_diagram(s01v1111,     Dchgs=(-1,+1), permutations=[(+1,(0,1)),(-1,(1,0))]),
    "s01v0001":      build_diagram(s01v0001,     Dchgs=(-2,+2), permutations=[(+1,(0,1)),(+1,(1,0))]),
    "s01v0111":      build_diagram(s01v0111,     Dchgs=(-2,+2), permutations=[(+1,(0,1)),(+1,(1,0))]),
    "s01v0011":      build_diagram(s01v0011,     Dchgs=(-3,+3), permutations=[(+1,(0,1)),(-1,(1,0))]),
    "s01s01v1100":   build_diagram(s01s01v1100,  Dchgs=(0,0),   permutations=[(+1,(0,1)),(+1,(1,0))]),
    "s01s10v0000":   build_diagram(s01s10v0000,  Dchgs=(0,0),   permutations=[(+1,(0,1)),(+1,(1,0))]),
    "s01s10v0101":   build_diagram(s01s10v0101,  Dchgs=(0,0),   permutations=[(+1,(0,1))]),
    "s01s01v0100":   build_diagram(s01s01v0100,  Dchgs=(-1,+1), permutations=[(+1,(0,1)),(-1,(1,0))]),
    "s01s01v1101":   build_diagram(s01s01v1101,  Dchgs=(-1,+1), permutations=[(+1,(0,1)),(-1,(1,0))]),
    "s01s10v0001":   build_diagram(s01s10v0001,  Dchgs=(-1,+1), permutations=[(+1,(0,1)),(-1,(1,0))]),
    "s01s10v0100":   build_diagram(s01s10v0100,  Dchgs=(+1,-1), permutations=[(+1,(0,1)),(-1,(1,0))]),
    "s01s01v0000":   build_diagram(s01s01v0000,  Dchgs=(-2,+2), permutations=[(+1,(0,1)),(+1,(1,0))]),
    "s01s01v0101":   build_diagram(s01s01v0101,  Dchgs=(-2,+2), permutations=[(+1,(0,1)),(+1,(1,0))]),
    "s01s01v1111":   build_diagram(s01s01v1111,  Dchgs=(-2,+2), permutations=[(+1,(0,1)),(+1,(1,0))]),
    "s01s10v0011":   build_diagram(s01s10v0011,  Dchgs=(-2,+2), permutations=[(+1,(0,1)),(+1,(1,0))]),
    "s01s01v0001":   build_diagram(s01s01v0001,  Dchgs=(-3,+3), permutations=[(+1,(0,1)),(-1,(1,0))]),
    "s01s01v0111":   build_diagram(s01s01v0111,  Dchgs=(-3,+3), permutations=[(+1,(0,1)),(-1,(1,0))]),
    "s01s01v0011":   build_diagram(s01s01v0011,  Dchgs=(-4,+4), permutations=[(+1,(0,1)),(+1,(1,0))]),
}
