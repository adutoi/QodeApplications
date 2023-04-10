#!/bin/csh
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

# Usage:  run.sh <distance in angstrom> <basis (eg, 6-31g, aug-cc-pvtz)>

set distance = $1
set basis    = $2

echo running Be
python3 input_gen.py
./scf.sh Be.in $basis

echo running Be2
set dd = `echo $distance | tr . _`          # change 5.0 to 5_0, etc
python3 input_gen.py $dd
./scf.sh Be2_{$dd}ang.in $basis 


