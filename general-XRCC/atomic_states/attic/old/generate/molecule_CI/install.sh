#!/bin/bash
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
# set -x

echo '-----------------------------------------------------------------'

for file in 'generic_hamiltonian.c' 'LMO_frozen_buildTwoBodyVec.c' 'LMO_frozen_Hamiltonian.c' 'LMO_frozen_Hamiltonian_filter.c'
do
		main_name=$(echo $file | cut -d. -f1)
		echo 'Compile ' $main_name.o 
		echo 'Build   ' $main_name.so 
		gcc -std=c99 -c -fPIC -O2 -fopenmp -lm $main_name.c -o $main_name.o
		gcc -std=c99 -shared  -O2 -fopenmp -lm $main_name.o -o $main_name.so
		echo '-----------------------------------------------------------------'
done

echo 'Compile ' LMO_frozen_ct_oneAtomBasisToTwoAtomBasisVec.o
echo 'Build   ' LMO_frozen_ct_oneAtomBasisToTwoAtomBasisVec.so
gcc -std=c99 -c -fPIC -O2 -fopenmp -lm                 LMO_frozen_ct_oneAtomBasisToTwoAtomBasisVec.c -o LMO_frozen_ct_oneAtomBasisToTwoAtomBasisVec.o
gcc -std=c99 -shared  -O2 -fopenmp -lm -lblas -llapack LMO_frozen_ct_oneAtomBasisToTwoAtomBasisVec.o -o LMO_frozen_ct_oneAtomBasisToTwoAtomBasisVec.so
echo '-----------------------------------------------------------------'

rm -f *.o
exit 0
