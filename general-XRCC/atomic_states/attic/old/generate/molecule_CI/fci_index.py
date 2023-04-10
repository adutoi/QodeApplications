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
from math import factorial
import numpy as np



class fci_index:

	def __init__(self, num_elec, num_spin_orbs):
		self.combination_matrix = [ [-1 for j in range(num_spin_orbs)] for i in range(num_elec) ]
		for i in range(num_elec):
			for j in range(i, num_spin_orbs):
				self.combination_matrix[i][j] = int( ( factorial(j) // factorial(j-i) ) // factorial(i) )
		self.combination_matrix = tuple(self.combination_matrix)	# ???
		self.num_spin_orbs = num_spin_orbs

	def getComboMat(self):
		""" returns a matrix listing the number of configurations of n_elec (rows) in n_orbs (columns) ... only the upper triangle is defined """
		return np.array(self.combination_matrix, dtype=np.int64)

	def __call__(self, config):
		""" returns the index of a configuration given by the list of orbital indices in config """
		index_num = 0
		# First orbital:
		for n in range(1, config[0]+1):
			index_num += self.combination_matrix[len(config)-1][self.num_spin_orbs-n]
		# The rest:
		for i in range(len(config)-1):
			for n in range(config[i]+2, config[i+1]+1):
				index_num +=  self.combination_matrix[len(config)-i-2][self.num_spin_orbs-n] 
		return index_num
