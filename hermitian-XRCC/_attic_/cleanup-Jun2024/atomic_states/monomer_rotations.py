#    (C) Copyright 2018 Anthony D. Dutoi and Yuhong Liu
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
import numpy
import qode

def subtract(int2):  return int2[1]-int2[0]



def monomer_rotations(vec, compression, frag_idx):

	# Parse dimensions
	n_states  = { 2:subtract(frag_idx[2]),  1:subtract(frag_idx[1]),  3:subtract(frag_idx[3]) }
	frag_dim  = n_states[2] + n_states[1] + n_states[3]
	super_dim = frag_dim**2

	# Set up vector spaces for monomer states
	FragA = qode.math.linear_inner_product_space(qode.math.numpy_space.real_traits(frag_dim))
	FragB = qode.math.linear_inner_product_space(qode.math.numpy_space.real_traits(frag_dim))

	# For each state of one monomer, compute the state of the other monomer, with which that state of the first is entangled
	Avecs = [ FragB.member(numpy.array(vec[i:i+frag_dim:1] )) for i in range(0,super_dim,frag_dim) ]
	Bvecs = [ FragA.member(numpy.array(vec[i:super_dim:frag_dim])) for i in range(0,frag_dim,1)    ]

	# The reduced density matrix for each monomer is then a matrix of overlaps of the forgoing states
	rhoA = numpy.array([[(a1|a2) for a2 in Avecs] for a1 in Avecs])
	rhoB = numpy.array([[(b1|b2) for b2 in Bvecs] for b1 in Bvecs])

	# Really, what I should do is zero out coherences that mix symmetries on a monomer to avoid allowing
	# asymmetrically biased states, but for now, just average the right and left atom
	rho  = (rhoA + rhoB) / 2

	# Make sure rho really is block diagonal by particle number
	part_num = [2]*n_states[2] + [1]*n_states[1] + [3]*n_states[3]
	for i in range(frag_dim):
		for j in range(frag_dim):
			if part_num[i]!=part_num[j]:
				if abs(rho[i,j])>1e-15:  raise Exception("something seriously wrong? {}".format(rho[i,j]))

	# Diagonalize each block
	rvals, rvecs = [], {}
	for n in (2,1,3):
		beg, end = frag_idx[n]
		if beg!=end:  vals, vecs = numpy.linalg.eigh(rho[beg:end,beg:end])
		else:         vals, vecs = numpy.array([]), numpy.array([[]])
		rvecs[n] = vecs.T.tolist()				# Keep eigenvectors sorted by particle number
		rvals += [(r,n,i) for i,r in enumerate(vals)]		# Collect all the eigenvalues together, but leave a map back to the associated vectors

	# Sort all eigenvalues by size (and check none are negative, save small noise)
	rvals = list(reversed(sorted( rvals, key=lambda p: p[0] )))
	for r,_1,_2 in rvals:
		if r<-1e-15:  raise Exception("something horribly wrong?: {}".format(r))

	# How many states to keep? ... also a tag for the output file to identify how it was made
	states_per_frag = 0
	if   compression[0]=="nstates":
		tag = "nstates="+compression[1]
		states_per_frag = int(compression[1])
	elif compression[0]=="thresh":
		tag = "thresh="+compression[1]
		population_thresh = float(compression[1])
		for r,_1,_2 in rvals:
			if r>population_thresh:  states_per_frag += 1
	else:  raise Exception("unknown compression algorithm: {}".format(compression[0]))

	# The mononmer state transforms, blocked by particle number
	u = { 2:[], 1:[], 3:[] }
	d = { 2:[], 1:[], 3:[] }
	for r,n,i in rvals[:states_per_frag]:
		u[n] += [rvecs[n][i]]
		d[n] += [r]

	return u, d, tag
