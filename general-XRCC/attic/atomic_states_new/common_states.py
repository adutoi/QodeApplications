import os
import sys
import pickle
import numpy
from qode.util import sort_eigen

# Usage:  python common_states.py <thresh> <geometry 1> [<geometry 2> [<geometry 3> [...]]]



def parse_val(v, thresh):
    if v<-thresh:  raise AssertionError
    if v< thresh:  return 0,0
    else:          return numpy.sqrt(v), 1/numpy.sqrt(v)

def rt_inv_rt(M):
    """ takes the +1/2 and -1/2 power of a symmetric matrix M, where eigenvectors of M with small eigenvalues are projected out """
    thresh = 1e-12	# hard coded threshold!!!
    vals, vecs = numpy.linalg.eigh(M)
    rtvals  = []
    irtvals = []
    for v in vals:
        rtv, irtv = parse_val(v, thresh)
        rtvals  += [rtv]
        irtvals += [irtv]
    rt     = vecs @ numpy.diag(rtvals)  @ vecs.T
    inv_rt = vecs @ numpy.diag(irtvals) @ vecs.T
    return rt, inv_rt



os.mkdir("common-{}".format(sys.argv[1]))

for n_elec in 1,2,3:	# For a given number of electrons on the monomer ...
    r = []
    u = []
    z = []
    for dist in sys.argv[2:]:	# Collect all the monomer states and weights from the different geometries
        tmp = pickle.load(open("{}/d.pickle".format(dist),"rb"))[n_elec]
        for t in tmp:  r += [t]							# the weights
        tmp = pickle.load(open("{}/u.pickle".format(dist),"rb"))[n_elec]
        for t in tmp:  u += [t]							# the states in the monomer-eigenstate basis
        tmp = numpy.load("{}/Z_{}e.npy".format(dist,n_elec)).T
        for t in tmp:  z += [t]							# the states in the full configuration basis
    r = numpy.array(r)
    u = numpy.array(u)
    z = numpy.array(z).T
    S = u.dot(u.T)		# overlap matrix of all states for all geometries (for given electron number)
    rtS, irtS = rt_inv_rt(S)
    vals, vecs = sort_eigen(numpy.linalg.eigh(rtS.dot(numpy.diag(r).dot(rtS))), "descending")	# diagnonalize the pseudo density operator in the symmetrically orthogonalized basis
    chi = irtS.dot(vecs)									# transform eigenvectors back to rep in original set of states
    n_states = 0
    for v in vals:
        if v>float(sys.argv[1]): n_states += 1		# determine how many states are above the threshold for retaining
    new_z = z.dot(chi[:,:n_states])		# build the states we want to keep in the configuration basis
    numpy.save("common-{}/Z_{}e.npy".format(sys.argv[1],n_elec), new_z)	# dump back to disk
