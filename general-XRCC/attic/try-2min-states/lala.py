#    (C) Copyright 2019 Anthony D. Dutoi
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
import numpy
from   qode.util import sort_eigen


def tens_pdt_2(list1, list2):	# the elements of each list are tuples, which will be concatenated
    items = []
    for item1 in list1:
        for item2 in list2:
            items += [item1 + item2]
    return items

def tens_pdt(lists):
    latter = [(i,) for i in lists[-1]]
    for ll in reversed(lists[:-1]):
        prior = [(i,) for i in ll]
        latter = tens_pdt_2(prior, latter)
    return latter

def build_H(H, frag_indices):
    monomer_Hamiltonians, dimer_Couplings, trimer_Couplings = H
    # frag_indices has one entry for each fragment, which is a tuple giving the absolute index of the fragment, the number of states defined for that fragment, and a list of the indices to use for it here
    frag_ids, spf, state_indices = list(zip(*frag_indices))	# spf stands for "states per fragment"
    basis = tens_pdt(state_indices)
    dim = len(basis)
    Hmat = numpy.zeros((dim,dim))
    for x,I in enumerate(basis):
        for y,J in enumerate(basis):
            transitions = [(m,i,j) for m,(i,j) in enumerate(zip(I,J)) if i!=j]
            if len(transitions)==0:
                for m,M in enumerate(frag_ids):
                    Hmat[x,y] += monomer_Hamiltonians[M][I[m],J[m]]	# Implicitly, we know that I[m]=J[m]
                    for n,N in list(enumerate(frag_ids))[m+1:]:
                        Hmat[x,y] += dimer_Couplings[M][N][I[m]*spf[n]+I[n], J[m]*spf[n]+J[n]]	# Implicitly, we know that I[m]=J[m] and I[n]=J[n]
            if len(transitions)==1:
                (m, mi, mj), = transitions
                M = frag_ids[m]
                Hmat[x,y] += monomer_Hamiltonians[M][mi,mj]
                for n,N in list(enumerate(frag_ids))[:m]:
                    Hmat[x,y] += dimer_Couplings[N][M][I[n]*spf[m]+mi, J[n]*spf[m]+mj]		# Implicitly, we know ...
                for n,N in list(enumerate(frag_ids))[m+1:]:
                    Hmat[x,y] += dimer_Couplings[M][N][mi*spf[n]+I[n], mj*spf[n]+J[n]]		# ... that I[n]=J[n]
            if len(transitions)==2:
                (m, mi, mj),(n, ni, nj) = transitions
                M = frag_ids[m]
                N = frag_ids[n]
                Hmat[x,y] += dimer_Couplings[M][N][mi*spf[n]+ni, mj*spf[n]+nj]
                for o,O in list(enumerate(frag_ids))[:m]:
                    Hmat[x,y] += trimer_Couplings[O][M][N][(I[o]*spf[m]+mi)*spf[n]+ni, (J[o]*spf[m]+mj)*spf[n]+nj]	# Implicitly, ...
                for o,O in list(enumerate(frag_ids))[m+1:n]:
                    Hmat[x,y] += trimer_Couplings[M][O][N][(mi*spf[o]+I[o])*spf[n]+ni, (mj*spf[o]+J[o])*spf[n]+nj]	# ... we know that ...
                for o,O in list(enumerate(frag_ids))[n+1:]:
                    Hmat[x,y] += trimer_Couplings[M][N][O][(mi*spf[n]+ni)*spf[o]+I[o], (mj*spf[n]+nj)*spf[o]+J[o]]	# ... I[o]=J[o]
            if len(transitions)==3:
                (m, mi, mj),(n, ni, nj),(o, oi, oj) = transitions
                M = frag_ids[m]
                N = frag_ids[n]
                O = frag_ids[o]
                Hmat[x,y] += trimer_Couplings[M][N][O][(mi*spf[n]+ni)*spf[o]+oi, (mj*spf[n]+nj)*spf[o]+oj]
    return Hmat





def fci(H, out, name):
    monomer_Hamiltonians, _, _ = H
    N_frag = len(monomer_Hamiltonians)
    frag_indices = []
    for m,h in enumerate(monomer_Hamiltonians):
        frag_indices += [(m, h.shape[0], list(range(h.shape[0])))]

    out.log("Building ...")
    Hmat = build_H(H, frag_indices)
    out.log("Diagonalizing ...")
    vals, vecs = sort_eigen(numpy.linalg.eig(Hmat))

    numpy.save("vals_{}.npy".format(name),vals)
    numpy.save("vecs_{}.npy".format(name),vecs)

    E = vals[0]
    out.log("FCI Energy  =", E)
    return E
