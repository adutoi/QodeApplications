#    (C) Copyright 2023, 2024 Anthony D. Dutoi and Marco Bauer
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
import time
from timer import timer
from qode.math.tensornet import scalar_value
from frag_resolve import frag_resolve

p, q, r, s, t, u, v, w = "pqrstuvw"    # some contraction indices for easier reading



##########
# Here are the implementations of the actual diagrams.
# They must take the arguments (supersys_info, subsys_chgs), but after that, it is up to you.
# It should return a list of kernels that takes state indices (for specified fragment charges) their relevant permutations.
# Don't forget to update the "catalog" dictionary at the end.
##########



# monomer diagram

# pqsr,pqrs-> :  ccaa0  v0000
def v0000(X, i0,j0):
    prefactor = 1
    return scalar_value( prefactor * X.ccaa0pqsr_Vpqrs[i0,j0] )
    #return scalar_value( prefactor * X.ccaa0[i0,j0](p,q,s,r) @ X.v0000(p,q,r,s) )



# dimer diagrams

# pr,qs,pqrs-> :  ca0  ca1  v0101
def v0101(X, i0,i1,j0,j1):
    prefactor = 4
    return scalar_value( prefactor * X.ca1[i1,j1](q,s) @ X.ca0pr_Vp1r1[i0,j0](q,s) )
    #return scalar_value( prefactor * X.ca0[i0,j0](p,r) @ X.ca1[i1,j1](q,s) @ X.v0101(p,q,r,s) )

# pqs,r,pqrs-> :  cca0  a1  v0010
def v0010(X, i0,i1,j0,j1):
    prefactor = 2 * (-1)**(X.n_i1 + X.P)
    return scalar_value( prefactor * X.a1[i1,j1](r) @ X.cca0pqs_Vpq1s[i0,j0](r) )
    #return scalar_value( prefactor * X.cca0[i0,j0](p,q,s) @ X.a1[i1,j1](r) @ X.v0010(p,q,r,s) )

# p,qsr,pqrs-> :  c0  caa1  v0111
def v0111(X, i0,i1,j0,j1):
    prefactor = 2 * (-1)**(X.n_i1 + X.P)
    return scalar_value( prefactor * X.c0[i0,j0](p) @ X.caa1qsr_V0qrs[i1,j1](p) )
    #return scalar_value( prefactor * X.c0[i0,j0](p) @ X.caa1[i1,j1](q,s,r) @ X.v0111(p,q,r,s) )

# pq,sr,pqrs-> :  cc0  aa1  v0011
def v0011(X, i0,i1,j0,j1):
    prefactor = 1
    return scalar_value( prefactor * X.aa1[i1,j1](s,r) @ X.cc0pq_Vpq11[i0,j0](r,s) )
    #return scalar_value( prefactor * X.cc0[i0,j0](p,q) @ X.aa1[i1,j1](s,r) @ X.v0011(p,q,r,s) )

# qtsr,pu,tu,pqrs-> :  ccaa0  ca1  s01  v1000
def s01v1000(X, i0,i1,j0,j1):
    prefactor = 2
    return scalar_value( prefactor * X.ca1Xu_S0u[i1,j1](p,t) @ X.ccaa0qXsr_V1qrs[i0,j0](t,p) )
    #return scalar_value( prefactor * X.ca1[i1,j1](p,u) @ X.s01(t,u) @ X.ccaa0qXsr_V1qrs[i0,j0](t,p) )
    #return scalar_value( prefactor * X.ccaa0[i0,j0](q,t,s,r) @ X.ca1[i1,j1](p,u) @ X.s01(t,u) @ X.v1000(p,q,r,s) )

# tr,pqus,tu,pqrs-> :  ca0  ccaa1  s01  v1101
def s01v1101(X, i0,i1,j0,j1):
    prefactor = 2
    return scalar_value( prefactor * X.ca0tX_St1[i0,j0](r,u) @ X.ccaa1pqXs_Vpq0s[i1,j1](u,r) )
    #return scalar_value( prefactor * X.ca0[i0,j0](t,r) @ X.s01(t,u) @ X.ccaa1pqXs_Vpq0s[i1,j1](u,r) )
    #return scalar_value( prefactor * X.ca0[i0,j0](t,r) @ X.ccaa1[i1,j1](p,q,u,s) @ X.s01(t,u) @ X.v1101(p,q,r,s) )

# pqtsr,u,tu,pqrs-> :  cccaa0  a1  s01  v0000
def s01v0000(X, i0,i1,j0,j1):
    prefactor = (-1)**(X.n_i1 + X.P)
    return scalar_value( prefactor * X.a1u_S0u[i1,j1](t) @ X.cccaa0pqXsr_Vpqrs[i0,j0](t) )
    #return scalar_value( prefactor * X.a1[i1,j1](u) @ X.s01(t,u) @ X.cccaa0pqXsr_Vpqrs[i0,j0](t) )
    #return scalar_value( prefactor * X.cccaa0[i0,j0](p,q,t,s,r) @ X.a1[i1,j1](u) @ X.s01(t,u) @ X.v0000(p,q,r,s) )

# t,pqusr,tu,pqrs-> :  c0  ccaaa1  s01  v1111
def s01v1111(X, i0,i1,j0,j1):
    prefactor = (-1)**(X.n_i1 + X.P)
    return scalar_value( prefactor * X.c0t_St1[i0,j0](u) @ X.ccaaa1pqXsr_Vpqrs[i1,j1](u) )
    #return scalar_value( prefactor * X.c0[i0,j0](t) @ X.s01(t,u) @ X.ccaaa1pqXsr_Vpqrs[i1,j1](u) )
    #return scalar_value( prefactor * X.c0[i0,j0](t) @ X.ccaaa1[i1,j1](p,q,u,s,r) @ X.s01(t,u) @ X.v1111(p,q,r,s) )

# ptr,qus,tu,pqrs-> :  cca0  caa1  s01  v0101
def s01v0101(X, i0,i1,j0,j1):
    prefactor = 4 * (-1)**(X.n_i1 + X.P)
    return scalar_value( prefactor * X.caa1XuX_S0u[i1,j1](q,s,t) @ X.cca0pXr_Vp1r1[i0,j0](t,q,s) )
    #return scalar_value( prefactor * X.caa1[i1,j1](q,u,s) @ X.s01(t,u) @ X.cca0pXr_Vp1r1[i0,j0](t,q,s) )
    #return scalar_value( prefactor * X.cca0[i0,j0](p,t,r) @ X.caa1[i1,j1](q,u,s) @ X.s01(t,u) @ X.v0101(p,q,r,s) )

# tsr,pqu,tu,pqrs-> :  caa0  cca1  s01  v1100
def s01v1100(X, i0,i1,j0,j1):
    prefactor = (-1)**(X.n_i1 + X.P)
    return scalar_value( prefactor * X.caa0tXX_St1[i0,j0](s,r,u) @ X.cca1pqX_Vpq00[i1,j1](u,r,s) )
    #return scalar_value( prefactor * X.caa0[i0,j0](t,s,r) @ X.s01(t,u) @ X.cca1pqX_Vpq00[i1,j1](u,r,s) )
    #return scalar_value( prefactor * X.caa0[i0,j0](t,s,r) @ X.cca1[i1,j1](p,q,u) @ X.s01(t,u) @ X.v1100(p,q,r,s) )

# pqts,ur,tu,pqrs-> :  ccca0  aa1  s01  v0010
def s01v0010(X, i0,i1,j0,j1):
    prefactor = -2
    return scalar_value( prefactor * X.aa1uX_S0u[i1,j1](r,t) @ X.ccca0pqXs_Vpq1s[i0,j0](t,r) )
    #return scalar_value( prefactor * X.aa1[i1,j1](u,r) @ X.s01(t,u) @ X.ccca0pqXs_Vpq1s[i0,j0](t,r) )
    #return scalar_value( prefactor * X.ccca0[i0,j0](p,q,t,s) @ X.aa1[i1,j1](u,r) @ X.s01(t,u) @ X.v0010(p,q,r,s) )

# pt,qusr,tu,pqrs-> :  cc0  caaa1  s01  v0111
def s01v0111(X, i0,i1,j0,j1):
    prefactor = -2
    return scalar_value( prefactor * X.cc0Xt_St1[i0,j0](p,u) @ X.caaa1qXsr_V0qrs[i1,j1](u,p) )
    #return scalar_value( prefactor * X.cc0[i0,j0](p,t) @ X.s01(t,u) @ X.caaa1qXsr_V0qrs[i1,j1](u,p) )
    #return scalar_value( prefactor * X.cc0[i0,j0](p,t) @ X.caaa1[i1,j1](q,u,s,r) @ X.s01(t,u) @ X.v0111(p,q,r,s) )

# pqt,usr,tu,pqrs-> :  ccc0  aaa1  s01  v0011
def s01v0011(X, i0,i1,j0,j1):
    prefactor = (-1)**(X.n_i1 + X.P)
    return scalar_value( prefactor * X.ccc0[i0,j0](p,q,t) @ X.aaa1[i1,j1](u,s,r) @ X.s01(t,u) @ X.v0011(p,q,r,s) )



##########
# A dictionary catalog.  the string association lets users specify active diagrams at the top level.
##########


def build_diagrams(contraction, Dchgs=(0,), permutations=((0,),)):
    def get_diagrams(supersys_info, subsys_chgs):
        label = contraction.__name__
        if label not in supersys_info.timings:  supersys_info.timings[label] = timer()
        def get_diagram(X):
            def diagram(*ij_args):
                t0 = time.time()
                result = contraction(X, *ij_args)
                supersys_info.timings[label] += (time.time() - t0)
                return result
            return diagram
        diagrams = []
        for permutation in permutations:
            X = frag_resolve(supersys_info, subsys_chgs, permutation)
            yes = True
            for m,Dchg in enumerate(Dchgs):
                if X.Dchg[m]!=Dchg:  yes = False
            if yes:
                diagrams += [(get_diagram(X), permutation)]
            else:
                diagrams += [None]
        return diagrams
    return get_diagrams


catalog = {}

catalog[1] = {
    "v0000":    build_diagrams(v0000)
}

catalog[2] = {
    "v0101":    build_diagrams(v0101,    Dchgs=( 0, 0), permutations=[(0,1)]),
    "v0010":    build_diagrams(v0010,    Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "v0111":    build_diagrams(v0111,    Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "v0011":    build_diagrams(v0011,    Dchgs=(-2,+2), permutations=[(0,1),(1,0)]),
    "s01v1101": build_diagrams(s01v1101, Dchgs=( 0, 0), permutations=[(0,1),(1,0)]),
    "s01v1000": build_diagrams(s01v1000, Dchgs=( 0, 0), permutations=[(0,1),(1,0)]),
    "s01v0101": build_diagrams(s01v0101, Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "s01v1100": build_diagrams(s01v1100, Dchgs=(+1,-1), permutations=[(0,1),(1,0)]),
    "s01v1111": build_diagrams(s01v1111, Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "s01v0000": build_diagrams(s01v0000, Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "s01v0010": build_diagrams(s01v0010, Dchgs=(-2,+2), permutations=[(0,1),(1,0)]),
    "s01v0111": build_diagrams(s01v0111, Dchgs=(-2,+2), permutations=[(0,1),(1,0)]),
    "s01v0011": build_diagrams(s01v0011, Dchgs=(-3,+3), permutations=[(0,1),(1,0)])
}
