#    (C) Copyright 2023 Anthony D. Dutoi and Marco Bauer
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

from qode.util               import recursive_looper, struct
from qode.util.dynamic_array import dynamic_array
from qode.math.tensornet     import tl_tensor, scalar_value
from Sv_precontract import precontract


# helper function to do repetitive manipulations of data passed from above
#    densities and integrals are the full arrays for the supersystem
#    subsystem is always in acending order, naming the fragment indices of interest here
#    charges gives the bra and ket charges (as a 2-tuple) for each such respective fragment in the subsystem
#    permutation gives a potential reordering of the fragments in the subsystem


def loops(n_indices, idx_range):
    return [(m,range(idx_range)) for m in range(n_indices)]

def map_subsystem(array, subsystem):
    def _map_subsystem(*indices):
        absolute_indices = tuple(subsystem[index] for index in indices)
        if len(absolute_indices)==1:  absolute_indices = absolute_indices[0]
        return array[absolute_indices]
    return _map_subsystem

def Dchg_rule(charges):
    def _Dchg_rule(*indices):
        index = indices[0]
        chg_i, chg_j = charges[index]
        return chg_i - chg_j
    return _Dchg_rule

def density_rule(densities, label, charges):
    def _density_rule(*indices):
        index = indices[0]
        try:
            rho = densities[index][label][charges[index]]    # charges[index] is the bra and ket charge
        except KeyError:
            rho = None    # eventually return an object whose __getitem__ member reports exactly what is missing (in case access is attempted)
        return rho
    return _density_rule

def precon_rule(contract_cache, label, subsystem, charges):
    precon = contract_cache[label]
    n_densities = len(label.split("_")) - 1
    def _precon_rule(*indices):
        rho_charges = tuple(charges[m] for m in indices[:n_densities])
        indices = tuple(subsystem[m] for m in indices)
        if len(indices)==1:  indices = indices[0]
        contraction = precon[indices]
        try:
            for braket_charge in rho_charges:
                contraction = contraction[braket_charge]
        except KeyError:
            contraction = None    # eventually return an object whose __getitem__ member reports exactly what is missing (in case access is attempted)
        return contraction
    return _precon_rule

class _parameters(object):
    def __init__(self, supersys_info, subsystem, charges, request, permutation=(0,)):
        n_frag = len(subsystem)
        self.P = 0 if permutation==(0,1) else 1        # needs to be generalized for n>=3.
        # Some diagrams need to know the number of e- in the ket for the combined "latter" frags of the un(!)permuted subsystem
        n_i_label = ""    # explicitly label which are included in the "latter" frags (to store all possibilities)
        n_i = 0           # start at 0 before looping backwards over frags in outer loop below (order of that loop otherwise irrelevant)
        for m in reversed(range(n_frag)):     # m0 is fragment index in permuted subsystem, m0_ is its index in unpermuted subsystem
            n_i_label = str(m) + n_i_label            # incrementally build the label for electron count of (variable) "latter" frags
            chg_i, _  = charges[m]                    # bra and ket charge of m0-th frag (double duty for m0, looping backwards over all frags in physical order)
            n_i += supersys_info.densities[subsystem[m]]['n_elec'][chg_i]      # number of electrons in "latter" frags (so far) in physical (unpermuted) order
            self.__dict__["n_i"+n_i_label] = n_i%2    # include only the parity of the number of electrons in the latter frags of the un(!)permuted subsystem
        subsystem = [subsystem[m] for m in permutation]
        charges   = [  charges[m] for m in permutation]
        #
        densities = dynamic_array(map_subsystem(supersys_info.densities,   subsystem), [range(n_frag)])
        S         = dynamic_array(map_subsystem(supersys_info.integrals.S, subsystem), [range(n_frag)]*2)
        V         = dynamic_array(map_subsystem(supersys_info.integrals.V, subsystem), [range(n_frag)]*4)
        Dchg      = dynamic_array(Dchg_rule(charges),                                  [range(n_frag)])

        for label in request.precontract:
            frag_count = label.count("#")
            precontract = dynamic_array(precon_rule(supersys_info.contract_cache, label, subsystem, charges), [range(n_frag)]*frag_count)
            recursive_looper(loops(frag_count,n_frag), self.assign(label, precontract))

        recursive_looper(loops(2,n_frag), self.assign("s##",      S))
        recursive_looper(loops(4,n_frag), self.assign("v####",    V))
        recursive_looper(loops(1,n_frag), self.assign("Dchg#", Dchg))
        for label in request.rho:
            rho = dynamic_array(density_rule(densities,label[:-1],charges), [range(n_frag)])
            recursive_looper(loops(1,n_frag), self.assign(label, rho))
    def assign(self, label, array):
        label = label.replace("#","{}")
        def _assign(*indices):
            self.__dict__[label.format(*indices)] = array[indices]
        return _assign


##########
# Here are the implementations of the actual diagrams.
# They must take the arguments (supersys_info, subsystem, charges), but after that, it is up to you.
# It should return a list of kernels that takes state indices (for specified fragment charges) their relevant permutations.
# Don't forget to update the "catalog" dictionary at the end.
##########

p, q, r, s, t, u, v, w = "pqrstuvw"    # some contraction indices for easier reading

# monomer diagram

# pqsr,pqrs-> :  ccaa0  v0000
def v0000(supersys_info, subsystem, charges):
    request = struct(rho=["ccaa#"], precontract=["ccaa#pqsr_Vpqrs"])
    X = _parameters(supersys_info, subsystem, charges, request)
    prefactor = 1
    def diagram(i0,j0):
        return scalar_value( prefactor * X.ccaa0pqsr_Vpqrs[i0,j0] )
        #return scalar_value( prefactor * X.ccaa0[i0,j0](p,q,s,r) @ X.v0000(p,q,r,s) )
    if X.Dchg0==0:
        return [(diagram, (0,))]
    else:
        return [(None, None)]



# dimer diagrams

# pr,qs,pqrs-> :  ca0  ca1  v0101
def v0101(supersys_info, subsystem, charges):
    request = struct(rho=["ca#"], precontract=["ca#pr_Vp#r#"])
    X = _parameters(supersys_info, subsystem, charges, request, permutation=(0,1))
    prefactor = 4
    def diagram(i0,i1,j0,j1):
        return scalar_value( prefactor * X.ca1[i1,j1](q,s) @ X.ca0pr_Vp1r1[i0,j0](q,s) )
        #return scalar_value( prefactor * X.ca0[i0,j0](p,r) @ X.ca1[i1,j1](q,s) @ X.v0101(p,q,r,s) )
    if X.Dchg0==0 and X.Dchg1==0:
        return [(diagram, (0,1))]
    else:
        return [(None, None)]

# pqs,r,pqrs-> :  cca0  a1  v0010
def v0010(supersys_info, subsystem, charges):
    result01 = _v0010(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _v0010(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _v0010(supersys_info, subsystem, charges, permutation):
    request = struct(rho=["cca#", "a#"], precontract=["cca#pqs_Vpq#s"])
    X = _parameters(supersys_info, subsystem, charges, request, permutation)
    prefactor = 2 * (-1)**(X.n_i1 + X.P)
    def diagram(i0,i1,j0,j1):
        return scalar_value( prefactor * X.a1[i1,j1](r) @ X.cca0pqs_Vpq1s[i0,j0](r) )
        #return scalar_value( prefactor * X.cca0[i0,j0](p,q,s) @ X.a1[i1,j1](r) @ X.v0010(p,q,r,s) )
    if X.Dchg0==-1 and X.Dchg1==+1:
        return diagram, permutation
    else:
        return None, None

# p,qsr,pqrs-> :  c0  caa1  v0111
def v0111(supersys_info, subsystem, charges):
    result01 = _v0111(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _v0111(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _v0111(supersys_info, subsystem, charges, permutation):
    request = struct(rho=["c#", "caa#"], precontract=["caa#qsr_V#qrs"])
    X = _parameters(supersys_info, subsystem, charges, request, permutation)
    prefactor = 2 * (-1)**(X.n_i1 + X.P)
    def diagram(i0,i1,j0,j1):
        return scalar_value( prefactor * X.c0[i0,j0](p) @ X.caa1qsr_V0qrs[i1,j1](p) )
        #return scalar_value( prefactor * X.c0[i0,j0](p) @ X.caa1[i1,j1](q,s,r) @ X.v0111(p,q,r,s) )
    if X.Dchg0==-1 and X.Dchg1==+1:
        return diagram, permutation
    else:
        return None, None

# pq,sr,pqrs-> :  cc0  aa1  v0011
def v0011(supersys_info, subsystem, charges):
    result01 = _v0011(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _v0011(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _v0011(supersys_info, subsystem, charges, permutation):
    request = struct(rho=["cc#", "aa#"], precontract=["cc#pq_Vpq##"])
    X = _parameters(supersys_info, subsystem, charges, request, permutation)
    prefactor = 1
    def diagram(i0,i1,j0,j1):
        return scalar_value( prefactor * X.aa1[i1,j1](s,r) @ X.cc0pq_Vpq11[i0,j0](r,s) )
        #return scalar_value( prefactor * X.cc0[i0,j0](p,q) @ X.aa1[i1,j1](s,r) @ X.v0011(p,q,r,s) )
    if X.Dchg0==-2 and X.Dchg1==+2:
        return diagram, permutation
    else:
        return None, None

# qtsr,pu,tu,pqrs-> :  ccaa0  ca1  s01  v1000
def s01v1000(supersys_info, subsystem, charges):
    result01 = _s01v1000(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _s01v1000(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01v1000(supersys_info, subsystem, charges, permutation):
    request = struct(rho=["ca#", "ccaa#"], precontract=["ccaa#qXsr_V#qrs"])
    X = _parameters(supersys_info, subsystem, charges, request, permutation)
    prefactor = 2
    def diagram(i0,i1,j0,j1):
        return scalar_value( prefactor * X.ca1[i1,j1](p,u) @ X.s01(t,u) @ X.ccaa0qXsr_V1qrs[i0,j0](t,p) )
        #return scalar_value( prefactor * X.ccaa0[i0,j0](q,t,s,r) @ X.ca1[i1,j1](p,u) @ X.s01(t,u) @ X.v1000(p,q,r,s) )
    if X.Dchg0==0 and X.Dchg1==0:
        return diagram, permutation
    else:
        return None, None

# tr,pqus,tu,pqrs-> :  ca0  ccaa1  s01  v1101
def s01v1101(supersys_info, subsystem, charges):
    result01 = _s01v1101(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _s01v1101(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01v1101(supersys_info, subsystem, charges, permutation):
    request = struct(rho=["ca#", "ccaa#"], precontract=["ccaa#pqXs_Vpq#s"])
    X = _parameters(supersys_info, subsystem, charges, request, permutation)
    prefactor = 2
    def diagram(i0,i1,j0,j1):
        return scalar_value( prefactor * X.ca0[i0,j0](t,r) @ X.s01(t,u) @ X.ccaa1pqXs_Vpq0s[i1,j1](u,r) )
        #return scalar_value( prefactor * X.ca0[i0,j0](t,r) @ X.ccaa1[i1,j1](p,q,u,s) @ X.s01(t,u) @ X.v1101(p,q,r,s) )
    if X.Dchg0==0 and X.Dchg1==0:
        return diagram, permutation
    else:
        return None, None

# pqtsr,u,tu,pqrs-> :  cccaa0  a1  s01  v0000
def s01v0000(supersys_info, subsystem, charges):
    result01 = _s01v0000(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _s01v0000(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01v0000(supersys_info, subsystem, charges, permutation):
    request = struct(rho=["cccaa#", "a#"], precontract=["cccaa#pqXsr_Vpqrs"])
    X = _parameters(supersys_info, subsystem, charges, request, permutation)
    prefactor = (-1)**(X.n_i1 + X.P)
    def diagram(i0,i1,j0,j1):
        return scalar_value( prefactor * X.a1[i1,j1](u) @ X.s01(t,u) @ X.cccaa0pqXsr_Vpqrs[i0,j0](t) )
        #return scalar_value( prefactor * X.cccaa0[i0,j0](p,q,t,s,r) @ X.a1[i1,j1](u) @ X.s01(t,u) @ X.v0000(p,q,r,s) )
    if X.Dchg0==-1 and X.Dchg1==+1:
        return diagram, permutation
    else:
        return None, None

# t,pqusr,tu,pqrs-> :  c0  ccaaa1  s01  v1111
def s01v1111(supersys_info, subsystem, charges):
    result01 = _s01v1111(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _s01v1111(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01v1111(supersys_info, subsystem, charges, permutation):
    request = struct(rho=["c#", "ccaaa#"], precontract=["ccaaa#pqXsr_Vpqrs"])
    X = _parameters(supersys_info, subsystem, charges, request, permutation)
    prefactor = (-1)**(X.n_i1 + X.P)
    def diagram(i0,i1,j0,j1):
        return scalar_value( prefactor * X.c0[i0,j0](t) @ X.s01(t,u) @ X.ccaaa1pqXsr_Vpqrs[i1,j1](u) )
        #return scalar_value( prefactor * X.c0[i0,j0](t) @ X.ccaaa1[i1,j1](p,q,u,s,r) @ X.s01(t,u) @ X.v1111(p,q,r,s) )
    if X.Dchg0==-1 and X.Dchg1==+1:
        return diagram, permutation
    else:
        return None, None

# ptr,qus,tu,pqrs-> :  cca0  caa1  s01  v0101
def s01v0101(supersys_info, subsystem, charges):
    result01 = _s01v0101(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _s01v0101(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01v0101(supersys_info, subsystem, charges, permutation):
    request = struct(rho=["cca#", "caa#"], precontract=["cca#pXr_Vp#r#"])
    X = _parameters(supersys_info, subsystem, charges, request, permutation)
    prefactor = 4 * (-1)**(X.n_i1 + X.P)
    def diagram(i0,i1,j0,j1):
        return scalar_value( prefactor * X.caa1[i1,j1](q,u,s) @ X.s01(t,u) @ X.cca0pXr_Vp1r1[i0,j0](t,q,s) )
        #return scalar_value( prefactor * X.cca0[i0,j0](p,t,r) @ X.caa1[i1,j1](q,u,s) @ X.s01(t,u) @ X.v0101(p,q,r,s) )
    if X.Dchg0==-1 and X.Dchg1==+1:
        return diagram, permutation
    else:
        return None, None

# tsr,pqu,tu,pqrs-> :  caa0  cca1  s01  v1100
def s01v1100(supersys_info, subsystem, charges):
    result01 = _s01v1100(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _s01v1100(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01v1100(supersys_info, subsystem, charges, permutation):
    request = struct(rho=["cca#", "caa#"], precontract=["cca#pqX_Vpq##"])
    X = _parameters(supersys_info, subsystem, charges, request, permutation)
    prefactor = (-1)**(X.n_i1 + X.P)
    def diagram(i0,i1,j0,j1):
        return scalar_value( prefactor * X.caa0[i0,j0](t,s,r) @ X.s01(t,u) @ X.cca1pqX_Vpq00[i1,j1](u,r,s) )
        #return scalar_value( prefactor * X.caa0[i0,j0](t,s,r) @ X.cca1[i1,j1](p,q,u) @ X.s01(t,u) @ X.v1100(p,q,r,s) )
    if X.Dchg0==+1 and X.Dchg1==-1:
        return diagram, permutation
    else:
        return None, None

# pqts,ur,tu,pqrs-> :  ccca0  aa1  s01  v0010
def s01v0010(supersys_info, subsystem, charges):
    result01 = _s01v0010(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _s01v0010(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01v0010(supersys_info, subsystem, charges, permutation):
    request = struct(rho=["ccca#", "aa#"], precontract=["ccca#pqXs_Vpq#s"])
    X = _parameters(supersys_info, subsystem, charges, request, permutation)
    prefactor = -2
    def diagram(i0,i1,j0,j1):
        return scalar_value( prefactor * X.aa1[i1,j1](u,r) @ X.s01(t,u) @ X.ccca0pqXs_Vpq1s[i0,j0](t,r) )
        #return scalar_value( prefactor * X.ccca0[i0,j0](p,q,t,s) @ X.aa1[i1,j1](u,r) @ X.s01(t,u) @ X.v0010(p,q,r,s) )
    if X.Dchg0==-2 and X.Dchg1==+2:
        return diagram, permutation
    else:
        return None, None

# pt,qusr,tu,pqrs-> :  cc0  caaa1  s01  v0111
def s01v0111(supersys_info, subsystem, charges):
    result01 = _s01v0111(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _s01v0111(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01v0111(supersys_info, subsystem, charges, permutation):
    request = struct(rho=["cc#", "caaa#"], precontract=["caaa#qXsr_V#qrs"])
    X = _parameters(supersys_info, subsystem, charges, request, permutation)
    prefactor = -2
    def diagram(i0,i1,j0,j1):
        return scalar_value( prefactor * X.cc0[i0,j0](p,t) @ X.s01(t,u) @ X.caaa1qXsr_V0qrs[i1,j1](u,p) )
        #return scalar_value( prefactor * X.cc0[i0,j0](p,t) @ X.caaa1[i1,j1](q,u,s,r) @ X.s01(t,u) @ X.v0111(p,q,r,s) )
    if X.Dchg0==-2 and X.Dchg1==+2:
        return diagram, permutation
    else:
        return None, None

# pqt,usr,tu,pqrs-> :  ccc0  aaa1  s01  v0011
def s01v0011(supersys_info, subsystem, charges):
    result01 = _s01v0011(supersys_info, subsystem, charges, permutation=(0,1))
    result10 = _s01v0011(supersys_info, subsystem, charges, permutation=(1,0))
    return [result01, result10]
def _s01v0011(supersys_info, subsystem, charges, permutation):
    request = struct(rho=["ccc#", "aaa#"], precontract=[])
    X = _parameters(supersys_info, subsystem, charges, request, permutation)
    prefactor = (-1)**(X.n_i1 + X.P)
    def diagram(i0,i1,j0,j1):
        return scalar_value( prefactor * X.ccc0[i0,j0](p,q,t) @ X.aaa1[i1,j1](u,s,r) @ X.s01(t,u) @ X.v0011(p,q,r,s) )
    if X.Dchg0==-3 and X.Dchg1==+3:
        return diagram, permutation
    else:
        return None, None



##########
# A dictionary catalog.  the string association lets users specify active diagrams at the top level.
##########

catalog = {}

catalog[1] = {
    "v0000": v0000
}

catalog[2] = {
    "v0101":    v0101,
    "v0010":    v0010,
    "v0111":    v0111,
    "v0011":    v0011,
    "s01v1101": s01v1101,
    "s01v1000": s01v1000,
    "s01v0101": s01v0101,
    "s01v1100": s01v1100,
    "s01v1111": s01v1111,
    "s01v0000": s01v0000,
    "s01v0010": s01v0010,
    "s01v0111": s01v0111,
    "s01v0011": s01v0011
}
