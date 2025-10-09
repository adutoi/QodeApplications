#    (C) Copyright 2023, 2025 Anthony D. Dutoi and Marco Bauer
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
from qode.math.tensornet import scalar_value, raw
from build_diagram       import build_diagram
from diagram_hack        import state_indices, no_result

p, q, r, s, t, u, v, w = "pqrstuvw"    # some contraction indices for easier reading



##########
# Here are the implementations of the actual diagrams.
# They must take the arguments (X, i0,i1,...,j0,j1,...) where and instance of frag_resolve (see build_diagram),
# which provides all of the input tensors and/or intermediate contractions.  These functions then return a scalar
# which is the evaluated diagram.  Don't forget to update the "catalog" dictionary at the end.
##########

# monomer diagram

def u000(X):
    i0, j0, = 0, 1
    return 1 * raw(
        #  X.ca0(i0,j0,p,q)
        #@ X.u0_00(p,q)
        X.ca0pq_U0pq
        )

# dimer diagrams

def u100(X, special_processing=None):
    (i0s,j0s),(i1s,j1s) = X.n_states[0], X.n_states[1]
    result = numpy.zeros((i0s, i1s, j0s, j1s))
    def get_standard(tensor):  # updates tensor in place
        i0, j0 = 0, 1
        res = 1 * raw(
            #  X.ca0(i0,j0,p,q)
            #@ X.u1_00(p,q)
            X.ca0pq_U1pq
            )
        for i1 in range(i1s):
            j1 = i1
            tensor[:,i1,:,j1] = res
    if special_processing is None:
        get_standard(result)
    elif special_processing <= 1:
        if no_result(X, contract="ket"):
            result = []
        else:
            if special_processing == 0 and i1s == j1s:  # state with state on frag 1 -> dirac_delta
                get_standard(result)
            elif special_processing == 1 and i1s == j1s:  # state with state on frag 1 -> dirac_delta with i0 and i1 permuted
                #for i1 in range(i1s):
                #    j1 = i1
                #    result[:,i1,:,j1] = 1 * raw( X.ca0pq_U1pq )
                get_standard(result)
                result = numpy.transpose(result, (1,0,2,3))  # 2 and 3 dont matter, because they are traced out at the end
            elif special_processing == 0 and i1s != j1s:
                # do this, since backend might not always be nunmpy...best use would be backend independent post processing though
                result[:,:,:,:] = 1 * raw( X.ca0pq_U1pq(0,2) @ X.KetCoeffs1(3,1) )
            elif special_processing == 1 and i1s != j1s:
                # do this, since backend might not always be nunmpy...best use would be backend independent post processing though
                result[:,:,:,:] = 1 * raw( X.ca0pq_U1pq(0,2) @ X.KetCoeffs1(3,1) )  # 2 and 3 dont matter, because they are traced out at the end
                result = numpy.transpose(result, (1,0,2,3))
            else:
                raise ValueError("One of the if statements needs to be True!")
            result = numpy.einsum("abii->ab", result)
    elif special_processing > 1:
        if no_result(X, contract="bra"):
            result = []
        else:
            if special_processing == 2 and i1s == j1s:  # state with state on frag 1 -> dirac_delta
                get_standard(result)
            elif special_processing == 3 and i1s == j1s:  # state with state on frag 1 -> dirac_delta with i0 and i1 permuted
                #for i1 in range(i1s):
                #    j1 = i1
                #    result[:,i1,:,j1] = 1 * raw( X.ca0pq_U1pq )
                get_standard(result)
                result = numpy.transpose(result, (0,1,3,2))  # 2 and 3 dont matter, because they are traced out at the end
            elif special_processing == 2 and i1s != j1s:
                # do this, since backend might not always be nunmpy...best use would be backend independent post processing though
                result[:,:,:,:] = 1 * raw( X.ca0pq_U1pq(0,2) @ X.KetCoeffs1(1,3) )
            elif special_processing == 3 and i1s != j1s:
                # do this, since backend might not always be nunmpy...best use would be backend independent post processing though
                result[:,:,:,:] = 1 * raw( X.ca0pq_U1pq(0,2) @ X.KetCoeffs1(1,3) )  # 2 and 3 dont matter, because they are traced out at the end
                result = numpy.transpose(result, (0,1,3,2))
            else:
                raise ValueError("One of the if statements needs to be True!")
            result = numpy.einsum("iiab->ab", result)
    else:
        raise ValueError(f"special processing {special_processing} can not be handled")
    return result
    #if i1==j1:
    #    return 1 * raw(
    #          X.ca0(i0,j0,p,q)
    #        @ X.u1_00(p,q)
    #        )
    #else:
    #    return 0

def u001(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (-1)**(X.n_j0 + X.P) * raw(
        #  X.c0(i0,j0,p)
        #@ X.a1(i1,j1,q)
        #@ X.u0_01(p,q)
          X.c0(i0,j0,p)
        @ X.a1q_U00q(i1,j1,p)
        )

def u101(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (-1)**(X.n_j0 + X.P) * raw(
        #  X.c0(i0,j0,p)
        #@ X.a1(i1,j1,q)
        #@ X.u1_01(p,q)
          X.c0(i0,j0,p)
        @ X.a1q_U10q(i1,j1,p)
        )

def s01u010(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return -1 * raw(
        #  X.ca0(i0,j0,t,q)
        #@ X.ca1(i1,j1,p,u)
        #@ X.s01(t,u)
        #@ X.u0_10(p,q)
          X.ca0tX_St1(i0,j0,q,u)
        @ X.ca1pX_U0p0(i1,j1,u,q)
        )

def s01u000(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (-1)**(X.n_j0 + X.P + 1) * raw(
        #  X.cca0(i0,j0,p,t,q)
        #@ X.a1(i1,j1,u)
        #@ X.s01(t,u)
        #@ X.u0_00(p,q)
          X.cca0pXq_U0pq(i0,j0,t)
        @ X.a1u_S0u(i1,j1,t)
        )

def s01u011(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (-1)**(X.n_j0 + X.P + 1) * raw(
        #  X.c0(i0,j0,t)
        #@ X.caa1(i1,j1,p,u,q)
        #@ X.s01(t,u)
        #@ X.u0_11(p,q)
          X.c0t_St1(i0,j0,u)
        @ X.caa1pXq_U0pq(i1,j1,u)
        )

def s01u001(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return 1 * raw(
        #  X.cc0(i0,j0,p,t)
        #@ X.aa1(i1,j1,u,q)
        #@ X.s01(t,u)
        #@ X.u0_01(p,q)
          X.cc0Xt_St1(i0,j0,p,u)
        @ X.aa1Xq_U00q(i1,j1,u,p)
        )

def s01u110(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return -1 * raw(
        #  X.ca0(i0,j0,t,q)
        #@ X.ca1(i1,j1,p,u)
        #@ X.s01(t,u)
        #@ X.u1_10(p,q)
          X.ca0tX_St1(i0,j0,q,u)
        @ X.ca1pX_U1p0(i1,j1,u,q)
        )

def s01u100(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (-1)**(X.n_j0 + X.P + 1) * raw(
        #  X.cca0(i0,j0,p,t,q)
        #@ X.a1(i1,j1,u)
        #@ X.s01(t,u)
        #@ X.u1_00(p,q)
          X.cca0pXq_U1pq(i0,j0,t)
        @ X.a1u_S0u(i1,j1,t)
        )

def s01u111(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (-1)**(X.n_j0 + X.P + 1) * raw(
        #  X.c0(i0,j0,t)
        #@ X.caa1(i1,j1,p,u,q)
        #@ X.s01(t,u)
        #@ X.u1_11(p,q)
          X.c0t_St1(i0,j0,u)
        @ X.caa1pXq_U1pq(i1,j1,u)
        )

def s01u101(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return 1 * raw(
        #  X.cc0(i0,j0,p,t)
        #@ X.aa1(i1,j1,u,q)
        #@ X.s01(t,u)
        #@ X.u1_01(p,q)
          X.cc0Xt_St1(i0,j0,p,u)
        @ X.aa1Xq_U10q(i1,j1,u,p)
        )

def s01s10u000(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return -1 * raw(
        #  X.ccaa0(i0,j0,p,t,w,q)
        #@ X.ca1(i1,j1,v,u)
        #@ X.s01(t,u)
        #@ X.s10(v,w)
        #@ X.u0_00(p,q)
          X.ccaa0pXXq_U0pq(i0,j0,t,w)
        @ X.ca1Xu_S0u(i1,j1,v,t)
        @ X.s10(v,w)
        )

def s01s01u010(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (1/2) * (-1)**(X.n_j0 + X.P + 1) * raw(
        #  X.cca0(i0,j0,t,v,q)
        #@ X.caa1(i1,j1,p,w,u)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.u0_10(p,q)
          X.cca0tXX_St1(i0,j0,v,q,u)
        @ X.caa1XwX_S0w(i1,j1,p,u,v)
        @ X.u0_10(p,q)
        )

def s01s10u001(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (-1)**(X.n_j0 + X.P + 1) * raw(
        #  X.cca0(i0,j0,p,t,w)
        #@ X.caa1(i1,j1,v,u,q)
        #@ X.s01(t,u)
        #@ X.s10(v,w)
        #@ X.u0_01(p,q)
          X.cca0XtX_St1(i0,j0,p,w,u)
        @ X.caa1vXX_Sv0(i1,j1,u,q,w)
        @ X.u0_01(p,q)
        )

def s01s01u000(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (1/2) * raw(
        #  X.ccca0(i0,j0,p,t,v,q)
        #@ X.aa1(i1,j1,w,u)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.u0_00(p,q)
          X.ccca0pXXq_U0pq(i0,j0,t,v)
        @ X.aa1Xu_S0u(i1,j1,w,t)
        @ X.s01(v,w)
        )

def s01s01u011(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (1/2) * raw(
        #  X.cc0(i0,j0,t,v)
        #@ X.caaa1(i1,j1,p,w,u,q)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.u0_11(p,q)
          X.cc0tX_St1(i0,j0,v,u)
        @ X.caaa1pXXq_U0pq(i1,j1,w,u)
        @ X.s01(v,w)
        )

def s01s01u001(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (1/2) * (-1)**(X.n_j0 + X.P) * raw(
          X.ccc0(i0,j0,p,t,v)
        @ X.aaa1(i1,j1,w,u,q)
        @ X.s01(t,u)
        @ X.s01(v,w)
        @ X.u0_01(p,q)
        )

def s01s10u100(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return -1 * raw(
        #  X.ccaa0(i0,j0,p,t,w,q)
        #@ X.ca1(i1,j1,v,u)
        #@ X.s01(t,u)
        #@ X.s10(v,w)
        #@ X.u1_00(p,q)
          X.ccaa0pXXq_U1pq(i0,j0,t,w)
        @ X.ca1Xu_S0u(i1,j1,v,t)
        @ X.s10(v,w)
        )

def s01s01u110(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (1/2) * (-1)**(X.n_j0 + X.P + 1) * raw(
        #  X.cca0(i0,j0,t,v,q)
        #@ X.caa1(i1,j1,p,w,u)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.u1_10(p,q)
          X.cca0tXX_St1(i0,j0,v,q,u)
        @ X.caa1XwX_S0w(i1,j1,p,u,v)
        @ X.u1_10(p,q)
        )

def s01s10u101(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (-1)**(X.n_j0 + X.P + 1) * raw(
        #  X.cca0(i0,j0,p,t,w)
        #@ X.caa1(i1,j1,v,u,q)
        #@ X.s01(t,u)
        #@ X.s10(v,w)
        #@ X.u1_01(p,q)
          X.cca0XtX_St1(i0,j0,p,w,u)
        @ X.caa1vXX_Sv0(i1,j1,u,q,w)
        @ X.u1_01(p,q)
        )

def s01s01u100(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (1/2) * raw(
        #  X.ccca0(i0,j0,p,t,v,q)
        #@ X.aa1(i1,j1,w,u)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.u1_00(p,q)
          X.ccca0pXXq_U1pq(i0,j0,t,v)
        @ X.aa1Xu_S0u(i1,j1,w,t)
        @ X.s01(v,w)
        )

def s01s01u111(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (1/2) * raw(
        #  X.cc0(i0,j0,t,v)
        #@ X.caaa1(i1,j1,p,w,u,q)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.u1_11(p,q)
          X.cc0tX_St1(i0,j0,v,u)
        @ X.caaa1pXXq_U1pq(i1,j1,w,u)
        @ X.s01(v,w)
        )

def s01s01u101(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (1/2) * (-1)**(X.n_j0 + X.P) * raw(
          X.ccc0(i0,j0,p,t,v)
        @ X.aaa1(i1,j1,w,u,q)
        @ X.s01(t,u)
        @ X.s01(v,w)
        @ X.u1_01(p,q)
        )



##########
# A dictionary catalog.  The string association lets users specify active diagrams at the top level
# for each kind of integral (symmetric, biorthogonal, ...).  The first argument to build_diagram() is
# one of the functions found above, which performs the relevant contraction for the given diagram for
# a fixed permutation (adjusted externally).
##########

catalog = {}

catalog[1] = {
    "u000":  build_diagram(u000)
}
catalog[2] = {
    "u100":          build_diagram(u100,         Dchgs=(0,0),   permutations=[(0,1),(1,0)]),
    "u001":          build_diagram(u001,         Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "u101":          build_diagram(u101,         Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "s01u010":       build_diagram(s01u010,      Dchgs=(0,0),   permutations=[(0,1),(1,0)]),
    "s01u000":       build_diagram(s01u000,      Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "s01u011":       build_diagram(s01u011,      Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "s01u001":       build_diagram(s01u001,      Dchgs=(-2,+2), permutations=[(0,1),(1,0)]),
    "s01u110":       build_diagram(s01u110,      Dchgs=(0,0),   permutations=[(0,1),(1,0)]),
    "s01u100":       build_diagram(s01u100,      Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "s01u111":       build_diagram(s01u111,      Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "s01u101":       build_diagram(s01u101,      Dchgs=(-2,+2), permutations=[(0,1),(1,0)]),
    "s01s10u000":    build_diagram(s01s10u000,   Dchgs=(0,0),   permutations=[(0,1),(1,0)]),
    "s01s01u010":    build_diagram(s01s01u010,   Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "s01s10u001":    build_diagram(s01s10u001,   Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "s01s01u000":    build_diagram(s01s01u000,   Dchgs=(-2,+2), permutations=[(0,1),(1,0)]),
    "s01s01u011":    build_diagram(s01s01u011,   Dchgs=(-2,+2), permutations=[(0,1),(1,0)]),
    "s01s01u001":    build_diagram(s01s01u001,   Dchgs=(-3,+3), permutations=[(0,1),(1,0)]),
    "s01s10u100":    build_diagram(s01s10u100,   Dchgs=(0,0),   permutations=[(0,1),(1,0)]),
    "s01s01u110":    build_diagram(s01s01u110,   Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "s01s10u101":    build_diagram(s01s10u101,   Dchgs=(-1,+1), permutations=[(0,1),(1,0)]),
    "s01s01u100":    build_diagram(s01s01u100,   Dchgs=(-2,+2), permutations=[(0,1),(1,0)]),
    "s01s01u111":    build_diagram(s01s01u111,   Dchgs=(-2,+2), permutations=[(0,1),(1,0)]),
    "s01s01u101":    build_diagram(s01s01u101,   Dchgs=(-3,+3), permutations=[(0,1),(1,0)]),
}
