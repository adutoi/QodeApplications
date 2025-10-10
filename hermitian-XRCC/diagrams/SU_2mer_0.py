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
import numpy    # needs to go when u100 finally cleaned up
from qode.math.tensornet import raw
from .diagram_hack import state_indices, no_result

p, q, r, s, t, u, v, w = "pqrstuvw"    # some contraction indices for easier reading



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
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (-1)**(X.n_j0) * raw(
        #  X.c0(i0,j0,p)
        #@ X.a1(i1,j1,q)
        #@ X.u0_01(p,q)
          X.c0(i0,j0,p)
        @ X.a1q_U00q(i1,j1,p)
        )

def u101(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)    # = 0, 1, 2, 3
    return (-1)**(X.n_j0) * raw(
        #  X.c0(i0,j0,p)
        #@ X.a1(i1,j1,q)
        #@ X.u1_01(p,q)
          X.c0(i0,j0,p)
        @ X.a1q_U10q(i1,j1,p)
        )
