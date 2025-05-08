#    (C) Copyright 2024 Anthony D. Dutoi
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

# This is effectively defunct as it does not finish in any reasonable amount of time.
# If the idea is every resurrected, look in .../Qode/Calculations/2024-Jun-422172c150ab-multi_term

import numpy
from qode.math.tensornet import raw, evaluate
from qode.math import space, numpy_space, vector_set, gram_schmidt

def _SVD(M, big, print_info):
    A = M
    T_left, T_right = None, None
    n_left, n_right = M.shape
    if n_left>n_right and n_left>big:
        S = space.linear_inner_product_space(numpy_space.real_traits(n_left))
        T_left = numpy.array(M.T)                                 # copy M -> T
        vecs = vector_set(S, [S.member(T_left[i,:]) for i in range(n_right)])    # work with rows of T in place
        gram_schmidt(vecs, normalize=True, n_times=n_right)
        if print_info:
            print("ON deviation = ", vecs.orthonormality())
        M = T_left @ M
    if n_right>n_left and n_right>big:
        S = space.linear_inner_product_space(numpy_space.real_traits(n_right))
        T_right = numpy.array(M.T)                                # copy M -> T
        vecs = vector_set(S, [S.member(T_right[:,i]) for i in range(n_left)])    # work with columns of T in place
        gram_schmidt(vecs, normalize=True, n_times=n_left)
        if print_info:
            print("ON deviation = ", vecs.orthonormality())
        M = M @ T_right
    U, s, Vh = numpy.linalg.svd(M)

    if print_info:
        Z = U[:,:len(s)] @ numpy.diag(s) @ Vh[:len(s),:]
        print("(0) Error =", numpy.linalg.norm(M), numpy.linalg.norm(M-Z))

    if T_left is not None:
        U  = T_left.T @ U
    if T_right is not None:
        Vh = Vh @ T_right.T

    if print_info:
        Z = U[:,:len(s)] @ numpy.diag(s) @ Vh[:len(s),:]
        print("(1) Error =", numpy.linalg.norm(A), numpy.linalg.norm(A-Z))

    return U, s, Vh

def svd(tensor, indices_A, indices_B=None, normalized=False, big=10**5, print_info=False, tens_wrap=None):
    nparray_Nd = raw(tensor)
    if indices_B is None:  indices_B = []
    all_free_indices = list(indices_A) + list(indices_B)
    if list(sorted(all_free_indices))!=list(range(len(nparray_Nd.shape))):
        raise RuntimeError("dimension mismatch")
    if len(indices_A)==0 or len(indices_B)==0:
        return tens_wrap(nparray_Nd)
    #
    nparray_Nd = nparray_Nd.transpose(all_free_indices)
    shape_A = nparray_Nd.shape[:len(indices_A)]
    shape_B = nparray_Nd.shape[len(indices_A):]
    M = nparray_Nd.reshape((numpy.prod(shape_A), numpy.prod(shape_B)))
    #
    U, s, Vh = _SVD(M, big, print_info)
    d = len(s)
    #
    if normalized:
        Shalf = 1
    else:
        Shalf = numpy.sqrt(s)
    A = (   U[:,:d] * Shalf).T
    B = (Vh.T[:,:d] * Shalf).T
    #
    A = tens_wrap(A.reshape([d] + list(shape_A)))
    B = tens_wrap(B.reshape([d] + list(shape_B)))
    #
    return A, B, s

def _single_term_recur(tensor, n_axes, tens_wrap):
    A, B, w = svd(tensor, [0], list(range(1,n_axes)), tens_wrap=tens_wrap)
    n_axes -= 1
    if n_axes>1:
        indices = [0] + [slice(None)]*n_axes
        return A[0,:], *_single_term_recur(B[tuple(indices)], n_axes, tens_wrap)
    else:
        return A[0,:], B[0,:]

def single_term(tensor, num_c, num_a, tens_wrap):
    permute = {
        0:  None,
        1:  {(0,): +1},
        2:  {(0,1): +1, (1,0): -1},
        3:  {(0,1,2): +1, (0,2,1): -1,  (1,0,2): -1, (1,2,0): +1, (2,1,0): -1, (2,0,1): +1},
    }
    primitives = list(enumerate(_single_term_recur(tensor, num_c+num_a, tens_wrap)))
    i,P = primitives[0]
    value0 = P(i)
    for i,P in primitives[1:]:
        value0 @= P(i)
    c_permutations = permute[num_c]
    a_indices = list(range(num_c, num_c+num_a))
    a_permutations = {tuple([a_indices[i] for i in I]): p for I,p in permute[num_a].items()}
    all_permutations = {}
    for c_perm,pc in c_permutations.items():
        for a_perm,ac in a_permutations.items():
            all_permutations[(*c_perm,*a_perm)] = pc*ac
    value = value0
    for perm,p in list(all_permutations.items())[1:]:
        value += p * value0(*perm)
    return value

def multi_term(tensor, num_c, num_a, thresh, tens_wrap):
    value = single_term(tensor, num_c, num_a, tens_wrap)
    test = evaluate(value)
    difference = evaluate(tensor - test)
    error = numpy.linalg.norm(numpy.array(raw(difference)))
    while error>thresh:
        print(error)
        delta = single_term(difference, num_c, num_a, tens_wrap)
        value += delta
        test = evaluate(test + delta)
        difference = evaluate(tensor - test)
        error = numpy.linalg.norm(numpy.array(raw(difference)))
    print(error)
    return value
