import readline
import pickle
import numpy
import tensorly
from qode.math.tensornet import raw, evaluate, scalar_value, tl_tensor
from qode.math import space, numpy_space, vector_set, gram_schmidt

class empty(object):
    pass



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

def svd(tensor, indices_A, indices_B=None, normalized=False, big=10**5, print_info=False):
    nparray_Nd = raw(tensor)
    if indices_B is None:  indices_B = []
    all_free_indices = list(indices_A) + list(indices_B)
    if list(sorted(all_free_indices))!=list(range(len(nparray_Nd.shape))):
        raise RuntimeError("dimension mismatch")
    if len(indices_A)==0 or len(indices_B)==0:
        return tl_tensor(nparray_Nd)
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
    A = tl_tensor(A.reshape([d] + list(shape_A)))
    B = tl_tensor(B.reshape([d] + list(shape_B)))
    #
    return A, B, s

def complement(tensor, indices, big=10**5, print_info=False):
    nparray_Nd = raw(tensor)
    all_free_indices = list(range(len(nparray_Nd.shape)))
    indices_A = indices
    indices_B = []
    for i in indices_A:
        if i not in all_free_indices:  raise RuntimeError("dimension mismatch")
    for i in all_free_indices:
        if i not in indices_A:  indices_B += [i]
    all_free_indices = list(indices_A) + list(indices_B)
    nparray_Nd = nparray_Nd.transpose(all_free_indices)
    shape_A = nparray_Nd.shape[:len(indices_A)]
    shape_B = nparray_Nd.shape[len(indices_A):]
    if numpy.prod(shape_A)>=numpy.prod(shape_B):
        raise RuntimeError("basis already (over)complete")
    M = nparray_Nd.reshape((numpy.prod(shape_A), numpy.prod(shape_B)))
    ONerror = numpy.linalg.norm(M @ M.T - numpy.identity(numpy.prod(shape_A)))
    print(f"deviation of given set from orthonormality = {ONerror}")
    #
    U, s, Vh = _SVD(M, big, print_info)
    d = len(s)
    #
    B = Vh[d:,:]
    B = tl_tensor(B.reshape([numpy.prod(shape_B)-d] + list(shape_B)))
    return B


def stack(tensors):
    shape = tensors[0].shape
    stacked_shape = [len(tensors)] + list(shape)
    nparray_Nd = numpy.zeros_like(raw(tensors[0]), shape=stacked_shape)
    for i,tensor in enumerate(tensors):
        if tensor.shape!=shape:  raise RuntimeError("cannot stack tensors of different shapes")
        nparray_Nd[i, ...] = raw(tensor)
    return tl_tensor(nparray_Nd)



p,q,r,s = "pqrs"

norm = lambda x:  numpy.linalg.norm(raw(x))
prec = lambda n:  numpy.set_printoptions(precision=n)
save = lambda f:  readline.write_history_file(f)




print("""\
Be = pickle.load(open("rho/Be631g-new-1e-6.pkl","rb"))
i,j = 0,0
c     = Be.rho['c'    ][-1,0][i,j]
cca   = Be.rho['cca'  ][-1,0][i,j]
cccaa = Be.rho['cccaa'][-1,0][i,j]
c1, c2a1, w1 = svd(cca, [0], [1,2])
""")

prec(3)



def _single_term_recur(tensor, n_axes):
    A, B, w = svd(tensor, [0], list(range(1,n_axes)))
    n_axes -= 1
    if n_axes>1:
        indices = [0] + [slice(None)]*n_axes
        return A[0,:], *_single_term_recur(B[tuple(indices)], n_axes)
    else:
        return A[0,:], B[0,:]

permute = {
    0:  None,
    1:  {(0,): +1},
    2:  {(0,1): +1, (1,0): -1},
    3:  {(0,1,2): +1, (0,2,1): -1,  (1,0,2): -1, (1,2,0): +1, (2,1,0): -1, (2,0,1): +1},
}

def single_term(tensor, num_c, num_a):
    primitives = list(enumerate(_single_term_recur(tensor, num_c+num_a)))
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
    return evaluate(value)

def multi_term(tensor, num_c, num_a, thresh=1e-1):
    value = single_term(tensor, num_c, num_a)
    difference = evaluate(tensor - value)
    error = norm(difference)
    while error>thresh:
        print(error)
        value += single_term(difference, num_c, num_a)
        difference = evaluate(tensor - value)
        error = norm(difference)
    print(error)

