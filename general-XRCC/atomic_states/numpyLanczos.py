#    (C) Copyright 2018 Yuhong Liu
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
import numpy as np
import qode
import new_traits
from   math  import sqrt



# orginal function to wrap: 
#     def lowest_eigen(H,v,thresh,dim=10,autocomplete=False):
# 
# print(type(h), isinstance(h, np.ndarray), type(hmat), isinstance(hmat, np.matrixlib.defmatrix.matrix))




def eigh(mat, num_eig=1, thresh=1e-6):
    if not isinstance(mat,np.ndarray):
        mat = np.array(mat)
    vec    = [ np.random.random( mat.shape[0] ) for i in range(num_eig) ]
    traits = qode.math.numpy_space.real_traits( mat.shape[0] )
    Space  = qode.math.linear_inner_product_space(traits)
    H      = Space.lin_op(mat)
    v      = [ Space.member(item) for item in vec ]
    input_mat_dim = mat.shape[0]
    dim    = input_mat_dim // num_eig 
    if dim < 1:
        dim = 1
    if dim > 50:
        dim = 50
    dim = 10
    print('Mat Dim =', input_mat_dim, '; num of eigen =', num_eig, '; Lanczos Dim =', dim)
    evpairs = list( qode.math.lanczos.lowest_eigen(H,v,thresh,dim=dim) )
    E = np.array( [ item[0] for item in evpairs ] )
    U = np.matrix( [ item[1].v for item in evpairs ] )
    U = U.T
    print("Lowest Eigenvalue from Lanczos Extraction: ", E[0])
    print('-'*80)
    for i in range(E.shape[0]):
        print('| Extracted Eigenvalue[%d] =' %(i), E[i])
    print('-'*80)
    return E,U


def Hv_eigh(mat, num_eig=1, thresh=1e-6):
    print('Hv mat object type =', type(mat))
    vec    = [ np.random.random( mat.states.shape[0] ) for i in range(num_eig) ]
    Space  = qode.math.linear_inner_product_space(new_traits.ham_traits())
    H      = Space.lin_op(mat)
    v      = [ Space.member(item) for item in vec ]
    input_mat_dim = mat.states.shape[0]
    dim    = input_mat_dim // num_eig 
    if dim < 1:
        dim = 1
    if dim > 50:
        dim = 50
    dim = 10
    print('Mat Dim =', input_mat_dim, '; num of eigen =', num_eig, '; Lanczos Dim =', dim)
    evpairs = list( qode.math.lanczos.lowest_eigen(H,v,thresh,dim=dim, block_action=num_eig) )
    E = np.array( [ item[0] for item in evpairs ] )
    U = np.matrix( [ item[1].v for item in evpairs ] )
    U = U.T
    print("Lowest Eigenvalue from Lanczos Extraction: ", E[0])
    print('-'*80)
    for i in range(E.shape[0]):
        print('| Extracted Eigenvalue[%d] =' %(i), E[i])
    print('-'*80)
    return E,U



if __name__ ==  "__main__":
    import sys
    mat   = np.load(sys.argv[1])
    n_eig = 10
    E,U   = np.linalg.eigh(np.matrix(mat))
    # print(E[0:20])

    e,u = lanczos_eig(mat, n_eig)
    # print('Numpy Lowest Eigenvalue =', E[0])
    for i in range(n_eig):
        print('np: %.10f vs Lan: %.10f %%error[%d] =' %(E[i], e[i], i) , abs(E[i] - e[i])/abs(E[i]) )
