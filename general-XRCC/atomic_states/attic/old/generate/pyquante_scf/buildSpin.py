#    (C) Copyright 2018 Anthony D. Dutoi
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
# from qode.SCF_diag_full.matrix_operation import build_size_mat,printline,printnpline,writenpline
import numpy as np
from math import sqrt

# This code builds big matrices with spins taken into considerations.
# Format of spins: (T and N)
#  alpha alpha, alpha beta
#  beta  alpha, beta  beta

def build_loop(M,N):
    'M:small mat; N:new big mat'
    size = len(M)
    size_n = len(N)
    
    for i in range(size_n):
        for j in range(size_n):
            if (i<= size - 1 and j<= size - 1):
                N[i,j]=M[i,j]
            elif (i>= size and j>= size):
                N[i,j]=M[i-size,j-size]    
    return N


def index_to_spin(k,chspin):
    if k < chspin:
        spin = 1
    else:
        spin = 2
        k -= chspin
    return k,spin

def spin_to_spa(i,j,k,l,M,norbi):
    
    i,si = index_to_spin(i,norbi)
    j,sj = index_to_spin(j,norbi)
    k,sk = index_to_spin(k,norbi)
    l,sl = index_to_spin(l,norbi)

    if (si == sk) and (sj == sl):
        repvalue = M[i*norbi+j,k*norbi+l]
    else:
        repvalue = 0.0
    return repvalue



def build_rep_loop(M,N):
    'M:small mat; N:new big mat'
    n = 2*int(sqrt(len(M)))
    norbi = n // 2
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    N[i*n+j,k*n+l] = spin_to_spa(i,j,k,l,M,norbi)
                    
    return N

# def spin_mat(M):
#     size = len(M)
#     # N = build_size_mat(size*2)
#     N = np.matrix(np.zeros((size*2,size*2)))
#     N = build_loop(M,N)
#     return N

def spin_rep_mat(M):
    size = len(M)
    # N = build_size_mat(size*4) # This size is correct.
    N = np.matrix(np.zeros((size*4,size*4)))
    N = build_rep_loop(M,N)
    return N

# #================= Main Function ============================

# def spin_main(T_on, N_on, V_on):
#     # If you diagonalize S_sp, and you NORMALIZE it!
#     # You will get the same Unitary Matrix as the doubled U_sp.
    
#     T_sp = spin_mat(T_on)
#     N_sp = spin_mat(N_on)
#     V_sp = spin_rep_mat(V_on)
#     #writenpline(V_sp,'correct-V.txt')
    

#     return T_sp,N_sp,V_sp









