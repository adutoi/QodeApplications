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
# Run under Python 3.x
#
import sys
import numpy as np

def transform_mat(h,V,T,N,C):
	from transformVMat import transformV
	print("\nTransforming h,v,T,N,C into converged basis...\n")
	return C.T * h * C,     \
	       transformV(V,C), \
	       C.T * T * C,     \
	       C.T * N * C

def two_idx_mat_add_spin(Mat):
	dim = Mat.shape[0]
	return np.matrix( np.vstack(( np.hstack(( Mat.copy(), np.zeros((dim,dim)) )) , np.hstack(( np.zeros((dim,dim)), Mat.copy() )) )) )

def four_idx_mat_add_spin(Mat):
	# for Electronic Repulsion Matrix
	from buildSpin import spin_rep_mat
	return spin_rep_mat(Mat)

def add_spin(S, h, V, T, N, C):
	print("\nAdding alpha, beta spins into matrices C, S, h, V, T, N.")
	return two_idx_mat_add_spin(S),  \
	       two_idx_mat_add_spin(h),  \
	       four_idx_mat_add_spin(V), \
	       two_idx_mat_add_spin(T),  \
	       two_idx_mat_add_spin(N),  \
	       two_idx_mat_add_spin(C)

def save_ON_spin_mat(prefix, s,h,v,T,N,C):
	print("\nSaving C, S, h, V, T and N Info in SPIN ORBITAL BASIS for {}".format(prefix))
	np.save(prefix + "_S_spin.npy", s)
	np.save(prefix + "_h_spin.npy", h)
	np.save(prefix + "_V_spin.npy", v)
	np.save(prefix + "_T_spin.npy", T)
	np.save(prefix + "_N_spin.npy", N)
	np.save(prefix + "_C_HF_spin.npy", C)
	print("Info Saved...\n")


def transform_main(prefix, S_mat, h_mat, V_mat, T_mat, N_mat, C_mat):
	h_on,  V_on,  T_on,  N_on                = transform_mat(   h_mat, V_mat, T_mat, N_mat, C_mat )	
	S_sp,  h_sp,  V_sp,  T_sp,  N_sp,  C_sp  = add_spin( S_mat, h_on , V_on , T_on , N_on , C_mat )
	save_ON_spin_mat(prefix, S_sp, h_sp, V_sp, T_sp, N_sp, C_sp)

if __name__ == "__main__":
	prefix_in = sys.argv[1]
	fields = prefix_in.split("_")
	if fields[0]=="Be2":  prefix_out = fields[0] +"_"+ fields[1] +"."+ "_".join(fields[2:])
	else:                 prefix_out = prefix_in
	transform_main(  prefix_out,                     \
	                 np.matrix(np.load(prefix_in + '_S.npy')), \
	                 np.matrix(np.load(prefix_in + '_h.npy')), \
	                 np.matrix(np.load(prefix_in + '_V.npy')), \
	                 np.matrix(np.load(prefix_in + '_T.npy')), \
	                 np.matrix(np.load(prefix_in + '_N.npy')), \
	                 np.matrix(np.load(prefix_in + '_C_HF.npy'))
	              )
