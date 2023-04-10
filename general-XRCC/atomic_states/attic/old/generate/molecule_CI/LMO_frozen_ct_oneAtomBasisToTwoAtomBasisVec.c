/*   (C) Copyright 2018 Anthony D. Dutoi
 *
 *   This file is part of Qode.
 *
 *   Qode is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   Qode is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with Qode.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <stdio.h>
#include "omp.h"

// must be square matrix to compute determinant.
//
//
double det(int dim, double* Mat)  // dim is leading dimension.
{
	void dgetrf_( int* M, int* N, double* A, int* LDA, int* IPIV, int* INFO );

	if (dim == 1)
	{
		return Mat[0];
	}
	// else if (dim < 1)
	// {
	// 	puts("error occured, invalid dimension of determinant.");
	// 	return 0.0;
	// }
	else
	{
		int m = dim, n = dim, lda = dim, info;
		int ipiv[dim];
		double result = 1.0;

		dgetrf_(&m, &n, Mat, &lda, ipiv, &info);

		// if (info < 0)
		// {
		// 	printf("INFO = %d Error Occured, check lapack docs\n", info);
		// }

		for (int i=0; i<dim; i++)
		{
			if (ipiv[i] == i+1)
			{
				result *= Mat[i*dim+i];
			}
			else
			{
				result *= -Mat[i*dim+i];
			}
		}
		return result;
	}
}


double compute_det(long long i, long long j, long long num_spin_orbs_atom, long long num_val_elec_atom, long long* Be_fci_states, long long* Be2_fci_states, double* S_Atomic_overlap)
{
	int dim = (int) num_val_elec_atom;
	double detM[dim*dim];
	for (int k=0; k<dim; k++)
	{
		for (int l=0; l<dim; l++)
		{
			// printf("i=%lld, k=%lld, j=%lld, l=%lld, Be2_fci_states = %lld, Be_fci_states = %lld, S index = %lld\n", i,k,j,l, Be2_fci_states[i * num_val_elec_atom + k],Be_fci_states[j * num_val_elec_atom + l], Be2_fci_states[i * num_val_elec_atom + k] * num_spin_orbs_atom + Be_fci_states[j * num_val_elec_atom + l] );
			// printf("%lld, %lld, %lld, %lld, S = %.10lf\n", i,k,j,l, S_Atomic_overlap[ Be2_fci_states[i * dim + k] * num_spin_orbs_atom + Be_fci_states[j * dim + l] ] ); 
			detM[k * dim + l] = S_Atomic_overlap[ Be2_fci_states[i * dim + k] * num_spin_orbs_atom + Be_fci_states[j * dim + l] ]; 
		}
	}

	// for (int i=0; i<dim*dim; i++)
	// {
	// 	printf("%.10lf ", detM[i]);
	// }
	// puts(" ");

	return det(dim, detM);
}





void one_atom_basis_to_two_atom_basis(long long num_spin_orbs_atom, long long num_val_elec_atom,  long long Be_n_state, long long Be2_n_state, \
	                                  long long* Be_fci_states, long long* Be2_fci_states, double* S_Atomic_overlap, double* U, long long num_vec, double* atomVec, int nthd )
{
    omp_set_dynamic(0);
    omp_set_num_threads(nthd);

    #pragma omp parallel for
    for (long long i=0; i<Be2_n_state; i++)
    {
    	for (long long j=0; j<Be_n_state; j++)
    	{
    		double detM = compute_det(i, j, num_spin_orbs_atom, num_val_elec_atom, Be_fci_states, Be2_fci_states, S_Atomic_overlap);
    		// printf("i=%lld j=%lld det=%.10lf\n", i, j, detM);
    		for (long long k=0; k<num_vec; k++)
    		{
                #pragma omp atomic 
    			atomVec[i*num_vec + k] += detM * U[j*num_vec +k];
    		}
    	}
    }
}




// def compute_det(first_idx, second_idx, num_valence_elec_atom, Be_fci_states, Be2_fci_states, S_Atomic_overlap):
// 	dimension = num_valence_elec_atom
// 	C_det = np.zeros((dimension,dimension))
// 	for k in range(dimension):
// 		for l in range(dimension):
// 			# print('Be2 FCI', Be2_fci_states[first_idx,k], 'Be FCI', Be_fci_states[second_idx,l
// 			C_det[k,l] = S_Atomic_overlap[Be2_fci_states[first_idx, k], Be_fci_states[second_idx, l]]
// 	return np.linalg.det(C_det)





// # def compute_row(ith, num_valence_elec_atom, Be_n_state, Be_fci_states, Be2_fci_states, S_Atomic_overlap):
// def compute_row(*args):
// 	ith, num_valence_elec_atom, Be_n_state, Be_fci_states, Be2_fci_states, S_Atomic_overlap = args[0]
// 	results_row = np.matrix(np.zeros((1,Be_n_state)))
// 	for j in range(Be_n_state):
// 		results_row[0,j] = compute_det(ith,j, num_valence_elec_atom, Be_fci_states, Be2_fci_states, S_Atomic_overlap)
// 	return results_row


// # count = 0
// # batch = 100
// # while count < Be2_n_state_1e:
// # 	myPool0  = Pool(20)
// # 	if count + batch < Be2_n_state_1e:
// # 		rows  = myPool0.map(compute_row, [[i, num_valence_elec_atom-1, Be_n_state_1e, Be_1e_fci_states, Be2_1e_fci_states, S_Atomic_overlap] for i in range(count, count + batch)] )
// # 	else:
// # 		rows  = myPool0.map(compute_row, [[i, num_valence_elec_atom-1, Be_n_state_1e, Be_1e_fci_states, Be2_1e_fci_states, S_Atomic_overlap] for i in range(count, Be2_n_state_1e)] )
// # 	myPool0.terminate()
// # 	print('batch =', count, count + len(rows))
// # 	for i in range(len(rows)):
// # 		atomVec1e[count+i,:] = rows[i] * U_1e
// # 	count += len(rows)


// # for i in range(Be2_n_state_3e):
// # 	print('3e vec',i,'orb')
// # 	ith_row = compute_row(i, num_valence_elec_atom+1, Be_n_state_3e, Be_3e_fci_states, Be2_3e_fci_states, S_Atomic_overlap)
// # 	atomVec3e[i,:] = ith_row * U_3e

















