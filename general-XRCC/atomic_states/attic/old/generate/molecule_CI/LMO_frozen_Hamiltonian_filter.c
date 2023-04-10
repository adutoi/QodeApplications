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
#include <stdlib.h>
#include <math.h>
#include "omp.h"
#include <time.h>


// THIS IS A DERIVED COPY FROM LMO_frozen_ct_Hamiltonian_multiple_vec.c

void load_vector( double* matrix, long long dim1, long long dim2, double* vector, long long col_index )
{
    if (col_index < dim2)
    {
        for (long long i=0; i<dim1; i++)
        {
            matrix[ i*dim2 + col_index ] = vector[i];
        }
    }
}

int compare (const void* a, const void* b) // Sorting Functions for C stdlib qsort...
{
  return ( *(long long*)a - *(long long*)b );
}


int sorted_single_state(long long num_elec, long long* ref_state, long long gd_position, long long excited, long long* sorted_state) 
{
    int phase = 1;
    for (long long i=0; i<num_elec; i++)
    {
        sorted_state[i] = ref_state[i];
    }
    sorted_state[gd_position] = excited;
    long long index = gd_position;
    long long temp;
    while (index < num_elec-1 && sorted_state[index] > sorted_state[index+1])
    {
        temp                  = sorted_state[index+1];
        sorted_state[index+1] = sorted_state[index];
        sorted_state[index]   = temp;
        index++;
        phase *= -1;
    }
    while (index > 0 && sorted_state[index] < sorted_state[index-1])
    {
        temp                  = sorted_state[index-1];
        sorted_state[index-1] = sorted_state[index];
        sorted_state[index]   = temp;
        index--;      
        phase *= -1;
    }
    return phase;
}


int sorted_double_state(long long num_elec,      long long* ref_state  , \
                         long long gd_position1,  long long gd_position2, \
                         long long excited1,      long long excited2    , long long* sorted_state)
{
    int phase = 1;
    for (long long i=0; i<num_elec; i++)
    {
        sorted_state[i] = ref_state[i];
    }
    sorted_state[gd_position1] = excited1;
    sorted_state[gd_position2] = excited2;
    long long index1 = gd_position1, index2 = gd_position2;
    long long temp;
    while (index2 < num_elec-1 && sorted_state[index2] > sorted_state[index2+1])  // if run, this is an easy case
    {
        temp                   = sorted_state[index2+1];
        sorted_state[index2+1] = sorted_state[index2];
        sorted_state[index2]   = temp;
        index2++;
        phase *= -1;
    }
    while (index1 > 0 && sorted_state[index1] < sorted_state[index1-1])    // if run, this is an easy case
    {
        temp                   = sorted_state[index1-1];
        sorted_state[index1-1] = sorted_state[index1];
        sorted_state[index1]   = temp;
        index1--;
        phase *= -1;     
    }
    // None of above run makes this a little complicated, but excited1 < excited2 is true.
    while (index2 > index1+1 && sorted_state[index2] < sorted_state[index2-1])
    {
        temp                   = sorted_state[index2-1];
        sorted_state[index2-1] = sorted_state[index2];
        sorted_state[index2]   = temp;
        index2--;
        phase *= -1;     
    }
    while (index1 < index2-1 && sorted_state[index1] > sorted_state[index1+1])
    {
        temp                   = sorted_state[index1+1];
        sorted_state[index1+1] = sorted_state[index1];
        sorted_state[index1]   = temp;
        index1++;
        phase *= -1;
    }
    return phase;
}





// FCI Vector Indexing Function
//
//
long long configToFciIndex(long long num_elec, long long num_spin_orbs, long long* C_ConfigArray, long long* C_ComboMatrix)
{
    long long index_num = 0;
    for(long long n=1; n<C_ConfigArray[0]+1; n++)
    {
        index_num += C_ComboMatrix[num_elec*num_spin_orbs - n];
    }
    for(long long i=0; i<num_elec-1; i++)
    {
        for(long long n=C_ConfigArray[i]+2; n<C_ConfigArray[i+1]+1; n++)
        {
            index_num += C_ComboMatrix[(num_elec-i-1)*num_spin_orbs - n];
        }
    }
    return index_num;
}



long long two_loop_index(long long i, long long j, long long dim)
{
    // return the index of summation over i and j for all i < j.
    return (2*dim-i-1) * i /2 + (j - i - 1);
}


long long valence_index_to_total_index(long long index, long long num_val_orbs_atom, long long num_core_elec_atom)
{
    if (index < num_val_orbs_atom)  // orbs from Atom A
    { return index + num_core_elec_atom; }
    else                            // orbs from Atom B
    { return index + num_core_elec_atom*2; }
}


long long total_index_to_valence_index(long long index, long long num_spat_orbs_atom, long long num_core_elec_atom)
{
    if (index < num_spat_orbs_atom)  // orbs from Atom A
    { return index - num_core_elec_atom; }
    else                            // orbs from Atom B
    { return index - num_core_elec_atom*2; }
}



long long find_1e_state_index( long long  num_val_elec , long long num_val_orbs,   long long  num_spin_orbs_atom, long long num_core_elec_atom, \
                               long long* ref_val_state, long long* C_ComboMatrix, long long  orig,               long long excited )
{
    long long sorted_val_state[num_val_elec];
    long long index = -1;
    long long temp;
    for (long long i=0; i<num_val_elec; i++)
    {
        if (ref_val_state[i] != orig)
        {
            sorted_val_state[i] = total_index_to_valence_index( ref_val_state[i], num_spin_orbs_atom, num_core_elec_atom );
        }
        else
        {
            sorted_val_state[i] = total_index_to_valence_index( excited, num_spin_orbs_atom, num_core_elec_atom );
            index = i;
        }
    }
    if (index != -1)  // sorting is needed if valence electron is being replaced.
    {
        while (index < num_val_elec-1 && sorted_val_state[index] > sorted_val_state[index+1])
        {
            temp                      = sorted_val_state[index+1];
            sorted_val_state[index+1] = sorted_val_state[index];
            sorted_val_state[index]   = temp;
            index++;
        }
        while (index > 0 && sorted_val_state[index] < sorted_val_state[index-1])
        {
            temp                      = sorted_val_state[index-1];
            sorted_val_state[index-1] = sorted_val_state[index];
            sorted_val_state[index]   = temp;
            index--;      
        }
    }
    // qsort( sorted_val_state, num_val_elec, sizeof(long long), compare );
    return configToFciIndex( num_val_elec, num_val_orbs, sorted_val_state, C_ComboMatrix );
}



long long find_2e_state_index( long long  num_val_elec , long long num_val_orbs,   long long  num_spin_orbs_atom, long long num_core_elec_atom, \
                               long long* ref_val_state, long long* C_ComboMatrix, long long  orig1,              long long orig2,       \
                               long long  excited1,      long long  excited2 )
{
    long long sorted_val_state[num_val_elec];
    long long index1 = -1, index2 = -1;
    long long temp;
    for (long long i=0; i<num_val_elec; i++)
    {
        if (ref_val_state[i] != orig1)
        {
            if (ref_val_state[i] != orig2)
            {
                sorted_val_state[i] = total_index_to_valence_index( ref_val_state[i], num_spin_orbs_atom, num_core_elec_atom );
            }
            else
            {
                sorted_val_state[i] = total_index_to_valence_index( excited2, num_spin_orbs_atom, num_core_elec_atom );
                index2 = i;
            }
        }
        else
        {
            sorted_val_state[i] = total_index_to_valence_index( excited1, num_spin_orbs_atom, num_core_elec_atom );  
            index1 = i;  
        }
    }

    if (index1 != -1) 
    {
        if (index2 != -1) // need to sort both
        {
            while (index2 < num_val_elec-1 && sorted_val_state[index2] > sorted_val_state[index2+1])  // if run, this is an easy case
            {
                temp                       = sorted_val_state[index2+1];
                sorted_val_state[index2+1] = sorted_val_state[index2];
                sorted_val_state[index2]   = temp;
                index2++;
            }
            while (index1 > 0 && sorted_val_state[index1] < sorted_val_state[index1-1])    // if run, this is an easy case
            {
                temp                       = sorted_val_state[index1-1];
                sorted_val_state[index1-1] = sorted_val_state[index1];
                sorted_val_state[index1]   = temp;
                index1--;      
            }
            // None of above run makes this a little complicated, but excited1 < excited2 is true.
            while (index2 > index1+1 && sorted_val_state[index2] < sorted_val_state[index2-1])
            {
                temp                       = sorted_val_state[index2-1];
                sorted_val_state[index2-1] = sorted_val_state[index2];
                sorted_val_state[index2]   = temp;
                index2--;      
            }
            while (index1 < index2-1 && sorted_val_state[index1] > sorted_val_state[index1+1])
            {
                temp                       = sorted_val_state[index1+1];
                sorted_val_state[index1+1] = sorted_val_state[index1];
                sorted_val_state[index1]   = temp;
                index1++;
            }
        }
        else     // need to sort the first orbital.
        {
            while (index1 < num_val_elec-1 && sorted_val_state[index1] > sorted_val_state[index1+1])
            {
                temp                       = sorted_val_state[index1+1];
                sorted_val_state[index1+1] = sorted_val_state[index1];
                sorted_val_state[index1]   = temp;
                index1++;
            }
            while (index1 > 0 && sorted_val_state[index1] < sorted_val_state[index1-1])
            {
                temp                       = sorted_val_state[index1-1];
                sorted_val_state[index1-1] = sorted_val_state[index1];
                sorted_val_state[index1]   = temp;
                index1--;      
            }
        }
    }
    else
    {
        if (index2 != -1) // need to sort second orbital.
        {
        while (index2 < num_val_elec-1 && sorted_val_state[index2] > sorted_val_state[index2+1])
        {
            temp                       = sorted_val_state[index2+1];
            sorted_val_state[index2+1] = sorted_val_state[index2];
            sorted_val_state[index2]   = temp;
            index2++;
        }
        while (index2 > 0 && sorted_val_state[index2] < sorted_val_state[index2-1])
        {
            temp                       = sorted_val_state[index2-1];
            sorted_val_state[index2-1] = sorted_val_state[index2];
            sorted_val_state[index2]   = temp;
            index2--;      
        }
        }
    }
    // qsort( sorted_val_state, num_val_elec, sizeof(long long), compare );
    return configToFciIndex(num_val_elec, num_val_orbs, sorted_val_state, C_ComboMatrix );
}





int is_core_elec(long long orb, long long num_core_elec_atom, long long* alpha_core_elec, long long* beta_core_elec)
{
    for (long long i=0; i<num_core_elec_atom; i++)
    {
        if (orb == alpha_core_elec[i] || orb == beta_core_elec[i])
        {
            return 1; // is indeed core electrion.
        }
    }
    return 0; // not core electron.
}


int violate_1e_pauli_exclusion(long long* state, long long dim, long long gd_position, long long excited)
{
    for (long long i=0; i<dim; i++)
    {
        if (state[i] == excited && i != gd_position)
        {
            return 1;  // Violating Pauli Exclution Principle, return True immediately.
        }
    }
    return 0; // Not Violating Pauli Exclusion Principle
}


int violate_2e_pauli_exclusion(long long* state, long long dim, long long gd_position1, long long gd_position2,\
                                                                long long excited1,     long long excited2)
{
    for (long long i=0; i<dim; i++)
    {
        if (state[i] == excited1 || state[i] == excited2)
        {
            if (i == gd_position1 || i == gd_position2 )
            {}
            else
            {
                return 1;  // Violating Pauli Exclution Principle, return True immediately.
            }   
        }
    }
    return 0; // Not Violating Pauli Exclusion Principle
}




// Module Main Routine
//
//
void compute_HPsi(long long  num_elec,      
                  long long  num_core_elec, 
                  long long  num_spin_orbs, 
                  long long  num_fci_states,
                  double*    C_Psi,           // Column Vectors
                  long long* C_FciConfigs,
                  long long* C_FciValConfigs,
                  double*    C_h_mat,      
                  double*    C_V_mat,      
                  double*    HPsi,            // Column Vectors
                  long long* C_ComboMatrix, 
                  long long  num_fci_vec, 
                  int nthd)
{
    omp_set_dynamic(0);
    omp_set_num_threads(nthd);

    time_t timer0, timer1;
    time(&timer0);

    // puts("Running C Code Hamiltonian.");
    // printf("num_elec = %lld\nnum_core_elec = %lld\nnum_spin_orbs = %lld\nnum_fci_states = %lld\nNum of Threads = %d\n", num_elec, num_core_elec, num_spin_orbs, num_fci_states, nthd);
    long long num_virtual  = num_spin_orbs - num_elec;
    long long num_val_elec = num_elec - num_core_elec;
    long long num_val_orbs = num_spin_orbs - num_core_elec;
    long long num_elec_atom      = num_elec     / 2;
    long long num_virtual_atom   = num_virtual  / 2;
    long long num_core_elec_atom = num_core_elec/ 2;
    long long num_spin_orbs_atom = num_spin_orbs/ 2;
    long long alpha_core_elec[num_core_elec_atom], beta_core_elec[num_core_elec_atom];
    for (long long i=0; i<num_core_elec_atom; i++)
    {
        alpha_core_elec[i] = i;
        beta_core_elec[i]  = i + num_spin_orbs_atom; 
    }



    // Here Filter Out Significant V values ( fabs > thresh )
    double    sig_thresh = 1.0e-10;
    long long idx;
    long long num_pairs = num_spin_orbs * (num_spin_orbs-1) /2;
    double    h_mat_value, V_mat_value;
    // long long count;

    long long* pairs_p  = (long long*) malloc( sizeof(long long) * num_pairs * num_pairs );
    long long* pairs_q  = (long long*) malloc( sizeof(long long) * num_pairs * num_pairs );
    long long* num_sigs = (long long*) malloc( sizeof(long long) * num_pairs );
    double   * sig_V    = (double*)    malloc( sizeof(double)    * num_pairs * num_pairs );

    if (pairs_p == NULL || pairs_q == NULL || num_sigs == NULL)
    {
        puts("Dynamic Allocation Failed.");
        return;
    }

    for (long long i=0; i<num_pairs; i++) // initialized to zero
    {
        num_sigs[i] = 0;
    }

    for (long long r=0; r<num_spin_orbs; r++) 
    {
        for (long long s=r+1; s<num_spin_orbs; s++) 
        {
            idx = two_loop_index(r, s, num_spin_orbs);
            for (long long p=0; p<num_spin_orbs; p++) 
            {
                for (long long q=p+1; q<num_spin_orbs; q++) 
                {
                    V_mat_value = C_V_mat[ (r * num_spin_orbs + s) * num_spin_orbs * num_spin_orbs + p * num_spin_orbs + q ] - \
                                  C_V_mat[ (r * num_spin_orbs + s) * num_spin_orbs * num_spin_orbs + q * num_spin_orbs + p ];
                    if ( fabs( V_mat_value ) > sig_thresh ) 
                    {
                        pairs_p[idx * num_pairs + num_sigs[idx]] = p;
                        pairs_q[idx * num_pairs + num_sigs[idx]] = q;
                        sig_V[  idx * num_pairs + num_sigs[idx]] = V_mat_value;
                        num_sigs[idx]++;
                    }
                }  
            }
        }
    }

    // Here builds the list of significant H_pq
    //
    long long* h_pairs    = (long long*) malloc( sizeof(long long) * num_spin_orbs * num_spin_orbs );
    long long* h_num_sigs = (long long*) malloc( sizeof(long long) * num_spin_orbs );
    double*    sig_h      = (double*)    malloc( sizeof(double)    * num_spin_orbs * num_spin_orbs );

    if (h_pairs == NULL || h_num_sigs == NULL)
    {
        puts("Dynamic Allocation Failed.");
        return;
    }

    for (long long i=0; i<num_spin_orbs; i++) // initialized to zero
    {
        h_num_sigs[i] = 0;
    }

    for (long long p=0; p<num_spin_orbs; p++) 
    {
        for (long long q=0; q<num_spin_orbs; q++) 
        {
            h_mat_value = C_h_mat[p * num_spin_orbs + q];
            if ( fabs(h_mat_value) > sig_thresh )
            {
                h_pairs[ p * num_spin_orbs + h_num_sigs[p] ] = q;
                sig_h[   p * num_spin_orbs + h_num_sigs[p] ] = h_mat_value;
                h_num_sigs[p]++;
            }
        }
    }


    // Main Loop of kets
    //
    //
    #pragma omp parallel for
    for (long long n=0; n<num_fci_states; n++)
    {
        int any_significant_C = 0;
        long long x=0;
        while ( !any_significant_C && x<num_fci_vec)
        {
            if ( fabs(C_Psi[n * num_fci_vec + x]) > sig_thresh )
            {
                any_significant_C = 1;
            }
                x++;
        }

        if (any_significant_C)
        {
            // printf("%lld-th ket\n", n);
            long long* ket_state     = &C_FciConfigs[n*num_elec];
            long long* ket_val_state = &C_FciValConfigs[n*num_val_elec];
            long long  sorted_state[num_elec];
            long long  valence_state[num_val_elec];
            long long  state_index; 
            long long  gd_position1, gd_position2, ex_position1, ex_position2; 
            long long  orig1, orig2, excited1, excited2;
            double h_value;
            int    sign;
            double sig_h_value, sig_V_value;
            long long index;

            int gd1_is_core, gd2_is_core, ex1_is_core, ex2_is_core;
            // Loop Over 1e: Ref + Singles

            // int is_core_elec(long long orig, long long num_core_elec_atom, long long* alpha_core_elec, long long beta_core_elec)
            
            for(long long i=0; i<num_elec; i++)
            {
                orig1       = ket_state[i];
                gd1_is_core = (is_core_elec(orig1, num_core_elec_atom, alpha_core_elec, beta_core_elec) ? 1 : 0);
                
                for(long long j=0; j<h_num_sigs[ orig1 ]; j++)
                {
                    excited1    = h_pairs[orig1 * num_spin_orbs + j];
                    ex1_is_core = (is_core_elec(excited1, num_core_elec_atom, alpha_core_elec, beta_core_elec) ? 1 : 0);

                    if ( (gd1_is_core  == ex1_is_core) && !violate_1e_pauli_exclusion(ket_state, num_elec, i, excited1))
                    {
                        sign        = sorted_single_state( num_elec, ket_state, i, excited1, sorted_state );
                        state_index = find_1e_state_index( num_val_elec, num_val_orbs, num_spin_orbs_atom, num_core_elec_atom, ket_val_state, C_ComboMatrix, orig1, excited1 );
                        h_value     = sign * sig_h[  orig1 * num_spin_orbs + j];
                      
                        // #pragma omp parallel for
                        for(long long m=0; m<num_fci_vec; m++)
                        {
                            #pragma omp atomic 
                            HPsi[state_index * num_fci_vec + m] += C_Psi[n * num_fci_vec + m] * h_value;
                        }
                    }  // if 
                }  // j
            }      // i



            // Loop Over 2e: Ref + Singles + Doubles

            for(long long i=0; i<num_elec; i++)
            {    
                orig1       = ket_state[i];
                gd1_is_core = (is_core_elec(orig1, num_core_elec_atom, alpha_core_elec, beta_core_elec) ? 1 : 0);
                for (long long j=i+1; j<num_elec; j++) 
                {
                    orig2 = ket_state[j];
                    gd2_is_core = (is_core_elec(orig2, num_core_elec_atom, alpha_core_elec, beta_core_elec) ? 1 : 0);
                    
                    index = two_loop_index(orig1, orig2, num_spin_orbs);

                    for (long long k=0; k<num_sigs[index]; k++)
                    {

                        excited1 = pairs_p[index * num_pairs + k];
                        excited2 = pairs_q[index * num_pairs + k];

                        if (gd1_is_core && orig1 !=excited1 && orig1 != excited2)
                        {}   // orig1 is core but no core in excited ones
                        else
                        {
                            if (gd2_is_core && orig2 != excited1 && orig2 != excited2)
                            {}   // orig2 is core but no core in excited ones
                            else
                            {
                                if (!violate_2e_pauli_exclusion( ket_state, num_elec, i, j, excited1, excited2 ))
                                {
                                    gd_position1 = i;
                                    gd_position2 = j;

                                    sign        = sorted_double_state( num_elec, ket_state, i, j, excited1, excited2, sorted_state );                                
                                    h_value     = sign * sig_V[index * num_pairs + k];
                                    state_index = find_2e_state_index( num_val_elec, num_val_orbs, num_spin_orbs_atom, num_core_elec_atom, ket_val_state, C_ComboMatrix, orig1, orig2, excited1, excited2);

                                    // #pragma omp parallel for
                                    for(long long m=0; m<num_fci_vec; m++)
                                    {
                                        #pragma omp atomic 
                                        HPsi[state_index * num_fci_vec + m] += C_Psi[n * num_fci_vec + m] * h_value;
                                    }
                                } // pauli exclusion if
                            } 
                        }
                    }
                }
            }

        } // if any C_Psi are significant.

    } // ket for loop


    free(pairs_p); free(pairs_q);    free(num_sigs); free(sig_V);
    free(h_pairs); free(h_num_sigs); free(sig_h);



    time(&timer1);
    printf("Total Time = %lf seconds\n", difftime(timer1, timer0));

}

        
