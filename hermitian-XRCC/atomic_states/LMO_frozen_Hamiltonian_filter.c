/*   (C) Copyright 2018, 2019 Yuhong Liu and Anthony Dutoi
 *
 *   This file is part of QodeApplications.
 *
 *   QodeApplications is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   QodeApplications is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with QodeApplications.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdlib.h>    // Holy crap!, there are malloc calls in here.  Try to get them out, or at least check carefully that all allocations are freed.
#include <stdio.h>     // And this is a side-effect of the dynamic allocations too (search for 'puts' to see)
#include <math.h>
#include "omp.h"
#include "PyC_types.h"



// THIS IS A DERIVED COPY FROM LMO_frozen_ct_Hamiltonian_multiple_vec.c

void load_vector( Double* matrix, BigInt dim1, BigInt dim2, Double* vector, BigInt col_index )
{
    if (col_index < dim2)
    {
        for (BigInt i=0; i<dim1; i++)
        {
            matrix[ i*dim2 + col_index ] = vector[i];
        }
    }
}

int compare (const void* a, const void* b) // Sorting Functions for C stdlib qsort...
{
  return ( *(BigInt*)a - *(BigInt*)b );
}


int sorted_single_state(BigInt num_elec, BigInt* ref_state, BigInt gd_position, BigInt excited, BigInt* sorted_state)
{
    int phase = 1;
    for (BigInt i=0; i<num_elec; i++)
    {
        sorted_state[i] = ref_state[i];
    }
    sorted_state[gd_position] = excited;
    BigInt index = gd_position;
    BigInt temp;
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


int sorted_Double_state(BigInt num_elec,      BigInt* ref_state  , \
                        BigInt gd_position1,  BigInt gd_position2, \
                        BigInt excited1,      BigInt excited2    , BigInt* sorted_state)
{
    int phase = 1;
    for (BigInt i=0; i<num_elec; i++)
    {
        sorted_state[i] = ref_state[i];
    }
    sorted_state[gd_position1] = excited1;
    sorted_state[gd_position2] = excited2;
    BigInt index1 = gd_position1, index2 = gd_position2;
    BigInt temp;
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
BigInt configToFciIndex(BigInt num_elec, BigInt num_spin_orbs, BigInt* C_ConfigArray, BigInt* C_ComboMatrix)
{
    BigInt index_num = 0;
    for(BigInt n=1; n<C_ConfigArray[0]+1; n++)
    {
        index_num += C_ComboMatrix[num_elec*num_spin_orbs - n];
    }
    for(BigInt i=0; i<num_elec-1; i++)
    {
        for(BigInt n=C_ConfigArray[i]+2; n<C_ConfigArray[i+1]+1; n++)
        {
            index_num += C_ComboMatrix[(num_elec-i-1)*num_spin_orbs - n];
        }
    }
    return index_num;
}



BigInt two_loop_index(BigInt i, BigInt j, BigInt dim)
{
    // return the index of summation over i and j for all i < j.
    return (2*dim-i-1) * i /2 + (j - i - 1);
}


BigInt valence_index_to_total_index(BigInt index, BigInt num_val_orbs_atom, BigInt num_core_elec_atom)
{
    if (index < num_val_orbs_atom)  // orbs from Atom A
    { return index + num_core_elec_atom; }
    else                            // orbs from Atom B
    { return index + num_core_elec_atom*2; }
}


BigInt total_index_to_valence_index(BigInt index, BigInt num_spat_orbs_atom, BigInt num_core_elec_atom)
{
    if (index < num_spat_orbs_atom)  // orbs from Atom A
    { return index - num_core_elec_atom; }
    else                            // orbs from Atom B
    { return index - num_core_elec_atom*2; }
}



BigInt find_1e_state_index( BigInt  num_val_elec , BigInt num_val_orbs,   BigInt  num_spin_orbs_atom, BigInt num_core_elec_atom, \
                            BigInt* ref_val_state, BigInt* C_ComboMatrix, BigInt  orig,               BigInt excited )
{
    BigInt sorted_val_state[num_val_elec];
    BigInt index = -1;
    BigInt temp;
    for (BigInt i=0; i<num_val_elec; i++)
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
    // qsort( sorted_val_state, num_val_elec, sizeof(BigInt), compare );
    return configToFciIndex( num_val_elec, num_val_orbs, sorted_val_state, C_ComboMatrix );
}



BigInt find_2e_state_index( BigInt  num_val_elec , BigInt num_val_orbs,   BigInt  num_spin_orbs_atom, BigInt num_core_elec_atom, \
                            BigInt* ref_val_state, BigInt* C_ComboMatrix, BigInt  orig1,              BigInt orig2,       \
                            BigInt  excited1,      BigInt  excited2 )
{
    BigInt sorted_val_state[num_val_elec];
    BigInt index1 = -1, index2 = -1;
    BigInt temp;
    for (BigInt i=0; i<num_val_elec; i++)
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
    // qsort( sorted_val_state, num_val_elec, sizeof(BigInt), compare );
    return configToFciIndex(num_val_elec, num_val_orbs, sorted_val_state, C_ComboMatrix );
}





int is_core_elec(BigInt orb, BigInt num_core_elec_atom, BigInt* alpha_core_elec, BigInt* beta_core_elec)
{
    for (BigInt i=0; i<num_core_elec_atom; i++)
    {
        if (orb == alpha_core_elec[i] || orb == beta_core_elec[i])
        {
            return 1; // is indeed core electrion.
        }
    }
    return 0; // not core electron.
}


int violate_1e_pauli_exclusion(BigInt* state, BigInt dim, BigInt gd_position, BigInt excited)
{
    for (BigInt i=0; i<dim; i++)
    {
        if (state[i] == excited && i != gd_position)
        {
            return 1;  // Violating Pauli Exclution Principle, return True immediately.
        }
    }
    return 0; // Not Violating Pauli Exclusion Principle
}


int violate_2e_pauli_exclusion(BigInt* state, BigInt dim, BigInt gd_position1, BigInt gd_position2,\
                                                                BigInt excited1,     BigInt excited2)
{
    for (BigInt i=0; i<dim; i++)
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
void compute_HPsi(PyInt   num_elec_py,
                  PyInt   num_core_elec_py,
                  PyInt   num_spin_orbs_py,
                  PyInt   num_fci_states_py,
                  Double* C_Psi,           // Column Vectors
                  BigInt* C_FciConfigs,
                  BigInt* C_FciValConfigs,
                  Double* C_h_mat,
                  Double* C_V_mat,
                  Double* HPsi,            // Column Vectors
                  BigInt* C_ComboMatrix,
                  PyInt   num_fci_vec_py,
                  PyInt   nthd)
{
    BigInt num_elec       = num_elec_py;	//
    BigInt num_core_elec  = num_core_elec_py;	//
    BigInt num_spin_orbs  = num_spin_orbs_py;	// Just an artifact of PyC being introduced after this code was first written
    BigInt num_fci_states = num_fci_states_py;	//
    BigInt num_fci_vec    = num_fci_vec_py;	//

    omp_set_dynamic(0);
    omp_set_num_threads(nthd);

    //time_t timer0, timer1;
    //time(&timer0);

    // puts("Running C Code Hamiltonian.");
    // printf("num_elec = %lld\nnum_core_elec = %lld\nnum_spin_orbs = %lld\nnum_fci_states = %lld\nNum of Threads = %d\n", num_elec, num_core_elec, num_spin_orbs, num_fci_states, nthd);
    BigInt num_virtual  = num_spin_orbs - num_elec;
    BigInt num_val_elec = num_elec - num_core_elec;
    BigInt num_val_orbs = num_spin_orbs - num_core_elec;
    BigInt num_elec_atom      = num_elec     / 2;
    BigInt num_virtual_atom   = num_virtual  / 2;
    BigInt num_core_elec_atom = num_core_elec/ 2;
    BigInt num_spin_orbs_atom = num_spin_orbs/ 2;
    BigInt alpha_core_elec[num_core_elec_atom], beta_core_elec[num_core_elec_atom];
    for (BigInt i=0; i<num_core_elec_atom; i++)
    {
        alpha_core_elec[i] = i;
        beta_core_elec[i]  = i + num_spin_orbs_atom;
    }



    // Here Filter Out Significant V values ( fabs > thresh )
    Double    sig_thresh = 1.0e-10;
    BigInt idx;
    BigInt num_pairs = num_spin_orbs * (num_spin_orbs-1) /2;
    Double    h_mat_value, V_mat_value;
    // BigInt count;

    BigInt* pairs_p  = (BigInt*) malloc( sizeof(BigInt) * num_pairs * num_pairs );
    BigInt* pairs_q  = (BigInt*) malloc( sizeof(BigInt) * num_pairs * num_pairs );
    BigInt* num_sigs = (BigInt*) malloc( sizeof(BigInt) * num_pairs );
    Double* sig_V    = (Double*) malloc( sizeof(Double) * num_pairs * num_pairs );

    if (pairs_p == NULL || pairs_q == NULL || num_sigs == NULL)
    {
        puts("Dynamic Allocation Failed.");
        return;
    }

    for (BigInt i=0; i<num_pairs; i++) // initialized to zero
    {
        num_sigs[i] = 0;
    }

    for (BigInt r=0; r<num_spin_orbs; r++)
    {
        for (BigInt s=r+1; s<num_spin_orbs; s++)
        {
            idx = two_loop_index(r, s, num_spin_orbs);
            for (BigInt p=0; p<num_spin_orbs; p++)
            {
                for (BigInt q=p+1; q<num_spin_orbs; q++)
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
    BigInt* h_pairs    = (BigInt*) malloc( sizeof(BigInt) * num_spin_orbs * num_spin_orbs );
    BigInt* h_num_sigs = (BigInt*) malloc( sizeof(BigInt) * num_spin_orbs );
    Double* sig_h      = (Double*) malloc( sizeof(Double) * num_spin_orbs * num_spin_orbs );

    if (h_pairs == NULL || h_num_sigs == NULL)
    {
        puts("Dynamic Allocation Failed.");
        return;
    }

    for (BigInt i=0; i<num_spin_orbs; i++) // initialized to zero
    {
        h_num_sigs[i] = 0;
    }

    for (BigInt p=0; p<num_spin_orbs; p++)
    {
        for (BigInt q=0; q<num_spin_orbs; q++)
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
    for (BigInt n=0; n<num_fci_states; n++)
    {
        int any_significant_C = 0;
        BigInt x=0;
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
            BigInt* ket_state     = &C_FciConfigs[n*num_elec];
            BigInt* ket_val_state = &C_FciValConfigs[n*num_val_elec];
            BigInt  sorted_state[num_elec];
            BigInt  valence_state[num_val_elec];
            BigInt  state_index;
            BigInt  gd_position1, gd_position2, ex_position1, ex_position2;
            BigInt  orig1, orig2, excited1, excited2;
            Double  h_value;
            int     sign;
            Double  sig_h_value, sig_V_value;
            BigInt  index;

            int gd1_is_core, gd2_is_core, ex1_is_core, ex2_is_core;
            // Loop Over 1e: Ref + Singles

            // int is_core_elec(BigInt orig, BigInt num_core_elec_atom, BigInt* alpha_core_elec, BigInt beta_core_elec)

            for(BigInt i=0; i<num_elec; i++)
            {
                orig1       = ket_state[i];
                gd1_is_core = (is_core_elec(orig1, num_core_elec_atom, alpha_core_elec, beta_core_elec) ? 1 : 0);

                for(BigInt j=0; j<h_num_sigs[ orig1 ]; j++)
                {
                    excited1    = h_pairs[orig1 * num_spin_orbs + j];
                    ex1_is_core = (is_core_elec(excited1, num_core_elec_atom, alpha_core_elec, beta_core_elec) ? 1 : 0);

                    if ( (gd1_is_core  == ex1_is_core) && !violate_1e_pauli_exclusion(ket_state, num_elec, i, excited1))
                    {
                        sign        = sorted_single_state( num_elec, ket_state, i, excited1, sorted_state );
                        state_index = find_1e_state_index( num_val_elec, num_val_orbs, num_spin_orbs_atom, num_core_elec_atom, ket_val_state, C_ComboMatrix, orig1, excited1 );
                        h_value     = sign * sig_h[  orig1 * num_spin_orbs + j];

                        // #pragma omp parallel for
                        for(BigInt m=0; m<num_fci_vec; m++)
                        {
                            #pragma omp atomic
                            HPsi[state_index * num_fci_vec + m] += C_Psi[n * num_fci_vec + m] * h_value;
                        }
                    }  // if
                }  // j
            }      // i



            // Loop Over 2e: Ref + Singles + Doubles

            for(BigInt i=0; i<num_elec; i++)
            {
                orig1       = ket_state[i];
                gd1_is_core = (is_core_elec(orig1, num_core_elec_atom, alpha_core_elec, beta_core_elec) ? 1 : 0);
                for (BigInt j=i+1; j<num_elec; j++)
                {
                    orig2 = ket_state[j];
                    gd2_is_core = (is_core_elec(orig2, num_core_elec_atom, alpha_core_elec, beta_core_elec) ? 1 : 0);

                    index = two_loop_index(orig1, orig2, num_spin_orbs);

                    for (BigInt k=0; k<num_sigs[index]; k++)
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

                                    sign        = sorted_Double_state( num_elec, ket_state, i, j, excited1, excited2, sorted_state );
                                    h_value     = sign * sig_V[index * num_pairs + k];
                                    state_index = find_2e_state_index( num_val_elec, num_val_orbs, num_spin_orbs_atom, num_core_elec_atom, ket_val_state, C_ComboMatrix, orig1, orig2, excited1, excited2);

                                    // #pragma omp parallel for
                                    for(BigInt m=0; m<num_fci_vec; m++)
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



    //time(&timer1);
    //printf("Total Time = %lf seconds\n", difftime(timer1, timer0));

}
