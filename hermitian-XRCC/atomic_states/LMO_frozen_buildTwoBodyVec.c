/*   (C) Copyright 2018, 2019 Yuhong Liu and Anthony Dutoi
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

#include <stdlib.h>    // need qsort
#include "omp.h"
#include "PyC_types.h"



int compare (const void* a, const void* b) {return ( *(BigInt*)a - *(BigInt*)b );}

//
//  This turns A and B vectors into dimer FCI vector
//



Double get_sign(BigInt* unsorted_config, BigInt dim)
{
	int sign = 1;
	for(BigInt i=0; i<dim; i++)
	{
		for(BigInt j=0; j<i; j++)
		{
			if (unsorted_config[j] > unsorted_config[i])
			{
				sign *= -1;
			}
		}
	}
	return (Double)sign;
}

BigInt configToFciIndex(BigInt* config, BigInt num_elec, BigInt num_spin_orbs, BigInt* combo_matrix)
{
    BigInt index_num = 0;
    for(BigInt n=1; n<config[0]+1; n++)
    {
        index_num += combo_matrix[num_elec*num_spin_orbs - n];
    }
    for(BigInt i=0; i<num_elec-1; i++)
    {
        for(BigInt n=config[i]+2; n<config[i+1]+1; n++)
        {
            index_num += combo_matrix[(num_elec-i-1)*num_spin_orbs - n];
        }
    }
    return index_num;
}

int no_pauli_exclusion(BigInt* configA, BigInt* configB, BigInt num_elec_A, BigInt num_elec_B)
{
	for(BigInt i=0; i<num_elec_A; i++)
	{
		for(BigInt j=0; j<num_elec_B; j++)
		{
			if ( configA[i] == configB[j] )
			{
				return 0;	// False:  this DOES violate the Pauli exclusion principle
			}
		}
	}
	return 1;
}



void Transform_ABvecs_Into_TwoBody_Basis( Double* TwoBodyVec,
                                          PyInt   num_spin_orbs,
                                          PyInt   num_core_elec,
                                          PyInt   num_valence_elec_A,
                                          PyInt   num_valence_elec_B,
                                          Double* atomA_state,
                                          Double* atomB_state,
                                          BigInt* atomA_configs,
                                          BigInt* atomB_configs,
                                          PyInt   num_atomA_configs,
                                          PyInt   num_atomB_configs,
                                          BigInt* comboMat )
{
	BigInt num_valence_elec  = num_valence_elec_A + num_valence_elec_B;
	BigInt num_valence_orbs  = num_spin_orbs - num_core_elec;
	BigInt num_spat_orbs     = num_spin_orbs / 2;
	BigInt num_core_spat_orb = num_core_elec / 2;

	/*
	printf("TwoBodyVec         = %p \n",TwoBodyVec);
	printf("num_spin_orbs      = %d \n",num_spin_orbs);
	printf("num_core_elec      = %d \n",num_core_elec);
	printf("num_valence_elec_A = %d \n",num_valence_elec_A);
	printf("num_valence_elec_B = %d \n",num_valence_elec_B);
	printf("atomA_state        = %p \n",atomA_state);
	printf("atomB_state        = %p \n",atomB_state);
	printf("atomA_configs      = %p \n",atomA_configs);
	printf("atomB_configs      = %p \n",atomB_configs);
	printf("num_atomA_configs  = %d \n",num_atomA_configs);
	printf("num_atomB_configs  = %d \n",num_atomB_configs);
	printf("comboMat           = %p \n",comboMat);
	printf("num_valence_elec   = %d \n",num_valence_elec);
	printf("num_valence_orbs   = %d \n",num_valence_orbs);
	printf("num_spat_orbs      = %d \n",num_spat_orbs);
	printf("num_core_spat_orb  = %d \n",num_core_spat_orb);
	*/

	#pragma omp parallel for
	for(BigInt i=0; i<num_atomA_configs; i++)
	{
		for(BigInt j=0; j<num_atomB_configs; j++)
		{
			if ( no_pauli_exclusion( &atomA_configs[i*num_valence_elec_A ], &atomB_configs[j*num_valence_elec_B], \
				                          (BigInt)num_valence_elec_A,            (BigInt)num_valence_elec_B ) )
			{
				BigInt dimer_config[num_valence_elec];	// must be inside loop because of OMP

				for (BigInt k=0; k<num_valence_elec_A; k++) {dimer_config[k]                    = atomA_configs[i*num_valence_elec_A+k];}
				for (BigInt k=0; k<num_valence_elec_B; k++) {dimer_config[k+num_valence_elec_A] = atomB_configs[j*num_valence_elec_B+k];}

				for (BigInt k=0; k<num_valence_elec; k++)
				{
					if (dimer_config[k] > num_spat_orbs) {dimer_config[k] -= 2*num_core_spat_orb;}
					else                                 {dimer_config[k] -=   num_core_spat_orb;}
				}

				Double sign = get_sign(dimer_config, num_valence_elec);
				qsort( dimer_config, num_valence_elec, sizeof(BigInt), compare );

				BigInt FCIindex = configToFciIndex( dimer_config, num_valence_elec, num_valence_orbs, comboMat );

				#pragma omp atomic
				TwoBodyVec[ FCIindex ] += sign * atomA_state[i] * atomB_state[j];
			} // pauli allowed
		} // Atom B
	} // Atom A

	return;
}
