/*   (C) Copyright 2018, 2019 Yuhong Liu Anthony Dutoi
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

#include <stdlib.h>    // need abs
#include <math.h>
#include "omp.h"
#include "PyC_types.h"



Double analyze_sign(BigInt* diff_location1, BigInt* diff_location2, BigInt num_loc)
{
	Double sign = 1.0;
	for (BigInt i=0; i<num_loc; i++)
	{
		sign *= pow( -1.0, abs(diff_location1[i] - diff_location2[i]) );
	}
	return sign;
}


BigInt different_orb_info(BigInt  num_elec,       BigInt* bra_state,      BigInt* ket_state, \
	                         BigInt* diff_location1, BigInt* diff_location2, BigInt* diff_orbs1, BigInt* diff_orbs2)
{
	BigInt num_diff_orb = 0;
	BigInt count, found_in_ket, found_in_bra;

	// for (BigInt i=0; i<num_elec; i++)
	// {
	// 	printf("%lld ", bra_state[i]);
	// }
	// printf(" -- ");
	// for (BigInt i=0; i<num_elec; i++)
	// {
	// 	printf("%lld ", ket_state[i]);
	// }
	// puts(" ");

	count = 0;
	for (BigInt i=0; i<num_elec; i++)
	{
		found_in_ket = 0;
		for (BigInt j=0; j<num_elec; j++)
		{
			if (bra_state[i] == ket_state[j])
			{
				found_in_ket = 1;
			}
		}
		// printf("found in ket = %lld; i = %lld; j= %lld\n", found_in_ket,i,j);
		if (!found_in_ket)
		{
			num_diff_orb++;
			if (num_diff_orb < 3)
			{
				diff_location1[count] = i;
				diff_orbs1[count]     = bra_state[i];
				count++;
			}
			else
			{
				return -1;
			}
		}
	}

	count = 0;
	for (BigInt i=0; i<num_elec; i++)
	{
		found_in_bra = 0;
		for (BigInt j=0; j<num_elec; j++)
		{
			if (bra_state[j] == ket_state[i])
			{
				found_in_bra = 1;
			}
		}
		if (!found_in_bra)
		{
			diff_location2[count] = i;
			diff_orbs2[count]     = ket_state[i];
			count++;
		}
	}
	return num_diff_orb;
}



Double generic_hamiltonian(BigInt* bra_state, BigInt* ket_state, BigInt num_spin_orbs, BigInt num_elec, Double* h_mat, Double* V_mat)
{
	Double sign;
	BigInt diff_location1[2], diff_location2[2], diff_orbs1[2], diff_orbs2[2];
	BigInt num_diff_orb = different_orb_info(num_elec, bra_state, ket_state, diff_location1, diff_location2, diff_orbs1, diff_orbs2);

	// printf("-------------    num_diff_orb = %lld   --------------\n", num_diff_orb );
	// for (BigInt i=0; i<num_diff_orb; i++)
	// {
	// 	printf("bra[%lld] = %lld \n", diff_location1[i], diff_orbs1[i]);
	// }

	// for (BigInt i=0; i<num_diff_orb; i++)
	// {
	// 	printf("ket[%lld] = %lld \n", diff_location2[i], diff_orbs2[i]);
	// }

	if (num_diff_orb == -1)
	{
		// puts("zero term");
		return 0.0;
	}
	else if (num_diff_orb == 2)
	{
		// puts("two diffs");
		sign = analyze_sign(diff_location1, diff_location2, num_diff_orb);
		return sign * ( V_mat[ (diff_orbs1[0]*num_spin_orbs+diff_orbs1[1])*num_spin_orbs*num_spin_orbs + diff_orbs2[0]*num_spin_orbs+diff_orbs2[1] ] -\
			            V_mat[ (diff_orbs1[0]*num_spin_orbs+diff_orbs1[1])*num_spin_orbs*num_spin_orbs + diff_orbs2[1]*num_spin_orbs+diff_orbs2[0] ] );
	}
	else if (num_diff_orb == 1)
	{
		// puts("one diffs");
		sign = analyze_sign(diff_location1, diff_location2, num_diff_orb);
		Double h_value = h_mat[ diff_orbs1[0] * num_spin_orbs + diff_orbs2[0] ];
		for (BigInt i=0; i<num_elec; i++)
		{
			h_value += ( V_mat[ (diff_orbs1[0]*num_spin_orbs+ket_state[i])*num_spin_orbs*num_spin_orbs + diff_orbs2[0]*num_spin_orbs + ket_state[i]  ] -\
			             V_mat[ (diff_orbs1[0]*num_spin_orbs+ket_state[i])*num_spin_orbs*num_spin_orbs + ket_state[i] *num_spin_orbs + diff_orbs2[0] ] );
		}
		return sign * h_value;	
	}
	else if (num_diff_orb == 0)
	{
		// puts("no diffs");
		Double h_value = 0.0;
		for (BigInt i=0; i<num_elec; i++)
		{
			h_value += h_mat[ ket_state[i]*(num_spin_orbs+1) ];
		}
		for (BigInt i=0; i<num_elec; i++)
		{
			for (BigInt j=0; j<num_elec; j++)
			{
				h_value += 0.5 * ( V_mat[ (ket_state[i]*num_spin_orbs+ket_state[j])*num_spin_orbs*num_spin_orbs + ket_state[i]*num_spin_orbs + ket_state[j] ] -\
			                       V_mat[ (ket_state[i]*num_spin_orbs+ket_state[j])*num_spin_orbs*num_spin_orbs + ket_state[j]*num_spin_orbs + ket_state[i] ] );
			}
		}
		return h_value;
	}
}


void generic_fci(PyInt num_fci_states, BigInt* fci_states, PyInt num_spin_orbs, PyInt num_elec, Double* h_mat, Double* V_mat, Double* FCI_H)
{
    omp_set_dynamic(0);
    omp_set_num_threads(64);

    #pragma omp parallel for
	for (BigInt i=0; i<num_fci_states; i++)
	{
		for(BigInt j=0; j<num_fci_states; j++)  // real symmetric. Compute Upper Right Matrix
		{
	        #pragma omp atomic
			FCI_H[i * num_fci_states + j] += generic_hamiltonian(&fci_states[i*num_elec], &fci_states[j*num_elec], (BigInt)num_spin_orbs, (BigInt)num_elec, h_mat, V_mat);
		}
	}
}
