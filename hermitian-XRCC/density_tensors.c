/*   (C) Copyright 2018, 2019, 2023 Anthony D. Dutoi and Yuhong Liu
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
//#include <stdio.h>
#include "PyC_types.h"



/*
 * Given a determinant configuration (stored as an ascending-ordered list of indices of occupied orbitals),
 * these functions determine the index of that configuration within an enumeration of all such configurations.
 */

// This is the same as finding the index of a general configuration where all orbitals are active
BigInt find_valence_config_index(BigInt* val_config, BigInt n_val_elec, BigInt n_val_spin_orbs, BigInt* combinatorics)
	{
	BigInt index = 0;
	if (n_val_elec>0)	// zero electrons is sane, but this should never be called with negative number of electrons or fewer orbitals than electrons
		{
		for(BigInt n=1;  n<val_config[0]+1;  n++)
			{
			index += combinatorics[n_val_elec*n_val_spin_orbs - n];
			}
		for(BigInt i=0;  i<n_val_elec-1;  i++)
			{
			for(BigInt n=val_config[i]+2;  n<val_config[i+1]+1;  n++)
				{
				index += combinatorics[(n_val_elec-i-1)*n_val_spin_orbs - n];
				}
			}
		}
	return index;
	}

BigInt find_config_index(BigInt* config, BigInt n_elec, BigInt n_orbs, BigInt n_core, BigInt* combinatorics)
	{
	BigInt n_val_elec      =  n_elec - n_core*2;
	BigInt n_val_spin_orbs = (n_orbs - n_core)*2;
	BigInt val_config[n_val_elec];
	BigInt val_e = 0;
	for (BigInt i=0;  i<n_elec;  i++)
		{
		BigInt idx = config[i];
		if      (idx>=n_core        && idx<n_orbs  ) {val_config[val_e++] = idx -   n_core;}
		else if (idx>=n_orbs+n_core && idx<2*n_orbs) {val_config[val_e++] = idx - 2*n_core;}
		}
	return find_valence_config_index(val_config, n_val_elec, n_val_spin_orbs, combinatorics);
	}



/*
 * Given a determinant configuration (stored as an ascending-ordered list of indices of occupied orbitals),
 * these functions will modify that configuration to delete or insert a given index.  The number of swaps
 * needed to move the index to or from the end of the list is reflected in the parity, which is +1 or -1.
 * Pauli violations return zero.  The last two functions can verify that a configuration has restricted
 * core occupancy.
 */

void duplicate_config(BigInt* target_config, BigInt* source_config, BigInt n_elec)
	{
	for (BigInt e=0;  e<n_elec;  e++) {target_config[e] = source_config[e];}
	return;
	}

BigInt annihilate(BigInt a, BigInt* config, BigInt n_elec)
	{
	BigInt parity = 0;
	BigInt x = 0;
	while (x<n_elec && config[x]!=a) {x++;}		// order of tests joined by && matters!
	if (x!=n_elec)
		{
		parity = 1;
		for (BigInt y=x+1;  y<n_elec;  y++)
			{
			config[y-1] = config[y];
			parity *= -1;
			}
		}
	return parity;
	}

BigInt create(BigInt c, BigInt* config, BigInt n_elec)	// config should have length n_elec+1 to account for new electron (last slot is meaningless on input)
	{
	BigInt parity = 0;
	BigInt x = 0;
	while (x<n_elec && config[x]<c) {x++;}		// order of tests joined by && matters!
	if (x==n_elec || config[x]!=c)
		{
		parity = 1;
		for (BigInt y=n_elec;  y>x;  y--)
			{
			config[y] = config[y-1];
			parity *= -1;
			}
		config[x] = c;
		}
	return parity;
	}

BigInt core_occupied(BigInt idx, BigInt* config, BigInt n_elec)
	{
	BigInt value = 0;
	for (BigInt n=0;  n<n_elec;  n++)
		{
		if (config[n]==idx) {value = 1;}
		}
	return value;
	}

BigInt all_core_occupied(BigInt* config, BigInt n_elec, BigInt n_orbs, BigInt n_core)
	{
	BigInt value = 1;
	for (BigInt i=0;  i<n_core;  i++)
		{
		if (!core_occupied(i,        config, n_elec)) {value = 0;}
		if (!core_occupied(n_orbs+i, config, n_elec)) {value = 0;}
		}
	return value;
	}



/*
 * For a given (non-zero) combination of field operators and bra-ket configurations, the contribution to each relevant tensor is computed.
 */

void z_contract(Double storage[], BigInt parity, BigInt index, BigInt tensor_size, BigInt bra_chg_idx, BigInt ket_chg_idx, BigInt n_states[], Double* z_list[], BigInt n_configs[], BigInt P, BigInt Q)
	{
	for (BigInt I=0;  I<n_states[bra_chg_idx];  I++)
		{
		Double zI_P = z_list[bra_chg_idx][I*n_configs[bra_chg_idx] + P];
		for (BigInt J=0;  J<n_states[ket_chg_idx];  J++)
			{
			Double zJ_Q = z_list[ket_chg_idx][J*n_configs[ket_chg_idx] + Q];
			Double* tensor = storage + (I*n_states[ket_chg_idx] + J)*tensor_size;
			tensor[index] += parity * zI_P * zJ_Q;
			}
		}
	}



/*
 * These operate by looping over ket configurations and field-operator indices, then determining the index of the state produced by that combination
 */

void a_tensor(Double storage[],  PyInt  bra_chg_idx_py,  PyInt   ket_chg_idx_py,
              BigInt n_elec[],   BigInt n_states[],      Double* z_list[],         BigInt n_configs[],  BigInt* configs[],
              PyInt  n_orbs_py,  PyInt  n_core_py,       BigInt* combinatorics[],
              PyInt  n_threads_py)
	{
	BigInt bra_chg_idx = bra_chg_idx_py;	//
	BigInt ket_chg_idx = ket_chg_idx_py;	//
	BigInt n_orbs      = n_orbs_py;		// Just an artifact of PyC being introduced after this code was first written
	BigInt n_core      = n_core_py;		//
	BigInt n_threads   = n_threads_py;	//

	BigInt n_e = n_elec[ket_chg_idx];
	BigInt dim = 2*n_orbs;     // tensor dimensions have length equal to number of spin orbitals
	for (BigInt Q=0;  Q<n_configs[ket_chg_idx];  Q++)
		{
		//printf("a %d\n", Q);
		//fflush(stdout);
		BigInt* config_Q = configs[ket_chg_idx] + Q*n_e;
		BigInt config_pQ[n_e];
		for (BigInt p=0;  p<dim;  p++)
			{
			duplicate_config(config_pQ, config_Q, n_e);
			BigInt parity_pQ = annihilate(p, config_pQ, n_e);
			if (parity_pQ!=0)
				{
				if (all_core_occupied(config_pQ, n_e-1, n_orbs, n_core))
					{
					// by construction the number of electrons in config_pQ must be n_elec[bra_chg_idx]
					BigInt P = find_config_index(config_pQ, n_elec[bra_chg_idx], n_orbs, n_core, combinatorics[bra_chg_idx]);
					z_contract(storage, parity_pQ, p, dim, bra_chg_idx, ket_chg_idx, n_states, z_list, n_configs, P, Q);
					}
				}
			}
		}
	return;
	}

void c_tensor(Double storage[],  PyInt  bra_chg_idx_py,  PyInt   ket_chg_idx_py,
              BigInt n_elec[],   BigInt n_states[],      Double* z_list[],         BigInt n_configs[],  BigInt* configs[],
              PyInt  n_orbs_py,  PyInt  n_core_py,       BigInt* combinatorics[],
              PyInt  n_threads_py)
	{
	BigInt bra_chg_idx = bra_chg_idx_py;	//
	BigInt ket_chg_idx = ket_chg_idx_py;	//
	BigInt n_orbs      = n_orbs_py;		// Just an artifact of PyC being introduced after this code was first written
	BigInt n_core      = n_core_py;		//
	BigInt n_threads   = n_threads_py;	//

	BigInt n_e = n_elec[ket_chg_idx];
	BigInt dim = 2*n_orbs;     // tensor dimensions have length equal to number of spin orbitals
	for (BigInt Q=0;  Q<n_configs[ket_chg_idx];  Q++)
		{
		//printf("c %d\n", Q);
		//fflush(stdout);
		BigInt* config_Q = configs[ket_chg_idx] + Q*n_e;
		BigInt config_pQ[n_e+1];
		for (BigInt p=0;  p<dim;  p++)
			{
			duplicate_config(config_pQ, config_Q, n_e);
			BigInt parity_pQ = create(p, config_pQ, n_e);
			if (parity_pQ!=0)
				{
				if (all_core_occupied(config_pQ, n_e+1, n_orbs, n_core))
					{
					// by construction the number of electrons in config_pQ must be n_elec[bra_chg_idx]
					BigInt P = find_config_index(config_pQ, n_elec[bra_chg_idx], n_orbs, n_core, combinatorics[bra_chg_idx]);
					z_contract(storage, parity_pQ, p, dim, bra_chg_idx, ket_chg_idx, n_states, z_list, n_configs, P, Q);
					}
				}
			}
		}
	return;
	}

void ca_tensor(Double storage[],  PyInt  bra_chg_idx_py,  PyInt   ket_chg_idx_py,
               BigInt n_elec[],   BigInt n_states[],      Double* z_list[],         BigInt n_configs[],  BigInt* configs[],
               PyInt  n_orbs_py,  PyInt  n_core_py,       BigInt* combinatorics[],
               PyInt  n_threads_py)
	{
	BigInt bra_chg_idx = bra_chg_idx_py;	//
	BigInt ket_chg_idx = ket_chg_idx_py;	//
	BigInt n_orbs      = n_orbs_py;		// Just an artifact of PyC being introduced after this code was first written
	BigInt n_core      = n_core_py;		//
	BigInt n_threads   = n_threads_py;	//

	BigInt n_e = n_elec[ket_chg_idx];
	BigInt dim = 2*n_orbs;     // tensor dimensions have length equal to number of spin orbitals
	for (BigInt Q=0;  Q<n_configs[ket_chg_idx];  Q++)
		{
		//printf("ca %d\n", Q);
		//fflush(stdout);
		BigInt* config_Q = configs[ket_chg_idx] + Q*n_e;
		BigInt config_qQ[n_e];
		for (BigInt q=0;  q<dim; q++)
			{
			duplicate_config(config_qQ, config_Q, n_e);
			BigInt parity_qQ = annihilate(q, config_qQ, n_e);
			if (parity_qQ!=0)
				{
				BigInt config_pqQ[n_e];
				for (BigInt p=0;  p<dim;  p++)
					{
					duplicate_config(config_pqQ, config_qQ, n_e-1);
					BigInt parity_pqQ = create(p, config_pqQ, n_e-1) * parity_qQ;
					if (parity_pqQ!=0)
						{
						if (all_core_occupied(config_pqQ, n_e, n_orbs, n_core))
							{
							// by construction the number of electrons in config_pqQ must be n_elec[bra_chg_idx]
							BigInt P = find_config_index(config_pqQ, n_elec[bra_chg_idx], n_orbs, n_core, combinatorics[bra_chg_idx]);
							BigInt pq = p*dim + q;
							z_contract(storage, parity_pqQ, pq, dim*dim, bra_chg_idx, ket_chg_idx, n_states, z_list, n_configs, P, Q);
							}
						}
					}
				}
			}
		}
	return;
	}

void aa_tensor(Double storage[],  PyInt  bra_chg_idx_py,  PyInt   ket_chg_idx_py,
               BigInt n_elec[],   BigInt n_states[],      Double* z_list[],         BigInt n_configs[],  BigInt* configs[],
               PyInt  n_orbs_py,  PyInt  n_core_py,       BigInt* combinatorics[],
               PyInt  n_threads_py)
	{
	BigInt bra_chg_idx = bra_chg_idx_py;	//
	BigInt ket_chg_idx = ket_chg_idx_py;	//
	BigInt n_orbs      = n_orbs_py;		// Just an artifact of PyC being introduced after this code was first written
	BigInt n_core      = n_core_py;		//
	BigInt n_threads   = n_threads_py;	//

	BigInt n_e = n_elec[ket_chg_idx];
	BigInt dim = 2*n_orbs;     // tensor dimensions have length equal to number of spin orbitals
	for (BigInt Q=0;  Q<n_configs[ket_chg_idx];  Q++)
		{
		//printf("aa %d\n", Q);
		//fflush(stdout);
		BigInt* config_Q = configs[ket_chg_idx] + Q*n_e;
		BigInt config_qQ[n_e];
		for (BigInt q=0;  q<dim; q++)
			{
			duplicate_config(config_qQ, config_Q, n_e);
			BigInt parity_qQ = annihilate(q, config_qQ, n_e);
			if (parity_qQ!=0)
				{
				BigInt config_pqQ[n_e-1];
				for (BigInt p=0;  p<dim;  p++)
					{
					duplicate_config(config_pqQ, config_qQ, n_e-1);
					BigInt parity_pqQ = annihilate(p, config_pqQ, n_e-1) * parity_qQ;
					if (parity_pqQ!=0)
						{
						if (all_core_occupied(config_pqQ, n_e-2, n_orbs, n_core))
							{
							// by construction the number of electrons in config_pqQ must be n_elec[bra_chg_idx]
							BigInt P = find_config_index(config_pqQ, n_elec[bra_chg_idx], n_orbs, n_core, combinatorics[bra_chg_idx]);
							BigInt pq = p*dim + q;
							z_contract(storage, parity_pqQ, pq, dim*dim, bra_chg_idx, ket_chg_idx, n_states, z_list, n_configs, P, Q);
							}
						}
					}
				}
			}
		}
	return;
	}

void cc_tensor(Double storage[],  PyInt  bra_chg_idx_py,  PyInt   ket_chg_idx_py,
               BigInt n_elec[],   BigInt n_states[],      Double* z_list[],         BigInt n_configs[],  BigInt* configs[],
               PyInt  n_orbs_py,  PyInt  n_core_py,       BigInt* combinatorics[],
               PyInt  n_threads_py)
	{
	BigInt bra_chg_idx = bra_chg_idx_py;	//
	BigInt ket_chg_idx = ket_chg_idx_py;	//
	BigInt n_orbs      = n_orbs_py;		// Just an artifact of PyC being introduced after this code was first written
	BigInt n_core      = n_core_py;		//
	BigInt n_threads   = n_threads_py;	//

	BigInt n_e = n_elec[ket_chg_idx];
	BigInt dim = 2*n_orbs;     // tensor dimensions have length equal to number of spin orbitals
	for (BigInt Q=0;  Q<n_configs[ket_chg_idx];  Q++)
		{
		//printf("cc %d\n", Q);
		//fflush(stdout);
		BigInt* config_Q = configs[ket_chg_idx] + Q*n_e;
		BigInt config_qQ[n_e+1];
		for (BigInt q=0;  q<dim; q++)
			{
			duplicate_config(config_qQ, config_Q, n_e);
			BigInt parity_qQ = create(q, config_qQ, n_e);
			if (parity_qQ!=0)
				{
				BigInt config_pqQ[n_e+2];
				for (BigInt p=0;  p<dim;  p++)
					{
					duplicate_config(config_pqQ, config_qQ, n_e+1);
					BigInt parity_pqQ = create(p, config_pqQ, n_e+1) * parity_qQ;
					if (parity_pqQ!=0)
						{
						if (all_core_occupied(config_pqQ, n_e+2, n_orbs, n_core))
							{
							// by construction the number of electrons in config_pqQ must be n_elec[bra_chg_idx]
							BigInt P = find_config_index(config_pqQ, n_elec[bra_chg_idx], n_orbs, n_core, combinatorics[bra_chg_idx]);
							BigInt pq = p*dim + q;
							z_contract(storage, parity_pqQ, pq, dim*dim, bra_chg_idx, ket_chg_idx, n_states, z_list, n_configs, P, Q);
							}
						}
					}
				}
			}
		}
	return;
	}

void caa_tensor(Double storage[],  PyInt  bra_chg_idx_py,  PyInt   ket_chg_idx_py,
                BigInt n_elec[],   BigInt n_states[],      Double* z_list[],         BigInt n_configs[],  BigInt* configs[],
                PyInt  n_orbs_py,  PyInt  n_core_py,       BigInt* combinatorics[],
                PyInt  n_threads_py)
	{
	BigInt bra_chg_idx = bra_chg_idx_py;	//
	BigInt ket_chg_idx = ket_chg_idx_py;	//
	BigInt n_orbs      = n_orbs_py;		// Just an artifact of PyC being introduced after this code was first written
	BigInt n_core      = n_core_py;		//
	BigInt n_threads   = n_threads_py;	//

	BigInt n_e = n_elec[ket_chg_idx];
	BigInt dim = 2*n_orbs;     // tensor dimensions have length equal to number of spin orbitals
	for (BigInt Q=0;  Q<n_configs[ket_chg_idx];  Q++)
		{
		//printf("caa %d\n", Q);
		//fflush(stdout);
		BigInt* config_Q = configs[ket_chg_idx] + Q*n_e;
		BigInt config_rQ[n_e];
		for (BigInt r=0;  r<dim; r++)
			{
			duplicate_config(config_rQ, config_Q, n_e);
			BigInt parity_rQ = annihilate(r, config_rQ, n_e);
			if (parity_rQ!=0)
				{
				BigInt config_qrQ[n_e-1];
				for (BigInt q=0;  q<dim; q++)
					{
					duplicate_config(config_qrQ, config_rQ, n_e-1);
					BigInt parity_qrQ = annihilate(q, config_qrQ, n_e-1) * parity_rQ;
					if (parity_qrQ!=0)
						{
						BigInt config_pqrQ[n_e-1];
						for (BigInt p=0;  p<dim;  p++)
							{
							duplicate_config(config_pqrQ, config_qrQ, n_e-2);
							BigInt parity_pqrQ = create(p, config_pqrQ, n_e-2) * parity_qrQ;
							if (parity_pqrQ!=0)
								{
								if (all_core_occupied(config_pqrQ, n_e-1, n_orbs, n_core))
									{
									// by construction the number of electrons in config_pqrQ must be n_elec[bra_chg_idx]
									BigInt P = find_config_index(config_pqrQ, n_elec[bra_chg_idx], n_orbs, n_core, combinatorics[bra_chg_idx]);
									BigInt pqr = (p*dim + q)*dim + r;
									z_contract(storage, parity_pqrQ, pqr, dim*dim*dim, bra_chg_idx, ket_chg_idx, n_states, z_list, n_configs, P, Q);
									}
								//else {printf("\n");}
								}
							//else {printf("\n");}
							}
						}
					//else {printf("\n");}
					}
				}
			//else {printf("\n");}
			}
		}
	return;
	}

void cca_tensor(Double storage[],  PyInt  bra_chg_idx_py,  PyInt   ket_chg_idx_py,
                BigInt n_elec[],   BigInt n_states[],      Double* z_list[],         BigInt n_configs[],  BigInt* configs[],
                PyInt  n_orbs_py,  PyInt  n_core_py,       BigInt* combinatorics[],
                PyInt  n_threads_py)
	{
	BigInt bra_chg_idx = bra_chg_idx_py;	//
	BigInt ket_chg_idx = ket_chg_idx_py;	//
	BigInt n_orbs      = n_orbs_py;		// Just an artifact of PyC being introduced after this code was first written
	BigInt n_core      = n_core_py;		//
	BigInt n_threads   = n_threads_py;	//

	BigInt n_e = n_elec[ket_chg_idx];
	BigInt dim = 2*n_orbs;     // tensor dimensions have length equal to number of spin orbitals
	for (BigInt Q=0;  Q<n_configs[ket_chg_idx];  Q++)
		{
		//printf("cca %d\n", Q);
		//fflush(stdout);
		BigInt* config_Q = configs[ket_chg_idx] + Q*n_e;
		BigInt config_rQ[n_e];
		for (BigInt r=0;  r<dim; r++)
			{
			duplicate_config(config_rQ, config_Q, n_e);
			BigInt parity_rQ = annihilate(r, config_rQ, n_e);
			if (parity_rQ!=0)
				{
				BigInt config_qrQ[n_e];
				for (BigInt q=0;  q<dim; q++)
					{
					duplicate_config(config_qrQ, config_rQ, n_e-1);
					BigInt parity_qrQ = create(q, config_qrQ, n_e-1) * parity_rQ;
					if (parity_qrQ!=0)
						{
						BigInt config_pqrQ[n_e+1];
						for (BigInt p=0;  p<dim;  p++)
							{
							duplicate_config(config_pqrQ, config_qrQ, n_e);
							BigInt parity_pqrQ = create(p, config_pqrQ, n_e) * parity_qrQ;
							if (parity_pqrQ!=0)
								{
								if (all_core_occupied(config_pqrQ, n_e+1, n_orbs, n_core))
									{
									// by construction the number of electrons in config_pqrQ must be n_elec[bra_chg_idx]
									BigInt P = find_config_index(config_pqrQ, n_elec[bra_chg_idx], n_orbs, n_core, combinatorics[bra_chg_idx]);
									BigInt pqr = (p*dim + q)*dim + r;
									z_contract(storage, parity_pqrQ, pqr, dim*dim*dim, bra_chg_idx, ket_chg_idx, n_states, z_list, n_configs, P, Q);
									}
								}
							}
						}
					}
				}
			}
		}
	return;
	}

void ccaa_tensor(Double storage[],  PyInt  bra_chg_idx_py,  PyInt   ket_chg_idx_py,
                 BigInt n_elec[],   BigInt n_states[],      Double* z_list[],         BigInt n_configs[],  BigInt* configs[],
                 PyInt  n_orbs_py,  PyInt  n_core_py,       BigInt* combinatorics[],
                 PyInt  n_threads_py)
	{
	BigInt bra_chg_idx = bra_chg_idx_py;	//
	BigInt ket_chg_idx = ket_chg_idx_py;	//
	BigInt n_orbs      = n_orbs_py;		// Just an artifact of PyC being introduced after this code was first written
	BigInt n_core      = n_core_py;		//
	BigInt n_threads   = n_threads_py;	//

	BigInt n_e = n_elec[ket_chg_idx];
	BigInt dim = 2*n_orbs;     // tensor dimensions have length equal to number of spin orbitals
	for (BigInt Q=0;  Q<n_configs[ket_chg_idx];  Q++)
		{
		//printf("ccaa %d\n", Q);
		//fflush(stdout);
		BigInt* config_Q = configs[ket_chg_idx] + Q*n_e;
		BigInt config_sQ[n_e];
		for (BigInt s=0;  s<dim; s++)
			{
			duplicate_config(config_sQ, config_Q, n_e);
			BigInt parity_sQ = annihilate(s, config_sQ, n_e);
			if (parity_sQ!=0)
				{
				BigInt config_rsQ[n_e-1];
				for (BigInt r=0;  r<dim; r++)
					{
					duplicate_config(config_rsQ, config_sQ, n_e-1);
					BigInt parity_rsQ = annihilate(r, config_rsQ, n_e-1) * parity_sQ;
					if (parity_rsQ!=0)
						{
						BigInt config_qrsQ[n_e-1];
						for (BigInt q=0;  q<dim; q++)
							{
							duplicate_config(config_qrsQ, config_rsQ, n_e-2);
							BigInt parity_qrsQ = create(q, config_qrsQ, n_e-2) * parity_rsQ;
							if (parity_qrsQ!=0)
								{
								BigInt config_pqrsQ[n_e];
								for (BigInt p=0;  p<dim;  p++)
									{
									duplicate_config(config_pqrsQ, config_qrsQ, n_e-1);
									BigInt parity_pqrsQ = create(p, config_pqrsQ, n_e-1) * parity_qrsQ;
									if (parity_pqrsQ!=0)
										{
										if (all_core_occupied(config_pqrsQ, n_e, n_orbs, n_core))
											{
											// by construction the number of electrons in config_pqrsQ must be n_elec[bra_chg_idx]
											BigInt P = find_config_index(config_pqrsQ, n_elec[bra_chg_idx], n_orbs, n_core, combinatorics[bra_chg_idx]);
											BigInt pqrs = ((p*dim + q)*dim + r)*dim + s;
											z_contract(storage, parity_pqrsQ, pqrs, dim*dim*dim*dim, bra_chg_idx, ket_chg_idx, n_states, z_list, n_configs, P, Q);
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	return;
	}

// Added in Apr. 2023 (ADD) -----------------------------------------------------------------------------------------------------

void caaa_tensor(Double storage[],  PyInt  bra_chg_idx_py,  PyInt   ket_chg_idx_py,
                 BigInt n_elec[],   BigInt n_states[],      Double* z_list[],         BigInt n_configs[],  BigInt* configs[],
                 PyInt  n_orbs_py,  PyInt  n_core_py,       BigInt* combinatorics[],
                 PyInt  n_threads_py)
	{
	BigInt bra_chg_idx = bra_chg_idx_py;	//
	BigInt ket_chg_idx = ket_chg_idx_py;	//
	BigInt n_orbs      = n_orbs_py;		// Just an artifact of PyC being introduced after this code was first written
	BigInt n_core      = n_core_py;		//
	BigInt n_threads   = n_threads_py;	//

	BigInt n_e = n_elec[ket_chg_idx];
	BigInt dim = 2*n_orbs;     // tensor dimensions have length equal to number of spin orbitals
	for (BigInt Q=0;  Q<n_configs[ket_chg_idx];  Q++)
		{
		//printf("caaa %d\n", Q);
		//fflush(stdout);
		BigInt* config_Q = configs[ket_chg_idx] + Q*n_e;
		BigInt config_sQ[n_e];
		for (BigInt s=0;  s<dim; s++)
			{
			duplicate_config(config_sQ, config_Q, n_e);
			BigInt parity_sQ = annihilate(s, config_sQ, n_e);
			if (parity_sQ!=0)
				{
				BigInt config_rsQ[n_e-1];
				for (BigInt r=0;  r<dim; r++)
					{
					duplicate_config(config_rsQ, config_sQ, n_e-1);
					BigInt parity_rsQ = annihilate(r, config_rsQ, n_e-1) * parity_sQ;
					if (parity_rsQ!=0)
						{
						BigInt config_qrsQ[n_e-2];
						for (BigInt q=0;  q<dim; q++)
							{
							duplicate_config(config_qrsQ, config_rsQ, n_e-2);
							BigInt parity_qrsQ = annihilate(q, config_qrsQ, n_e-2) * parity_rsQ;
							if (parity_qrsQ!=0)
								{
								BigInt config_pqrsQ[n_e-2];
								for (BigInt p=0;  p<dim;  p++)
									{
									duplicate_config(config_pqrsQ, config_qrsQ, n_e-3);
									BigInt parity_pqrsQ = create(p, config_pqrsQ, n_e-3) * parity_qrsQ;
									if (parity_pqrsQ!=0)
										{
										if (all_core_occupied(config_pqrsQ, n_e-2, n_orbs, n_core))
											{
											// by construction the number of electrons in config_pqrsQ must be n_elec[bra_chg_idx]
											BigInt P = find_config_index(config_pqrsQ, n_elec[bra_chg_idx], n_orbs, n_core, combinatorics[bra_chg_idx]);
											BigInt pqrs = ((p*dim + q)*dim + r)*dim + s;
											z_contract(storage, parity_pqrsQ, pqrs, dim*dim*dim*dim, bra_chg_idx, ket_chg_idx, n_states, z_list, n_configs, P, Q);
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	return;
	}

void ccca_tensor(Double storage[],  PyInt  bra_chg_idx_py,  PyInt   ket_chg_idx_py,
                 BigInt n_elec[],   BigInt n_states[],      Double* z_list[],         BigInt n_configs[],  BigInt* configs[],
                 PyInt  n_orbs_py,  PyInt  n_core_py,       BigInt* combinatorics[],
                 PyInt  n_threads_py)
	{
	BigInt bra_chg_idx = bra_chg_idx_py;	//
	BigInt ket_chg_idx = ket_chg_idx_py;	//
	BigInt n_orbs      = n_orbs_py;		// Just an artifact of PyC being introduced after this code was first written
	BigInt n_core      = n_core_py;		//
	BigInt n_threads   = n_threads_py;	//

	BigInt n_e = n_elec[ket_chg_idx];
	BigInt dim = 2*n_orbs;     // tensor dimensions have length equal to number of spin orbitals
	for (BigInt Q=0;  Q<n_configs[ket_chg_idx];  Q++)
		{
		//printf("ccca %d\n", Q);
		//fflush(stdout);
		BigInt* config_Q = configs[ket_chg_idx] + Q*n_e;
		BigInt config_sQ[n_e];
		for (BigInt s=0;  s<dim; s++)
			{
			duplicate_config(config_sQ, config_Q, n_e);
			BigInt parity_sQ = annihilate(s, config_sQ, n_e);
			if (parity_sQ!=0)
				{
				BigInt config_rsQ[n_e];
				for (BigInt r=0;  r<dim; r++)
					{
					duplicate_config(config_rsQ, config_sQ, n_e-1);
					BigInt parity_rsQ = create(r, config_rsQ, n_e-1) * parity_sQ;
					if (parity_rsQ!=0)
						{
						BigInt config_qrsQ[n_e+1];
						for (BigInt q=0;  q<dim; q++)
							{
							duplicate_config(config_qrsQ, config_rsQ, n_e);
							BigInt parity_qrsQ = create(q, config_qrsQ, n_e) * parity_rsQ;
							if (parity_qrsQ!=0)
								{
								BigInt config_pqrsQ[n_e+2];
								for (BigInt p=0;  p<dim;  p++)
									{
									duplicate_config(config_pqrsQ, config_qrsQ, n_e+1);
									BigInt parity_pqrsQ = create(p, config_pqrsQ, n_e+1) * parity_qrsQ;
									if (parity_pqrsQ!=0)
										{
										if (all_core_occupied(config_pqrsQ, n_e+2, n_orbs, n_core))
											{
											// by construction the number of electrons in config_pqrsQ must be n_elec[bra_chg_idx]
											BigInt P = find_config_index(config_pqrsQ, n_elec[bra_chg_idx], n_orbs, n_core, combinatorics[bra_chg_idx]);
											BigInt pqrs = ((p*dim + q)*dim + r)*dim + s;
											z_contract(storage, parity_pqrsQ, pqrs, dim*dim*dim*dim, bra_chg_idx, ket_chg_idx, n_states, z_list, n_configs, P, Q);
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	return;
	}

void cccaa_tensor(Double storage[],  PyInt  bra_chg_idx_py,  PyInt   ket_chg_idx_py,
                  BigInt n_elec[],   BigInt n_states[],      Double* z_list[],         BigInt n_configs[],  BigInt* configs[],
                  PyInt  n_orbs_py,  PyInt  n_core_py,       BigInt* combinatorics[],
                  PyInt  n_threads_py)
	{
	BigInt bra_chg_idx = bra_chg_idx_py;	//
	BigInt ket_chg_idx = ket_chg_idx_py;	//
	BigInt n_orbs      = n_orbs_py;		// Just an artifact of PyC being introduced after this code was first written
	BigInt n_core      = n_core_py;		//
	BigInt n_threads   = n_threads_py;	//

	BigInt n_e = n_elec[ket_chg_idx];
	BigInt dim = 2*n_orbs;     // tensor dimensions have length equal to number of spin orbitals
	for (BigInt Q=0;  Q<n_configs[ket_chg_idx];  Q++)
		{
		//printf("cccaa %d\n", Q);
		//fflush(stdout);
		BigInt* config_Q = configs[ket_chg_idx] + Q*n_e;
		BigInt config_sQ[n_e];
		for (BigInt s=0;  s<dim; s++)
			{
			duplicate_config(config_sQ, config_Q, n_e);
			BigInt parity_sQ = annihilate(s, config_sQ, n_e);
			if (parity_sQ!=0)
				{
				BigInt config_rsQ[n_e-1];
				for (BigInt r=0;  r<dim; r++)
					{
					duplicate_config(config_rsQ, config_sQ, n_e-1);
					BigInt parity_rsQ = annihilate(r, config_rsQ, n_e-1) * parity_sQ;
					if (parity_rsQ!=0)
						{
						BigInt config_qrsQ[n_e-1];
						for (BigInt q=0;  q<dim; q++)
							{
							duplicate_config(config_qrsQ, config_rsQ, n_e-2);
							BigInt parity_qrsQ = create(q, config_qrsQ, n_e-2) * parity_rsQ;
							if (parity_qrsQ!=0)
								{
								BigInt config_pqrsQ[n_e];
								for (BigInt p=0;  p<dim;  p++)
									{
									duplicate_config(config_pqrsQ, config_qrsQ, n_e-1);
									BigInt parity_pqrsQ = create(p, config_pqrsQ, n_e-1) * parity_qrsQ;
									if (parity_pqrsQ!=0)
										{
										BigInt config_opqrsQ[n_e+1];
										for (BigInt o=0;  o<dim;  o++)
											{
											duplicate_config(config_opqrsQ, config_pqrsQ, n_e);
											BigInt parity_opqrsQ = create(o, config_opqrsQ, n_e) * parity_pqrsQ;
											if (parity_opqrsQ!=0)
												{
												if (all_core_occupied(config_opqrsQ, n_e+1, n_orbs, n_core))
													{
													// by construction the number of electrons in config_opqrsQ must be n_elec[bra_chg_idx]
													BigInt P = find_config_index(config_opqrsQ, n_elec[bra_chg_idx], n_orbs, n_core, combinatorics[bra_chg_idx]);
													BigInt opqrs = (((o*dim + p)*dim + q)*dim + r)*dim + s;
													z_contract(storage, parity_opqrsQ, opqrs, dim*dim*dim*dim*dim, bra_chg_idx, ket_chg_idx, n_states, z_list, n_configs, P, Q);
													}
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	return;
	}

void ccaaa_tensor(Double storage[],  PyInt  bra_chg_idx_py,  PyInt   ket_chg_idx_py,
                  BigInt n_elec[],   BigInt n_states[],      Double* z_list[],         BigInt n_configs[],  BigInt* configs[],
                  PyInt  n_orbs_py,  PyInt  n_core_py,       BigInt* combinatorics[],
                  PyInt  n_threads_py)
	{
	BigInt bra_chg_idx = bra_chg_idx_py;	//
	BigInt ket_chg_idx = ket_chg_idx_py;	//
	BigInt n_orbs      = n_orbs_py;		// Just an artifact of PyC being introduced after this code was first written
	BigInt n_core      = n_core_py;		//
	BigInt n_threads   = n_threads_py;	//

	BigInt n_e = n_elec[ket_chg_idx];
	BigInt dim = 2*n_orbs;     // tensor dimensions have length equal to number of spin orbitals
	for (BigInt Q=0;  Q<n_configs[ket_chg_idx];  Q++)
		{
		//printf("ccaaa %d\n", Q);
		//fflush(stdout);
		BigInt* config_Q = configs[ket_chg_idx] + Q*n_e;
		BigInt config_sQ[n_e];
		for (BigInt s=0;  s<dim; s++)
			{
			duplicate_config(config_sQ, config_Q, n_e);
			BigInt parity_sQ = annihilate(s, config_sQ, n_e);
			if (parity_sQ!=0)
				{
				BigInt config_rsQ[n_e-1];
				for (BigInt r=0;  r<dim; r++)
					{
					duplicate_config(config_rsQ, config_sQ, n_e-1);
					BigInt parity_rsQ = annihilate(r, config_rsQ, n_e-1) * parity_sQ;
					if (parity_rsQ!=0)
						{
						BigInt config_qrsQ[n_e-2];
						for (BigInt q=0;  q<dim; q++)
							{
							duplicate_config(config_qrsQ, config_rsQ, n_e-2);
							BigInt parity_qrsQ = annihilate(q, config_qrsQ, n_e-2) * parity_rsQ;
							if (parity_qrsQ!=0)
								{
								BigInt config_pqrsQ[n_e-2];
								for (BigInt p=0;  p<dim;  p++)
									{
									duplicate_config(config_pqrsQ, config_qrsQ, n_e-3);
									BigInt parity_pqrsQ = create(p, config_pqrsQ, n_e-3) * parity_qrsQ;
									if (parity_pqrsQ!=0)
										{
										BigInt config_opqrsQ[n_e-1];
										for (BigInt o=0;  o<dim;  o++)
											{
											duplicate_config(config_opqrsQ, config_pqrsQ, n_e-2);
											BigInt parity_opqrsQ = create(o, config_opqrsQ, n_e-2) * parity_pqrsQ;
											if (parity_opqrsQ!=0)
												{
												if (all_core_occupied(config_opqrsQ, n_e-1, n_orbs, n_core))
													{
													// by construction the number of electrons in config_opqrsQ must be n_elec[bra_chg_idx]
													BigInt P = find_config_index(config_opqrsQ, n_elec[bra_chg_idx], n_orbs, n_core, combinatorics[bra_chg_idx]);
													BigInt opqrs = (((o*dim + p)*dim + q)*dim + r)*dim + s;
													z_contract(storage, parity_opqrsQ, opqrs, dim*dim*dim*dim*dim, bra_chg_idx, ket_chg_idx, n_states, z_list, n_configs, P, Q);
													}
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	return;
	}
