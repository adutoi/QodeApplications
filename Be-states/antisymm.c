/*   (C) Copyright 2023 Anthony Dutoi
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
#include <string.h>       // memcpy()
#include "PyC_types.h"    // PyInt, BigInt

#ifdef _OPENMP
#include <omp.h>
#endif

#include <stdio.h>        // fprintf(stderr, "...")  Can be eliminated when debugged

void antisymmetrize(Double *tensor, PyInt dim, PyInt n_dim, PyInt** old_orderings, PyInt old_n_orderings, PyInt old_len_orderings, PyInt* strides, PyInt p_0, PyInt* old_phases, PyInt indent)
    {
    for (int x=0; x<indent; x++) {printf("  ");}
    printf("p_0 = %d\n", p_0);

    PyInt len_orderings = old_len_orderings + 1;
    PyInt n_orderings   = old_n_orderings * len_orderings;

    PyInt  orderings_x[n_orderings][len_orderings];
    PyInt* orderings[n_orderings];
    for (PyInt m=0; m<n_orderings; m++) {orderings[m] = orderings_x[m];}

    PyInt* insert[n_orderings];
    PyInt  phases[n_orderings];

    PyInt new_strides[len_orderings + 1];
    for (PyInt i=0; i<len_orderings; i++) {new_strides[i] = strides[i];}
    new_strides[len_orderings] = new_strides[len_orderings-1] * dim;

    PyInt m = 0;
    for (PyInt n=0; n<old_n_orderings; n++)
        {
        PyInt phase = 1;
        for (PyInt i=len_orderings-1; i>=0; i--)
            {
            phases[m] = old_phases[n] * phase;
            phase *= -1;
            insert[m] = &(orderings[m][i]);
            orderings[m][i] = -13;
            for (PyInt j=0;   j<i;             j++) {orderings[m][j] = old_orderings[n][j];}
            for (PyInt j=i+1; j<len_orderings; j++) {orderings[m][j] = old_orderings[n][j-1];}
            m++;
            }
        }

    for (PyInt p=p_0; p<dim; p++)
        {
        for (PyInt m=0; m<n_orderings; m++) {insert[m][0] = p;}
        if (len_orderings < n_dim)
            {
            antisymmetrize(tensor, dim, n_dim, orderings, n_orderings, len_orderings, new_strides, p+1, phases, indent+1);
            }
        else
            {
            PyInt idx_0 = 0;
            for (PyInt i=0; i<len_orderings; i++) {idx_0 += orderings[0][i] * strides[i];}
            for (PyInt m=1; m<n_orderings; m++)
               {
               PyInt idx = 0;
               for (PyInt i=0; i<len_orderings; i++) {idx += orderings[m][i] * strides[i];}
               for (PyInt k=0; k<strides[0]; k++) {tensor[idx+k] = phases[m] * tensor[idx_0+k];}
               }
            }
        }

    return;
    }








