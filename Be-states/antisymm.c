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

// This takes a tensor with an arbitrary number of axes (all of the same length), whereby the only
// meaningful elements are those where latter indices all have larger values than any of the former.
// The "elements" of the tensor may themselves be vectors.  When this function runs, it completes
// the tensor in an antisymmetric way by copying phased versions of the meaningful values.
//
void antisymmetrize(Double* tensor,               // the input/output tensor
                    PyInt   num_axes,             // the number of tensor axes
                    PyInt   len_axis,             // the length of the axes
                    PyInt*  strides,              // the vector length of the "elements" of the tensor (given as a reference!))
                    PyInt   p_0,                  // for recursive use.  0 on first input
                    PyInt   old_len_orderings,    // for recursive use.  0 on first input
                    PyInt   old_num_orderings,    // for recursive use.  1 on first input
                    PyInt** old_orderings,        // for recursive use.  NULL on first input
                    PyInt*  old_phases)           // for recursive use.  [1] on first input
    {
    PyInt len_orderings = old_len_orderings + 1;                // Dimensions of the array that holds ...
    PyInt num_orderings = old_num_orderings * len_orderings;    // ... different ordering of fixed indices.

    PyInt new_strides[len_orderings+1];                                       // This array translates multidimensional ...
    for (PyInt i=len_orderings; i>0; i--) {new_strides[i] = strides[i-1];}    // ... indices to linear index of 1-D array.
    new_strides[0] = strides[0] * len_axis;                                   // For API convenience, it is one iteration ahead.

    PyInt  orderings_x[num_orderings][len_orderings];                         // All this to keep pointers ...
    PyInt* orderings[num_orderings];                                          // ... compatible without a malloc call.
    for (PyInt m=0; m<num_orderings; m++) {orderings[m] = orderings_x[m];}    // Storage for the orderings of fixed indices.

    PyInt  phases[num_orderings];    // Phase associated with each ordering (relative to indices in ascending order.
    PyInt* insert[num_orderings];    // Keep track of places where new variable index (looped over in this layer) will go.
    PyInt m = 0;    // running index for new orderings
    for (PyInt n=0; n<old_num_orderings; n++)    // For each of the old orderings ...
        {
        PyInt phase = 1;
        for (PyInt i=len_orderings-1; i>=0; i--)    // ... insert the new index in each possible location.
            {
            phases[m] = phase * old_phases[n];
            phase *= -1;
            insert[m] = &(orderings[m][i]);         // can efficiently fill in these blanks later
            for (PyInt j=0;   j<i;             j++) {orderings[m][j] = old_orderings[n][j];}      // These parts are constant during ...
            for (PyInt j=i+1; j<len_orderings; j++) {orderings[m][j] = old_orderings[n][j-1];}    // ... looping below.
            m++;
            }
        }

    for (PyInt p=p_0; p<len_axis; p++)    // loop over the next index
        {
        for (PyInt m=0; m<num_orderings; m++) {insert[m][0] = p;}    // fill in its location in the orderings
        if (len_orderings < num_axes)     // If there is another index, keep going ...
            {
            antisymmetrize(tensor, num_axes, len_axis, new_strides, p+1, len_orderings, num_orderings, orderings, phases);
            }
        else                              // ... otherwise it is time to do the copying finally.
            {
            printf("==========\n");
            PyInt idx_0 = 0;                                                                    // Build the linear index ...
            for (PyInt i=0; i<len_orderings; i++) {printf("%7d", strides[i]);}
            printf("\n");
            for (PyInt i=0; i<len_orderings; i++) {printf("%7d", orderings[0][i]);
                idx_0 += orderings[0][i] * strides[i];}      // ... for ascending-order case.
            printf("   ->\n");
            for (PyInt m=1; m<num_orderings; m++)    // loop over other arrangements of the indices
                {
                PyInt idx = 0;                                                                  // Build the linear index ...
                for (PyInt i=0; i<len_orderings; i++) {printf("%7d", orderings[m][i]);
                    idx += orderings[m][i] * strides[i];}    // ... for the alternate arrangement.
                printf("   x %2d  (%d)\n", phases[m], strides[len_orderings-1]);
                //for (PyInt k=0; k<strides[len_orderings-1]; k++) {tensor[idx+k] = phases[m] * tensor[idx_0+k];}    // Fill in redundant elements (to within a phase).
                }
            printf("==========\n");
            }
        }

    return;
    }
