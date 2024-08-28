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
#include <stdlib.h>       // malloc(), free()
#include "PyC_types.h"    // PyInt, Double

// This takes a tensor with an arbitrary number of axes all of the same length, where the
// "elements" of the tensor may themselves be vectors.  It antisymmetrizes the tensor using
// the input values that are present for all permutations of the indices (so it would give
// zero if the input were a symmetric tensor).  In the standard case, where only one permutational
// "wedge" is non-zero, this would then increase the Frobenius norm of the tensor by a factor of
// the order of the permutational antisymmetry (2 for 2 indices, 6 for 3 indices, etc).
// The "undo" argument effectively implements an inverse (one choice of an infinite
// number of possibilities which will result in the same tensor if again antisymmetrized,
// implying a decrease in the Frobenius norm by the same factor).  It zeros out all
// out all but one permutational wedge (would be faster to write separate code with no phases,
// but more worried about human time than computer time right now).
//
void antisymmetrize_recur(Double** tensors,              // the input/output tensors
                          PyInt    num_tensors,          // how many tensors (of identical shape) we are simultaneously antisymmetrizing
                          PyInt    num_axes,             // the number of tensor axes
                          PyInt    len_axis,             // the length of the axes
                          PyInt    undo,                 // rather than copy phased values, replace those values with zero (presuming they are not)
                          PyInt*   strides,              // the vector length of the "elements" of the tensor (given as a reference!)
                          PyInt    p_0,                  // for recursive use.  0 on first input
                          PyInt    old_len_orderings,    // for recursive use.  0 on first input
                          PyInt    old_num_orderings,    // for recursive use.  1 on first input
                          PyInt**  old_orderings,        // for recursive use.  NULL on first input
                          PyInt*   old_phases)           // for recursive use.  [1] on first input
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
            antisymmetrize_recur(tensors, num_tensors, num_axes, len_axis, undo, new_strides, p+1, len_orderings, num_orderings, orderings, phases);
            }
        else                              // ... otherwise it is time to do the copying finally.
            {
            PyInt idx_0 = 0;                                                                    // Build the linear index ...
            for (PyInt i=0; i<len_orderings; i++) {idx_0 += orderings[0][i] * strides[i];}      // ... for ascending-order case.
            if (undo == 0)
                {
                for (PyInt m=1; m<num_orderings; m++)    // loop over other arrangements of the indices
                    {
                    PyInt idx = 0;                                                                  // Build the linear index ...
                    for (PyInt i=0; i<len_orderings; i++) {idx += orderings[m][i] * strides[i];}    // ... for the alternate arrangements.
                    for (PyInt t=0; t<num_tensors; t++)
                        {
                        for (PyInt k=0; k<strides[len_orderings-1]; k++)
                            {
                            tensors[t][idx_0+k] += phases[m] * tensors[t][idx+k];    // Build antisymmetrized value at ascending-indices location
                            }
                        }
                    }
                for (PyInt m=1; m<num_orderings; m++)    // loop over other arrangements of the indices
                    {
                    PyInt idx = 0;                                                                  // Build the linear index ...
                    for (PyInt i=0; i<len_orderings; i++) {idx += orderings[m][i] * strides[i];}    // ... for the alternate arrangements.
                    for (PyInt t=0; t<num_tensors; t++)
                        {
                        for (PyInt k=0; k<strides[len_orderings-1]; k++)
                            {
                            tensors[t][idx+k] = phases[m] * tensors[t][idx_0+k];    // Fill in phased redundant elements
                            }
                        }
                    }
                }
            else
                {
                for (PyInt m=1; m<num_orderings; m++)    // loop over other arrangements of the indices
                    {
                    PyInt idx = 0;                                                                  // Build the linear index ...
                    for (PyInt i=0; i<len_orderings; i++) {idx += orderings[m][i] * strides[i];}    // ... for the alternate arrangements.
                    for (PyInt t=0; t<num_tensors; t++)
                        {
                        for (PyInt k=0; k<strides[len_orderings-1]; k++)
                            {
                            tensors[t][idx+k] = 0;                                  // Zero out the element
                            }
                        }
                    }
                }
            }
        }

    return;
    }



void antisymmetry(Double** tensors,      // array of density tensors to antisymmetrize
                  PyInt    n_tensors,    // number of density tensors to antisymmetrize
                  PyInt    n_orbs,       // number of orbitals (edge dimension of tensors)
                  PyInt    n_create,     // number of creation operators
                  PyInt    n_annihil,    // number of annihilation operators
                  PyInt    undo)         // un-antisymmetrize, leaving only one index-ordered "wedge" of non-zero values
    {
    PyInt one    = 1;                                      // &one is a PyInt array with one element that is 1: ie, [1]
    PyInt stride = 1;                                      // stride = n_orbs^n_annihil, which is the size of ...
    for (int i=0; i<n_annihil; i++) {stride *= n_orbs;}    // ... the a-string subtensor for each value of c-string indices

    if (n_annihil > 1)    // first antisymmetrize the latter a-string tensor for all values of the c-string indices
        {
        PyInt n_subtensors = 1;                                      // n_subtensors = n_orbs^n_create, because there is ...
        for (int i=0; i<n_create;  i++) {n_subtensors *= n_orbs;}    // ... one a-string subtensor for each value of c-string indices
        Double** all_tensors = (Double**)malloc(n_tensors * n_subtensors * sizeof(Double*));  // storage for pointers to every subtensor of every tensor. Seg-fault with: Double* all_tensors[n_tensors * n_subtensors]

        int k = 0;    // a running index for all subtensor of every tensor
        for (int i=0; i<n_tensors; i++)    // loop over all top-level tensors
            {
            for (int j=0; j<n_subtensors; j++)    // loop over all subtensors of each top-level tensor
                {
                all_tensors[k] = tensors[i] + j*stride;    // store the pointer to that subtensor
                k++;                                       // increment the running index
                }
            }

        //
        antisymmetrize_recur(all_tensors, n_tensors*n_subtensors, n_annihil, n_orbs, undo, &one,    0, 0, 1, NULL, &one);

        free(all_tensors);
        }

    if (n_create > 1)    // then antisymmetrize the among the former indices by treating the entire a-string subtensor as an "element" of the c-string tensor
        {
        //
        antisymmetrize_recur(    tensors, n_tensors,              n_create,  n_orbs, undo, &stride, 0, 0, 1, NULL, &one);
        }

    return;
    }
