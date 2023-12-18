/*   (C) Copyright 2018, 2019, 2023 Yuhong Liu and Anthony Dutoi
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

/*
 * The basic operational principle is that an array of configurations is passed
 * along with the state vectors (having the same length).  These configurations
 * are bit strings, where each bit represents the occupation of an orbital.  The
 * action of an operator is implemented by bit-fiddling copies of the configuration
 * string to correspond to annihilation and creation operators, and then looking
 * up where the (rephased) generated configuration falls in the output vector.
 * This relies on the fact that the configuration strings are store in ascending
 * order if their bit-sequences are interpreted as binary integers.
 *
 * On the inside of this code (which differs from the convention on the outside)
 * we imagine the binary integers used to represent the configurations written
 * in the usual way, with the low bit to the right.  Therefore, to make things
 * conceptually consistent, it helps to imagine arrays written right-to-left
 * (opposite the natural direction for a left-to-right reader), so that the axes
 * of the integrals tensors associated with the same orbital sets as the bit
 * strings run in the same direction.  This is particularly convenient for the
 * *arrays* of configuration integers that need to be used to allow for systems
 * with more orbitals than can fit in a single BigInt; the lowest-order bits (that
 * is, the lowest-index orbitals) are represented in the 0-th element of such
 * an array, and so on, so it helps to think of the 1-st element as being to the
 * left of the 0-th, etc.  These configuration integers come in as one big block
 * (to ensure they are contiguous for efficiency) and get chopped up as they are
 * used.
 *
 * Ideas for making this better:
 * - allow input about what ranges of creation/annihilation have nonzero elements
 * - feed in integrals that account for frozen core while working only with active-space configs
 * - faster "find" function (run through Ham once to make lookup table for each element)
 * - spin symmetry
 */

            // The configs array is packed such that components of each config are contiguous, with
            // each of those then stored together consecutively and contiguously under one pointer.
            // Internally, the high-order components of each config follow the low-order components
            // (ie, the 1-st follows the 0-th in storage, as usual), despite what is said above about
            // conceptualizing the arrays as written backwards.  We needed to choose between this
            // small inconistency, or one where we interpret the << operator as a *rightward* bitshift
            // or interpreting low-index components of a config array as high-order bits, and this seemed
            // like the least of evils.  The inconsistency is not removable . . . yes it is, if all the storage is "backwards"


#include <string.h>       // memcpy()
#include <math.h>         // fabs()
#include "PyC_types.h"    // PyInt, BigInt
#ifdef _OPENMP
#include <omp.h>
#endif

#include <stdio.h>        // fprintf(stderr, "...")  Can be eliminated when debugged

#define OP_ACTION 1
#define COMPUTE_D 2

// The number of orbitals represented by a single BigInt.
// Self-standing function because needs to be accessible to python too for packing the configs arrays.
PyInt orbs_per_configint()
    {
    return 8*sizeof(BigInt) - 1;    // pretty safe to assume a byte is 8 bits.  Subtract 1 bit for sign.
    }



// The index of a config in an array of configs (given that each configuration requires n_configint BigInts).
// Returns -1 if config not in configs.  The last two arguments are the inclusive initial bounds.
// Also needs to be accessible to python.
// This is where the time goes!!
PyInt bisect_search(BigInt* config, BigInt* configs, PyInt n_configint, PyInt lower, PyInt upper)
    {
    PyInt   i, half;        // Get used repeatedly, ...
    BigInt  deviation;      // ... so just declare once ...
    BigInt* test_config;    // ... in the old-fashioned way.

    // Note that we will only ever care if deviation is >, ==, or < zero.  Therefore,
    // we test the high-order components of two configurations first and we only test
    // the ones below it if those are equal.  Otherwise, we already have our answer.

    // Make sure config is not below the lower bound
    deviation = 0;
    test_config = configs + (lower * n_configint);    // pointer arithmetic for start of test_config
    for (i=n_configint-1; i>=0 && deviation==0; i--)  {deviation = config[i] - test_config[i];}
    if (deviation < 0)  {return -1;}

    // Make sure config is not above the upper bound
    deviation = 0;
    test_config = configs + (upper * n_configint);    // pointer arithmetic for start of test_config
    for (i=n_configint-1; i>=0 && deviation==0; i--)  {deviation = config[i] - test_config[i];}
    if (deviation > 0)  {return -1;}

    // Narrow it down to a choice of one or two numbers.  So any time that the contents of this
    // loop runs, there are at least three choices, so that half will not be one of the boundaries.
    // It can happen that this loop ends with upper==lower, which just makes the two code blocks
    // that follow redundant (but that will not lead to a wrong result).
    while (upper-lower > 1)
        {
        half = (lower + upper) / 2;
        deviation = 0;
        test_config = configs + (half * n_configint);    // pointer arithmetic for start of test_config
        for (i=n_configint-1; i>=0 && deviation==0; i--)  {deviation = config[i] - test_config[i];}
        if      (deviation == 0)  {return half;}         // only happens if all components equal
        else if (deviation  > 0)  {lower = half+1;}
        else                      {upper = half-1;}
        }

    // If the config is the lower of the two (or one) choices, return that index.
    deviation = 0;
    test_config = configs + (lower * n_configint);    // pointer arithmetic for start of test_config
    for (i=n_configint-1; i>=0 && deviation==0; i--)  {deviation = config[i] - test_config[i];}
    if (deviation == 0)  {return lower;}              // only happens if all components equal

    // If the config is the upper of the two (or one) choices, return that index.
    deviation = 0;
    test_config = configs + (upper * n_configint);    // pointer arithmetic for start of test_config
    for (i=n_configint-1; i>=0 && deviation==0; i--)  {deviation = config[i] - test_config[i];}
    if (deviation == 0)  {return upper;}              // only happens if all components equal

    // Else, the default is to announce that the config is not in the configs array.
    return -1;
    }






void resolve(
int     mode,
PyInt   n_create,
PyInt   n_destroy,

Double *op_tensor,
PyInt   n_orbs,

Double *Psi_left,
Double *Psi_right,
int     Psi_left_0,
int     Psi_left_N,
int     Psi_right_0,
int     Psi_right_N,

int*    occupied,
int     n_occ,
int     occ_0,
int*    empty,
int     n_emt,
int     emt_0,
int*    cum_occ,

int     permute,
BigInt* config,
PyInt   config0_idx,

BigInt* configs,
PyInt   n_configs,
PyInt   n_configint,

BigInt  op_idx,
BigInt  stride,
PyFloat thresh,
int     factor
)
    {
    if (n_destroy != 0)
        {
        int    n_bits  = orbs_per_configint();            // number of bits/orbitals in a BigInt
        int    n_bytes_config = n_configint * sizeof(BigInt);    // number of bytes in a config array (for memcpy)
        int    n_bytes_cumocc = n_orbs * sizeof(int);
        BigInt p_config[n_configint];
        int    p_cum_occ[n_orbs];    // the number of orbitals at or below a given index that are occupied (for phase calculations)
	for (int p_=occ_0; p_<n_occ; p_++)
            {
            int p = occupied[p_];                          // absolute index of the occupied orbital p
            int Q = p / n_bits;
            int R = p % n_bits;
            int p_permute = permute + cum_occ[n_orbs-1] - cum_occ[p];
            memcpy(p_config, config, n_bytes_config);
            p_config[Q] = p_config[Q] ^ ((BigInt)1<<R);    // a copy of the original configuration without orbital q
            memcpy(p_cum_occ, cum_occ, n_bytes_cumocc);
            for (int i=p; i<n_orbs; i++) {p_cum_occ[i] -= 1;}
            empty[n_emt] = p;                              // now q is empty (and loop limit below accounts for this)
            resolve(mode, n_create, n_destroy-1, op_tensor, n_orbs, Psi_left, Psi_right, Psi_left_0, Psi_left_N, Psi_right_0, Psi_right_N, occupied, n_occ, p_+1, empty, n_emt+1, emt_0, p_cum_occ, p_permute, p_config, config0_idx, configs, n_configs, n_configint, op_idx+p*stride, stride*n_orbs, thresh, factor*n_destroy);
            }
        }
    else if (n_create != 0)
        {
        int    n_bits  = orbs_per_configint();            // number of bits/orbitals in a BigInt
        int    n_bytes_config = n_configint * sizeof(BigInt);    // number of bytes in a config array (for memcpy)
        int    n_bytes_cumocc = n_orbs * sizeof(int);
        BigInt p_config[n_configint];
        int    p_cum_occ[n_orbs];    // the number of orbitals at or below a given index that are occupied (for phase calculations)
	for (int p_=emt_0; p_<n_emt; p_++)
            {
            int p = empty[p_];                          // absolute index of the occupied orbital p
            int Q = p / n_bits;
            int R = p % n_bits;
            int p_permute = permute + cum_occ[n_orbs-1] - cum_occ[p];
            memcpy(p_config, config, n_bytes_config);
            p_config[Q] = p_config[Q] ^ ((BigInt)1<<R);    // a copy of the original configuration without orbital q
            memcpy(p_cum_occ, cum_occ, n_bytes_cumocc);
            for (int i=p; i<n_orbs; i++) {p_cum_occ[i] += 1;}
            resolve(mode, n_create-1, n_destroy, op_tensor, n_orbs, Psi_left, Psi_right, Psi_left_0, Psi_left_N, Psi_right_0, Psi_right_N, occupied, n_occ, occ_0, empty, n_emt, p_+1, p_cum_occ, p_permute, p_config, config0_idx, configs, n_configs, n_configint, op_idx+p*stride, stride*n_orbs, thresh, factor*n_create);
            }
        }
    else
        {
        if (mode == OP_ACTION)
            {
            Double val = factor * op_tensor[op_idx];
            if (fabs(val) > thresh)
                {
                PyInt op_config0_idx = bisect_search(config, configs, n_configint, 0, n_configs-1);    // THIS IS THE EXPENSIVE STEP!
                if (op_config0_idx != -1)
                    {
                    int phase = (permute%2) ? -1 : 1;
                    val *= phase;
                    for (int v=Psi_right_0; v<Psi_right_N; v++)
                        {
                        Double update = val * Psi_right[v*n_configs + config0_idx];
                        #pragma omp atomic
                        Psi_left[v*n_configs + op_config0_idx] += update;
                        }
                    }
                }
            }
        else  // mode == COMPUTE_D
            {
            }
        }
    return;
    }










void opPsi_1e(Double* op,            // tensor of matrix elements (integrals)
              Double* Psi,           // block of row vectors: input vectors to act on
              Double* opPsi,         // block of row vectors: incremented by output
              BigInt* configs,       // bitwise occupation strings stored as arrays of integers (packed in one contiguous block, per global comments above)
              PyInt   n_configint,   // the number of BigInts needed to store a configuration
              PyInt   n_orbs,        // edge dimension of the integrals tensor
              PyInt   vec_0,         // index of first vector in block to act upon
              PyInt   n_vecs,        // how many vectors we are acting on simultaneously
              PyInt   n_configs,     // how many configurations are there (call signature is ok as long as PyInt not longer than BigInt)
              PyFloat thresh,        // threshold for ignoring integrals and coefficients (avoiding expensive index search)
              PyInt   n_threads)     // number of OMP threads to spread the work over
    {
    omp_set_num_threads(n_threads);
    int    n_bits  = orbs_per_configint();            // number of bits/orbitals in a BigInt

    // "scratch" space that needs to be maximally n_orbs long, allocated once (per thread)
    int occupied[n_orbs];   // the orbitals that are occupied in a given configuration (not necessarily in order)
    int empty[n_orbs];      // the orbitals that are empty    in a given configuration (not necessarily in order)
    int cum_occ[n_orbs];    // the number of orbitals below a given index that are occupied (for phase calculations)

    #pragma omp parallel for private(occupied, empty, cum_occ)
    for (PyInt n=0; n<n_configs; n++)
        {
        Double biggest = 0;                   // The biggest n-th component of all vectors being acted on
        for (int v=vec_0; v<vec_0+n_vecs; v++)
            {
            Double size = fabs(Psi[v*n_configs + n]);
            if (size > biggest)  {biggest = size;}
            }

        if (biggest > thresh)    // all of this is skipped if the configuration has no significan coefficiencts
            {
            BigInt* config = configs + (n * n_configint);    // config[] is now an array of integers collectively holding the present configuration

            int n_occ = 0;    // count the number of occupied orbitals (also acts as a running index for cataloging their indices)
            int n_emt = 0;    // count the number of empty    orbitals (also acts as a running index for cataloging their indices)
            for (int i=0; i<n_orbs; i++)
                {
                int Q = i / n_bits;                                              // Which component of config does orbital i belong to? ...
                int R = i % n_bits;                                              // ... and which bit does it correspond to in that component?
                if (config[Q] & ((BigInt)1<<R))  {occupied[n_occ++] = i;}    // If bit R is "on" in component Q, it is occupied, ...
                else                             {   empty[n_emt++] = i;}    // ... otherwise it is empty
                cum_occ[i] = n_occ;    // after incrementing n_occ (so cumulative occupancy "counting this orb")
                }

            resolve(OP_ACTION, 1, 1, op, n_orbs, opPsi, Psi, vec_0, vec_0+n_vecs, vec_0, vec_0+n_vecs, occupied, n_occ, 0, empty, n_emt, 0, cum_occ, 0, config, n, configs, n_configs, n_configint, 0, 1, thresh/biggest, 1);
            }
        }
    return;
    }



void opPsi_2e(Double* op,            // tensor of matrix elements (integrals), assumed antisymmetrized
              Double* Psi,           // block of row vectors: input vectors to act on
              Double* opPsi,         // block of row vectors: incremented by output
              BigInt* configs,       // bitwise occupation strings stored as arrays of integers (packed in one contiguous block, per global comments above)
              PyInt   n_configint,   // the number of BigInts needed to store a configuration
              PyInt   n_orbs,        // edge dimension of the integrals tensor
              PyInt   vec_0,         // index of first vector in block to act upon
              PyInt   n_vecs,        // how many vectors we are acting on simultaneously
              PyInt   n_configs,     // how many configurations are there (call signature is ok as long as PyInt not longer than BigInt)
              PyFloat thresh,        // threshold for ignoring integrals and coefficients (avoiding expensive index search)
              PyInt   n_threads)     // number of OMP threads to spread the work over
    {
    omp_set_num_threads(n_threads);
    int    n_bits  = orbs_per_configint();            // number of bits/orbitals in a BigInt

    // "scratch" space that needs to be maximally n_orbs long, allocated once (per thread)
    int occupied[n_orbs];          // the orbitals that are occupied in a given configuration (not necessarily in order)
    int empty[n_orbs];             // the orbitals that are empty    in a given configuration (not necessarily in order)
    int cum_occ[n_orbs];    // the number of orbitals below a given index that are occupied (for phase calculations)

    #pragma omp parallel for private(occupied, empty, cum_occ)
    for (PyInt n=0; n<n_configs; n++)
        {
        Double biggest = 0;                   // The biggest n-th component of all vectors being acted on
        for (int v=vec_0; v<vec_0+n_vecs; v++)
            {
            Double size = fabs(Psi[v*n_configs + n]);
            if (size > biggest)  {biggest = size;}
            }

        if (biggest > thresh)    // all of this is skipped if the configuration has no significan coefficiencts
            {
            BigInt* config = configs + (n * n_configint);    // config[] is now an array of integers collectively holding the present configuration

            int n_occ = 0;    // count the number of occupied orbitals (also acts as a running index for cataloging their indices)
            int n_emt = 0;    // count the number of empty    orbitals (also acts as a running index for cataloging their indices)
            for (int i=0; i<n_orbs; i++)
                {
                int Q = i / n_bits;                                              // Which component of config does orbital i belong to? ...
                int R = i % n_bits;                                              // ... and which bit does it correspond to in that component?
                if (config[Q] & ((BigInt)1<<R))  {occupied[n_occ++] = i;}    // If bit R is "on" in component Q, it is occupied, ...
                else                             {   empty[n_emt++] = i;}    // ... otherwise it is empty
                cum_occ[i] = n_occ;    // after incrementing n_occ (so cumulative occupancy "counting this orb")
                }

            // mind the minus sign in the last argument!
            resolve(OP_ACTION, 2, 2, op, n_orbs, opPsi, Psi, vec_0, vec_0+n_vecs, vec_0, vec_0+n_vecs, occupied, n_occ, 0, empty, n_emt, 0, cum_occ, 0, config, n, configs, n_configs, n_configint, 0, 1, thresh/biggest, -1);
            }
        }
    return;
    }
