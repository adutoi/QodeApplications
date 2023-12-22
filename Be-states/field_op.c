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






void resolve_recur(int     mode,           // OP_ACTION or COMPUTE_D, depending on whether producing new states via operator action, or building densities
             PyInt   n_create,       // number of creation operators in the strings looped over (always vacuum normal ordered)
             PyInt   n_destroy,      // number of destruction operators in the strings looped over (always vacuum normal ordered)
             Double **op_tensor,      // tensor of matrix elements (integrals) either read for OP_ACTION or produces for COMPUTE_D
             PyInt   n_orbs,         // edge dimension of the integrals tensor
             Double **Psi_L,       // the states being produced (LHS of equation) for OP_ACTION; states in the bra for COMPUTE_D
             Double **Psi_R,      // the states being acted on (RHS of equation) for OP_ACTION; states in the ket for COMPUTE_D
             int     Psi_L_0,     // lowest index to use in array Psi_L
             int     Psi_L_N,     // highest index to use in array Psi_L
             int     Psi_R_0,    // lowest index to use in array Psi_R
             int     Psi_R_N,    // highest index to use in array Psi_R (for OP_ACTION, must have same number of states on left and right)
             int*    occupied,       // the orbitals that were occupied in the configuration at the *top* layer of recursion
             int     n_occ,          // the number of orbitals that were occupied at the *top* layer of recursion
             int*    empty,          // the orbitals that were empty (not necessarily in order) after the action of all the destruction operators
             int     n_emt,          // the number of orbitals that were empty after the action of all the destruction operators
             int*    cum_occ,        // the number of orbitals at or below a given absolute index that are occupied at present layer (for phase calculations)
             int     permute,        // keep track of the number of permutations associated with field-operator action (at present layer of recursion)
             BigInt* config,         // array of integers collectively holding the configuration being acted upon (at present layer of recursion)
             PyInt   config0_idx,    // index of the configuration being acted upon at the *top* layer of recursion
             BigInt* configs_L,        // bitwise occupation strings stored as arrays of integers (packed in one contiguous block, per global comments above)
             PyInt   n_configs_L,      // how many configurations are there
             PyInt   n_configint_L,    // the number of BigInts needed to store a configuration
             PyInt   n_configint_R,    // the number of BigInts needed to store a configuration
             BigInt  op_idx,         // recursively built index of the op_tensor array (must start as zero)
             BigInt  stride,         // the stride to be applied to each successive loop index in order to build op_idx
             PyFloat thresh,         // a threshold used to abort most expensive actions if not going to matter
             int     factor, int p_0)         // a symmetry factor to apply to matrix elements to avoid looping over redundant matrix elements (may come in as -1 to account for reordering)
    {
    int    n_bits  = orbs_per_configint();            // number of bits/orbitals in a BigInt
    int    n_bytes_config = n_configint_R * sizeof(BigInt);    // number of bytes in a config array (for memcpy)
    BigInt p_config[n_configint_R];
    int p_n;
    int* orb_list;
    if (n_destroy > 0)
        {
        p_n = n_occ;
        orb_list = occupied;
        }
    else
        {
        p_n = n_emt;
        orb_list = empty;
        }
    if (n_destroy + n_create > 1)
        {
        int    n_bytes_cumocc = n_orbs * sizeof(int);
        int    p_cum_occ[n_orbs];    // the number of orbitals at or below a given index that are occupied (for phase calculations)
        int    reset_p_0 = 0;
        int delta;
        int* other;
        if (n_destroy > 0)
            {
            delta = -1;
            other = empty + n_emt++;
            if (mode == OP_ACTION) {factor *= n_destroy;}
            n_destroy--;
            if (n_destroy == 0) {reset_p_0 = 1;}
            }
	else
            {
            delta = +1;
            other = occupied + n_occ++;
            if (mode == OP_ACTION) {factor *= n_create;}
            n_create--;
            }
        for (int p_=p_0; p_<p_n; p_++)
            {
            int p = orb_list[p_];                          // absolute index of the occupied orbital p
            int Q = p / n_bits;
            int R = p % n_bits;
            int p_permute = permute + cum_occ[n_orbs-1] - cum_occ[p];
            memcpy(p_config, config, n_bytes_config);
            p_config[Q] = p_config[Q] ^ ((BigInt)1<<R);    // a copy of the original configuration without orbital q
            memcpy(p_cum_occ, cum_occ, n_bytes_cumocc);
            for (int i=p; i<n_orbs; i++) {p_cum_occ[i] -= delta;}
            int q_0 = p_ + 1;
            if (reset_p_0)  {q_0 = 0;}
            other[0] = p;                              // now q is empty (and loop limit below accounts for this)
            resolve_recur(mode, n_create, n_destroy, op_tensor, n_orbs, Psi_L, Psi_R, Psi_L_0, Psi_L_N, Psi_R_0, Psi_R_N, occupied, n_occ, empty, n_emt, p_cum_occ, p_permute, p_config, config0_idx, configs_L, n_configs_L, n_configint_L, n_configint_R, op_idx+p*stride, stride*n_orbs, thresh, factor, q_0);
            }
        }
    else if (mode == OP_ACTION)
        {
        for (int p_=p_0; p_<p_n; p_++)
            {
            int p = orb_list[p_];                          // absolute index of the occupied orbital p
            Double val = factor * op_tensor[0][op_idx + p*stride];
            if (fabs(val) > thresh)
                {
                int Q = p / n_bits;
                int R = p % n_bits;
                memcpy(p_config, config, n_bytes_config);
                p_config[Q] = p_config[Q] ^ ((BigInt)1<<R);    // a copy of the original configuration without orbital q
                PyInt op_config0_idx = bisect_search(p_config, configs_L, n_configint_L, 0, n_configs_L-1);    // THIS IS THE EXPENSIVE STEP!
                if (op_config0_idx != -1)
                    {
                    int p_permute = permute + cum_occ[n_orbs-1] - cum_occ[p];
                    int phase = (p_permute%2) ? -1 : 1;
                    val *= phase;
                    for (int v=Psi_R_0; v<Psi_R_N; v++)
                        {
                        Double update = val * Psi_R[v][config0_idx];
                        #pragma omp atomic
                        Psi_L[v][op_config0_idx] += update;
                        }
                    }
                }
            }
        }
    else  // mode == COMPUTE_D
        {
        for (int p_=p_0; p_<p_n; p_++)
            {
            int p = orb_list[p_];                          // absolute index of the occupied orbital p
            int p_op_idx = op_idx + p*stride;
            int Q = p / n_bits;
            int R = p % n_bits;
            memcpy(p_config, config, n_bytes_config);
            p_config[Q] = p_config[Q] ^ ((BigInt)1<<R);    // a copy of the original configuration without orbital q
            PyInt op_config0_idx = bisect_search(p_config, configs_L, n_configint_L, 0, n_configs_L-1);    // THIS IS THE EXPENSIVE STEP!
            if (op_config0_idx != -1)
                {
                int p_permute = permute + cum_occ[n_orbs-1] - cum_occ[p];
                int phase = (p_permute%2) ? -1 : 1;
                int block = 0;
                for (int vL=Psi_L_0; vL<Psi_L_N; vL++)
                    {
                    Double Z = phase * Psi_L[vL][op_config0_idx];
                    for (int vR=Psi_R_0; vR<Psi_R_N; vR++)
                        {
                        Double update = Z * Psi_R[vR][config0_idx];
                        #pragma omp atomic
                        op_tensor[block++][op_idx]+= update;
                        }
                    }
                }
            }
        }
    return;
    }








void resolve(int     mode,
           PyInt   n_create,
           PyInt   n_destroy,
           Double** tensors,            // tensor of matrix elements (integrals), assumed antisymmetrized
           Double** Psi_L,           // block of row vectors: input vectors to act on
           Double** Psi_R,         // block of row vectors: incremented by output
           BigInt* configs_L,       // bitwise occupation strings stored as arrays of integers (packed in one contiguous block, per global comments above)
           BigInt* configs_R,       // bitwise occupation strings stored as arrays of integers (packed in one contiguous block, per global comments above)
           PyInt   n_configint_L,   // the number of BigInts needed to store a configuration
           PyInt   n_configint_R,   // the number of BigInts needed to store a configuration
           PyInt   n_orbs,        // edge dimension of the integrals tensor
           PyInt   vec_0,         // index of first vector in block to act upon
           PyInt   n_vecs_L,        // how many vectors we are acting on simultaneously
           PyInt   n_vecs_R,        // how many vectors we are acting on simultaneously
           PyInt   n_configs_L,     // how many configurations are there (call signature is ok as long as PyInt not longer than BigInt)
           PyInt   n_configs_R,     // how many configurations are there (call signature is ok as long as PyInt not longer than BigInt)
           PyFloat thresh,        // threshold for ignoring integrals and coefficients (avoiding expensive index search)
           PyInt   n_threads)     // number of OMP threads to spread the work over
    {
    omp_set_num_threads(n_threads);
    int n_bits = orbs_per_configint();            // number of bits/orbitals in a BigInt

    int permute = (n_destroy/2) % 2;

    #pragma omp parallel for
    for (PyInt n=0; n<n_configs_R; n++)
        {
        // "scratch" space that needs to be maximally n_orbs long, allocated once (per thread)
        int occupied[n_orbs];   // the orbitals that are occupied in a given configuration (not necessarily in order)
        int empty[n_orbs];      // the orbitals that are empty    in a given configuration (not necessarily in order)
        int cum_occ[n_orbs];    // the number of orbitals below a given index that are occupied (for phase calculations)

        Double biggest = 0;                   // The biggest n-th component of all vectors being acted on
        for (int v=vec_0; v<vec_0+n_vecs_R; v++)
            {
            Double size = fabs(Psi_R[v][n]);
            if (size > biggest)  {biggest = size;}
            }

        if (biggest > thresh)    // all of this is skipped if the configuration has no significan coefficiencts
            {
            BigInt* config = configs_R + (n * n_configint_R);    // config[] is now an array of integers collectively holding the present configuration

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

            resolve_recur(mode, n_create, n_destroy, tensors, n_orbs, Psi_L, Psi_R, vec_0, vec_0+n_vecs_L, vec_0, vec_0+n_vecs_R, occupied, n_occ, empty, n_emt, cum_occ, permute, config, n, configs_L, n_configs_L, n_configint_L, n_configint_R, 0, 1, thresh/biggest, 1, 0);
            }
        }
    return;
    }



void op_Psi(PyInt   n_elec,        // electron order of the operator
           Double* op,            // tensor of matrix elements (integrals), assumed antisymmetrized
           Double** opPsi,         // block of row vectors: incremented by output
           Double** Psi,           // block of row vectors: input vectors to act on
           BigInt* configs,       // bitwise occupation strings stored as arrays of integers (packed in one contiguous block, per global comments above)
           PyInt   n_configint,   // the number of BigInts needed to store a configuration
           PyInt   n_orbs,        // edge dimension of the integrals tensor
           PyInt   vec_0,         // index of first vector in block to act upon
           PyInt   n_vecs,        // how many vectors we are acting on simultaneously
           PyInt   n_configs,     // how many configurations are there (call signature is ok as long as PyInt not longer than BigInt)
           PyFloat thresh,        // threshold for ignoring integrals and coefficients (avoiding expensive index search)
           PyInt   n_threads)     // number of OMP threads to spread the work over
    {
    resolve(OP_ACTION, n_elec, n_elec, &op, opPsi, Psi, configs, configs, n_configint, n_configint, n_orbs, vec_0, n_vecs, n_vecs, n_configs, n_configs, thresh, n_threads);
    return;
    }

void densities(PyInt   n_create,      // number of creation operators
               PyInt   n_destroy,     // number of destruction operators
               Double** rho,            // pointers to storate for density tensors for each pair of states
               Double** bras,           // block of row vectors: input vectors to act on
               Double** kets,         // block of row vectors: incremented by output
               BigInt* configs_L,       // bitwise occupation strings stored as arrays of integers (packed in one contiguous block, per global comments above)
               BigInt* configs_R,       // bitwise occupation strings stored as arrays of integers (packed in one contiguous block, per global comments above)
               PyInt   n_configint_L,   // the number of BigInts needed to store a configuration
               PyInt   n_configint_R,   // the number of BigInts needed to store a configuration
               PyInt   n_orbs,        // edge dimension of the integrals tensor
               PyInt   n_bras,        // how many vectors we are acting on simultaneously
               PyInt   n_kets,        // how many vectors we are acting on simultaneously
               PyInt   n_configs_L,     // how many configurations are there (call signature is ok as long as PyInt not longer than BigInt)
               PyInt   n_configs_R,     // how many configurations are there (call signature is ok as long as PyInt not longer than BigInt)
               PyFloat thresh,        // threshold for ignoring integrals and coefficients (avoiding expensive index search)
               PyInt   n_threads)     // number of OMP threads to spread the work over
    {
    resolve(COMPUTE_D, n_create, n_destroy, rho, bras, kets, configs_L, configs_R, n_configint_L, n_configint_R, n_orbs, 0, n_bras, n_kets, n_configs_L, n_configs_R, thresh, n_threads);
    return;
    }
