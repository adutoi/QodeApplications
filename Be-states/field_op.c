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
 * The basic operational principle is that an array of configurations is given
 * along with state vectors (having the same length).  These configurations
 * are bit strings (perhaps consisting of multiple consecutive integers), where
 * each bit represents the occupation of an orbital.  The action of field
 * operators are implemented by bit flips on copies of the configuration, while
 * simultaneously keeping track of the phase (using the prepend convention and
 * the orbital ordering described below).  After action of a string of field
 * operators on a ket configuration, the location of the new configuration in
 * a configuration list is looked up, in order to find the bra onto which it
 * projects.  This look-up relies on the fact that the configuration strings
 * are stored in ascending order according to the interpretation of the bit
 * strings as itegers.  This step (implemented via bisection search) is the
 * most expensive part of this algorithm.
 *
 * On the inside of this code (which may differ from the outside) we imagine
 * the binary representation of integers used to represent the configurations
 * written in the usual way, with the low bit (representing the lowest-index
 * orbital) to the right.  Therefore, to make things conceptually consistent,
 * it helps to imagine arrays written right-to-left in storage (opposite the
 * natural direction for a left-to-right reader), so that the axes of the
 * integrals/density tensors associated with the same orbital sets as the bit
 * strings run in the same direction.  This is particularly convenient for the
 * *arrays* of configuration integers that need to be used to allow for systems
 * with more orbitals than can fit in a single BigInt; the lowest-order bits
 * (that is, the lowest-index orbitals) are represented in the 0-th element of
 * such an array, and so on, so it helps to think of the 1-st element as being
 * to the left of the 0-th, etc.  These configuration integers come in as one
 * big block (to ensure they are contiguous for efficiency) and get chopped up
 * as they are used.
 *
 * Ideas for making this better:
 * - allow input about what ranges of creation/annihilation have nonzero elements
 * - feed in integrals that account for frozen core while working only with active-space configs
 * - faster "find" function? (run through Ham once to make lookup table for each element)
 * - spin symmetry?
 */

#include <string.h>       // memcpy()
#include <math.h>         // fabs()
#include "PyC_types.h"    // PyInt, BigInt, Double
#ifdef _OPENMP
#include <omp.h>          // multithreading
#endif

// The operation of the recursive kernel for performing the action of an operator on a vector
// to produce a new vector or to build transition-density tensors between vectors is largely the
// same, so there iw one piece of code, where one of two modes is activated upon bottoming out.
#define OP_ACTION 1
#define COMPUTE_D 2



// The number of orbitals represented by a single BigInt.
// Self-standing function to be accessible to python for packing the configs arrays.
PyInt orbs_per_configint()
    {
    return 8*sizeof(BigInt) - 1;    // pretty safe to assume a byte is 8 bits.  Less 1 bit for sign.
    }



// The index of a config in an array of configs (given that each configuration requires
// n_configint BigInts).  Returns -1 if config not in configs.  The last two arguments
// are the inclusive initial bounds.
// Also needs to be accessible to python.
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




// The recursive kernel for looping over all orbital indices associated with a vacuum-normal-ordered
// string of field operators.  The looping goes over the last operators first (those closest to the
// ket) for a single ket configuration.  Only ascending-order index combinations are used for the creation
// and annihilation substrings, and only these integrals are read in OP_ACTION mode and only these density
// elements are produced in COMPUTE_D mode.
// The tricky part here, which leads to more lines of code than one might
// first think to write, is to get the recursion to bottom out at 1 loop instead of 0.  This is because
// things can be made more efficient if it is known that it is the inner-most loop, and it is the
// inner-most loop that actually matters.  See comments with top-level functions below for more context.
void resolve_recur(int      mode,             // OP_ACTION or COMPUTE_D, depending on whether using Psi_L for storing new states or as bras
                   PyInt    n_create,         // number of creation operators at present level of recursion
                   PyInt    n_annihil,        // number of annihilation operators at present level of recursion
                   Double** Psi_L,            // the states being produced (LHS of equation) for OP_ACTION; states in the bra (on left) for COMPUTE_D
                   PyInt    n_Psi_L,          // number of states in Psi_L (for OP_ACTION, must have n_Psi_L==n_Psi_R, below)
                   BigInt*  configs_L,        // configurations strings representing the basis for the states in Psi_L
                   PyInt    n_configs_L,      // the number of configurations in the basis configs_L
                   PyInt    n_configint_L,    // the number of BigInts needed to store a single configuration in configs_L
                   Double** Psi_R,            // the states being acted on (RHS of equation) for OP_ACTION; states in the ket (on right) for COMPUTE_D
                   PyInt    n_Psi_R,          // number of states in Psi_R
                   BigInt*  config,           // the ket (right-hand) configuration being acted upon at present layer of recursion
                   PyInt    config0_idx,      // index of the configuration (in the right-hand basis) acted upon at the *top* layer of recursion
                   PyInt    n_configint_R,    // the number of BigInts needed to store the ket configuration being acted on, at any level of recursion
                   Double** tensors,          // tensor of matrix elements (sole entry) for OP_ACTION, or storage for output (array of arrays) for COMPUTE_D
                   PyInt    n_orbs,           // edge dimension of the tensor(s)
                   int*     occupied,         // indices of orbitals that are occupied in the configuration at the present level of recursion (not necessarily in order)
                   int      n_occ,            // number of orbitals that are occupied at the present level of recursion
                   int*     empty,            // indices of orbitals that are empty in the configuration at the present level of recursion (not necessarily in order)
                   int      n_emt,            // number of orbitals that are empty at the present level of recursion
                   int*     cum_occ,          // the cumulative number of orbitals at or below a given index that are occupied at present level of recursion
                   int      permute,          // the number of permutations performed so far to satisfy the field-operator prepend convention for orbitals in descending order
                   int      op_idx,           // recursively built index for the tensors array (must start as 0, depends on orbital indices)
                   int      stride,           // the stride to be applied at this level of recursion in order to build op_idx (must start as 1)
                   int      factor,           // recursively built factor to avoid looping over redundant matrix elements (must start as 1 or -1, depending on global reordering)
                   int      p_0,              // recursively updated initial orbital index to avoid looping over redundant matrix elements (must start as 0)
                   Double   thresh)           // a threshold used to abort the most expensive actions if not going to matter
    {
    // Some admin that needs to be done for either mode at any level of recursion
    int    n_bits         = orbs_per_configint();              // number of bits/orbitals in a BigInt
    int    n_bytes_config = n_configint_R * sizeof(BigInt);    // number of bytes in the ket config (for memcpy)
    BigInt p_config[n_configint_R];                            // a place to store modified configurations
    int    p_n;
    int*   orb_list;
    if (n_annihil > 0)
        {
        p_n      = n_occ;       // upper limit of the orbital loop if doing annihlation operator
        orb_list = occupied;    // resolution of counting index to orbital index draws from occupied orbitals of present configurations
        }
    else
        {
        p_n      = n_emt;       // upper limit of the orbital loop if doing creation operator
        orb_list = empty;       // resolution of counting index to orbital index draws from empty orbitals of present configurations
        }

    if (n_annihil + n_create > 1)    // recursive part (there is still >1 loop to go)
        {
        int  n_bytes_cum_occ = n_orbs * sizeof(int);    // number of bytes in cum_occ array (for memcpy)
        int  p_cum_occ[n_orbs];                         // a place to store modified cum_occ arrays
        int  reset_p_0 = 0;
        int  occ_change;
        int* other_orb_list_entry;
        if (n_annihil > 0)
            {
            factor *= n_annihil;                          // increase the redundancy factor (only used for OP_ACTION)
            n_annihil--;                                  // there will be one less annihilation loop
            other_orb_list_entry = empty + n_emt++;       // we will eventually add the orbital index to the end of the empty array (whose length is incremented)
            occ_change = -1;                              // occupancies in cum_occ array will go down
            if (n_annihil == 0) {reset_p_0 = 1;}          // if this is the last annihilation operator we will reset the beginning index of the orbital loops (for creation loops)
            }
	else
            {
            factor *= n_create;                           // increase the redundancy factor (only used for OP_ACTION)
            n_create--;                                   // there will be one less creation loop
            other_orb_list_entry = occupied + n_occ++;    // we will eventually add the orbital index to the end of the occupied array (whose length is incremented). Ignored (see below)
            occ_change = +1;                              // occupancies in cum_occ array will go up
            }
        for (int p_=p_0; p_<p_n; p_++)    // loop over the "counting" index for either occupieds or empties (starting from the given value; see below)
            {
            int p = orb_list[p_];    // "absolute" index of the orbital
            int Q = p / n_bits;      // Q=quotient:  in which component of config is orbital p?
            int R = p % n_bits;      // R=remainder: which bit in ^this component is this orbital?
            int p_permute = permute + cum_occ[n_orbs-1] - cum_occ[p];     // how many permutations does it take to get to/from position p to the front (prepend convention)
            memcpy(p_config, config, n_bytes_config);                     // a copy of the original configuration ...
            p_config[Q] = p_config[Q] ^ ((BigInt)1<<R);                   // ... with occupancy of postion p flipped
            memcpy(p_cum_occ, cum_occ, n_bytes_cum_occ);                  // a copy of the cum_occ array ...
            for (int i=p; i<n_orbs; i++) {p_cum_occ[i] -= occ_change;}    // ... with occupancies appropriately altered
            int q_0 = p_ + 1;             // the beginning of the next loop states above the current index ...
            if (reset_p_0)  {q_0 = 0;}    // ... unless we are switching from annihilation to creatino operators
            other_orb_list_entry[0] = p;                              // now q is empty (and loop limit below accounts for this)
            // recur, passing through appropriately modified quantities (see below about inline updates)
            resolve_recur(mode, n_create, n_annihil, Psi_L, n_Psi_L, configs_L, n_configs_L, n_configint_L, Psi_R, n_Psi_R, p_config, config0_idx, n_configint_R, tensors, n_orbs, occupied, n_occ, empty, n_emt, p_cum_occ, p_permute, op_idx+p*stride, stride*n_orbs, factor, q_0, thresh);
            }
        }
    else if (mode == OP_ACTION)    // (bottom out)
        {
        for (int p_=p_0; p_<p_n; p_++)    // final orbital loop (see above)
            {
            int p = orb_list[p_];                                   // absolute index (see above)
            Double val = factor * tensors[0][op_idx + p*stride];    // finish building tensor index (done inline with recursion above) and get integral from only tensor
            if (fabs(val) > thresh)                                 // don't do anything if the integral is too small (thresh considers also ket coefficient)
                {
                int Q = p / n_bits;                            // build ...
                int R = p % n_bits;                            // ... modified configuration ...
                memcpy(p_config, config, n_bytes_config);      // ... as discussed ...
                p_config[Q] = p_config[Q] ^ ((BigInt)1<<R);    // ... above
                PyInt op_config0_idx = bisect_search(p_config, configs_L, n_configint_L, 0, n_configs_L-1);    // EXPENSIVE! -- find index of fully resolved operator on original config
                if (op_config0_idx != -1)    // don't do anything if action takes outside of space of configurations
                    {
                    int p_permute = permute + cum_occ[n_orbs-1] - cum_occ[p];    // final permutation and ...
                    int phase = (p_permute%2) ? -1 : 1;                          // ... multiplication by resulting phase ...
                    val *= phase;                                                // ... delayed until we know operation was nonzero
                    for (int v=0; v<n_Psi_R; v++)    // for each vector in the input set ...
                        {
                        Double update = val * Psi_R[v][config0_idx];    // ... connect the input configuration ...
                        #pragma omp atomic                              // ... (in a thread-safe way) ...
                        Psi_L[v][op_config0_idx] += update;             // ... to the slot of the output
                        }
                    }
                }
            }
        }
    else    // mode == COMPUTE_D (bottom out)
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
                for (int vL=0; vL<n_Psi_L; vL++)
                    {
                    Double Z = phase * Psi_L[vL][op_config0_idx];
                    for (int vR=0; vR<n_Psi_R; vR++)
                        {
                        Double update = Z * Psi_R[vR][config0_idx];
                        #pragma omp atomic
                        tensors[block++][op_idx]+= update;
                        }
                    }
                }
            }
        }

    return;
    }






// defintions types ordering


void resolve(int     mode,
           PyInt   n_create,
           PyInt   n_annihil,
           Double** tensors,            // tensor of matrix elements (integrals), assumed antisymmetrized
           Double** Psi_L,           // block of row vectors: input vectors to act on
           Double** Psi_R,         // block of row vectors: incremented by output
           BigInt* configs_L,       // bitwise occupation strings stored as arrays of integers (packed in one contiguous block, per global comments above)
           BigInt* configs_R,       // bitwise occupation strings stored as arrays of integers (packed in one contiguous block, per global comments above)
           PyInt   n_configint_L,   // the number of BigInts needed to store a configuration
           PyInt   n_configint_R,   // the number of BigInts needed to store a configuration
           PyInt   n_orbs,        // edge dimension of the integrals tensor
           PyInt   n_Psi_L,        // how many vectors we are acting on simultaneously
           PyInt   n_Psi_R,        // how many vectors we are acting on simultaneously
           PyInt   n_configs_L,     // how many configurations are there (call signature is ok as long as PyInt not longer than BigInt)
           PyInt   n_configs_R,     // how many configurations are there (call signature is ok as long as PyInt not longer than BigInt)
           PyFloat thresh,        // threshold for ignoring integrals and coefficients (avoiding expensive index search)
           PyInt   n_threads)     // number of OMP threads to spread the work over
    {
    omp_set_num_threads(n_threads);
    int n_bits = orbs_per_configint();            // number of bits/orbitals in a BigInt

    int permute = (n_annihil/2) % 2;

    #pragma omp parallel for
    for (PyInt n=0; n<n_configs_R; n++)
        {
        // "scratch" space that needs to be maximally n_orbs long, allocated once (per thread)
        int occupied[n_orbs];   // the orbitals that are occupied in a given configuration (not necessarily in order)
        int empty[n_orbs];      // the orbitals that are empty    in a given configuration (not necessarily in order)
        int cum_occ[n_orbs];    // the number of orbitals below a given index that are occupied (for phase calculations)

        Double biggest = 0;                   // The biggest n-th component of all vectors being acted on
        for (int v=0; v<n_Psi_R; v++)
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

            resolve_recur(mode, n_create, n_annihil, Psi_L, n_Psi_L, configs_L, n_configs_L, n_configint_L, Psi_R, n_Psi_R, config, n, n_configint_R, tensors, n_orbs, occupied, n_occ, empty, n_emt, cum_occ, permute, 0, 1, 1, 0, thresh/biggest);
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
           PyInt   n_Psi,        // how many vectors we are acting on simultaneously
           PyInt   n_configs,     // how many configurations are there (call signature is ok as long as PyInt not longer than BigInt)
           PyFloat thresh,        // threshold for ignoring integrals and coefficients (avoiding expensive index search)
           PyInt   n_threads)     // number of OMP threads to spread the work over
    {
    resolve(OP_ACTION, n_elec, n_elec, &op, opPsi, Psi, configs, configs, n_configint, n_configint, n_orbs, n_Psi, n_Psi, n_configs, n_configs, thresh, n_threads);
    return;
    }

void densities(PyInt   n_create,      // number of creation operators
               PyInt   n_annihil,     // number of destruction operators
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
    resolve(COMPUTE_D, n_create, n_annihil, rho, bras, kets, configs_L, configs_R, n_configint_L, n_configint_R, n_orbs, n_bras, n_kets, n_configs_L, n_configs_R, thresh, n_threads);
    return;
    }
