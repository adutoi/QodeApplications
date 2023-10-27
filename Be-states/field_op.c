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

    int n_bits  = orbs_per_configint();            // number of bits/orbitals in a BigInt
    int n_bytes = n_configint * sizeof(BigInt);    // number of bytes in a config array (for memcpy)

    // "scratch" space that needs to be maximally n_orbs long, allocated once (per thread)
    int occupied[n_orbs];          // the orbitals that are occupied in a given configuration (not necessarily in order)
    int empty[n_orbs];             // the orbitals that are empty    in a given configuration (not necessarily in order)
    int cumulative_occ[n_orbs];    // the number of orbitals below a given index that are occupied (for phase calculations)
    // "scratch" space for storing configurations generated by field-operator strings, allocated once (per thread)
    BigInt    q_config[n_configint];
    BigInt   pq_config[n_configint];

    #pragma omp parallel for private(occupied, empty, cumulative_occ, q_config, pq_config)
    for (PyInt n=0; n<n_configs; n++)
        {
        int any_significant = 0;    // ie, False.  There are no significant coefficients for this configuration in any vector
        int v = vec_0;
        while (v<vec_0+n_vecs && !any_significant)    // loop over the vectors we are acting on
            {if (fabs(Psi[(v++)*n_configs+n]) > thresh)  {any_significant = 1;}}

        if (any_significant)    // all of this is skipped if the configuration has no significan coefficiencts
            {
            BigInt* config = configs + (n * n_configint);    // config[] is now an array of integers collectively holding the present configuration

            int Q;    // integer quotient  (for repeated orbital<->bit mapping manipulations)
            int R;    // integer remainder (for repeated orbital<->bit mapping manipulations)

            int n_occ = 0;    // count the number of occupied orbitals (also acts as a running index for cataloging their indices)
            int n_emt = 0;    // count the number of empty    orbitals (also acts as a running index for cataloging their indices)
            for (int i=0; i<n_orbs; i++)
                {
                cumulative_occ[i] = n_occ;    // before incrementing n_occ (so cumulative occupancy "not counting this orb")
                Q = i / n_bits;                                              // Which component of config does orbital i belong to? ...
                R = i % n_bits;                                              // ... and which bit does it correspond to in that component?
                if (config[Q] & ((BigInt)1<<R))  {occupied[n_occ++] = i;}    // If bit R is "on" in component Q, it is occupied, ...
                else                             {   empty[n_emt++] = i;}    // ... otherwise it is empty
                }

            for (int q_=0; q_<n_occ; q_++)
                {
                int q = occupied[q_];                          // absolute index of the occupied orbital q
                Q = q / n_bits;
                R = q % n_bits;
                memcpy(q_config, config, n_bytes);
                q_config[Q] = q_config[Q] ^ ((BigInt)1<<R);    // a copy of the original configuration without orbital q
                empty[n_emt] = q;                              // now q is empty (and loop limit below accounts for this)
                for (int p_=0; p_<n_emt+1; p_++)
                    {
                    int p = empty[p_];
                    // Interpret the component being looped over as the operator:
                    //     op_pq * a+_p a_q
                    // This therefore loops over all q and p that lead to a nonzero
                    // action on config.
                    Double op_pq = op[p*n_orbs + q];
                    if (fabs(op_pq) > thresh)    // cull all of the inner operations if matrix element not significant
                        {
                        Q = p / n_bits;
                        R = p % n_bits;
                        memcpy(pq_config, q_config, n_bytes);
                        pq_config[Q] = pq_config[Q] ^ ((BigInt)1<<R);
                        PyInt m = bisect_search(pq_config, configs, n_configint, 0, n_configs-1);    // THIS IS THE EXPENSIVE STEP!
                        if (m != -1)
                            {
                            int permute = cumulative_occ[q] - cumulative_occ[p];
                            if (p>q)  {permute += 1;}    // correct for asymmetry in counting occs betweeen p and q
                            int phase = (permute%2) ? -1 : 1;
                            op_pq *= phase;
                            for (v=vec_0; v<vec_0+n_vecs; v++)
                                {
                                Double update = op_pq * Psi[v*n_configs+n];
                                #pragma omp atomic
                                opPsi[v*n_configs+m] += update;
                                }
                            }
                        }
                    }
                }
            }
        }
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

    int n_bits  = orbs_per_configint();            // number of bits/orbitals in a BigInt
    int n_bytes = n_configint * sizeof(BigInt);    // number of bytes in a config array (for memcpy)

    // "scratch" space that needs to be maximally n_orbs long, allocated once (per thread)
    int occupied[n_orbs];          // the orbitals that are occupied in a given configuration (not necessarily in order)
    int empty[n_orbs];             // the orbitals that are empty    in a given configuration (not necessarily in order)
    int cumulative_occ[n_orbs];    // the number of orbitals below a given index that are occupied (for phase calculations)
    // "scratch" space for storing configurations generated by field-operator strings, allocated once (per thread)
    BigInt    r_config[n_configint];
    BigInt   sr_config[n_configint];
    BigInt  psr_config[n_configint];
    BigInt pqsr_config[n_configint];

    #pragma omp parallel for private(occupied, empty, cumulative_occ, r_config, sr_config, psr_config, pqsr_config)
    for (PyInt n=0; n<n_configs; n++)
        {
        int any_significant = 0;    // ie, False.  There are no significant coefficients for this configuration in any vector
        int v = vec_0;
        while (v<vec_0+n_vecs && !any_significant)    // loop over the vectors we are acting on
            {if (fabs(Psi[(v++)*n_configs+n]) > thresh)  {any_significant = 1;}}

        if (any_significant)    // all of this is skipped if the configuration has no significan coefficiencts
            {
            BigInt* config = configs + (n * n_configint);    // config[] is now an array of integers collectively holding the present configuration

            int Q;    // integer quotient  (for repeated orbital<->bit mapping manipulations)
            int R;    // integer remainder (for repeated orbital<->bit mapping manipulations)

            int n_occ = 0;    // count the number of occupied orbitals (also acts as a running index for cataloging their indices)
            int n_emt = 0;    // count the number of empty    orbitals (also acts as a running index for cataloging their indices)
            for (int i=0; i<n_orbs; i++)
                {
                cumulative_occ[i] = n_occ;    // before incrementing n_occ (so cumulative occupancy "not counting this orb")
                Q = i / n_bits;                                              // Which component of config does orbital i belong to? ...
                R = i % n_bits;                                              // ... and which bit does it correspond to in that component?
                if (config[Q] & ((BigInt)1<<R))  {occupied[n_occ++] = i;}    // If bit R is "on" in component Q, it is occupied, ...
                else                             {   empty[n_emt++] = i;}    // ... otherwise it is empty
                }

            for (int r_=0; r_<n_occ; r_++)
                {
                int r = occupied[r_];                          // absolute index of the occupied orbital r
                Q = r / n_bits;
                R = r % n_bits;
                memcpy(r_config, config, n_bytes);
                r_config[Q] = r_config[Q] ^ ((BigInt)1<<R);    // a copy of the original configuration without orbital r
                empty[n_emt] = r;                              // now r is empty (and loop limits below account for this)
                for (int s_=r_+1; s_<n_occ; s_++)
                    {
                    int s = occupied[s_];
                    Q = s / n_bits;
                    R = s % n_bits;
                    memcpy(sr_config, r_config, n_bytes);
                    sr_config[Q] = sr_config[Q] ^ ((BigInt)1<<R);
                    empty[n_emt+1] = s;
                    for (int p_=0; p_<n_emt+2; p_++)
                        {
                        int pp = empty[p_];
                        Q = pp / n_bits;
                        R = pp % n_bits;
                        memcpy(psr_config, sr_config, n_bytes);
                        psr_config[Q] = psr_config[Q] ^ ((BigInt)1<<R);
                        for (int q_=p_+1; q_<n_emt+2; q_++)
                            {
                            int qq = empty[q_];
                            int p = pp;
                            int q = qq;
                            if (p > q)  {p = qq;  q = pp;}
                            // Interpret the component being looped over as the operator:
                            //     op_pqrs * a+_p a+_q a_s a_r
                            // such that the excitations are best interpreted as p<-r and
                            // q<-s, where p<q and r<s have been enforced above (and was
                            // usually the case anyway, unless refilling an originally
                            // occupied orbital).  This therefore loops over all ordered
                            // pairs (r,s) and (p,q) that lead to a nonzero action on
                            // config.  We keep track of pp and qq separately to avoid
                            // confusing the inner loops, especially the bitwise
                            // configuration-changing mechanism.
                            Double op_pqrs = 4 * op[((p*n_orbs + q)*n_orbs + r)*n_orbs + s];    // because integrals antisymmetrized
                            if (fabs(op_pqrs) > thresh)    // cull all of the inner operations if matrix element not significant
                                {
                                Q = qq / n_bits;
                                R = qq % n_bits;
                                memcpy(pqsr_config, psr_config, n_bytes);
                                pqsr_config[Q] = pqsr_config[Q] ^ ((BigInt)1<<R);
                                PyInt m = bisect_search(pqsr_config, configs, n_configint, 0, n_configs-1);    // THIS IS THE EXPENSIVE STEP!
                                if (m != -1)
                                    {
                                    int permute = (cumulative_occ[r] - cumulative_occ[p]) + (cumulative_occ[s] - cumulative_occ[q]);
                                    if (p>r)   {permute += 1;}    // correct for asymmetry in counting occs betweeen p and r
                                    if (q>s)   {permute += 1;}    // correct for asymmetry in counting occs betweeen q and s
                                    if (q==r)  {permute += 1;}    // corner case where refilled occ gets counted
                                    if (p>s)   {permute += 1;}    // one way that excitations can "cross"
                                    if (q<r)   {permute += 1;}    // another way they can "cross"
                                    int phase = (permute%2) ? -1 : 1;
                                    op_pqrs *= phase;
                                    for (v=vec_0; v<vec_0+n_vecs; v++)
                                        {
                                        Double update = op_pqrs * Psi[v*n_configs+n];
                                        #pragma omp atomic
                                        opPsi[v*n_configs+m] += update;
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
