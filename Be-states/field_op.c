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

#include <stdlib.h>       // exit()
#include <stdio.h>        // fprintf(stderr, "...")
#include <math.h>         // fabs()
#include "PyC_types.h"

#ifdef _OPENMP
#include <omp.h>
#endif

/*
 * Ideas for making this even better:
 * - allow input about what ranges of creation/annihilation have nonzero elements
 * - feed in integrals that account for frozen core while working only with active-space configs
 * - faster "find" function (run through Ham once to make lookup table for each element)
 * - spin symmetry
 */



PyInt bisect_search(BigInt config, BigInt* configs, PyInt lower, PyInt upper)
    {
    if ((config<configs[lower]) || (config>configs[upper]))  {return -1;}
    /***********************************************************************************
    // ********  The code appears to be just slightly faster *without* this ... but maybe context dependent?  so keep it around
    // make a guess as to where roughly the config is in the index range
    float  frac  = (config - configs[lower]) / (float)(configs[upper] - configs[lower]);
    BigInt delta = (BigInt)(frac * (upper - lower));
    BigInt guess = lower + delta;
    if (guess < lower)  {guess = lower;}
    if (guess > upper)  {guess = upper;}
    // figuring that our guess was either too high or too low, try to bound it
    if (config == configs[guess])
        {
        return guess;
        }
    else if (config < configs[guess])
        {
        upper = guess;
        frac  = (configs[upper] - config) / (float)(configs[upper] - configs[lower]);
        delta = 2 * (BigInt)(frac * (upper - lower));
        if (delta < 2)  {delta = 2;}
        guess = upper - delta;
        if (guess < lower)  {guess = lower;}
        while (config < configs[guess])
            {
            guess -= delta;
            delta *= 2;
            if (guess < lower)  {guess = lower;}
            }
        lower = guess;
        }
    else if (config > configs[guess])
        {
        lower = guess;
        frac  = (config - configs[lower]) / (float)(configs[upper] - configs[lower]);
        delta = 2 * (BigInt)(frac * (upper - lower));
        if (delta < 2)  {delta = 2;}
        guess = lower + delta;
        if (guess > upper)  {guess = upper;}
        while (config > configs[guess])
            {
            guess += delta;
            delta *= 2;
            if (guess > upper)  {guess = upper;}
            }
        upper = guess;
        }
    ***********************************************************************************/
    // once bounded, perform the bisection search (this code works even if everything above is deleted ... including the top line, though that one makes things faster)
    while (upper-lower > 1)
        {
        PyInt half = (lower + upper) / 2;
        BigInt deviation = config - configs[half];
        if      (deviation == 0)  {return half;}
        else if (deviation  > 0)  {lower = half+1;}
        else                      {upper = half-1;}
        }
    if      (config == configs[lower])  {return lower;}
    else if (config == configs[upper])  {return upper;}
    else                                {return    -1;}
    }

PyInt find_index(PyInt config, BigInt* configs, PyInt n_configs)  {return bisect_search((BigInt)config, configs, 0, n_configs-1);}



void opPsi_1e(Double* op,            // tensor of matrix elements (integrals)
              Double* Psi,           // block of row vectors: input vectors to act on
              Double* opPsi,         // block of row vectors: incremented by output
              BigInt* configs,       // bitwise occupation strings stored as integers ... so, max 64 orbs for FCI ;-) [no checking here!]
              PyInt   n_orbs,        // edge dimension of the integrals tensor.  cannot be bigger than the number of bits in a BigInt (64)
              PyInt   vec_0,         // index of first vector in block to act upon
              PyInt   n_vecs,        // how many vectors we are acting on simultaneously
              PyInt   n_configs,     // how many configurations are there (call signature is ok as long as PyInt not longer than BigInt)
              PyFloat thresh,        // threshold for ignoring integrals and coefficients (avoiding expensive index search)
              PyInt   n_threads)     // number of OMP threads to spread the work over
    {
    omp_set_num_threads(n_threads);

    // "scratch" space that needs to be maximally n_orbs long
    // It helps to imagine arrays written right-to-left (opposite the natural direction
    // for a left-to-right reader), so that they align with the convention of putting
    // low-order bits on the right (in the integers used to represent confgurations).
    int occupied[n_orbs];
    int empty[n_orbs];
    int cumulative_occ[n_orbs];

    #pragma omp parallel for private(occupied, empty, cumulative_occ)
    for (PyInt n=0; n<n_configs; n++)
        {
        int any_significant = 0;
        int v = vec_0;
        while (v<vec_0+n_vecs && !any_significant)
            {if (fabs(Psi[(v++)*n_configs+n]) > thresh)  {any_significant = 1;}}

        if (any_significant)
            {
            BigInt config = configs[n];

            int n_occ = 0;
            int n_emt = 0;
            for (int i=0; i<n_orbs; i++)
                {
                cumulative_occ[i] = n_occ;    // before incrementing (so "not counting this orb")
                if (((BigInt)1<<i) & config)  {occupied[n_occ++] = i;}
                else                          {   empty[n_emt++] = i;}
                }

            for (int q_=0; q_<n_occ; q_++)
                {
                int q = occupied[q_];
                BigInt q_config = config ^ ((BigInt)1<<q);
                empty[n_emt] = q;
                for (int p_=0; p_<n_emt+1; p_++)
                    {
                    int p = empty[p_];
                    // Interpret the component being looped over as the operator:
                    //     op_pq * a+_p a_q
                    // This therefore loops over all q and p that lead to a nonzero
                    // action on config.
                    Double op_pq = op[p*n_orbs + q];
                    if (fabs(op_pq) > thresh)
                        {
                        BigInt pq_config = q_config ^ ((BigInt)1<<p);
                        PyInt m = bisect_search(pq_config, configs, 0, n_configs-1);
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
              BigInt* configs,       // bitwise occupation strings stored as integers ... so, max 64 orbs for FCI ;-) [no checking here!]
              PyInt   n_orbs,        // edge dimension of the integrals tensor.  cannot be bigger than the number of bits in a BigInt (64)
              PyInt   vec_0,         // index of first vector in block to act upon
              PyInt   n_vecs,        // how many vectors we are acting on simultaneously
              PyInt   n_configs,     // how many configurations are there (call signature is ok as long as PyInt not longer than BigInt)
              PyFloat thresh,        // threshold for ignoring integrals and coefficients (avoiding expensive index search)
              PyInt   n_threads)     // number of OMP threads to spread the work over
    {
    omp_set_num_threads(n_threads);

    // "scratch" space that needs to be maximally n_orbs long
    // It helps to imagine arrays written right-to-left (opposite the natural direction
    // for a left-to-right reader), so that they align with the convention of putting
    // low-order bits on the right (in the integers used to represent confgurations).
    int occupied[n_orbs];
    int empty[n_orbs];
    int cumulative_occ[n_orbs];

    #pragma omp parallel for private(occupied, empty, cumulative_occ)
    for (PyInt n=0; n<n_configs; n++)
        {
        int any_significant = 0;
        int v = vec_0;
        while (v<vec_0+n_vecs && !any_significant)
            {if (fabs(Psi[(v++)*n_configs+n]) > thresh)  {any_significant = 1;}}

        if (any_significant)
            {
            BigInt config = configs[n];

            int n_occ = 0;
            int n_emt = 0;
            for (int i=0; i<n_orbs; i++)
                {
                cumulative_occ[i] = n_occ;    // before incrementing (so "not counting this orb")
                if (((BigInt)1<<i) & config)  {occupied[n_occ++] = i;}
                else                          {   empty[n_emt++] = i;}
                }

            for (int r_=0; r_<n_occ; r_++)
                {
                int r = occupied[r_];
                BigInt r_config = config ^ ((BigInt)1<<r);
                empty[n_emt] = r;
                for (int s_=r_+1; s_<n_occ; s_++)
                    {
                    int s = occupied[s_];
                    BigInt sr_config = r_config ^ ((BigInt)1<<s);
                    empty[n_emt+1] = s;
                    for (int p_=0; p_<n_emt+2; p_++)
                        {
                        int pp = empty[p_];
                        BigInt psr_config = sr_config ^ ((BigInt)1<<pp);
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
                            Double op_pqrs = 4 * op[((p*n_orbs + q)*n_orbs + r)*n_orbs + s];
                            if (fabs(op_pqrs) > thresh)
                                {
                                BigInt pqsr_config = psr_config ^ ((BigInt)1<<qq);
                                PyInt m = bisect_search(pqsr_config, configs, 0, n_configs-1);
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
