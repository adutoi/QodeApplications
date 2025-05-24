#    (C) Copyright 2023, 2024, 2025 Anthony D. Dutoi
# 
#    This file is part of QodeApplications.
# 
#    QodeApplications is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
# 
#    QodeApplications is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
# 
#    You should have received a copy of the GNU General Public License
#    along with QodeApplications.  If not, see <http://www.gnu.org/licenses/>.
#
import numpy
import qode
from qode.util     import struct
from qode.util.PyC import Double



def map_frag_dimer(nested):

    all_configs_0, all_configs_1 = {}, {}
    for config_1,configs_0 in nested:
        n1 = config_1.bit_count()
        if n1 not in all_configs_1:
            all_configs_1[n1] = list()
        all_configs_1[n1] += [config_1]
        for config_0 in configs_0:
            n0 = config_0.bit_count()
            if n0 not in all_configs_0:
                all_configs_0[n0] = set()
            all_configs_0[n0] |= {config_0}

    frag0_to_dimer, frag1_to_dimer = {}, {}
    for n in all_configs_0:
        all_configs_0[n] = sorted(all_configs_0[n])
        frag0_to_dimer[n] = [[] for _ in all_configs_0[n]]
    for n in all_configs_1:
        all_configs_1[n] = sorted(all_configs_1[n])
        frag1_to_dimer[n] = [[] for _ in all_configs_1[n]]
    dimer_to_frags = []

    P = 0
    for config_1,configs_0 in nested:
        n1 = config_1.bit_count()
        i1 = all_configs_1[n1].index(config_1)
        for config_0 in configs_0:
            n0 = config_0.bit_count()
            i0 = all_configs_0[n0].index(config_0)
            dimer_to_frags += [((n1,i1),(n0,i0))]
            frag1_to_dimer[n1][i1] += [P]
            frag0_to_dimer[n0][i0] += [P]
            P += 1

    return dimer_to_frags, (frag0_to_dimer, frag1_to_dimer), (all_configs_0, all_configs_1)



def frag_state_densities(dimer_state, dimer_to_frags, frags_to_dimer):
    (n0,i0),(n1,i1) = dimer_to_frags[0]
    n_elec = n0 + n1
    for (n0,i0),(n1,i1) in dimer_to_frags:
        if n0+n1!=n_elec:
            raise RuntimeError("dimer state is assumed to have definite particle number")
    dim_1 = {n:len(frags_to_dimer[1][n]) for n in frags_to_dimer[1]}
    dim_0 = {n:len(frags_to_dimer[0][n]) for n in frags_to_dimer[0]}
    rho_1 = {n:numpy.zeros((dim_1[n],dim_1[n])) for n in dim_1}
    rho_0 = {n:numpy.zeros((dim_0[n],dim_0[n])) for n in dim_0}
    for n in rho_0:
        for R_list in frags_to_dimer[1][n_elec-n]:
            for P in R_list:
                (n_i1,i1),(n_i0,i0) = dimer_to_frags[P]
                for Q in R_list:
                    (n_j1,j1),(n_j0,j0) = dimer_to_frags[Q]
                    rho_0[n][i0,j0] += dimer_state[P] * dimer_state[Q]    # should have n_i1==n_j1==n_elec-n and  i1==j1  and n_i0==n_j0==n
    for n in rho_1:
        for R_list in frags_to_dimer[0][n_elec-n]:
            for P in R_list:
                (n_i1,i1),(n_i0,i0) = dimer_to_frags[P]
                for Q in R_list:
                    (n_j1,j1),(n_j0,j0) = dimer_to_frags[Q]
                    rho_1[n][i1,j1] += dimer_state[P] * dimer_state[Q]    # should have n_i0==n_j0==n_elec-n and  i0==j0  and n_i1==n_j1==n
    return rho_0, rho_1



def trim_states(rho, statesthresh, n_elec_ref, configs, printout=print):

    evals_evecs, all_evals = {}, []
    for n,rho_n in rho.items():
        if rho_n is not None:
            evals, evecs = qode.util.sort_eigen(numpy.linalg.eigh(rho_n), order="descending")    # ordered by weight for given electron count
            evals_evecs[n] = (evals, evecs)
            all_evals += list(evals)

    # it is actually easier to work in terms of thresh because states globally ordered by electron count, not weight
    if statesthresh.nstates is None:     # using thresh (easy)
        thresh = statesthresh.thresh
        if thresh is None:
            raise RuntimeError("must define at least one criterion for keeping monomer states")
    elif statesthresh.thresh is None:    # using nstates (so define a surrogate thresh)
        all_evals = list(reversed(sorted(all_evals)))
        thresh = (all_evals[statesthresh.nstates-1] + all_evals[statesthresh.nstates]) / 2
    else:
        raise RuntimeError("cannot define both thresh and nstates as criteria for keeping monomer states")

    states = {}
    for n,(evals,evecs) in evals_evecs.items():
        chg = n_elec_ref - n
        n_config_n = len(evals)
        printout("n_config_n", n_config_n)
        for i,e in enumerate(evals):
            if e>thresh:
                if chg not in states:
                    states[chg] = struct(
                        configs = configs[n],
                        coeffs  = []
                    )
                tmp = numpy.zeros(n_config_n, dtype=Double.numpy, order="C")
                tmp[:] = evecs[:,i]
                states[chg].coeffs += [tmp]

    for chg,states_chg in states.items():
        num_states = len(states_chg.coeffs)
        if num_states>0:
            printout("{}: {} x {}".format(chg, num_states, states_chg.coeffs[0].shape))

    ref_chg, ref_idx = 0, 0
    state_indices = [(ref_chg,ref_idx)]    # List of all charge and state indices, reference state needs to be first, but otherwise irrelevant order
    for i in range(len(states[ref_chg].coeffs)):
        if   i!=ref_idx:  state_indices += [(ref_chg,i)]
    for chg in states:
        if chg!=ref_chg:  state_indices += [(chg,i) for i in range(len(states[chg].coeffs))]

    return states, state_indices
