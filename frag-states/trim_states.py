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
        print(chg)
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
