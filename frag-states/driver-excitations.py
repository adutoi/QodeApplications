#!/usr/bin/env python3
#    (C) Copyright 2025 Anthony D. Dutoi
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
import pickle
import numpy
from qode.util import struct, read_input, indented, no_print
from qode.math.tensornet import np_tensor, raw
from qode.many_body.fermion_field import combine_orb_lists, CI_methods
import input_env
import hamiltonian
import states
import densities

# these belong elsewhere
n_elec_ref = {"H": 1, "Be": 4, "C": 6}
n_spatial_orb = {
    "6-31G": {"H": 2, "Be": 9, "C": 9}
}



if __name__=="__main__":
    printout = indented(print, indent="    ")

    params = struct(    # defaults
        n_threads = 1,
        nstates   = None,
        thresh    = None,
        compress  = struct(method="SVD", divide="cc-aa"),
        nat_orbs  = False,
        abs_anti  = False,
        n_dimer_states = 1
    )
    params.update(read_input.from_command_line(namespace=input_env))

    frags = [
        struct(
            atoms = frag.atoms,
            n_elec_ref = sum(n_elec_ref[atom.element] for atom in frag.atoms) - frag.charge,    # "cation (+1)" and "anion (-1)" are interpreted relative to the reference
            basis = struct(
                AOcode = params.basis,
                n_spatial_orb = sum(n_spatial_orb[params.basis][atom.element] for atom in frag.atoms),
                core = frag.core    # indices of spatial MOs to freeze in CI portions
            )
        )
    for frag in params.frags]

    label  = "-".join("".join(atom.element for atom in frag.atoms) for frag in params.frags)
    label += "_{}_" + params.basis

    printout("Fragment HF")
    for frag in frags:
        hamiltonian.fragment_HF(frag, Sz=0, printout=indented(printout))    # modifies/initializes MOcoeffs and stores HF data in frag
    frags[1].basis.MOcoeffs = frags[0].basis.MOcoeffs    # insist identical for this code

    monomer_ints, dimer_ints = hamiltonian.dimer_integrals(frags, printout=printout)
    monomer_ints_c, dimer_ints_c = hamiltonian.dimer_integrals(frags, printout=printout, core_hack=True)

    S = dimer_ints.S
    adim = S.shape[0]//2
    sigma = S - numpy.identity(2*adim)
    U0, _, Vh1 = numpy.linalg.svd(sigma[:adim,adim:])
    T = numpy.zeros_like(S)
    T[:adim,:adim] = U0
    T[adim:,adim:] = Vh1.T
    sigma = T.T @ sigma @ T
    sigma_ints_1 = []
    sigma_ints_2 = []
    for p in range(adim):
        s = numpy.zeros_like(S)
        s[p,adim+p] = sigma[p,adim+p]
        s[adim+p,p] = sigma[adim+p,p]
        s = T @ s @ T.T
        sigma_ints_1 += [s]
        s = np_tensor(s)
        sigma_ints_2 += [(1/2) * raw(s(0,2) @ s(1,3))]
    dimer_ints.sigma_1 = sigma_ints_1
    dimer_ints.sigma_2 = sigma_ints_2

    printout("Monomer FCIs (just for fun/illustration)")
    occupied = range(frags[0].n_elec_ref // 2), range(frags[1].n_elec_ref // 2)
    occupied = (combine_orb_lists(occupied[0], occupied[0], frags[0].basis.n_spatial_orb),
                combine_orb_lists(occupied[1], occupied[1], frags[1].basis.n_spatial_orb))
    for m,frag in enumerate(frags):
        configs = CI_methods.monomer_configs(frag, Sz=0)
        (Eval, Evec), = CI_methods.lanczos_ground(monomer_ints_c[m], configs, occupied[m], thresh=1e-8, printout=indented(printout), n_threads=params.n_threads)

    printout("Dimer FCI")
    occupied = combine_orb_lists(occupied[0], occupied[1], 2*frags[0].basis.n_spatial_orb)
    occupied[-1] += 1    # break the symmetry of the (sole!) guess so that we can get all excited states
    printout(occupied)
    configs, nested = CI_methods.dimer_configs(*frags, Sz=0)
    results = CI_methods.lanczos_ground(dimer_ints_c, configs, occupied, n_states=params.n_dimer_states, thresh=1e-8, printout=indented(printout), n_threads=params.n_threads, full_ints=dimer_ints)
    Eval,Evec = results[-1]    # analyze the highest state from this run in the way we used to do the ground state

    dimer_to_frags, frags_to_dimer, frag_configs = states.map_frag_dimer(nested)
    frag_rhos = states.frag_state_densities(Evec.v, dimer_to_frags, frags_to_dimer)
    rhos = {n: (1/2)*(frag_rhos[0][n] + frag_rhos[1][n]) for n in frag_rhos[0]}
    frag_rhos = [rhos, rhos]

    # comment out a bunch of things below because we only need one of the atoms to be dumped, but we need the states

    statesthresh = params("nstates thresh")
    printout("Optimal states for fragment 0")
    frags[0].states, frags[0].state_indices = states.trim_states(frag_rhos[0], statesthresh, frags[0].n_elec_ref, frag_configs[0], printout=indented(printout))
    #printout("Optimal states for fragment 1")
    #frags[1].states, frags[1].state_indices = states.trim_states(frag_rhos[1], statesthresh, frags[1].n_elec_ref, frag_configs[1], printout=indented(printout))

    #label += "_nth"
    #printout("Reduced density tensors")
    #label += "_" + densities.build_tensors(frags, thresh=1e-30, options=params("compress nat_orbs abs_anti"), printout=indented(printout), n_threads=params.n_threads)

    #frags[0].states    = None    # otherwise huge files
    #frags[1].states    = None    # otherwise huge files
    pickle.dump(frags[0], open(f"rho/{label}.pkl".format(0), "wb"))    # users responsibility to softlink rho/ to different volume if desired
    #pickle.dump(frags[1], open(f"rho/{label}.pkl".format(1), "wb"))    # users responsibility to softlink rho/ to different volume if desired
