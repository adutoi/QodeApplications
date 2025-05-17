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
from qode.util import struct, read_input
import input_env
import states
import densities
# these belong elsewhere
n_elec_ref = {"H": 1, "Be": 4, "C": 6}
n_spatial_orb = {
    "6-31G": {"H": 2, "Be": 9, "C": 9}
}

if __name__=="__main__":

    params = struct(    # defaults
        n_threads = 1,
        nstates   = None,
        thresh    = None,
        compress  = "SVD,cc-aa",
        nat_orbs  = False,
        abs_anti  = False
    )
    params.update(read_input.from_command_line(namespace=input_env))

    frags = [
        struct(
            atoms = frag.atoms,
            n_elec_ref = sum(n_elec_ref[atom.element] for atom in frag.atoms),    # "cation (+1)" and "anion (-1)" are interpreted relative to the reference
            basis = struct(
                AOcode = params.basis,
                n_spatial_orb = sum(n_spatial_orb[params.basis][atom.element] for atom in frag.atoms),
                core = frag.core    # indices of spatial MOs to freeze in CI portions
            )
        )
    for frag in params.frags]

    label  = "-".join("".join(atom.element for atom in frag.atoms) for frag in params.frags)
    label += "_" + params.basis
    label += "_" + states.get_optimal(frags, params("nstates thresh"), n_threads=params.n_threads)
    label += "_" + densities.build_tensors(frags, thresh=1e-30, options=params("compress nat_orbs abs_anti"), n_threads=params.n_threads)

    frags[0].states = None
    pickle.dump(frags[0], open(f"rho/{label}.pkl", "wb"))    # users responsibility to softlink rho/ to different volume if desired
