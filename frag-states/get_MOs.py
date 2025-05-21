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
import numpy
from qode.util import struct, logger, indented, no_print
from qode.many_body.self_consistent_field.fermionic import RHF_RoothanHall_Nonorthogonal, RHF_RoothanHall_Orthonormal
from get_ints import get_ints

def fragment_HF(frag, printout=print, archive=None):
    if archive is None:  archive = struct()
    archive.printout = logger(printout)
    if frag.multiplicity!=1 or frag.n_elec_ref%2!=0:
        raise NotImplementedError("currently can only handle singlets")

    frag.basis.MOcoeffs = numpy.identity(frag.basis.n_spatial_orb)    # integrals will be for spatial AOs
    symm_ints, bior_ints, nuc_rep = get_ints([frag], spin_ints=False, printout=indented(no_print))
    N, S, T, U, V = nuc_rep[0,0], symm_ints.S[0,0], symm_ints.T[0,0], symm_ints.U[0,0,0], symm_ints.V[0,0,0,0]
    archive.printout("AO integrals initiated")

    HF_result = RHF_RoothanHall_Nonorthogonal(frag.n_elec_ref, (S, T+U, V), thresh=1e-12)
    HF_result.n_elec_dn = frag.n_elec_ref // 2
    HF_result.n_elec_up = HF_result.n_elec_dn
    HF_result.occupied_dn, n, i = list(frag.basis.core), len(frag.basis.core), 0
    while n<HF_result.n_elec_dn:
        if i not in frag.basis.core:
            HF_result.occupied_dn += [i]
            n += 1
        i += 1
    HF_result.occupied_up = HF_result.occupied_dn
    archive.result = HF_result
    archive.printout(f"Hartree-Fock completed.  Total HF energy = {HF_result.energy}")

    frag.HartreeFock = HF_result
    frag.basis.MOcoeffs = frag.HartreeFock.MO_coeffs

