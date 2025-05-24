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
from qode.atoms.integrals.fragments import unblock_2, unblock_last2, unblock_4
from qode.many_body.self_consistent_field.fermionic import RHF_RoothanHall_Nonorthogonal
from qode.many_body.fermion_field import combine_orb_lists
from get_ints import get_ints

def fragment_HF(frag, Sz=0, printout=print, archive=None):
    if archive is None:  archive = struct()
    archive.printout = logger(printout)

    frag.basis.MOcoeffs = numpy.identity(frag.basis.n_spatial_orb)    # integrals will be for spatial AOs
    symm_ints, bior_ints, nuc_rep = get_ints([frag], spin_ints=False, printout=indented(no_print))
    N, S, T, U, V = nuc_rep[0,0], symm_ints.S[0,0], symm_ints.T[0,0], symm_ints.U[0,0,0], symm_ints.V[0,0,0,0]
    archive.printout("AO integrals initiated")

    if Sz!=0 or frag.n_elec_ref%2!=0:
        raise NotImplementedError("currently can only handle Sz=0 states")
    E, e, C = RHF_RoothanHall_Nonorthogonal(frag.n_elec_ref, (S, T+U, V), thresh=1e-12)    # assumes Sz=0 (also, singlet)
    archive.printout(f"Hartree-Fock completed.  Total HF energy = {E}")

    frag.basis.MOcoeffs = C



def mask_core(h, V, core):
    if core is not None:
        orbs = list(range(h.shape[0]))
        for p in orbs:
            for q in orbs:
                if (q in core) and (p!=q):  h[p,q] = 0
                if (p in core) and (p!=q):  h[p,q] = 0
                for r in orbs:
                    for s in orbs:
                        if (r in core) and (p!=r) and (q!=r):  V[p,q,r,s] = 0
                        if (s in core) and (p!=s) and (q!=s):  V[p,q,r,s] = 0
                        if (p in core) and (p!=r) and (p!=s):  V[p,q,r,s] = 0
                        if (q in core) and (q!=r) and (q!=s):  V[p,q,r,s] = 0
    return


def dimer_integrals(frags, printout=print):
    symm_ints, bior_ints, nuc_rep = get_ints(frags, printout=indented(no_print))

    core = (combine_orb_lists(frags[0].basis.core, frags[0].basis.core, frags[0].basis.n_spatial_orb),
            combine_orb_lists(frags[1].basis.core, frags[1].basis.core, frags[1].basis.n_spatial_orb))

    def _monomer_integrals(m):
        N, T, U, V = nuc_rep[m,m], symm_ints.T[m,m], symm_ints.U[m,m,m], symm_ints.V[m,m,m,m]
        h = T + U
        V = V + 0
        mask_core(h, V, core[m])
        return N, h, V
    monomer_ints = [_monomer_integrals(m) for m in range(2)]

    core = combine_orb_lists(core[0], core[1], 2*frags[0].basis.n_spatial_orb)

    N = nuc_rep[0,0] + nuc_rep[1,1] + nuc_rep[0,1]
    T = unblock_2(    bior_ints.T, frags, spin_orbs=True)
    U = unblock_last2(bior_ints.U, frags, spin_orbs=True)
    V = unblock_4(    bior_ints.V, frags, spin_orbs=True)
    h = T + U[0] + U[1]

    mask_core(h, V, core)
    dimer_ints = N, h, V

    return monomer_ints, dimer_ints

