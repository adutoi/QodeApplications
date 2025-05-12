#    (C) Copyright 2023, 2024 Anthony D. Dutoi
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

from qode.util import struct
from qode.util.PyC import Double
from qode.many_body.self_consistent_field.fermionic import RHF_RoothanHall_Nonorthogonal, RHF_RoothanHall_Orthonormal
from qode.atoms.integrals.fragments import unblock_2, unblock_last2, unblock_4

from get_ints import get_ints
import psi4_check

from CI_space_traits import CI_space_traits
import field_op
import field_op_ham
import configurations

import densities



def lala(basis):

    basis_label, n_spatial_orb = basis

    frag = struct(
        atoms = [("Be",[0,0,0])],
        n_elec_ref = 4,	# The number of electrons in the reference state of the monomer ("cation (+1)" and "anion (-1)" and technically interpreted relative to the reference, not zero, as would be the chemical definition)
        basis = struct(
            AOcode = basis_label,
            n_spatial_orb = n_spatial_orb,
            MOcoeffs = numpy.identity(n_spatial_orb),    # rest of code assumes spin-restricted orbitals
            core = [0]	# indices of spatial MOs to freeze in CI portions
        )
    )

    num_spatial  = frag.basis.n_spatial_orb
    num_spin     = 2*num_spatial
    core         = frag.basis.core + [c+num_spatial for c in frag.basis.core]
    num_core     = len(core)
    num_elec_ref = frag.n_elec_ref - num_core

    configs = {}

    configs[ 0] = field_op.packed_configs(configurations.all_configs(num_spin, num_elec_ref,   frozen_occ_orbs=core))
    configs[+1] = field_op.packed_configs(configurations.all_configs(num_spin, num_elec_ref-1, frozen_occ_orbs=core))
    configs[-1] = field_op.packed_configs(configurations.all_configs(num_spin, num_elec_ref+1, frozen_occ_orbs=core))

    print(len(configs[ 0]))
    print(len(configs[+1]))
    print(len(configs[-1]))

    aa, caaa, a, caa, ccaaa, ca, ccaa = {}, {}, {}, {}, {}, {}, {}

    num_elec_ref += num_core

    aa[+1,-1]    = field_op.determinant_densities("aa",    num_spin, num_elec_ref+1, configs[+1], configs[-1])
    caaa[+1,-1]  = field_op.determinant_densities("caaa",  num_spin, num_elec_ref+1, configs[+1], configs[-1])

    a[+1, 0]     = field_op.determinant_densities("a",     num_spin, num_elec_ref,   configs[+1], configs[ 0])
    caa[+1, 0]   = field_op.determinant_densities("caa",   num_spin, num_elec_ref,   configs[+1], configs[ 0])
    ccaaa[+1, 0] = field_op.determinant_densities("ccaaa", num_spin, num_elec_ref,   configs[+1], configs[ 0])
    a[ 0,-1]     = field_op.determinant_densities("a",     num_spin, num_elec_ref+1, configs[ 0], configs[-1])
    caa[ 0,-1]   = field_op.determinant_densities("caa",   num_spin, num_elec_ref+1, configs[ 0], configs[-1])
    ccaaa[ 0,-1] = field_op.determinant_densities("ccaaa", num_spin, num_elec_ref+1, configs[ 0], configs[-1])

    ca[ 0, 0]    = field_op.determinant_densities("ca",    num_spin, num_elec_ref,   configs[ 0], configs[ 0])
    ccaa[ 0, 0]  = field_op.determinant_densities("ccaa",  num_spin, num_elec_ref,   configs[ 0], configs[ 0])
    ca[+1,+1]    = field_op.determinant_densities("ca",    num_spin, num_elec_ref-1, configs[+1], configs[+1])
    ccaa[+1,+1]  = field_op.determinant_densities("ccaa",  num_spin, num_elec_ref-1, configs[+1], configs[+1])
    ca[-1,-1]    = field_op.determinant_densities("ca",    num_spin, num_elec_ref+1, configs[-1], configs[-1])
    ccaa[-1,-1]  = field_op.determinant_densities("ccaa",  num_spin, num_elec_ref+1, configs[-1], configs[-1])

    return aa, caaa, a, caa, ccaaa, ca, ccaa
