#    (C) Copyright 2023 Anthony D. Dutoi
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

import sys
import numpy
import qode

from qode.many_body.self_consistent_field.fermionic import RHF_RoothanHall_Nonorthogonal, RHF_RoothanHall_Orthonormal
from get_ints import get_ints
from qode.atoms.integrals.fragments import unblock_2, unblock_last2, unblock_4
import psi4_check

from CI_space_traits import CI_space_traits
import field_op_ham
import configurations

from qode.util.PyC import Double

class _empty(object):  pass

n_threads = 1
dist = float(sys.argv[1])
if len(sys.argv)==3:  n_threads = int(sys.argv[2])


frag0 = _empty()
frag0.atoms = [("Be",[0,0,0])]
frag0.n_elec_ref = 4	# The number of electrons in the reference state of the monomer ("cation (+1)" and "anion (-1)" and technically interpreted relative to the reference, not zero, as would be the chemical definition)
frag0.basis = _empty()
frag0.basis.AOcode = "6-31G"
frag0.basis.n_spatial_orb = 9
frag0.basis.MOcoeffs = numpy.identity(frag0.basis.n_spatial_orb)    # rest of code assumes spin-restricted orbitals
frag0.basis.core = [0]	# indices of spatial MOs to freeze in CI portions



psi4_check.print_HF_energy(
    "".join("{} {} {} {}\n".format(A,x,y,z) for A,(x,y,z) in frag0.atoms),
    frag0.basis.AOcode
    )

symm_ints, bior_ints, nuc_rep = get_ints([frag0], spin_ints=False)
N, S, T, U, V = nuc_rep[0,0], symm_ints.S[0,0], symm_ints.T[0,0], symm_ints.U[0,0,0], symm_ints.V[0,0,0,0]
E, e, frag0.basis.MOcoeffs = RHF_RoothanHall_Nonorthogonal(frag0.n_elec_ref, (S, T+U, V), thresh=1e-12)
print(E)

symm_ints, bior_ints, nuc_rep = get_ints([frag0], spin_ints=False)
N, S, T, U, V = nuc_rep[0,0], symm_ints.S[0,0], symm_ints.T[0,0], symm_ints.U[0,0,0], symm_ints.V[0,0,0,0]
E, e, _ = RHF_RoothanHall_Orthonormal(frag0.n_elec_ref, (T+U, V), thresh=1e-12)
print(E)



frag1 = _empty()
frag1.atoms = [("Be",[0,0,dist])]
frag1.n_elec_ref = 4
frag1.basis = _empty()
frag1.basis.AOcode = "6-31G"
frag1.basis.n_spatial_orb = 9
frag1.basis.MOcoeffs = frag0.basis.MOcoeffs
frag1.basis.core = [0]

symm_ints, bior_ints, nuc_rep = get_ints([frag0,frag1])

num_elec_atom_dn = frag0.n_elec_ref // 2
num_elec_atom_up = frag0.n_elec_ref - num_elec_atom_dn
num_spatial_atom = frag0.basis.n_spatial_orb



dn_configs_atom = configurations.all_configs(num_spatial_atom, num_elec_atom_dn-len(frag0.basis.core), frozen_occ_orbs=frag0.basis.core)
up_configs_atom = configurations.all_configs(num_spatial_atom, num_elec_atom_up-len(frag0.basis.core), frozen_occ_orbs=frag0.basis.core)
configs_atom    = configurations.tensor_product_configs([up_configs_atom,up_configs_atom], [num_spatial_atom,num_spatial_atom])

N, S, T, U, V = nuc_rep[0,0], symm_ints.S[0,0], symm_ints.T[0,0], symm_ints.U[0,0,0], symm_ints.V[0,0,0,0]
h = T + U


shift = 60
factor = 2**shift
new_configs_atom = [config*factor for config in configs_atom]

dim = h.shape[0]
new_h = numpy.zeros((dim+shift,dim+shift), dtype=Double.numpy)
new_V = numpy.zeros((dim+shift,dim+shift,dim+shift,dim+shift), dtype=Double.numpy)
new_h[shift:shift+dim,shift:shift+dim] = h
new_V[shift:shift+dim,shift:shift+dim,shift:shift+dim,shift:shift+dim] = V




CI_space_atom = qode.math.linear_inner_product_space(CI_space_traits(new_configs_atom))
H     = CI_space_atom.lin_op(field_op_ham.Hamiltonian(new_h,new_V, n_threads=n_threads))
guess = CI_space_atom.member(CI_space_atom.aux.basis_vec([shift+0,shift+1,shift+9,shift+10]))

print((guess|H|guess) + N)
(Eval,Evec), = qode.math.lanczos.lowest_eigen(H, [guess], thresh=1e-8)
print("\nE_gs = {}\n".format(Eval+N))
