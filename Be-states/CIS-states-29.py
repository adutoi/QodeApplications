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
import densities
import pickle
class empty(object):  pass

n_threads = 1
dist = float(sys.argv[1])
if len(sys.argv)==3:  n_threads = int(sys.argv[2])



# Set up fragments (performing a couple of consistency checks along the way)

frag0 = empty()
frag0.atoms = [("Be",[0,0,0])]
frag0.n_elec_ref = 4	# The number of electrons in the reference state of the monomer ("cation (+1)" and "anion (-1)" and technically interpreted relative to the reference, not zero, as would be the chemical definition)
frag0.basis = empty()
frag0.basis.AOcode = "6-31G"
frag0.basis.n_spatial_orb = 9
frag0.basis.MOcoeffs = numpy.identity(frag0.basis.n_spatial_orb)    # rest of code assumes spin-restricted orbitals
frag0.basis.core = []	# indices of spatial MOs to freeze in CI portions

psi4_check.print_HF_energy(
    "".join("{} {} {} {}\n".format(A,x,y,z) for A,(x,y,z) in frag0.atoms),
    frag0.basis.AOcode
    )

symm_ints, bior_ints, nuc_rep = get_ints([frag0], spin_ints=False)
N, S, T, U, V = nuc_rep[0,0], symm_ints.S[0,0], symm_ints.T[0,0], symm_ints.U[0,0,0], symm_ints.V[0,0,0,0]
E, e, frag0.basis.MOcoeffs = RHF_RoothanHall_Nonorthogonal(frag0.n_elec_ref, (S, T+U, V), thresh=1e-12)    # changes frag0 (eg, in next line)
print(E)

symm_ints, bior_ints, nuc_rep = get_ints([frag0], spin_ints=False)
N, S, T, U, V = nuc_rep[0,0], symm_ints.S[0,0], symm_ints.T[0,0], symm_ints.U[0,0,0], symm_ints.V[0,0,0,0]
E, e, _ = RHF_RoothanHall_Orthonormal(frag0.n_elec_ref, (T+U, V), thresh=1e-12)
print(E)

frag1 = empty()
frag1.atoms = [("Be",[0,0,dist])]
frag1.n_elec_ref = 4
frag1.basis = empty()
frag1.basis.AOcode = "6-31G"
frag1.basis.n_spatial_orb = 9
frag1.basis.MOcoeffs = frag0.basis.MOcoeffs    # use MO coeffs from frag0
frag1.basis.core = [0]

# shortcuts
num_elec_atom_dn = frag0.n_elec_ref // 2
num_elec_atom_up = frag0.n_elec_ref - num_elec_atom_dn
num_spatial_atom = frag0.basis.n_spatial_orb

# screw it, do this manually (CIS for atoms)
configs_atom = [
0b000000011000000011,
0b000000011000000110,
0b000000011000001010,
0b000000011000010010,
0b000000011000100010,
0b000000011001000010,
0b000000011010000010,
0b000000011100000010,
0b000000011000000101,
0b000000011000001001,
0b000000011000010001,
0b000000011000100001,
0b000000011001000001,
0b000000011010000001,
0b000000011100000001,
0b000000110000000011,
0b000001010000000011,
0b000010010000000011,
0b000100010000000011,
0b001000010000000011,
0b010000010000000011,
0b100000010000000011,
0b000000101000000011,
0b000001001000000011,
0b000010001000000011,
0b000100001000000011,
0b001000001000000011,
0b010000001000000011,
0b100000001000000011
]
configs_atom = sorted(configs_atom)


# Dimer calculation

symm_ints, bior_ints, nuc_rep = get_ints([frag0,frag1])

N = nuc_rep[0,0] + nuc_rep[1,1] + nuc_rep[0,1]
T = unblock_2(    bior_ints.T, [frag0,frag1], spin_orbs=True)
U = unblock_last2(bior_ints.U, [frag0,frag1], spin_orbs=True)
V = unblock_4(    bior_ints.V, [frag0,frag1], spin_orbs=True)
h = T + U[0] + U[1]

dimer_core = frag0.basis.core + [c+frag0.basis.n_spatial_orb for c in frag1.basis.core]

core = dimer_core + [c+2*frag0.basis.n_spatial_orb for c in dimer_core]
orbs = list(range(4*9))
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

configs_dimer = configurations.tensor_product_configs([configs_atom, configs_atom], [2*num_spatial_atom, 2*num_spatial_atom])

CI_space_dimer = qode.math.linear_inner_product_space(CI_space_traits(configs_dimer))
H     = CI_space_dimer.lin_op(field_op_ham.Hamiltonian(h,V, n_threads=n_threads))
guess = CI_space_dimer.member(CI_space_dimer.aux.basis_vec([0,1,9,10,18,19,27,28]))

print((guess|H|guess) + N)
(Eval,Evec), = qode.math.lanczos.lowest_eigen(H, [guess], thresh=1e-8)
print("\nE_gs = {}\n".format(Eval+N))






states = {}
chg = 0
states[chg] = empty()
states[chg].configs = configs_atom    # there are 29 configs
states[chg].coeffs  = []
evecs = numpy.identity(29)
for i in range(29):
    tmp = numpy.zeros(29, dtype=Double.numpy)
    tmp[:] = evecs[:,i]
    states[chg].coeffs += [tmp]

for chg,states_chg in states.items():
    num_states = len(states_chg.coeffs)
    if num_states>0:
        print("{}: {} x {}".format(chg, num_states, states_chg.coeffs[0].shape))
        #for config in states_chg.configs:
        #    print("  {:018b}".format(config))

frag0.rho = densities.build_tensors(states, 2*frag0.basis.n_spatial_orb, frag0.n_elec_ref, thresh=1e-30, n_threads=n_threads)

ref_chg, ref_idx = 0, 0
frag0.state_indices = [(ref_chg,ref_idx)]                # List of all charge and state indices, reference state needs to be first, but otherwise irrelevant order
for i in range(len(states[ref_chg].coeffs)):
    if   i!=ref_idx:  frag0.state_indices += [(ref_chg,i)]
for chg in states:
    if chg!=ref_chg:  frag0.state_indices += [(chg,i) for i in range(len(states[chg].coeffs))]

pickle.dump(frag0, open("/scratch/adutoi/Be631g-CIS.pkl", "wb"))
