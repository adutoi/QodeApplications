#    (C) Copyright 2018, 2019 Anthony D. Dutoi
# 
#    This file is part of Qode.
# 
#    Qode is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
# 
#    Qode is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
# 
#    You should have received a copy of the GNU General Public License
#    along with Qode.  If not, see <http://www.gnu.org/licenses/>.
#

import sys
import numpy
import multiprocessing
from   qode.util     import parallel, output, textlog
from   qode.util.PyC import import_C, Double
import excitonic
from   build_density_tensors import build_density_tensors
from   block_ints import block_ints

contract = import_C("H_contractions", flags="-O2")
contract.monomer.return_type(float)
contract.dimer_2min2pls.return_type(float)
contract.dimer_1min1pls.return_type(float)
contract.dimer_00.return_type(float)
contract.dimer_1pls1min.return_type(float)
contract.dimer_2pls2min.return_type(float)



# This class manages the many different tensor contractions of primitive integrals and fragment density tensors,
# which are needed to build the renormalized supersystem Hamiltonian matrix.  Its member functions are called
# from inside loops over fragments and states of those fragments, building matrix elements one at a time.
class build_matrix_elements(object):
    def __init__(self, rho, h, V, n_orb, n_elec):    # n_elec refers to the fragments in their reference states
        n_orb = [2*n for n in n_orb]                 # convert to spin orbitals (does not alter original list)
        n_orb_tot = sum(n_orb)
        self.data = rho, h, V, n_orb, n_orb_tot, n_elec
    def monomer(self, fragment, I, J):
        rho, h, V, n_orb, n_orb_tot, n_elec = self.data
        m = fragment
        i_m_chg, i_m_idx = I
        j_m_chg, j_m_idx = J
        if i_m_chg == j_m_chg:
            n_orb = n_orb[m]
            Rca   = rho[m]["ca"  ][(i_m_chg, j_m_chg)][i_m_idx][j_m_idx]
            Rccaa = rho[m]["ccaa"][(i_m_chg, j_m_chg)][i_m_idx][j_m_idx]
            return contract.monomer(n_orb, Rca, Rccaa, h[m][m], V[m][m][m][m])
        else:
            return 0
    def dimer(self, fragments, I, J):
        rho, h, V, n_orb, n_orb_tot, n_elec = self.data
        m1, m2 = fragments
        i_m1, i_m2 = I
        j_m1, j_m2 = J
        i_m1_chg, i_m1_idx = i_m1
        i_m2_chg, i_m2_idx = i_m2
        j_m1_chg, j_m1_idx = j_m1
        j_m2_chg, j_m2_idx = j_m2
        if (i_m1_chg+i_m2_chg == j_m1_chg+j_m2_chg) and (i_m1_chg >= j_m1_chg-2) and (i_m1_chg <= j_m1_chg+2):
            n_orb1 = n_orb[m1]
            n_orb2 = n_orb[m2]
            if i_m1_chg == j_m1_chg-2:
                Rcc1 = rho[m1]["cc"][(i_m1_chg, j_m1_chg)][i_m1_idx][j_m1_idx]
                Raa2 = rho[m2]["aa"][(i_m2_chg, j_m2_chg)][i_m2_idx][j_m2_idx]
                return contract.dimer_2min2pls(n_orb1, n_orb2, Rcc1, Raa2, V[m1][m1][m2][m2])
            if i_m1_chg == j_m1_chg-1:
                Rc1   = rho[m1]["c"  ][(i_m1_chg, j_m1_chg)][i_m1_idx][j_m1_idx]
                Rcca1 = rho[m1]["cca"][(i_m1_chg, j_m1_chg)][i_m1_idx][j_m1_idx]
                Ra2   = rho[m2]["a"  ][(i_m2_chg, j_m2_chg)][i_m2_idx][j_m2_idx]
                Rcaa2 = rho[m2]["caa"][(i_m2_chg, j_m2_chg)][i_m2_idx][j_m2_idx]
                sign = -(-1)**((n_elec[m2]+j_m2_chg) % 2)
                return sign * contract.dimer_1min1pls(n_orb1, n_orb2, Rc1, Rcca1, Ra2, Rcaa2, h[m1][m2], V[m1][m1][m1][m2], V[m1][m2][m2][m2])
            if i_m1_chg == j_m1_chg:
                Rca1 = rho[m1]["ca"][(i_m1_chg, j_m1_chg)][i_m1_idx][j_m1_idx]
                Rca2 = rho[m2]["ca"][(i_m2_chg, j_m2_chg)][i_m2_idx][j_m2_idx]
                return contract.dimer_00(n_orb1, n_orb2, Rca1, Rca2, V[m1][m2][m1][m2])
            if i_m1_chg == j_m1_chg+1:
                Ra1   = rho[m1]["a"  ][(i_m1_chg, j_m1_chg)][i_m1_idx][j_m1_idx]
                Rcaa1 = rho[m1]["caa"][(i_m1_chg, j_m1_chg)][i_m1_idx][j_m1_idx]
                Rc2   = rho[m2]["c"  ][(i_m2_chg, j_m2_chg)][i_m2_idx][j_m2_idx]
                Rcca2 = rho[m2]["cca"][(i_m2_chg, j_m2_chg)][i_m2_idx][j_m2_idx]
                sign = +(-1)**((n_elec[m2]+j_m2_chg) % 2)
                return sign * contract.dimer_1pls1min(n_orb1, n_orb2, Ra1, Rcaa1, Rc2, Rcca2, h[m2][m1], V[m2][m2][m2][m1], V[m2][m1][m1][m1])
            if i_m1_chg == j_m1_chg+2:
                Raa1 = rho[m1]["aa"][(i_m1_chg, j_m1_chg)][i_m1_idx][j_m1_idx]
                Rcc2 = rho[m2]["cc"][(i_m2_chg, j_m2_chg)][i_m2_idx][j_m2_idx]
                return contract.dimer_2pls2min(n_orb1, n_orb2, Raa1, Rcc2, V[m2][m2][m1][m1])
        else:
                return 0



# Presently, we are testing only on homogeneous systems (all fragments are chemically identical).
# Here are some (temporarily) hard-coded system parameters for frozen-core 6-31G Be atoms.
n_elec_frag = 4    # The number of electrons when the fragment is neutral
n_orb_frag  = 9    # The number of *spatial* orbitals for each fragment
n_core_frag = 1    # The number of frozen (uncorrelated) core orbitals

# Furthermore, our integrals files are presently only for dimers, so hardcode this here
n_frag = 2



# Here we load up the system-wide primitive description of the physical interactions of individual electrons.
# These will be contracted with descriptions of the fragment states below to give a compressed ("renormalized")
# description of interactions between entire fragments.  This is presently hardcoded to look for specific systems
# (Be dimers) only varying the distance between them.
Be_C = numpy.load("atomic_states/data/Be_C.npy")
h, V = block_ints(sys.argv[1], Be_C)	# h = 1-electron integrals (kinetic energy, nuclear attraction); V = 2-electron integrals (electron repulsion)
print("integrals loaded")

# Load descriptions of the fragment many-electron states.  Then process them (using build_density_tensors) to arrive at
# quantities suitable for contracting with the integrals loaded above.  Since we are presently only working with
# homogeneous systems, only one "example" fragment is loaded.
class states(object): pass      #
neutral = states()              # Small dictionaries to hold symbolic representations of fragment electron configurations
cation  = states()              # and their associated numerical coefficients in the fragment wavefunctions.
anion   = states()              #
neutral.coeffs  = numpy.load("atomic_states/H/16-115-550/{}/Z_2e.npy".format(sys.argv[2])).T   #
neutral.configs = numpy.load("atomic_states/data/configs_2e-stable.npy")                       # In a supersystem, any given
cation.coeffs   = numpy.load("atomic_states/H/16-115-550/{}/Z_1e.npy".format(sys.argv[2])).T   # fragment may be found as neutral
cation.configs  = numpy.load("atomic_states/data/configs_1e-stable.npy")                       # or with an extra or missing electron
anion.coeffs    = numpy.load("atomic_states/H/16-115-550/{}/Z_3e.npy".format(sys.argv[2])).T   # (anionic or cationic, respectively).
anion.configs   = numpy.load("atomic_states/data/configs_3e-stable.npy")                       #
rho_frag = build_density_tensors({0:neutral, +1:cation, -1:anion}, n_orb_frag, n_core_frag)       # Compute the density tensors
n_states_frag = { 0:neutral.coeffs.shape[0], +1:cation.coeffs.shape[0], -1:anion.coeffs.shape[0]} # Number of fragment states for each fragment charge (0,+1,-1)
print("rho computed")



# Set up the homogeneous supersystem by simply making many pointers to the information for a single fragment.
# The information that distinguishes among them (e.g., who is next to to whom) is carried by the interaction
# integrals.
n_elec   =   [n_elec_frag]*n_frag
n_orb    =    [n_orb_frag]*n_frag
rho      =      [rho_frag]*n_frag
n_states = [n_states_frag]*n_frag



# Instantiate the engine that computes the matrix elements.
# (n_orb will eventually not be necessary, when the integrals get passed in a more structured way.)
# Wrap internal functions for easier multiprocessing behavior.
compute = build_matrix_elements(rho, h, V, n_orb, n_elec)
def compute_monomer(args):
    M, i, j, I, J = args
    return M, i, j, compute.monomer(M, I, J)
def compute_dimer(args):
    M, N, i, j, I, J = args
    return M, N, i, j, compute.dimer((M,N), I, J)



if __name__ == '__main__':

    # Allocate storage for all of the matrix elements to be computed.
    frag_dims = [sum(n for chg,n in chg_n.items()) for chg_n in n_states]
    H1 = []
    H2 = []
    for M,dimM in enumerate(frag_dims):
        H1 += [numpy.zeros((dimM,dimM), dtype=Double.numpy)]
        H2row = []
        for N,dimN in enumerate(frag_dims):
            if M<N:  H2row += [numpy.zeros((dimM*dimM, dimN*dimN), dtype=Double.numpy)]
            else:    H2row += [None]
        H2 += [H2row]

    # List all charges and states indices for each subsystem.
    state_indices = []
    for chg_n in n_states:
        state_indices_M = []
        for chg,n in chg_n.items():
            for i in range(n):
                state_indices_M += [(chg,i)]
        state_indices += [state_indices_M]

    # Set up the renormalized monomer terms.
    args1 = []
    for M,state_indices_M in enumerate(state_indices):
        for i,I in enumerate(state_indices_M):
            for j,J in enumerate(state_indices_M):
                #H1[M][i,j] = compute.monomer(M, I, J)
                args1 += [(M, i, j, I, J)]
        print("Set up H1[{}]".format(M))

    # Set up the renormalized dimer terms.
    args2 = []
    for M,state_indices_M in enumerate(state_indices):
        for N,state_indices_N in list(enumerate(state_indices))[M+1:]:
            tens_pdt_basis = []
            for iM in state_indices_M:
                for iN in state_indices_N:
                    tens_pdt_basis += [(iM,iN)]
            for i,I in enumerate(tens_pdt_basis):
                for j,J in enumerate(tens_pdt_basis):
                    args2 += [(M, N, i, j, I, J)]
            print("Set up H2[{}][{}]".format(M,N))

    # Compute all the terms in parallel
    pool = multiprocessing.Pool(30)
    results1 = pool.map(compute_monomer,args1)
    for M, i, j, H1Mij     in results1:  H1[M][i,j]    = H1Mij
    print("Finished H1")
    results2 = pool.map(compute_dimer,  args2)
    for M, N, i, j, H2MNij in results2:  H2[M][N][i,j] = H2MNij
    print("Finished H2")



    # Compute the XR2-CCSD energy
    out, resources = output(log=textlog(echo=True)), parallel.resources(1)
    E_e = excitonic.ccsd((H1,H2), out, resources)
    E_n = 4*4 * 0.52917721067 / float(sys.argv[1])
    out.log("\nTotal Excitonic CCSD Energy = ", E_n + E_e)
