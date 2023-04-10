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

# Use within a Psi4 conda environment for integrals

import sys
import numpy
import multiprocessing
from   qode.util     import parallel, output, textlog
from   qode.util.PyC import Double
from   build_H import build_matrix_elements
import lala
from   Be631g_compress import monomer_data as Be
from   get_ints import get_ints



# Read information about the supersystem from the command line (assume a homogeneous system of identical, evenly spaced fragments in a line)
# and load up some hard-coded info for frozen-core, generalized CI 6-31G Be atoms, most importantly reduced descriptions of the fragment
# many-electron states that are suitable for contracting with the integrals.
exec(open(sys.argv[1]+".in","r").read())
core = [core_0, core_1]
n_states = [n_states_0, n_states_1]



# Assemble the supersystem from the displaced fragments
BeN = [Be((0,0,m*displacement), core[m]) for m in range(n_frag)]
integrals, nuc_repulsion = get_ints(BeN)
for m,frag in enumerate(BeN):  frag.load_states(states, integrals.V[m,m,m,m], n_states[m])

# Instantiate the engine that computes the matrix elements.  Wrap internal functions for easier multiprocessing behavior.
compute = build_matrix_elements(BeN, integrals, nuc_repulsion.matrix)
def compute_monomer(args):
    M, i, j, I, J = args
    return M, i, j, compute.monomer(M, I, J)
def compute_dimer(args):
    M, N, i, j, I, J = args
    return M, N, i, j, compute.dimer((M,N), I, J)
#def compute_trimer(args):
#    M, N, O, i, j, I, J = args
#    return M, N, O, i, j, compute.trimer((M,N,O), I, J)



if __name__ == '__main__':

    # Allocate storage for all of the matrix elements to be computed.
    frag_dims = [len(fragment.state_indices) for fragment in BeN]
    H1 = []
    H2 = []
    #H3 = []
    for M,dimM in enumerate(frag_dims):
        H1 += [numpy.zeros((dimM,dimM), dtype=Double.numpy)]
        H2row = []
        #H3row = []
        for N,dimN in enumerate(frag_dims):
            if M<N:
                H2row += [numpy.zeros((dimM*dimN, dimM*dimN), dtype=Double.numpy)]
                #H3elm = []
                #for O,dimO in enumerate(frag_dims):
                #    if N<O:
                #        H3elm += [numpy.zeros((dimM*dimN*dimO, dimM*dimN*dimO), dtype=Double.numpy)]
                #    else:
                #        H3elm += [None]
                #H3row += [H3elm]
            else:
                H2row += [None]
                #H3row += [None]
        H2 += [H2row]
        #H3 += [H3row]

    # Set up the renormalized monomer terms.
    args1 = []
    for M,frag in enumerate(BeN):
        for i,I in enumerate(frag.state_indices):
            for j,J in enumerate(frag.state_indices):
                args1 += [(M, i, j, I, J)]
        print("Set up H1[{}]".format(M))

    # Set up the renormalized dimer terms.
    args2 = []
    for M,fragM in enumerate(BeN):
        for N,fragN in list(enumerate(BeN))[M+1:]:
            tens_pdt_basis = []
            for iM in fragM.state_indices:
                for iN in fragN.state_indices:
                    tens_pdt_basis += [(iM,iN)]
            for i,I in enumerate(tens_pdt_basis):
                for j,J in enumerate(tens_pdt_basis):
                    args2 += [(M, N, i, j, I, J)]
            print("Set up H2[{}][{}]".format(M,N))

    # Set up the renormalized trimer terms.
    #args3 = []
    #for M,fragM in enumerate(BeN):
    #    for N,fragN in list(enumerate(BeN))[M+1:]:
    #        for O,fragO in list(enumerate(BeN))[N+1:]:
    #            print(M,N,O)
    #            tens_pdt_basis = []
    #            for iM in fragM.state_indices:
    #                for iN in fragN.state_indices:
    #                    for iO in fragO.state_indices:
    #                        tens_pdt_basis += [(iM,iN,iO)]
    #            for i,I in enumerate(tens_pdt_basis):
    #                print(i, "of", len(tens_pdt_basis))
    #                for j,J in enumerate(tens_pdt_basis):
    #                    args3 += [(M, N, O, i, j, I, J)]
    #            print("Set up H3[{}][{}][{}]".format(M,N,O))

    # Compute all the terms in parallel
    pool = multiprocessing.Pool(30)
    results1 = pool.map(compute_monomer,args1)
    #results1 = [compute_monomer(arg) for arg in args1]
    for M, i, j, H1Mij     in results1:  H1[M][i,j]    = H1Mij
    print("Finished H1")
    results2 = pool.map(compute_dimer,  args2)
    #results2 = [compute_dimer(arg) for arg in args2]
    for M, N, i, j, H2MNij in results2:  H2[M][N][i,j] = H2MNij
    print("Finished H2")
    #results3 = pool.map(compute_trimer,  args3)
    #for M, N, O, i, j, H3MNOij in results3:  H3[M][N][O][i,j] = H3MNOij
    #print("Finished H3")



    # Compute the XR2-FCI energy
    out, resources = output(log=textlog(echo=True)), parallel.resources(1)
    E = lala.fci((H1,H2,None), out, sys.argv[1])
    out.log("\nTotal Excitonic FCI  Energy = ", E)
