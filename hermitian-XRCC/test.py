#    (C) Copyright 2023 Anthony D. Dutoi and Marco Bauer
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
import torch
import tensorly as tl
import qode.util
import qode.math
import excitonic
import diagrammatic_expansion   # defines information structure for housing results of diagram evaluations
import XR_term                  # knows how to use ^this information to pack a matrix for use in XR model
import S_diagrams               # contains definitions of actual diagrams needed for S operator in BO rep
import Sn_diagrams              # contains definitions of actual diagrams needed for SH operator in BO rep
import St_diagrams              # contains definitions of actual diagrams needed for SH operator in BO rep
import Su_diagrams              # contains definitions of actual diagrams needed for SH operator in BO rep
import Sv_diagrams              # contains definitions of actual diagrams needed for SH operator in BO rep
from   get_ints import get_ints
from   Be631g   import monomer_data as Be
from tendot import tendot
#import ray
import time
import pickle

torch.set_num_threads(4)
tl.set_backend("pytorch")
#tl.set_backend("numpy")


def Be2_XR(displacement):
    #########
    # Load data
    #########

    # Information about the Be2 supersystem
    n_frag       = 2
    #displacement = sys.argv[1]
    states       = "load=states:16-115-550:thresh=1e-6:4.5:u.pickle/4.5"
    n_states     = ("all","all","all","all","all")

    # "Assemble" the supersystem from the displaced fragments
    BeN = [Be((0,0,m*float(displacement))) for m in range(int(n_frag))]

    # Load states and get integrals
    #for m,frag in enumerate(BeN):  frag.load_states(states, n_states)      # load the density tensors
    BeN[0].load_states(states, n_states)
    symm_ints, bior_ints, nuc_rep = get_ints(BeN)

    BeN_rho = [BeN[0].rho for i in range(len(BeN))]

    #print(BeN[0].rho["ca"][0,0][0][0])
    #pickle.dump([BeN[0].rho[i] for i in ["c", "a", "ca"]], open("density_c_a_ca.pkl", mode="wb"))
    pickle.dump(BeN[0].coeffs, open("state_coeffs.pkl", mode="wb"))
    raise ValueError("stop right here")    

    #start_dens_import = time.time()
    #print("starting making densities tl.tensors")
    #for string in BeN[0].rho.keys():
    #    if string in ["n_elec", "n_states"]:
    #            continue
    #    for chgs in BeN[0].rho[string].keys():
    #        for i in range(len(BeN[0].rho[string][chgs])):
    #            for j in range(len(BeN[0].rho[string][chgs][i])):
    #                BeN[0].rho[string][chgs][i][j] = tl.tensor(BeN[0].rho[string][chgs][i][j], dtype=tl.float64)
    #print("done importing", time.time() - start_dens_import)

    #D = [BeN[i].rho["ca"][0,0][0][0] for i in range(len(BeN))]
    D = [BeN_rho[i]["ca"][0,0][0][0] for i in range(len(BeN))]

    # h
    #integrals.h = {}
    #integrals.h[0, 0] = integrals.T[0, 0] + integrals.U[0, 0, 0] + integrals.U[1, 0, 0]
    #integrals.h[0, 1] = integrals.T[0, 1] + integrals.U[0, 0, 1] + integrals.U[1, 0, 1]
    #integrals.h[1, 0] = integrals.T[1, 0] + integrals.U[0, 1, 0] + integrals.U[1, 1, 0]
    #integrals.h[1, 1] = integrals.T[1, 1] + integrals.U[0, 1, 1] + integrals.U[1, 1, 1]
    # f
    #two_p_mean_field = {(0, 0): 2 * (  numpy.einsum("sr,prqs->pq", D[0], integrals.V[0, 0, 0, 0])
    #                                 + numpy.einsum("sr,prqs->pq", D[1], integrals.V[0, 1, 0, 1])),
    #                    (0, 1): 2 * (  numpy.einsum("sr,prqs->pq", D[0], integrals.V[0, 0, 1, 0])
    #                                 + numpy.einsum("sr,prqs->pq", D[1], integrals.V[0, 1, 1, 1])),
    #                    (1, 0): 2 * (  numpy.einsum("sr,prqs->pq", D[0], integrals.V[1, 0, 0, 0])
    #                                 + numpy.einsum("sr,prqs->pq", D[1], integrals.V[1, 1, 0, 1])),
    #                    (1, 1): 2 * (  numpy.einsum("sr,prqs->pq", D[0], integrals.V[1, 0, 1, 0])
    #                                 + numpy.einsum("sr,prqs->pq", D[1], integrals.V[1, 1, 1, 1]))}
    # lack of 0011 and 1100 contributions (they don't belong in here, but maybe compensate for them and for all other charged contributions by contracting with transition densities)
    #integrals.f = {key: integrals.h[key] + two_p_mean_field[key] for key in integrals.h}
    """
    two_p_mean_field_bior= {(0, 0): 2 * (  tendot(D[0], bior_ints.V[0, 0, 0, 0], axes=([0, 1], [3, 1]))
                                    ),#+ tendot(D[1], bior_ints.V[0, 1, 0, 1], axes=([0, 1], [3, 1]))),
                            (0, 1): 2 * (  tendot(D[0], bior_ints.V[0, 0, 1, 0], axes=([0, 1], [3, 1]))
                                    + tendot(D[1], bior_ints.V[0, 1, 1, 1], axes=([0, 1], [3, 1]))),
                            (1, 0): 2 * (  tendot(D[0], bior_ints.V[1, 0, 0, 0], axes=([0, 1], [3, 1]))
                                    + tendot(D[1], bior_ints.V[1, 1, 0, 1], axes=([0, 1], [3, 1]))),
                            (1, 1): 2 * (#  tendot(D[0], bior_ints.V[1, 0, 1, 0], axes=([0, 1], [3, 1]))
                                    + tendot(D[1], bior_ints.V[1, 1, 1, 1], axes=([0, 1], [3, 1])))}
    #symm_ints_fock = numpy.array([[symm_ints.T[m0, m1] + two_p_mean_field_symm[(m0, m1)] for m0 in range(n_frag)] for m1 in range(n_frag)], dtype=object)
    #two_p_mean_bior = torch.tensor([[two_p_mean_field_bior[(m0, m1)] for m0 in range(n_frag)] for m1 in range(n_frag)], dtype=torch.double)
    #symm_ints_fock[0][0] = symm_ints.T[0, 0] + two_p_mean_field_symm[(0, 0)]
    #symm_ints_fock[1][1] = symm_ints.T[0, 0] + two_p_mean_field_symm[(1, 1)]

    two_p_mean_field_bior_half= {(0, 0): 2 * (  tendot(D[0], bior_ints.V_half[0, 0, 0, 0], axes=([0, 1], [3, 1]))
                                    + tendot(D[1], bior_ints.V_half[0, 1, 0, 1], axes=([0, 1], [3, 1]))),
                            (0, 1): 2 * (  tendot(D[0], bior_ints.V_half[0, 0, 1, 0], axes=([0, 1], [3, 1]))
                                    + tendot(D[1], bior_ints.V_half[0, 1, 1, 1], axes=([0, 1], [3, 1]))),
                            (1, 0): 2 * (  tendot(D[0], bior_ints.V_half[1, 0, 0, 0], axes=([0, 1], [3, 1]))
                                    + tendot(D[1], bior_ints.V_half[1, 1, 0, 1], axes=([0, 1], [3, 1]))),
                            (1, 1): 2 * (  tendot(D[0], bior_ints.V_half[1, 0, 1, 0], axes=([0, 1], [3, 1]))
                                    + tendot(D[1], bior_ints.V_half[1, 1, 1, 1], axes=([0, 1], [3, 1])))}
    #bior_ints_fock = numpy.array([[bior_ints.T[m0, m1] + two_p_mean_field_bior[(m0, m1)] for m0 in range(n_frag)] for m1 in range(n_frag)], dtype=object)
    #two_p_mean_bior_half = torch.tensor([[two_p_mean_field_bior_half[(m0, m1)] for m0 in range(n_frag)] for m1 in range(n_frag)], dtype=torch.double)
    #bior_ints_fock[0][0] = bior_ints.T[0, 0] + two_p_mean_field_bior[(0, 0)]
    #bior_ints_fock[1][1] = bior_ints.T[1, 1] + two_p_mean_field_bior[(1, 1)]
    """
    # The engines that build the terms
    #BeN_rho = [frag.rho for frag in BeN]   # diagrammatic_expansion.blocks should take BeN directly? (n_states and n_elec one level higher)

    """
    print("element of cccaa", BeN_rho[0]["cccaa"][(0, 1)][0][0][0, 0, 0, 0, 0])

    for m in range(len(BeN_rho)):
        for op_string in BeN_rho[m]:
            if len(op_string) != 5:
                continue
            for charges in BeN_rho[m][op_string]:
                for i in range(len(BeN_rho[m][op_string][charges])):
                    for j in range(len(BeN_rho[m][op_string][charges][i])):
                        iter = len([BeN_rho[m][op_string][charges][i][j][:, 0, 0, 0, 0]])
                        for k in range(iter):
                            #rand = tl.tensor(numpy.random.rand(18, 18, 18, 18))
                            BeN_rho[m][op_string][charges][i][j][k, :, :, :, :] *= (1.0 + 1e-3 * (-1 + 2 * tl.tensor(numpy.random.random_sample())))#+= 1e-3 * rand
                            #for l in range(iter):
                            #    for m in range(iter):
                            #        for n in range(iter):
                            #            for o in range(iter):
                            #                rand = tl.tensor(numpy.random.random_sample())
                            #                BeN_rho[m][op_string][charges][i][j][k, l, m, n, o] *= (1.0 + rand * 1e-2)#+= 1 * rand * BeN_rho[m][op_string][charges][i][j][k, l, m, n, o]
                                            #if k == l == m == n == o == i == j == 0:
                                            #    BeN_rho[m][op_string][charges][i][j][k, l, m, n, o] += 0.0001


    print("element of cccaa with rand on top", BeN_rho[0]["cccaa"][(0, 1)][0][0][0, 0, 0, 0, 0])
    """
    """
    orb_length = len(BeN_rho[0]["ccaaa"][(+1, 0)][0][0])

    dummy_ccaaa_00 = tl.copy(BeN_rho[0]["ccaaa"][(+1, 0)][0][0])

    for i in range(orb_length):
        for j in range(orb_length):
            for k in range(orb_length):
                for l in range(orb_length):
                    for m in range(orb_length):
                        dummy_ccaaa_00[i,j,k,l,m] *= (1.0 + 5e-3 * (-1 + 2 * tl.tensor(numpy.random.random_sample())))

    print("diff norm of statistical sample for ccaaa_00 with 1e-3 and norm of approx ccaaa_00",
          tl.norm(BeN_rho[0]["ccaaa"][(+1, 0)][0][0] - dummy_ccaaa_00), tl.norm(dummy_ccaaa_00))

    #print("0 to 1 for a", BeN_rho[0]["a"][(0, -1)][0][1])
    #print("0 to 1 for c", BeN_rho[0]["c"][(0, 1)][0][1])
    #print("0 to 1 for ca", BeN_rho[0]["ca"][(0, 0)][0][1])
    
    a_01 = BeN_rho[0]["a"][(+1, 0)][0][0]
    c_01 = BeN_rho[0]["c"][(0, 1)][0][0]
    ca_01 = BeN_rho[0]["ca"][(0, 0)][0][0]
    ca_01_decomp = tl.tenalg.outer((c_01, a_01))

    ca_diff = ca_01 - ca_01_decomp
    print("diff of ca - c outer a", tl.norm(ca_diff), tl.norm(ca_01), tl.norm(ca_01_decomp))

    #print("norms of following contracts", tl.norm(numpy.einsum("ia,i->a", ca_01, c_01)), tl.norm(numpy.einsum("ia->a", ca_01)))
    #print("contract ca to a", tl.norm(a_01 - numpy.einsum("ia,i->a", ca_01, c_01)), tl.norm(a_01 - 0.5 * numpy.einsum("ia->a", ca_01)), tl.norm(a_01))
    #print(0.5 * numpy.einsum("ia->a", ca_01))
    #print(a_01)
    #print(numpy.einsum("ia,i->a", ca_01, c_01))

    caa_01 = BeN_rho[0]["caa"][(+1, 0)][0][0]
    kappa1_3 = tl.tenalg.outer((c_01, a_01, a_01))
    kappa1_3 = 0.5 * (kappa1_3 - tl.moveaxis(kappa1_3, 1, 2))
    kappa21 = tl.tenalg.outer((ca_diff, a_01))
    kappa21 = 0.5 * (kappa21 - tl.moveaxis(kappa21, 1, 2))
    caa_01_decomp = (2/3) * (3 * kappa21 + kappa1_3)
    caa_diff = caa_01 - caa_01_decomp
    print("diff of caa - decomp", tl.norm(caa_diff), tl.norm(caa_01), tl.norm(caa_01_decomp))

    caa_decomp2 = tl.tenalg.outer((ca_01, a_01))
    #for n in range(1, 2):
    #    ca_0n = BeN_rho[0]["ca"][(0, 0)][0][n]
    #    a_n0 = BeN_rho[0]["a"][(0, -1)][n][0]
    #    caa_decomp2 += tl.tenalg.outer((ca_0n, a_n0))
    caa_decomp2 = 2 * 0.5 * (caa_decomp2 - tl.moveaxis(caa_decomp2, 1, 2))
    print("diff off easy decomp for caa", tl.norm(caa_01 - caa_decomp2), tl.norm(caa_01), tl.norm(caa_decomp2))
    caa_diff2 = caa_01 - caa_decomp2

    print("rdm experiment")
    ccaa_00 = BeN_rho[0]["ccaa"][(0, 0)][0][0]
    ca_00 = BeN_rho[0]["ca"][(0, 0)][0][0]
    #ccaa_00_decomp = tl.tenalg.outer((ca_00, ca_00))
    #ccaa_00_decomp = numpy.einsum("ib,ja->ijab", ca_00, ca_00)
    #ccaa_00_decomp = 2 * (1/4) * (ccaa_00_decomp - tl.moveaxis(ccaa_00_decomp, 0, 1) - tl.moveaxis(ccaa_00_decomp, 2, 3) + tl.moveaxis(ccaa_00_decomp, [0, 2], [1, 3]))
    #print("2rdm in standard representation", tl.norm(ccaa_00 - ccaa_00_decomp), tl.norm(ccaa_00), tl.norm(ccaa_00_decomp))
    #ccaa_diff = ccaa_00 - ccaa_00_decomp

    ccaaa_00 = BeN_rho[0]["ccaaa"][(+1, 0)][0][0]
    ccaa_00 = BeN_rho[0]["ccaa"][(0, 0)][0][0]
    ccaaa_00_decomp_ccaa_a = 3 * (1/12) * (numpy.einsum("ijab,c->ijabc", ccaa_00, a_01) - numpy.einsum("ijab,c->ijacb", ccaa_00, a_01) - numpy.einsum("ijab,c->ijcba", ccaa_00, a_01)
                    + numpy.einsum("ijab,c->ijcab", ccaa_00, a_01) - numpy.einsum("ijab,c->ijbac", ccaa_00, a_01) + numpy.einsum("ijab,c->ijbca", ccaa_00, a_01)
                    - numpy.einsum("ijab,c->jiabc", ccaa_00, a_01) + numpy.einsum("ijab,c->jiacb", ccaa_00, a_01) + numpy.einsum("ijab,c->jicba", ccaa_00, a_01)
                    - numpy.einsum("ijab,c->jicab", ccaa_00, a_01) + numpy.einsum("ijab,c->jibac", ccaa_00, a_01) - numpy.einsum("ijab,c->jibca", ccaa_00, a_01))
    ccaaa_00_decomp_caa_ca = 3 * (1/12) * (numpy.einsum("iab,jc->ijcab", caa_01, ca_00) - numpy.einsum("iab,jc->jicab", caa_01, ca_00) - numpy.einsum("iab,jc->ijacb", caa_01, ca_00)
                    + numpy.einsum("iab,jc->jiacb", caa_01, ca_00) - numpy.einsum("iab,jc->ijbac", caa_01, ca_00) + numpy.einsum("iab,jc->jibac", caa_01, ca_00)
                    + numpy.einsum("iab,jc->ijabc", caa_01, ca_00) - numpy.einsum("iab,jc->jiabc", caa_01, ca_00) - numpy.einsum("iab,jc->ijcba", caa_01, ca_00)
                    + numpy.einsum("iab,jc->jicba", caa_01, ca_00) + numpy.einsum("iab,jc->ijbca", caa_01, ca_00) - numpy.einsum("iab,jc->jibca", caa_01, ca_00))
    print("diff easy ccaaa ccaa a", tl.norm(ccaaa_00 - ccaaa_00_decomp_ccaa_a), tl.norm(ccaaa_00), tl.norm(ccaaa_00_decomp_ccaa_a))
    print("diff easy ccaaa caa ca", tl.norm(ccaaa_00 - ccaaa_00_decomp_caa_ca), tl.norm(ccaaa_00), tl.norm(ccaaa_00_decomp_caa_ca))
    ccaaa_00_decomp_summed = ccaaa_00_decomp_ccaa_a + ccaaa_00_decomp_caa_ca
    print("diff easy ccaaa summed", tl.norm(ccaaa_00 - ccaaa_00_decomp_summed), tl.norm(ccaaa_00), tl.norm(ccaaa_00_decomp_summed))
    ccaaa_00_decomp_averaged = 0.5 * ccaaa_00_decomp_summed
    print("diff easy ccaaa averaged", tl.norm(ccaaa_00 - ccaaa_00_decomp_averaged), tl.norm(ccaaa_00), tl.norm(ccaaa_00_decomp_averaged))
    #BeN_rho[0]["ccaaa"][(0, -1)][0][0] = ccaaa_00_decomp_averaged
    #BeN_rho[1]["ccaaa"][(0, -1)][0][0] = ccaaa_00_decomp_averaged
    #k21_ccaa_a = 3 * (1/12) * (numpy.einsum("ijab,c->ijabc", ccaa_diff, a_01) - numpy.einsum("ijab,c->ijacb", ccaa_diff, a_01) - numpy.einsum("ijab,c->ijcba", ccaa_diff, a_01)
    #                  + numpy.einsum("ijab,c->ijcab", ccaa_diff, a_01) - numpy.einsum("ijab,c->ijbac", ccaa_diff, a_01) + numpy.einsum("ijab,c->ijbca", ccaa_diff, a_01)
    #                  - numpy.einsum("ijab,c->jiabc", ccaa_diff, a_01) + numpy.einsum("ijab,c->jiacb", ccaa_diff, a_01) + numpy.einsum("ijab,c->jicba", ccaa_diff, a_01)
    #                  - numpy.einsum("ijab,c->jicab", ccaa_diff, a_01) + numpy.einsum("ijab,c->jibac", ccaa_diff, a_01) - numpy.einsum("ijab,c->jibca", ccaa_diff, a_01))
    #k1_3_ccaaa = 
    """
    """
    print("should be particle number", numpy.einsum("ii->", ca_00))
    print(a_01)

    # build 3p cumulant approximation from 4p cumulant terms (see mazziotti paper)
    # one could also think of e.g. contracting ccaa and ca over three indices to get to caa, but for now we leave it out
    # we further focus on just ccaa a now, so we keep L3_2 ccaa to use a to get ccaaa, 3D_uc only ccaaa_00_decomp_ccaa_a and L4_3 with ccaa and caa
    L2_3_ccaa = - numpy.einsum("xixb,ja->ijab", ccaa_00, ca_00) + numpy.einsum("ixxb,ja->ijab", ccaa_00, ca_00) + numpy.einsum("xibx,ja->ijab", ccaa_00, ca_00) - \
        numpy.einsum("ixbx,ja->ijab", ccaa_00, ca_00) + numpy.einsum("ii->", ca_00) * ccaa_00 + \
        numpy.einsum("xiab,jx->ijab", ccaa_00, ca_00) - numpy.einsum("ixab,jx->ijab", ccaa_00, ca_00) - numpy.einsum("ijxb,xa->ijab", ccaa_00, ca_00) + \
        numpy.einsum("ijbx,xa->ijab", ccaa_00, ca_00)
    #L2_3_caa = - numpy.einsum("xixb,ja->ijab", ccaa_00, ca_00)




    #ccaa_01 = BeN_rho[0]["ccaa"][(0, 0)][0][1]
    #ca_01 = BeN_rho[0]["ca"][(0, 0)][0][1]
    #ccaa_01_decomp = tl.tenalg.outer((ca_01, ca_01))
    #ccaa_01_decomp = (1/4) * (ccaa_01_decomp - tl.moveaxis(ccaa_01_decomp, 0, 1) - tl.moveaxis(ccaa_01_decomp, 2, 3) + tl.moveaxis(ccaa_01_decomp, [0, 2], [1, 3]))

    #print("diff of ccaa - ca outer ca antisymmetrized", tl.norm(ccaa_01 - ccaa_01_decomp), tl.norm(ccaa_01), tl.norm(ccaa_01_decomp))

    print("rtm experiment")
    ccaa_01 = BeN_rho[0]["ccaa"][(0, 0)][0][7]
    ca_01 = BeN_rho[0]["ca"][(0, 0)][0][7]
    ccaa_01_decomp = numpy.einsum("ib,ja->ijab", ca_00, ca_01)
    ccaa_01_decomp = (1/4) * (ccaa_01_decomp - tl.moveaxis(ccaa_01_decomp, 0, 1) - tl.moveaxis(ccaa_01_decomp, 2, 3) + tl.moveaxis(ccaa_01_decomp, [0, 2], [1, 3]))
    print("2rtm in standard representation", tl.norm(ccaa_01 - ccaa_01_decomp), tl.norm(ccaa_01), tl.norm(ccaa_01_decomp))

    ccaa_01 = BeN_rho[0]["ccaa"][(0, 0)][0][8]
    ca_01 = BeN_rho[0]["ca"][(0, 0)][0][8]
    ccaa_01_decomp = numpy.einsum("ib,ja->ijab", ca_00, ca_01)
    ccaa_01_decomp = (1/4) * (ccaa_01_decomp - tl.moveaxis(ccaa_01_decomp, 0, 1) - tl.moveaxis(ccaa_01_decomp, 2, 3) + tl.moveaxis(ccaa_01_decomp, [0, 2], [1, 3]))
    print("2rtm in standard representation", tl.norm(ccaa_01 - ccaa_01_decomp), tl.norm(ccaa_01), tl.norm(ccaa_01_decomp))

    ccaa_01 = BeN_rho[0]["ccaa"][(0, 0)][0][7]
    ca_01 = BeN_rho[0]["ca"][(0, 0)][0][7]
    ca_11 = BeN_rho[0]["ca"][(0, 0)][7][7]
    ccaa_01_decomp = numpy.einsum("ib,ja->ijab", ca_01, ca_11)
    ccaa_01_decomp = (1/4) * (ccaa_01_decomp - tl.moveaxis(ccaa_01_decomp, 0, 1) - tl.moveaxis(ccaa_01_decomp, 2, 3) + tl.moveaxis(ccaa_01_decomp, [0, 2], [1, 3]))
    print("2rtm in standard representation", tl.norm(ccaa_01 - ccaa_01_decomp), tl.norm(ccaa_01), tl.norm(ccaa_01_decomp))

    ccaa_01 = BeN_rho[0]["ccaa"][(0, 0)][0][8]
    ca_01 = BeN_rho[0]["ca"][(0, 0)][0][8]
    ca_11 = BeN_rho[0]["ca"][(0, 0)][8][8]
    ccaa_01_decomp = numpy.einsum("ib,ja->ijab", ca_01, ca_11)
    ccaa_01_decomp = (1/4) * (ccaa_01_decomp - tl.moveaxis(ccaa_01_decomp, 0, 1) - tl.moveaxis(ccaa_01_decomp, 2, 3) + tl.moveaxis(ccaa_01_decomp, [0, 2], [1, 3]))
    print("2rtm in standard representation", tl.norm(ccaa_01 - ccaa_01_decomp), tl.norm(ccaa_01), tl.norm(ccaa_01_decomp))

    for i in range(2):
        ccaaa_01 = BeN_rho[0]["ccaaa"][(0, -1)][0][i]
        ccaaa_00_decomp_ccaa_a = tl.zeros_like(ccaaa_01)
        for n in range(8):
            #if n == 1:
            #    continue
            #ccaa_00 = BeN_rho[0]["ccaa"][(0, 0)][n][0]
            #a_01 = BeN_rho[0]["a"][(0, -1)][1][n]
            #ccaaa_00_decomp_ccaa_a += 0.5 * 3 * (1/12) * (numpy.einsum("ijab,c->ijabc", ccaa_00, a_01) - numpy.einsum("ijab,c->ijacb", ccaa_00, a_01) - numpy.einsum("ijab,c->ijcba", ccaa_00, a_01)
            #                + numpy.einsum("ijab,c->ijcab", ccaa_00, a_01) - numpy.einsum("ijab,c->ijbac", ccaa_00, a_01) + numpy.einsum("ijab,c->ijbca", ccaa_00, a_01)
            #                - numpy.einsum("ijab,c->jiabc", ccaa_00, a_01) + numpy.einsum("ijab,c->jiacb", ccaa_00, a_01) + numpy.einsum("ijab,c->jicba", ccaa_00, a_01)
            #                - numpy.einsum("ijab,c->jicab", ccaa_00, a_01) + numpy.einsum("ijab,c->jibac", ccaa_00, a_01) - numpy.einsum("ijab,c->jibca", ccaa_00, a_01))
            ccaa_00 = BeN_rho[0]["ccaa"][(0, 0)][0][n]
            a_01 = BeN_rho[0]["a"][(0, -1)][n][i]
            ccaaa_00_decomp_ccaa_a += 0.5 * 3 * (1/12) * (numpy.einsum("ijab,c->ijabc", ccaa_00, a_01) - numpy.einsum("ijab,c->ijacb", ccaa_00, a_01) - numpy.einsum("ijab,c->ijcba", ccaa_00, a_01)
                            + numpy.einsum("ijab,c->ijcab", ccaa_00, a_01) - numpy.einsum("ijab,c->ijbac", ccaa_00, a_01) + numpy.einsum("ijab,c->ijbca", ccaa_00, a_01)
                            - numpy.einsum("ijab,c->jiabc", ccaa_00, a_01) + numpy.einsum("ijab,c->jiacb", ccaa_00, a_01) + numpy.einsum("ijab,c->jicba", ccaa_00, a_01)
                            - numpy.einsum("ijab,c->jicab", ccaa_00, a_01) + numpy.einsum("ijab,c->jibac", ccaa_00, a_01) - numpy.einsum("ijab,c->jibca", ccaa_00, a_01))
        ccaaa_00_diff = ccaaa_01 - ccaaa_00_decomp_ccaa_a
        print(f"ccaaa 0{i}", tl.norm(ccaaa_00_diff), tl.norm(ccaaa_01), tl.norm(ccaaa_00_decomp_ccaa_a))

    ccaa_00 = BeN_rho[0]["ccaa"][(0, 0)][0][0]
    a_01 = BeN_rho[0]["a"][(0, -1)][0][1]
    ccaaa_01 = BeN_rho[0]["ccaaa"][(0, -1)][0][1]
    ccaaa_01_decomp_ccaa_a = 3 * (1/12) * (numpy.einsum("ijab,c->ijabc", ccaa_00, a_01) - numpy.einsum("ijab,c->ijacb", ccaa_00, a_01) - numpy.einsum("ijab,c->ijcba", ccaa_00, a_01)
                    + numpy.einsum("ijab,c->ijcab", ccaa_00, a_01) - numpy.einsum("ijab,c->ijbac", ccaa_00, a_01) + numpy.einsum("ijab,c->ijbca", ccaa_00, a_01)
                    - numpy.einsum("ijab,c->jiabc", ccaa_00, a_01) + numpy.einsum("ijab,c->jiacb", ccaa_00, a_01) + numpy.einsum("ijab,c->jicba", ccaa_00, a_01)
                    - numpy.einsum("ijab,c->jicab", ccaa_00, a_01) + numpy.einsum("ijab,c->jibac", ccaa_00, a_01) - numpy.einsum("ijab,c->jibca", ccaa_00, a_01))
    ccaaa_01_diff = ccaaa_01 - ccaaa_01_decomp_ccaa_a
    print("ccaaa 01", tl.norm(ccaaa_01_diff), tl.norm(ccaaa_01), tl.norm(ccaaa_01_decomp_ccaa_a))
    diff_ccaaa_ansatze = ccaaa_00_decomp_ccaa_a - ccaaa_01_decomp_ccaa_a
    print("ri ccaaa - and ccaaa from only gs", tl.norm(diff_ccaaa_ansatze), "remainder - upper remainder", tl.norm(diff_ccaaa_ansatze - ccaaa_01_diff))

    ccaa_00 = BeN_rho[0]["ccaa"][(0, 0)][0][0]
    a_01 = BeN_rho[0]["a"][(0, -1)][0][3]
    ccaaa_01 = BeN_rho[0]["ccaaa"][(0, -1)][0][3]
    ccaaa_00_decomp_ccaa_a = 3 * (1/12) * (numpy.einsum("ijab,c->ijabc", ccaa_00, a_01) - numpy.einsum("ijab,c->ijacb", ccaa_00, a_01) - numpy.einsum("ijab,c->ijcba", ccaa_00, a_01)
                    + numpy.einsum("ijab,c->ijcab", ccaa_00, a_01) - numpy.einsum("ijab,c->ijbac", ccaa_00, a_01) + numpy.einsum("ijab,c->ijbca", ccaa_00, a_01)
                    - numpy.einsum("ijab,c->jiabc", ccaa_00, a_01) + numpy.einsum("ijab,c->jiacb", ccaa_00, a_01) + numpy.einsum("ijab,c->jicba", ccaa_00, a_01)
                    - numpy.einsum("ijab,c->jicab", ccaa_00, a_01) + numpy.einsum("ijab,c->jibac", ccaa_00, a_01) - numpy.einsum("ijab,c->jibca", ccaa_00, a_01))
    ccaaa_01_diff = ccaaa_01 - ccaaa_00_decomp_ccaa_a
    print("ccaaa 03", tl.norm(ccaaa_01_diff), tl.norm(ccaaa_01), tl.norm(ccaaa_00_decomp_ccaa_a))

    ccaa_00 = BeN_rho[0]["ccaa"][(0, 0)][0][0]
    a_01 = BeN_rho[0]["a"][(0, -1)][0][5]
    ccaaa_01 = BeN_rho[0]["ccaaa"][(0, -1)][0][5]
    ccaaa_00_decomp_ccaa_a = 3 * (1/12) * (numpy.einsum("ijab,c->ijabc", ccaa_00, a_01) - numpy.einsum("ijab,c->ijacb", ccaa_00, a_01) - numpy.einsum("ijab,c->ijcba", ccaa_00, a_01)
                    + numpy.einsum("ijab,c->ijcab", ccaa_00, a_01) - numpy.einsum("ijab,c->ijbac", ccaa_00, a_01) + numpy.einsum("ijab,c->ijbca", ccaa_00, a_01)
                    - numpy.einsum("ijab,c->jiabc", ccaa_00, a_01) + numpy.einsum("ijab,c->jiacb", ccaa_00, a_01) + numpy.einsum("ijab,c->jicba", ccaa_00, a_01)
                    - numpy.einsum("ijab,c->jicab", ccaa_00, a_01) + numpy.einsum("ijab,c->jibac", ccaa_00, a_01) - numpy.einsum("ijab,c->jibca", ccaa_00, a_01))
    ccaaa_01_diff = ccaaa_01 - ccaaa_00_decomp_ccaa_a
    print("ccaaa 05", tl.norm(ccaaa_01_diff), tl.norm(ccaaa_01), tl.norm(ccaaa_00_decomp_ccaa_a))

    ccaa_00 = BeN_rho[0]["ccaa"][(0, 0)][0][0]
    a_01 = BeN_rho[0]["a"][(0, -1)][0][7]
    ccaaa_01 = BeN_rho[0]["ccaaa"][(0, -1)][0][7]
    ccaaa_00_decomp_ccaa_a = 3 * (1/12) * (numpy.einsum("ijab,c->ijabc", ccaa_00, a_01) - numpy.einsum("ijab,c->ijacb", ccaa_00, a_01) - numpy.einsum("ijab,c->ijcba", ccaa_00, a_01)
                    + numpy.einsum("ijab,c->ijcab", ccaa_00, a_01) - numpy.einsum("ijab,c->ijbac", ccaa_00, a_01) + numpy.einsum("ijab,c->ijbca", ccaa_00, a_01)
                    - numpy.einsum("ijab,c->jiabc", ccaa_00, a_01) + numpy.einsum("ijab,c->jiacb", ccaa_00, a_01) + numpy.einsum("ijab,c->jicba", ccaa_00, a_01)
                    - numpy.einsum("ijab,c->jicab", ccaa_00, a_01) + numpy.einsum("ijab,c->jibac", ccaa_00, a_01) - numpy.einsum("ijab,c->jibca", ccaa_00, a_01))
    ccaaa_01_diff = ccaaa_01 - ccaaa_00_decomp_ccaa_a
    print("ccaaa 07", tl.norm(ccaaa_01_diff), tl.norm(ccaaa_01), tl.norm(ccaaa_00_decomp_ccaa_a))

    caa_01 = BeN_rho[0]["caa"][(0, -1)][0][7]
    ca_00 = BeN_rho[0]["ca"][(0, 0)][0][0]
    ccaaa_01 = BeN_rho[0]["ccaaa"][(0, -1)][0][7]
    ccaaa_00_decomp_caa_ca = 3 * (1/12) * (numpy.einsum("iab,jc->ijcab", caa_01, ca_00) - numpy.einsum("iab,jc->jicab", caa_01, ca_00) - numpy.einsum("iab,jc->ijacb", caa_01, ca_00)
                    + numpy.einsum("iab,jc->jiacb", caa_01, ca_00) - numpy.einsum("iab,jc->ijbac", caa_01, ca_00) + numpy.einsum("iab,jc->jibac", caa_01, ca_00)
                    + numpy.einsum("iab,jc->ijabc", caa_01, ca_00) - numpy.einsum("iab,jc->jiabc", caa_01, ca_00) - numpy.einsum("iab,jc->ijcba", caa_01, ca_00)
                    + numpy.einsum("iab,jc->jicba", caa_01, ca_00) + numpy.einsum("iab,jc->ijbca", caa_01, ca_00) - numpy.einsum("iab,jc->jibca", caa_01, ca_00))
    ccaaa_01_diff = ccaaa_01 - ccaaa_00_decomp_caa_ca
    print("ccaaa 01 from caa and ca", tl.norm(ccaaa_01_diff), tl.norm(ccaaa_01), tl.norm(ccaaa_00_decomp_ccaa_a))


    #print("rtm experiment")
    #ccaa_01 = BeN_rho[0]["ccaa"][(0, 0)][0][1]
    #ca_01 = BeN_rho[0]["ca"][(0, 0)][0][1]
    #ccaa_01_decomp = numpy.einsum("ib,ja->ijab", ca_00, ca_00)
    #ccaa_01_decomp = (1/4) * (ccaa_01_decomp - tl.moveaxis(ccaa_01_decomp, 0, 1) - tl.moveaxis(ccaa_01_decomp, 2, 3) + tl.moveaxis(ccaa_01_decomp, [0, 2], [1, 3]))
    #print("2rtm in standard representation", tl.norm(ccaa_01 - ccaa_01_decomp), tl.norm(ccaa_01), tl.norm(ccaa_01_decomp))

    #eigvals, eigvecs = numpy.linalg.eigh(ca_00)
    #eigvals2, eigvecs2 = numpy.linalg.eig(ca_00)
    #print(numpy.linalg.norm(eigvals - eigvals2))
    #print(eigvals, eigvals2)
    #print(numpy.linalg.norm(eigvals))
    """




    """
    rho_decomp = [{op_string:{charges:[[tl.decomposition.tucker(BeN_rho[m][op_string][charges][i][j], 16)#, tol=1e-7)
                                    for j in range(len(BeN_rho[m][op_string][charges][i]))]
                                    for i in range(len(BeN_rho[m][op_string][charges]))]
                            for charges in BeN_rho[m][op_string]} for op_string in BeN_rho[m] if len(op_string) < 5 and len(op_string) > 1}
                for m in range(len(BeN_rho))]

    for i in range(len(rho_decomp)):
        for op_string in rho_decomp[i]:
            BeN_rho[i][op_string] = rho_decomp[i][op_string]
    """
    #for i in range(len(rho_decomp)):
    #    rho_decomp[i]["n_elec"] = BeN_rho[i]["n_elec"]
    #    rho_decomp[i]["n_states"] = BeN_rho[i]["n_states"]

    S_blocks       = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=symm_ints.S,                     diagrams=S_diagrams)
    Sn_blocks      = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=(symm_ints.S, nuc_rep),          diagrams=Sn_diagrams)
    St_blocks_symm = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=(symm_ints.S, symm_ints.T),      diagrams=St_diagrams)
    #St_blocks_bior_v_mean = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=(symm_ints.S, two_p_mean_field_bior),      diagrams=St_diagrams)
    #St_blocks_half_v_mean = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=(symm_ints.S, two_p_mean_field_bior_half),      diagrams=St_diagrams)
    Su_blocks_symm = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=(symm_ints.S, symm_ints.U),      diagrams=Su_diagrams)
    Sv_blocks_symm = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=(symm_ints.S, symm_ints.V),      diagrams=Sv_diagrams)
    St_blocks_bior = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=(symm_ints.S, bior_ints.T),      diagrams=St_diagrams)
    Su_blocks_bior = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=(symm_ints.S, bior_ints.U),      diagrams=Su_diagrams)
    Sv_blocks_bior = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=(symm_ints.S, bior_ints.V),      diagrams=Sv_diagrams)
    Sv_blocks_half = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=(symm_ints.S, bior_ints.V_half), diagrams=Sv_diagrams)

    # for comparison
    #body1_ref = numpy.load("reference/test-data-4.5/H1_0.npy")    # from old XR code (same as next line, but unaffected by hack)
    #body2_ref = numpy.load("reference/test-data-4.5/H2_0_1.npy")  # from old XR code (no S) hacked to use symmetric integrals
    #Sref, Href = {}, {}
    #for n_elec in [6,7,8,9,10]:  # full S and model-space-BO H from brute-force on dimer (ie, exact target, but one-body and two-body together)
    #    Sref[n_elec] = numpy.load("atomic_states/states/16-115-550/load=states:16-115-550:thresh=1e-6:4.5:u.pickle/{}/S-{}.npy".format(displacement,n_elec))
    #    Href[n_elec] = numpy.load("atomic_states/states/16-115-550/load=states:16-115-550:thresh=1e-6:4.5:u.pickle/{}/H-{}.npy".format(displacement,n_elec))

    #########
    # build and test
    #########

    # S (up to 1e-5 accuracy with all diagrams)
    if False:

        active_S_diagrams = {
            0:  ["identity"],
            2:  [
                "s01"#,
                #"s01s10", "s01s01",
                #"s01s01s10",
                #"s01s01s10s10", "s01s01s01s10"
                ]
        }
        S = {}
        S[6]  = XR_term.dimer_matrix(S_blocks, active_S_diagrams, (0,1), [(+1,+1)])
        S[7]  = XR_term.dimer_matrix(S_blocks, active_S_diagrams, (0,1), [(0,+1),(+1,0)])
        S[8]  = XR_term.dimer_matrix(S_blocks, active_S_diagrams, (0,1), [(0,0),(+1,-1),(-1,+1)])
        S[9]  = XR_term.dimer_matrix(S_blocks, active_S_diagrams, (0,1), [(0,-1),(-1,0)])
        S[10] = XR_term.dimer_matrix(S_blocks, active_S_diagrams, (0,1), [(-1,-1)])

        norms_by_p_number = []

        for n_elec in [6,7,8,9,10]:
            dim, _ = Sref[n_elec].shape
            Id = numpy.identity(dim)
            #Sref_norm = numpy.linalg.norm(Sref[n_elec]-Id)
            #S_norm = numpy.linalg.norm(S[n_elec]-Id)
            diff_norm = numpy.linalg.norm(S[n_elec]-Sref[n_elec])
            print(diff_norm)
            norms_by_p_number.append(diff_norm)
        #    print("{:2d} Frobenius norm of Sref and S:   ".format(n_elec), numpy.linalg.norm(Sref[n_elec]-Id), numpy.linalg.norm(S[n_elec]-Id))
        #    print("   Frobenius norm of S-Sref: ", numpy.linalg.norm(S[n_elec]-Sref[n_elec]))

    # H1 and H2 (check code structure through XR-CCSD calc with symmetric integrals for exact comparison)
    if False:

        all_monomer_charges = [0, +1, -1]
        H1 = []
        for m in [0, 1]:
            H1_m  = XR_term.monomer_matrix(Sn_blocks,      {1: ["n00"]},   m, all_monomer_charges)
            H1_m += XR_term.monomer_matrix(St_blocks_symm, {1: ["t00"]},   m, all_monomer_charges)
            H1_m += XR_term.monomer_matrix(Su_blocks_symm, {1: ["u000"]},  m, all_monomer_charges)
            H1_m += XR_term.monomer_matrix(Sv_blocks_symm, {1: ["v0000"]}, m, all_monomer_charges)
            print("1-body error {}:".format(m), numpy.linalg.norm(H1_m - body1_ref))
            H1 += [H1_m]

        vals, vecs = qode.util.sort_eigen(numpy.linalg.eig(H1[0]))
        print("1-body ground-state energy:", vals[0])

        all_dimer_charges = [(0,0), (0,+1), (0,-1), (+1,0), (+1,+1), (+1,-1), (-1,0), (-1,+1), (-1,-1)]
        H2blocked  = XR_term.dimer_matrix(Sn_blocks,      {2: ["n01"]},                              (0,1), all_dimer_charges)
        H2blocked += XR_term.dimer_matrix(St_blocks_symm, {2: ["t01"]},                              (0,1), all_dimer_charges)
        H2blocked += XR_term.dimer_matrix(Su_blocks_symm, {2: ["u001", "u101", "u100"]},             (0,1), all_dimer_charges)
        H2blocked += XR_term.dimer_matrix(Sv_blocks_symm, {2: ["v0101", "v0011", "v0010", "v0100"]}, (0,1), all_dimer_charges)

        ### well, this sucks.  reorder the states
        dims0 = [BeN[0].rho['n_states'][chg] for chg in [0,+1,-1]]
        dims1 = [BeN[1].rho['n_states'][chg] for chg in [0,+1,-1]]
        mapping2 = [[None]*sum(dims0) for _ in range(sum(dims1))]
        idx = 0
        beg0 = 0
        for dim0 in dims0:
            beg1 = 0
            for dim1 in dims1:
                for m in range(dim0):
                    for n in range(dim1):
                        mapping2[beg0+m][beg1+n] = idx
                        idx += 1
                beg1 += dim1
            beg0 += dim0
        mapping = []
        for m in range(sum(dims0)):
            for n in range(sum(dims1)):
                mapping += [mapping2[m][n]]
        H2 = numpy.zeros(H2blocked.shape)
        for i,i_ in enumerate(mapping):
            for j,j_ in enumerate(mapping):
                H2[i,j] = H2blocked[i_,j_]
        ###

        print("norms of H2, H2ref, and diff", numpy.linalg.norm(H2), numpy.linalg.norm(body2_ref), numpy.linalg.norm(H2-body2_ref))
        out, resources = qode.util.output(log=qode.util.textlog(echo=True)), qode.util.parallel.resources(1)
        E, T = excitonic.ccsd(([body1_ref,body1_ref],[[None,body2_ref]]), out, resources)
        out.log("\nTotal Excitonic CCSD Energy (ref)  = ", E)
        E, T = excitonic.ccsd((H1,[[None,H2]]), out, resources)
        out.log("\nTotal Excitonic CCSD Energy (test) = ", E)

    # Full H1+H2 (Marco-style: convenient for playing around with turning diagrams on/off)
    if True:
        """
        #active_Sn_diagrams_bo = {
        #                      1: [
        #                          "n00"
        #                         ],
        #                      2: [
        #                          "n01",
        #                          "s01n00", "s01n11", "s01n01"
        #                         ]
        #                     }
        active_St_diagrams_bo = {
                            #1: [
                            #    "t00"
                            #   ],
                            2: [
                                #"t01"#,
                                "s01t00", "s10t00",
                                "s10t01", "s01t01"
                                ]
                            }
        active_Su_diagrams_bo = {
                            #1: [
                            #    "u000"
                            #   ],
                            2: [
                                #"u100",
                                #"u001", "u101"#,
                                "s01u000", "s10u000",
                                "s01u100", "s10u100",
                                "s10u001", "s10u101", "s01u001", "s01u101"
                                ]
                            }
        active_Sv_diagrams_bo = {
                            #1: [
                            #    "v0000"
                            #   ],
                            2: [
                                #"v0101", "v0010", "v0100", "v0011"#,
                                #"s01v0000", "s10v0000",
                                "s01v0101", "s10v0010", "s01v0100", "s10v0011", "s01v0010", "s10v0100"#, "s01v0011"
                                ]
                            }
        
        #active_Sn_diagrams_symm = {
        #                      1: [
        #                          "n00"
        #                         ],
        #                      2: [
        #                          "n01",
        #                          "s01n00", "s01n11", "s01n01"
        #                         ]
        #                     }
        active_St_diagrams_symm = {
                            1: [
                                "t00"
                                ],
                            2: [
                                "t01"#,
                                #"s01t00", "s10t00",
                                #"s10t01", "s01t01"
                                ]
                            }
        active_Su_diagrams_symm = {
                            1: [
                                "u000"
                                ],
                            2: [
                                "u100",
                                "u001", "u101"#,
                                #"s01u000", "s10u000",
                                #"s01u100", "s10u100",
                                #"s10u001", "s10u101", "s01u001", "s01u101"
                                ]
                            }
        active_Sv_diagrams_symm = {
                            1: [
                                "v0000"
                                ],
                            2: [
                                "v0101", "v0010", "v0100", "v0011"#,
                                #"s01v0000", "s10v0000",
                                #"s01v0101", "s10v0010", "s01v0100", "s10v0011", "s01v0010", "s10v0100"#, "s01v0011"
                                ]
                            }

        dimer_charges = {
                        6:  [(+1, +1)],
                        7:  [(0, +1), (+1, 0)],
                        8:  [(0, 0), (+1, -1), (-1, +1)],
                        9:  [(0, -1), (-1, 0)],
                        10: [(-1, -1)]
                        }

        SH = {}

        for n_elec in [6,7,8,9,10]:
            SH[n_elec]   = XR_term.dimer_matrix(St_blocks_bior, active_St_diagrams_bo, (0,1), dimer_charges[n_elec])
            SH[n_elec]  += XR_term.dimer_matrix(Su_blocks_bior, active_Su_diagrams_bo, (0,1), dimer_charges[n_elec])
            SH[n_elec]  += XR_term.dimer_matrix(Sv_blocks_bior, active_Sv_diagrams_bo, (0,1), dimer_charges[n_elec])
            SH[n_elec]  += XR_term.dimer_matrix(St_blocks_symm, active_St_diagrams_symm, (0,1), dimer_charges[n_elec])
            SH[n_elec]  += XR_term.dimer_matrix(Su_blocks_symm, active_Su_diagrams_symm, (0,1), dimer_charges[n_elec])
            SH[n_elec]  -= XR_term.dimer_matrix(Sv_blocks_bior, active_Sv_diagrams_symm, (0,1), dimer_charges[n_elec])
            SH[n_elec]  += XR_term.dimer_matrix(Sv_blocks_half, active_Sv_diagrams_symm, (0,1), dimer_charges[n_elec])
        """
        """
        #
        SH[6]   = XR_term.dimer_matrix(St_blocks_bior, active_St_diagrams, (0,1), dimer_charges[6])
        SH[6]  += XR_term.dimer_matrix(Su_blocks_bior, active_Su_diagrams, (0,1), dimer_charges[6])
        SH[6]  += XR_term.dimer_matrix(Sv_blocks_bior, active_Sv_diagrams, (0,1), dimer_charges[6])
        #
        SH[7]   = XR_term.dimer_matrix(St_blocks_bior, active_St_diagrams, (0,1), dimer_charges[7])
        SH[7]  += XR_term.dimer_matrix(Su_blocks_bior, active_Su_diagrams, (0,1), dimer_charges[7])
        SH[7]  += XR_term.dimer_matrix(Sv_blocks_bior, active_Sv_diagrams, (0,1), dimer_charges[7])
        #
        SH[8]   = XR_term.dimer_matrix(St_blocks_bior, active_St_diagrams, (0,1), dimer_charges[8])
        SH[8]  += XR_term.dimer_matrix(Su_blocks_bior, active_Su_diagrams, (0,1), dimer_charges[8])
        SH[8]  += XR_term.dimer_matrix(Sv_blocks_bior, active_Sv_diagrams, (0,1), dimer_charges[8])
        #
        SH[9]   = XR_term.dimer_matrix(St_blocks_bior, active_St_diagrams, (0,1), dimer_charges[9])
        SH[9]  += XR_term.dimer_matrix(Su_blocks_bior, active_Su_diagrams, (0,1), dimer_charges[9])
        SH[9]  += XR_term.dimer_matrix(Sv_blocks_bior, active_Sv_diagrams, (0,1), dimer_charges[9])
        #
        SH[10]  = XR_term.dimer_matrix(St_blocks_bior, active_St_diagrams, (0,1), dimer_charges[10])
        SH[10] += XR_term.dimer_matrix(Su_blocks_bior, active_Su_diagrams, (0,1), dimer_charges[10])
        SH[10] += XR_term.dimer_matrix(Sv_blocks_bior, active_Sv_diagrams, (0,1), dimer_charges[10])
        """

        #for n_elec in [6,7,8,9,10]:
        #    H = qode.math.precise_numpy_inverse(S[n_elec]) @ SH[n_elec]# + SH_S[n_elec]
        #    print("{:2d} Frobenius norm of H, Href: ".format(n_elec), numpy.linalg.norm(H), numpy.linalg.norm(Href[n_elec]))
        #    print(   "   Frobenius norm of H-Href:  ",         numpy.linalg.norm(H-Href[n_elec]))
        #    vals, vecs = qode.util.sort_eigen(numpy.linalg.eig(Href[n_elec]))
        #    print("   2-body ground-state electronic energy (ref):    ", vals[0])
        #    vals, vecs = qode.util.sort_eigen(numpy.linalg.eig(H))
        #    print("   2-body ground-state electronic energy (test):   ", vals[0])

        n_elec = 8

        # charges under consideration
        #monomer_charges = [0, +1, -1]
        dimer_charges = {
                        6:  [(+1, +1)],
                        7:  [(0, +1), (+1, 0)],
                        8:  [(0, 0), (+1, -1), (-1, +1)],
                        9:  [(0, -1), (-1, 0)],
                        10: [(-1, -1)]
                        }
        #neutral_dimer_charges = {8 : [(0, 0)]}
        #all_dimer_charges = [(0,0), (0,+1), (0,-1), (+1,0), (+1,+1), (+1,-1), (-1,0), (-1,+1), (-1,-1)]

        """
        S    = XR_term.dimer_matrix(S_blocks, {
                            0: [
                                "identity"
                                ],
                            2: [
                                "s01"
                                #"s01s10", "s01s01",
                                #"s01s01s10",
                                #"s01s01s10s10", "s01s01s01s10"
                                ]
                            },  (0,1), dimer_charges[n_elec])

        Sinv = qode.math.precise_torch_inverse(S)

        
        SH  = XR_term.dimer_matrix(St_blocks_symm, {
                            1: [
                                "t00"
                                ],
                            2: [
                                "t01",
                                #"s01t00", "s10t00",
                                #"s10t01", "s01t01"
                                ]
                            }, (0,1), dimer_charges[n_elec])
        SH  += XR_term.dimer_matrix(Su_blocks_symm, {
                            1: [
                                "u000"
                                ],
                            2: [
                                "u100",
                                "u001", "u101",
                                #"s01u000", "s10u000",
                                #"s01u100", "s10u100",
                                #"s10u001", "s10u101", "s01u001", "s01u101"
                                ]
                            }, (0,1), dimer_charges[n_elec])

        SH  += XR_term.dimer_matrix(St_blocks_bior, {
                            1: [
                                #"t00"
                                ],
                            2: [
                                #"t01",
                                "s01t00", "s10t00",
                                "s10t01", "s01t01"
                                ]
                            }, (0,1), dimer_charges[n_elec])
        
        SH  += XR_term.dimer_matrix(Su_blocks_bior, {
                            1: [
                                #"u000"
                                ],
                            2: [
                                #"u100",
                                #"u001", "u101",
                                "s01u000", "s10u000",
                                "s01u100", "s10u100",
                                "s10u001", "s10u101", "s01u001", "s01u101"
                                ]
                            }, (0,1), dimer_charges[n_elec])
        
        SH  -= XR_term.dimer_matrix(Sv_blocks_bior, {
                            1: [
                                "v0000"
                                ],
                            2: [
                                "v0101", "v0010", "v0100", "v0011",
                                #"s01v0000", "s10v0000",
                                #"s01v0101", "s10v0010", "s01v0100", "s10v0011", "s01v0010", "s10v0100"#, "s01v0011"
                                ]
                            }, (0,1), dimer_charges[n_elec])

        SH  += XR_term.dimer_matrix(Sv_blocks_half, {
                            1: [
                                "v0000"
                                ],
                            2: [
                                "v0101", "v0010", "v0100", "v0011",
                                #"s01v0000", "s10v0000",
                                #"s01v0101", "s10v0010", "s01v0100", "s10v0011", "s01v0010", "s10v0100"#, "s01v0011"
                                ]
                            }, (0,1), dimer_charges[n_elec])

        SH  += XR_term.dimer_matrix(Sv_blocks_bior, {
                            1: [
                                #"v0000"
                                ],
                            2: [
                                #"v0101", "v0010", "v0100", "v0011",
                                "s01v0000", "s10v0000",
                                "s01v0101", "s10v0010", "s01v0100", "s10v0011", "s01v0010", "s10v0100"#, "s01v0011"
                                ]
                            }, (0,1), dimer_charges[n_elec])
        """

        S    = XR_term.dimer_matrix(S_blocks, {
                            0: [
                                "identity"
                                ],
                            2: [
                                #"s01"
                                #"s01s10", "s01s01",
                                #"s01s01s10",
                                #"s01s01s10s10", "s01s01s01s10"
                                ]
                            },  (0,1), dimer_charges[n_elec])
        #print("dimer_charges:", dimer_charges[n_elec])

        Sinv = qode.math.precise_torch_inverse(S)
        #Sinv = qode.math.precise_numpy_inverse(S)
        print(type(S))

        
        """
        SH  = XR_term.dimer_matrix(St_blocks_symm, {
                            1: [
                                "t00"
                                ],
                            2: [
                                "t01",
                                #"s01t00", "s10t00",
                                #"s10t01", "s01t01"
                                ]
                            }, (0,1), dimer_charges[n_elec])
        
        SH  += XR_term.dimer_matrix(Su_blocks_symm, {
                            1: [
                                "u000"
                                ],
                            2: [
                                "u100",
                                "u001", "u101",
                                #"s01u000", "s10u000",
                                #"s01u100", "s10u100",
                                #"s10u001", "s10u101", "s01u001", "s01u101"
                                ]
                            }, (0,1), dimer_charges[n_elec])
        """
        

        SH  = XR_term.dimer_matrix(St_blocks_bior, {
                            1: [
                                "t00"
                                ],
                            2: [
                                "t01"
                                #"s01t00", "s10t00",
                                #"s10t01", "s01t01"
                                ]
                            }, (0,1), dimer_charges[n_elec])
        
        SH  += XR_term.dimer_matrix(Su_blocks_bior, {
                            1: [
                                "u000"
                                ],
                            2: [
                                "u100",
                                "u001", "u101"
                                #"s01u000", "s10u000",
                                #"s01u100", "s10u100",
                                #"s10u001", "s10u101", "s01u001", "s01u101"
                                ]
                            }, (0,1), dimer_charges[n_elec])
        
        """
        SH  -= XR_term.dimer_matrix(Sv_blocks_bior, {
                            1: [
                                "v0000"
                                ],
                            2: [
                                "v0101", "v0010", "v0100", "v0011",
                                #"s01v0000", "s10v0000",
                                #"s01v0101", "s10v0010", "s01v0100", "s10v0011", "s01v0010", "s10v0100"#, "s01v0011"
                                ]
                            }, (0,1), dimer_charges[n_elec])

        SH  += XR_term.dimer_matrix(Sv_blocks_half, {
                            1: [
                                "v0000"
                                ],
                            2: [
                                "v0101", "v0010", "v0100", "v0011",
                                #"s01v0000", "s10v0000",
                                #"s01v0101", "s10v0010", "s01v0100", "s10v0011", "s01v0010", "s10v0100"#, "s01v0011"
                                ]
                            }, (0,1), dimer_charges[n_elec])
        """
        #SH  += XR_term.dimer_matrix(Sv_blocks_bior, {
        #                    1: [
        #                        "v0000"
        #                        ],
        #                    2: [
        #                        "v0101", "v0010", "v0100", "v0011",
        #                        #"s01v0000", "s10v0000",
        #                        #"s01v0101", "s10v0010", "s01v0100", "s10v0011", "s01v0010", "s10v0100"#, "s01v0011"
        #                        ]
        #                    }, (0,1), dimer_charges[n_elec])
        """
        
        
        SH  += XR_term.dimer_matrix(Sv_blocks_bior, {
                            1: [
                                "v0000"
                                ],
                            2: [
                                "v0101", 
                                "v0010", "v0100",
                                "v0011"
                                #"s01v0000", "s10v0000",
                                #"s01v0101", "s10v0010", "s01v0100", "s10v0011", "s01v0010", "s10v0100"#, "s01v0011"
                                ]
                            }, (0,1), dimer_charges[n_elec])
        """
        """
        SH  -= XR_term.dimer_matrix(St_blocks_bior_v_mean, {  # do this twice instead of introducing factor of two
                            1: [
                                "t00"
                                ],
                            2: [
                                "t01"
                                #"s01t00", "s10t00",
                                #"s10t01", "s01t01"
                                ]
                            }, (0,1), dimer_charges[n_elec])
        
        SH  -= XR_term.dimer_matrix(St_blocks_bior_v_mean, {  # do this twice instead of introducing factor of two
                            1: [
                                "t00"
                                ],
                            2: [
                                "t01"
                                #"s01t00", "s10t00",
                                #"s10t01", "s01t01"
                                ]
                            }, (0,1), dimer_charges[n_elec])
        
        SH  -= XR_term.dimer_matrix(St_blocks_half_v_mean, {  # do this twice instead of introducing factor of two
                            1: [
                                "t00"
                                ],
                            2: [
                                "t01"
                                #"s01t00", "s10t00",
                                #"s10t01", "s01t01"
                                ]
                            }, (0,1), dimer_charges[n_elec])
        
        SH  -= XR_term.dimer_matrix(St_blocks_half_v_mean, {  # do this twice instead of introducing factor of two
                            1: [
                                "t00"
                                ],
                            2: [
                                "t01"
                                #"s01t00", "s10t00",
                                #"s10t01", "s01t01"
                                ]
                            }, (0,1), dimer_charges[n_elec])
        """
        """
        SH  += XR_term.dimer_matrix(St_blocks_bior_v_mean, {
                            1: [
                                #"t00"
                                ],
                            2: [
                                #"t01",
                                "s01t00", "s10t00"#,
                                #"s10t01", "s01t01"
                                ]
                            }, (0,1), dimer_charges[n_elec])
        """
        SH  += XR_term.dimer_matrix(Sv_blocks_bior, {
                            1: [
                                "v0000"
                                ],
                            2: [
                                "v0101", "v0010", "v0100", "v0011"#,
                                #"s01v0000", "s10v0000",
                                #"s10v0010", "s01v0100", "s01v0010", "s10v0100",#, "s01v0011"
                                #"s10v0011", "s01v0101"
                                ]
                            }, (0,1), dimer_charges[n_elec])
        

        """
        SH  += XR_term.dimer_matrix(St_blocks_half_v_mean, {
                            1: [
                                #"t00"
                                ],
                            2: [
                                #"t01",
                                "s01t00", "s10t00",
                                "s10t01", "s01t01"
                                ]
                            }, (0,1), dimer_charges[n_elec])
        """
        #vals, vecs = qode.util.sort_eigen_np(numpy.linalg.eig((Sinv @ SH)))
        #vals, vecs = qode.util.sort_eigen(numpy.linalg.eig((SH)))
        #print(SH)
        #print(torch.linalg.eig((Sinv @ SH))[0])
        vals, vecs = qode.util.sort_eigen_torch(torch.linalg.eig((Sinv @ SH)))
        print("2-body ground-state energy:", numpy.real(numpy.array(vals[0])))
        print("ref = ", -31.10732242634624)
    return vals[0]  #None#sum(norms_by_p_number)   #vals[0]


#ray.init(num_cpus=1, ignore_reinit_error=True)
scan_result = [Be2_XR(i) for i in numpy.arange(4.5, 4.6, 0.1)]
print(numpy.real(numpy.array(scan_result)))
#ray.shutdown()
#numpy.save("scan_result_XR_FCI.npy", numpy.real(numpy.array(scan_result)))
# scan results from 3.0 to 7.0 in 0.1
# scan results wide from 1.0 to 7.2 in 0.2
# scan results extra points from 3.5 to 5.3 in 0.2


