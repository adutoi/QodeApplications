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

#torch.set_num_threads(4)
#tl.set_backend("pytorch")

#########
# Load data
#########

# Information about the Be2 supersystem
n_frag       = 2
displacement = sys.argv[1]
states       = "load=states:16-115-550:thresh=1e-6:4.5:u.pickle/4.5"
n_states     = ("all","all","all","all","all")

# "Assemble" the supersystem from the displaced fragments
BeN = [Be((0,0,m*float(displacement))) for m in range(int(n_frag))]

# Load states and get integrals
for m,frag in enumerate(BeN):  frag.load_states(states, n_states)      # load the density tensors
symm_ints, bior_ints, nuc_rep = get_ints(BeN)

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
#integrals.f = {key: integrals.h[key] + two_p_mean_field[key] for key in integrals.h}

# The engines that build the terms
BeN_rho = [frag.rho for frag in BeN]   # diagrammatic_expansion.blocks should take BeN directly? (n_states and n_elec one level higher)

rho_decomp = [{op_string:{charges:[[tl.decomposition.tucker(BeN_rho[m][op_string][charges][i][j], 16)#, tol=1e-6)
                                   for j in range(len(BeN_rho[m][op_string][charges][i]))]
                                  for i in range(len(BeN_rho[m][op_string][charges]))]
                         for charges in BeN_rho[m][op_string]} for op_string in BeN_rho[m] if len(op_string) == 4}# < 6 and len(op_string) > 1}
              for m in range(len(BeN_rho))]

for i in range(len(rho_decomp)):
    for op_string in rho_decomp[i]:
        BeN_rho[i][op_string] = rho_decomp[i][op_string]

#for i in range(len(rho_decomp)):
#    rho_decomp[i]["n_elec"] = BeN_rho[i]["n_elec"]
#    rho_decomp[i]["n_states"] = BeN_rho[i]["n_states"]

S_blocks       = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=symm_ints.S,                     diagrams=S_diagrams)
Sn_blocks      = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=(symm_ints.S, nuc_rep),          diagrams=Sn_diagrams)
St_blocks_symm = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=(symm_ints.S, symm_ints.T),      diagrams=St_diagrams)
Su_blocks_symm = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=(symm_ints.S, symm_ints.U),      diagrams=Su_diagrams)
Sv_blocks_symm = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=(symm_ints.S, symm_ints.V),      diagrams=Sv_diagrams)
St_blocks_bior = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=(symm_ints.S, bior_ints.T),      diagrams=St_diagrams)
Su_blocks_bior = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=(symm_ints.S, bior_ints.U),      diagrams=Su_diagrams)
Sv_blocks_bior = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=(symm_ints.S, bior_ints.V),      diagrams=Sv_diagrams)
Sv_blocks_half = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=(symm_ints.S, bior_ints.V_half), diagrams=Sv_diagrams)

# for comparison
#body1_ref = numpy.load("reference/test-data-4.5/H1_0.npy")    # from old XR code (same as next line, but unaffected by hack)
#body2_ref = numpy.load("reference/test-data-4.5/H2_0_1.npy")  # from old XR code (no S) hacked to use symmetric integrals
Sref, Href = {}, {}
for n_elec in [6,7,8,9,10]:  # full S and model-space-BO H from brute-force on dimer (ie, exact target, but one-body and two-body together)
    Sref[n_elec] = numpy.load("atomic_states/states/16-115-550/load=states:16-115-550:thresh=1e-6:4.5:u.pickle/{}/S-{}.npy".format(displacement,n_elec))
    Href[n_elec] = numpy.load("atomic_states/states/16-115-550/load=states:16-115-550:thresh=1e-6:4.5:u.pickle/{}/H-{}.npy".format(displacement,n_elec))

#########
# build and test
#########

# S (up to 1e-5 accuracy with all diagrams)
if True:

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

    for n_elec in [6,7,8,9,10]:
        dim, _ = Sref[n_elec].shape
        Id = numpy.identity(dim)
        print("{:2d} Frobenius norm of Sref and S:   ".format(n_elec), numpy.linalg.norm(Sref[n_elec]-Id), numpy.linalg.norm(S[n_elec]-Id))
        print("   Frobenius norm of S-Sref: ", numpy.linalg.norm(S[n_elec]-Sref[n_elec]))

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
    monomer_charges = [0, +1, -1]
    dimer_charges = {
                    6:  [(+1, +1)],
                    7:  [(0, +1), (+1, 0)],
                    8:  [(0, 0), (+1, -1), (-1, +1)],
                    9:  [(0, -1), (-1, 0)],
                    10: [(-1, -1)]
                    }
    all_dimer_charges = [(0,0), (0,+1), (0,-1), (+1,0), (+1,+1), (+1,-1), (-1,0), (-1,+1), (-1,-1)]

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

    Sinv = qode.math.precise_numpy_inverse(S)



    #SH   = XR_term.dimer_matrix(Sn_blocks, {
    #                    1: [
    #                        "n00"
    #                        ],
    #                    2: [
    #                        "n01",
    #                        "s01n00", "s01n11", "s01n01"
    #                        ]
    #                    }, (0,1), dimer_charges[n_elec])

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
    SH_noS  = XR_term.dimer_matrix(Sv_blocks_bior, {
                        1: [
                            "v0000"
                            ]
                        #2: [
                        #    "v0101", "v0010", "v0100", "v0011",
                        #    #"s01v0000", "s10v0000",
                        #    #"s01v0101", "s10v0010", "s01v0100", "s10v0011", "s01v0010", "s10v0100"#, "s01v0011"
                        #    ]
                        }, (0,1), dimer_charges[n_elec])
    """


    vals, vecs = qode.util.sort_eigen(numpy.linalg.eig(Sinv @ SH ))#+ SH_noS))
    print("2-body ground-state energy:", vals[0])
    print("ref = ", -31.10732242634624)
