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

# Usage:
#     python [-u] <this-file.py> <displacement> <rhos>
# where <rhos> can be the filestem of any one of the .pkl files in atomic_states/ prepared by Be631g.py.

import sys
import pickle
import numpy
#import torch
import tensorly
import qode.util
import qode.math
import excitonic
import diagrammatic_expansion   # defines information structure for housing results of diagram evaluations
import XR_term                  # knows how to use ^this information to pack a matrix for use in XR model
import S_diagrams               # contains definitions of actual diagrams needed for S operator in BO rep
#import Sn_diagrams              # contains definitions of actual diagrams needed for SH operator in BO rep
import St_diagrams              # contains definitions of actual diagrams needed for SH operator in BO rep
import Su_diagrams              # contains definitions of actual diagrams needed for SH operator in BO rep
import Sv_diagrams              # contains definitions of actual diagrams needed for SH operator in BO rep
import combo_diagram
#from   get_ints import get_ints
from qode.util import struct, timer
from precontract import precontract

# needed for unpickling?!
#class empty(object):  pass     # Basically just a dictionary class

#torch.set_num_threads(4)
#tensorly.set_backend("pytorch")

def get_xr_H(ints, dens, xr_order, bra_det=False):
    diag_timer = timer()
    precon_timer = timer()
    matrix_timer = timer()
    symm_ints, bior_ints, nuc_rep = ints#get_ints(BeN, project_core)

    # The engines that build the terms
    BeN_rho = dens  #[frag.rho for frag in BeN]   # diagrammatic_expansion.blocks should take BeN directly? (n_states and n_elec one level higher)
    contract_cache = precontract(BeN_rho, symm_ints.S, precon_timer)

    S_blocks       = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=symm_ints.S,                               diagrams=S_diagrams,  contract_cache=contract_cache, timings=diag_timer, precon_timings=precon_timer)
    St_blocks_symm = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=struct(S=symm_ints.S, T=symm_ints.T),      diagrams=St_diagrams, contract_cache=contract_cache, timings=diag_timer, precon_timings=precon_timer, bra_det=bra_det)
    Su_blocks_symm = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=struct(S=symm_ints.S, U=symm_ints.U),      diagrams=Su_diagrams, contract_cache=contract_cache, timings=diag_timer, precon_timings=precon_timer, bra_det=bra_det)
    Sv_blocks_symm = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=struct(S=symm_ints.S, V=symm_ints.V),      diagrams=Sv_diagrams, contract_cache=contract_cache, timings=diag_timer, precon_timings=precon_timer, bra_det=bra_det)
    St_blocks_bior = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=struct(S=symm_ints.S, T=bior_ints.T),      diagrams=St_diagrams, contract_cache=contract_cache, timings=diag_timer, precon_timings=precon_timer, bra_det=bra_det)
    Su_blocks_bior = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=struct(S=symm_ints.S, U=bior_ints.U),      diagrams=Su_diagrams, contract_cache=contract_cache, timings=diag_timer, precon_timings=precon_timer, bra_det=bra_det)
    Sv_blocks_bior = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=struct(S=symm_ints.S, V=bior_ints.V),      diagrams=Sv_diagrams, contract_cache=contract_cache, timings=diag_timer, precon_timings=precon_timer, bra_det=bra_det)
    #Sv_blocks_half = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=struct(S=symm_ints.S, V=bior_ints.V_half), diagrams=Sv_diagrams, contract_cache=contract_cache, timings=diag_timer, precon_timings=precon_timer, bra_det=bra_det)
    Sv_blocks_diff = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=struct(S=symm_ints.S, V=bior_ints.V_diff), diagrams=Sv_diagrams, contract_cache=contract_cache, timings=diag_timer, precon_timings=precon_timer, bra_det=bra_det)

    Combo_blocks_bior = diagrammatic_expansion.blocks(densities=BeN_rho, integrals=struct(S=symm_ints.S, T=bior_ints.T, U=bior_ints.U, V=bior_ints.V),      diagrams=combo_diagram, contract_cache=contract_cache, timings=diag_timer, precon_timings=precon_timer, bra_det=bra_det)

    # charges under consideration
    monomer_charges = [0, +1, -1]
    #dimer_charges = {
    #                6:  [(+1, +1)],
    #                7:  [(0, +1), (+1, 0)],
    #                8:  [(0, 0), (+1, -1), (-1, +1)],
    #                9:  [(0, -1), (-1, 0)],
    #                10: [(-1, -1)]
    #                }
    all_dimer_charges = [(0,0), (0,+1), (0,-1), (+1,0), (+1,+1), (+1,-1), (-1,0), (-1,+1), (-1,-1)]
    #all_dimer_charges = [(0,0), (+1,-1), (-1,+1)]

    #########
    # Build and test
    #########

    if xr_order == 0:
        print("starting H1")

        H1 = []
        for m in [0,1]:
            #H1_m  = XR_term.monomer_matrix(Sn_blocks, {
            #                    1: [
            #                        "n00"
            #                        ]
            #                    }, m, monomer_charges)

            H1_m = XR_term.monomer_matrix(St_blocks_bior, {
                                1: [
                                    "t00"
                                    ]
                                }, m, monomer_charges, matrix_timer)
            H1_m += XR_term.monomer_matrix(Su_blocks_bior, {
                                1: [
                                    "u000"
                                    ]
                                }, m, monomer_charges, matrix_timer)

            H1_m += XR_term.monomer_matrix(Sv_blocks_bior, {
                                1: [
                                    "v0000"
                                    ]
                                }, m, monomer_charges, matrix_timer)
            H1 += [H1_m]


        #print("starting S2")
        """
        S2     = XR_term.dimer_matrix(S_blocks, {
                                0: [
                                    "identity"
                                ]
                            },  (0,1), all_dimer_charges)

        S2inv = qode.math.precise_numpy_inverse(S2)
        """


        print("starting S2H2")

        print("starting T")
        S2H2  = XR_term.dimer_matrix(St_blocks_bior, {
                                2: [
                                    "t01"
                                ]
                            }, (0,1), all_dimer_charges, matrix_timer, bra_det=bra_det)
        print("starting U")
        S2H2  += XR_term.dimer_matrix(Su_blocks_bior, {
                                2: [
                                    "u100",
                                    "u001", "u101"
                                ]
                            }, (0,1), all_dimer_charges, matrix_timer, bra_det=bra_det)

        print("starting V")
        S2H2  += XR_term.dimer_matrix(Sv_blocks_bior, {
                                2: [
                                    "v0101", "v0010", "v0111", "v0011"
                                ]
                            }, (0,1), all_dimer_charges, matrix_timer, bra_det=bra_det)


        print("finished H build")

        #H2blocked = S2inv @ S2H2
        #print("hakish solution for now...S needs to be build with bra densities from H term in bra of S and ket of S, i.e. S_blocks requires different densities")
        H2blocked = S2H2
    elif xr_order == 1:
        print("build H1")

        H1 = []
        for m in [0,1]:
            H1_m  = XR_term.monomer_matrix(St_blocks_symm, {
                                1: [
                                    "t00"
                                    ]
                                }, m, monomer_charges, matrix_timer)
            H1_m += XR_term.monomer_matrix(Su_blocks_symm, {
                                1: [
                                    "u000"
                                    ]
                                }, m, monomer_charges, matrix_timer)

            H1_m += XR_term.monomer_matrix(Sv_blocks_symm, {
                                1: [
                                    "v0000"
                                    ]
                                }, m, monomer_charges, matrix_timer)

            H1 += [H1_m]



        print("build S2inv")

        S2     = XR_term.dimer_matrix(S_blocks, {
                                0: [
                                    "identity"
                                ],
                                2: [
                                    "s01"
                                ]
                            },  (0,1), all_dimer_charges, matrix_timer)

        S2inv = qode.math.precise_numpy_inverse(S2)



        print("build S2H2 (1e)")

        S2H2  = XR_term.dimer_matrix(St_blocks_symm, {
                            1: [
                                "t00"
                                ],
                            2: [
                                "t01"
                                ]
                            }, (0,1), all_dimer_charges, matrix_timer, bra_det=bra_det)
        S2H2 += XR_term.dimer_matrix(Su_blocks_symm, {
                            1: [
                                "u000"
                                ],
                            2: [
                                "u100",
                                "u001", "u101"
                                ]
                            }, (0,1), all_dimer_charges, matrix_timer, bra_det=bra_det)

        S2H2 += XR_term.dimer_matrix(St_blocks_bior, {
                            2: [
                                "s01t00", "s01t11",
                                #->#"s01t10", 
                                "s01t01"
                                ]
                            }, (0,1), all_dimer_charges, matrix_timer, bra_det=bra_det)
        S2H2 += XR_term.dimer_matrix(Su_blocks_bior, {
                            2: [
                                "s01u000", "s01u011",
                                "s01u100", "s01u111",
                                #->#"s01u010", "s01u110", 
                                "s01u001", "s01u101"
                                ]
                            }, (0,1), all_dimer_charges, matrix_timer, bra_det=bra_det)

        print("build S2H2 (2e)")

        #S2H2 -= XR_term.dimer_matrix(Sv_blocks_bior, {
        #                       1: [
        #                           "v0000"
        #                          ],
        #                       2: [
        #                           "v0101", "v0010", "v0111", "v0011"
        #                          ]
        #                      }, (0,1), all_dimer_charges)

        S2H2 += XR_term.dimer_matrix(Sv_blocks_diff, {
                            1: [
                                "v0000"
                                ],
                            2: [
                                "v0101", "v0010", "v0111", "v0011"
                                ]
                            }, (0,1), all_dimer_charges, matrix_timer, bra_det=bra_det)

        S2H2 += XR_term.dimer_matrix(Sv_blocks_bior, {
                            2: [
                                "s01v0000", "s01v1111",
                                "s01v0101", "s01v1101", 
                                #->#"s01v1000", 
                                "s01v1100", "s01v0010", "s01v0111"#, "s01v0011"
                                ]
                            }, (0,1), all_dimer_charges, matrix_timer, bra_det=bra_det)


        S2H2 += XR_term.dimer_matrix(Combo_blocks_bior, {
                            2: [
                                "combo"
                                ]
                            }, (0,1), all_dimer_charges, matrix_timer, bra_det=bra_det)

        print("Apply H")



        H2blocked = S2inv @ S2H2



        H2blocked -= XR_term.dimer_matrix(St_blocks_symm, {
                            1: [
                                "t00"
                                ]
                            }, (0,1), all_dimer_charges, matrix_timer, bra_det=bra_det)
        H2blocked -= XR_term.dimer_matrix(Su_blocks_symm, {
                            1: [
                                "u000"
                                ]
                            }, (0,1), all_dimer_charges, matrix_timer, bra_det=bra_det)

        H2blocked -= XR_term.dimer_matrix(Sv_blocks_symm, {
                            1: [
                                "v0000"
                                ]
                            }, (0,1), all_dimer_charges, matrix_timer, bra_det=bra_det)
    else:
        raise NotImplementedError("only xr in zeroth and first order are implemented")

    # well, this sucks.  reorder the states
    """
    dims0 = [BeN_rho[0]['n_states'][chg] for chg in [0,+1,-1]]
    dims1 = [BeN_rho[1]['n_states'][chg] for chg in [0,+1,-1]]
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
    print(H2blocked.shape)
    for i,i_ in enumerate(mapping):
        for j,j_ in enumerate(mapping):
            H2[i,j] = H2blocked[i_,j_]
    """
    # determine bra map
    dims0_bra = [BeN_rho[0]['n_states_bra'][chg] for chg in [0,+1,-1]]
    dims1_bra = [BeN_rho[1]['n_states_bra'][chg] for chg in [0,+1,-1]]
    mapping2 = [[None]*sum(dims1_bra) for _ in range(sum(dims0_bra))]  # dims were originally swapped here, which leads to false assignment, if dims0 != dims1
    idx = 0
    beg0 = 0
    for dim0 in dims0_bra:
        beg1 = 0
        for dim1 in dims1_bra:
            for m in range(dim0):
                for n in range(dim1):
                    mapping2[beg0+m][beg1+n] = idx
                    idx += 1
            beg1 += dim1
        beg0 += dim0
    mapping_bra = []
    for m in range(sum(dims0_bra)):
        for n in range(sum(dims1_bra)):
            mapping_bra += [mapping2[m][n]]

    # determine ket map
    if not bra_det:
        dims0_ket = [BeN_rho[0]['n_states'][chg] for chg in [0,+1,-1]]
        dims1_ket = [BeN_rho[1]['n_states'][chg] for chg in [0,+1,-1]]
        mapping2 = [[None]*sum(dims1_ket) for _ in range(sum(dims0_ket))]
        idx = 0
        beg0 = 0
        for dim0 in dims0_ket:
            beg1 = 0
            for dim1 in dims1_ket:
                for m in range(dim0):
                    for n in range(dim1):
                        mapping2[beg0+m][beg1+n] = idx
                        idx += 1
                beg1 += dim1
            beg0 += dim0
        mapping_ket = []
        for m in range(sum(dims0_ket)):
            for n in range(sum(dims1_ket)):
                mapping_ket += [mapping2[m][n]]

    H2 = numpy.zeros(H2blocked.shape)
    print(H2blocked.shape)
    if bra_det:
        for i,i_ in enumerate(mapping_bra):
            H2[i] = H2blocked[i_]
    else:
        for i,i_ in enumerate(mapping_bra):
            for j,j_ in enumerate(mapping_ket):
                H2[i,j] = H2blocked[i_,j_]

    return H1, H2

def get_xr_states(ints, dens, xr_order):
    H1, H2 = get_xr_H(ints, dens, xr_order)
    out, resources = qode.util.output(log=qode.util.textlog(echo=True)), qode.util.parallel.resources(1)
    #E, T = excitonic.ccsd((H1,[[None,H2],[None,None]]), out, resources)
    E, T = excitonic.fci((H1,[[None,H2],[None,None]]), out)
    #E += sum(nuc_rep[m1,m2] for m1 in range(2) for m2 in range(m1+1))
    out.log("\nTotal Excitonic CCSD Energy (test) (without nuc_rep) = ", E)

    #timings.print()

    return E, T

def get_xr_states_from_H(H1, H2):
    #H1, H2 = get_xr_H(ints, dens, xr_order)
    out, resources = qode.util.output(log=qode.util.textlog(echo=True)), qode.util.parallel.resources(1)
    #E, T = excitonic.ccsd((H1,[[None,H2],[None,None]]), out, resources)
    E, T = excitonic.fci((H1,[[None,H2],[None,None]]), out)
    #E += sum(nuc_rep[m1,m2] for m1 in range(2) for m2 in range(m1+1))
    out.log("\nTotal Excitonic CCSD Energy (test) (without nuc_rep) = ", E)

    #timings.print()

    return E, T
