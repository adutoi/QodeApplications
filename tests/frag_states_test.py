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
from qode.util import struct
from driver import main
import XR_tensor



# quick and dirty way to compare nested iterables of numbers with inhomogeneous lengths
def compare(A, B, **kwargs):
    try:
        for a,b in zip(A,B):
            compare(a, b, **kwargs)
    except:
        numpy.testing.assert_allclose(A, B, **kwargs)



# the test function
def test_it():

    # input parameters
    params = struct(
        n_threads  = 1,
        basis      = "6-31G",
        nstates    = None,
        thresh     = 1e-6,
        compress   = struct(method="SVD", divide="cc-aa"),
        nat_orbs   = False,
        abs_anti   = False,
        op_strings = {2:["aa"], 1:["a", "caa"], 0:["ca", "ccaa"]},
        frags = [
            struct(
                atoms=[struct(element="Be", position=[0, 0, 0])],
                core=[0],
                charge=0
            ),
            struct(
                atoms=[struct(element="Be", position=[0, 0, 4.5])],
                core=[0],
                charge=0
            )
        ]
    )

    # main is the function to be tested
    frags = main(params, test_only=True)

    # distill frags structure for comparison (ie, only norms of tensors)
    test = []
    for frag in frags:
        test_frag = []
        test_frag += [numpy.linalg.norm(frag.basis.MOcoeffs)]
        test_states = []
        for chg in frag.states:
            test_states_chg = []
            for state in frag.states[chg].coeffs:
                test_states_chg += [numpy.linalg.norm(state)]
            test_states += [(chg, test_states_chg)]
        test_frag += [test_states]
        test_frag += [frag.state_indices]
        test_rho = []
        for chg in frag.rho["n_elec"]:
            test_rho += [(chg, frag.rho["n_elec"][chg])]
        for chg in frag.rho["n_states"]:
            test_rho += [(chg, frag.rho["n_states"][chg])]
        for op_string in frag.rho:
            if op_string not in ("n_elec", "n_states"):
                for chgs in frag.rho[op_string]:
                    test_rho += [(chgs, numpy.linalg.norm(XR_tensor.raw(frag.rho[op_string][chgs])))]
        test_frag += [test_rho]
        test += [test_frag]

    # to generate updated reference data, uncomment this line and run file as a script 
    #print(test)

    # hard-coded reference data
    result = [
              # for first fragment
              [
               # norm of the MO coefficient matrix
               4.49942660733095,
               # norms of each of the states for each charge
               [
                (-1, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                ( 0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                (+1, [1.0, 1.0, 1.0, 1.0])
               ],
               # ordering of the state indices: (charge, state)
               [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (-1, 0), (-1, 1), (-1, 2), (-1, 3), (-1, 4), (-1, 5), (-1, 6), (-1, 7), (1, 0), (1, 1), (1, 2), (1, 3)],
               # contents of the densities structure
               [
                # number of electrons for each charge
                (-1, 5), (0,  4), (+1, 3),
                # number of states for each charge
                (-1, 8), (0, 11), (+1, 4),
                # norms of density tensors (bra and ket charges given, but density types suppressed)
                ((-1, -1), 8.157334276081299), ((0, 0), 9.648609608782913), ((1, 1), 4.898979485566358), ((-1, -1), 28.649710718177506), ((0, 0), 33.117447653188385), ((1, 1), 12.000000000000005), ((0, -1), 3.6896374025420644), ((1, 0), 3.966414348394967), ((0, -1), 14.304726940256698), ((1, 0), 12.285347823021073), ((1, -1), 5.789891151503158), ((-1, 0), 3.6896374025420644), ((0, 1), 3.966414348394967), ((-1, 0), 14.304726940256698), ((0, 1), 12.285347823021077), ((-1, 1), 5.7898911515031575)
               ],
              ],
              # repeat for second fragment
              [
               4.49942660733095,
               [
                (-1, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                ( 0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                (+1, [1.0, 1.0, 1.0, 1.0])
               ],
               [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (-1, 0), (-1, 1), (-1, 2), (-1, 3), (-1, 4), (-1, 5), (-1, 6), (-1, 7), (1, 0), (1, 1), (1, 2), (1, 3)],
               [
                (-1, 5), (0,  4), (+1, 3),
                (-1, 8), (0, 11), (1,  4),
                ((-1, -1), 8.157334276081299), ((0, 0), 9.648609608782913), ((1, 1), 4.898979485566358), ((-1, -1), 28.649710718177506), ((0, 0), 33.117447653188385), ((1, 1), 12.000000000000005), ((0, -1), 3.6896374025420644), ((1, 0), 3.966414348394967), ((0, -1), 14.304726940256698), ((1, 0), 12.285347823021073), ((1, -1), 5.789891151503158), ((-1, 0), 3.6896374025420644), ((0, 1), 3.966414348394967), ((-1, 0), 14.304726940256698), ((0, 1), 12.285347823021077), ((-1, 1), 5.7898911515031575)
               ]
              ]
             ]

    # make the comparison (raises exception if outside of tolerance)
    compare(test, result, atol=1e-10)



# can run the file as a script for debugging and/or generation of updated reference data
if __name__=="__main__":
    test_it()
