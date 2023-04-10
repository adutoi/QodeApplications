#    (C) Copyright 2023 Anthony D. Dutoi
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

import numpy

class _empty(object):  pass    # Basically just a dictionary

def _parameters2(densities, overlaps, subsystem, charges, permutation):
    # helper functions to do repetitive manipulations of data passed from above
    # needs to be generalized (should not be hard) and have "2" removed from its name ... or maybe it is better this way
    densities = [densities[m] for m in subsystem]
    overlaps = {(m1_,m2_):overlaps[m1,m2] for m2_,m2 in enumerate(subsystem) for m1_,m1 in enumerate(subsystem)}
    m1, m2 = 0, 1
    (chg_i1,chg_j1), (chg_i2,chg_j2) = charges
    n_i2 = densities[m2]['n_elec'][chg_i2]
    p = 0
    if permutation==(1,0):
        m1, m2 = 1, 0
        (chg_i2,chg_j2), (chg_i1,chg_j1) = (chg_i1,chg_j1), (chg_i2,chg_j2)
        p = 1
    data = _empty()
    data.Dchg_1 = chg_i1 - chg_j1
    data.Dchg_2 = chg_i2 - chg_j2
    if data.Dchg_1==0:
        data.ca1    = densities[m1]['ca'  ][chg_i1,chg_j1]
        data.ccaa1  = densities[m1]['ccaa'][chg_i1,chg_j1]
    if data.Dchg_1==-1:
        data.c1     = densities[m1]['c'   ][chg_i1,chg_j1]
        data.cca1   = densities[m1]['cca' ][chg_i1,chg_j1]
    if data.Dchg_1==+1:
        data.a1     = densities[m1]['a'   ][chg_i1,chg_j1]
        data.caa1   = densities[m1]['caa' ][chg_i1,chg_j1]
    if data.Dchg_1==-2:
        data.cc1    = densities[m1]['cc'  ][chg_i1,chg_j1]
        data.ccca1  = densities[m1]['ccca'][chg_i1,chg_j1]
    if data.Dchg_1==+2:
        data.aa1    = densities[m1]['aa'  ][chg_i1,chg_j1]
        data.caaa1  = densities[m1]['caaa'][chg_i1,chg_j1]
    if data.Dchg_2==0:
        data.ca2    = densities[m2]['ca'  ][chg_i2,chg_j2]
        data.ccaa2  = densities[m2]['ccaa'][chg_i2,chg_j2]
    if data.Dchg_2==-1:
        data.c2     = densities[m2]['c'   ][chg_i2,chg_j2]
        data.cca2   = densities[m2]['cca' ][chg_i2,chg_j2]
    if data.Dchg_2==+1:
        data.a2     = densities[m2]['a'   ][chg_i2,chg_j2]
        data.caa2   = densities[m2]['caa' ][chg_i2,chg_j2]
    if data.Dchg_2==-2:
        data.cc2    = densities[m2]['cc'  ][chg_i2,chg_j2]
        data.ccca2  = densities[m2]['ccca'][chg_i2,chg_j2]
    if data.Dchg_2==+2:
        data.aa2    = densities[m2]['aa'  ][chg_i2,chg_j2]
        data.caaa2  = densities[m2]['caaa'][chg_i2,chg_j2]
    data.sig12  = overlaps[m1,m2]
    data.sig21  = overlaps[m2,m1]
    data.n_i2   = n_i2%2
    data.p      = p
    return data



##########
# Here are the implementations of the actual diagrams.
# The public @staticmethods must take the arguments (densities, integrals, subsystem, charges), but after that, it is up to you.
# It should return a list of kernels that takes state indices (for specified fragment charges) their relevant permutations.
# Don't forget to update the "catalog" dictionary at the end.
##########

class body_0(object):

    @staticmethod
    def identity(densities, integrals, subsystem, charges):
        # Identity
        def diagram():
            return 1
        return [(diagram, [])]



class body_2(object):

    @staticmethod
    def order1_CT1(densities, integrals, subsystem, charges):
        result01 = body_2._order1_CT1(densities, integrals, subsystem, charges, permutation=(0,1))
        result10 = body_2._order1_CT1(densities, integrals, subsystem, charges, permutation=(1,0))
        return [result01, result10]
    @staticmethod
    def _order1_CT1(densities, integrals, subsystem, charges, permutation):
        # 1 * 1 * (1)<-(2)
        X = _parameters2(densities, integrals, subsystem, charges, permutation)
        if X.Dchg_1==-1 and X.Dchg_2==+1:
            prefactor = (-1)**(X.n_i2 + X.p)
            def diagram(i1,i2,j1,j2):
                partial =          numpy.tensordot(X.c1[i1][j1], X.sig12,      axes=([0],[0]))
                return prefactor * numpy.tensordot(partial,      X.a2[i2][j2], axes=([0],[0]))
            return diagram, permutation
        else:
            return None, None

    @staticmethod
    def order2_CT0(densities, integrals, subsystem, charges):
        # 1/2! * 1 * (1)<-->(2)
        X = _parameters2(densities, integrals, subsystem, charges, permutation=(0,1))
        if X.Dchg_1==0 and X.Dchg_2==0:
            prefactor = -1
            def diagram(i1,i2,j1,j2):
                partial =          numpy.tensordot(X.ca1[i1][j1], X.sig12,       axes=([0],[0]))
                partial =          numpy.tensordot(partial,       X.sig21,       axes=([0],[1]))
                return prefactor * numpy.tensordot(partial,       X.ca2[i2][j2], axes=([0,1],[1,0]))
            return [(diagram, (0,1))]
        else:
            return [(None, None)]

    @staticmethod
    def order2_CT2(densities, integrals, subsystem, charges):
        result01 = body_2._order2_CT2(densities, integrals, subsystem, charges, permutation=(0,1))
        result10 = body_2._order2_CT2(densities, integrals, subsystem, charges, permutation=(1,0))
        return [result01, result10]
    @staticmethod
    def _order2_CT2(densities, integrals, subsystem, charges, permutation):
        # 1/2! * 1 * (1)<-<-(2)
        X = _parameters2(densities, integrals, subsystem, charges, permutation)
        if X.Dchg_1==-2 and X.Dchg_2==+2:
            prefactor = 1/2.
            def diagram(i1,i2,j1,j2):
                partial =          numpy.tensordot(X.cc1[i1][j1], X.sig12,       axes=([0],[0]))
                partial =          numpy.tensordot(partial,       X.sig12,       axes=([0],[0]))
                return prefactor * numpy.tensordot(partial,       X.aa2[i2][j2], axes=([0,1],[1,0]))
            return diagram, permutation
        else:
            return None, None

    @staticmethod
    def order3_CT1(densities, integrals, subsystem, charges):
        result01 = body_2._order3_CT1(densities, integrals, subsystem, charges, permutation=(0,1))
        result10 = body_2._order3_CT1(densities, integrals, subsystem, charges, permutation=(1,0))
        return [result01, result10]
    @staticmethod
    def _order3_CT1(densities, integrals, subsystem, charges, permutation):
        # 1/3! * 3 * (1)<-<-->(2)
        X = _parameters2(densities, integrals, subsystem, charges, permutation)
        if X.Dchg_1==-1 and X.Dchg_2==+1:
            prefactor = (-1)**(X.n_i2 + X.p + 1) / 2.
            def diagram(i1,i2,j1,j2):
                partial =          numpy.tensordot(X.cca1[i1][j1], X.sig12,        axes=([0],[0]))
                partial =          numpy.tensordot(partial,        X.sig12,        axes=([0],[0]))
                partial =          numpy.tensordot(partial,        X.sig21,        axes=([0],[1]))
                return prefactor * numpy.tensordot(partial,        X.caa2[i2][j2], axes=([0,1,2],[2,1,0]))
            return diagram, permutation
        else:
            return None, None

    @staticmethod
    def order4_CT0(densities, integrals, subsystem, charges):
        # (1/4!) * 3 * (1)<-<-->->(2)
        X = _parameters2(densities, integrals, subsystem, charges, permutation=(0,1))
        if X.Dchg_1==0 and X.Dchg_2==0:
            prefactor = 1/4.
            def diagram(i1,i2,j1,j2):
                partial =          numpy.tensordot(X.ccaa1[i1][j1], X.sig12,         axes=([0],[0]))
                partial =          numpy.tensordot(partial,         X.sig12,         axes=([0],[0]))
                partial =          numpy.tensordot(partial,         X.sig21,         axes=([0],[1]))
                partial =          numpy.tensordot(partial,         X.sig21,         axes=([0],[1]))
                return prefactor * numpy.tensordot(partial,         X.ccaa2[i2][j2], axes=([0,1,2,3],[3,2,1,0]))
            return [(diagram, (0,1))]
        else:
            return [(None, None)]

    @staticmethod
    def order4_CT2(densities, integrals, subsystem, charges):
        result01 = body_2._order4_CT2(densities, integrals, subsystem, charges, permutation=(0,1))
        result10 = body_2._order4_CT2(densities, integrals, subsystem, charges, permutation=(1,0))
        return [result01, result10]
    @staticmethod
    def _order4_CT2(densities, integrals, subsystem, charges, permutation):
        # 1/4! * 4 * (1)<-<-<-->(2)
        X = _parameters2(densities, integrals, subsystem, charges, permutation)
        if X.Dchg_1==-2 and X.Dchg_2==+2:
            prefactor = -1 / 6.
            def diagram(i1,i2,j1,j2):
                partial =          numpy.tensordot(X.ccca1[i1][j1], X.sig12,         axes=([0],[0]))
                partial =          numpy.tensordot(partial,         X.sig12,         axes=([0],[0]))
                partial =          numpy.tensordot(partial,         X.sig12,         axes=([0],[0]))
                partial =          numpy.tensordot(partial,         X.sig21,         axes=([0],[1]))
                return prefactor * numpy.tensordot(partial,         X.caaa2[i2][j2], axes=([0,1,2,3],[3,2,1,0]))
            return diagram, permutation
        else:
            return None, None



##########
# A dictionary catalog.  the string association lets users specify active diagrams at the top level.
# would like to build automatically, but more difficult than expected to get function references correct
##########

catalog = {}

catalog[0] = {
"identity": body_0.identity
}

catalog[2] = {
"order1_CT1": body_2.order1_CT1,
"order2_CT0": body_2.order2_CT0,
"order2_CT2": body_2.order2_CT2,
"order3_CT1": body_2.order3_CT1,
"order4_CT0": body_2.order4_CT0,
"order4_CT2": body_2.order4_CT2
}

# e.g., does not work
#catalog[2] = {}
#for k,v in body_2.__dict__.items():
#    catalog[2][k] = v
