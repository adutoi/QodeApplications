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

import tensorly
from tendot_wrapper import tendot

class _empty(object):  pass    # Basically just a dictionary

def _parameters2(densities, overlaps, subsystem, charges, permutation):
    # helper functions to do repetitive manipulations of data passed from above
    # needs to be generalized (should not be hard) and have "2" removed from its name ... or maybe it is better this way
    densities = [densities[m] for m in subsystem]
    overlaps = {(m0_,m1_):overlaps[m0,m1] for m1_,m1 in enumerate(subsystem) for m0_,m0 in enumerate(subsystem)}
    m0, m1 = 0, 1
    (chg_i0,chg_j0), (chg_i1,chg_j1) = charges
    n_i1 = densities[m1]['n_elec'][chg_i1]
    P = 0
    if permutation==(1,0):
        m0, m1 = 1, 0
        (chg_i1,chg_j1), (chg_i0,chg_j0) = (chg_i0,chg_j0), (chg_i1,chg_j1)
        P = 1
    data = _empty()
    data.n_i1   = n_i1%2
    data.P      = P
    data.Dchg_0 = chg_i0 - chg_j0
    data.Dchg_1 = chg_i1 - chg_j1
    if data.Dchg_0==0:
        data.ca_0   = tensorly.tensor(densities[m0]['ca'  ][chg_i0,chg_j0])
        data.ccaa_0 = tensorly.tensor(densities[m0]['ccaa'][chg_i0,chg_j0])
    if data.Dchg_0==-1:
        data.c_0    = tensorly.tensor(densities[m0]['c'   ][chg_i0,chg_j0])
        data.cca_0  = tensorly.tensor(densities[m0]['cca' ][chg_i0,chg_j0])
    if data.Dchg_0==+1:
        data.a_0    = tensorly.tensor(densities[m0]['a'   ][chg_i0,chg_j0])
        data.caa_0  = tensorly.tensor(densities[m0]['caa' ][chg_i0,chg_j0])
    if data.Dchg_0==-2:
        data.cc_0   = tensorly.tensor(densities[m0]['cc'  ][chg_i0,chg_j0])
        data.ccca_0 = tensorly.tensor(densities[m0]['ccca'][chg_i0,chg_j0])
    if data.Dchg_0==+2:
        data.aa_0   = tensorly.tensor(densities[m0]['aa'  ][chg_i0,chg_j0])
        data.caaa_0 = tensorly.tensor(densities[m0]['caaa'][chg_i0,chg_j0])
    if data.Dchg_1==0:
        data.ca_1   = tensorly.tensor(densities[m1]['ca'  ][chg_i1,chg_j1])
        data.ccaa_1 = tensorly.tensor(densities[m1]['ccaa'][chg_i1,chg_j1])
    if data.Dchg_1==-1:
        data.c_1    = tensorly.tensor(densities[m1]['c'   ][chg_i1,chg_j1])
        data.cca_1  = tensorly.tensor(densities[m1]['cca' ][chg_i1,chg_j1])
    if data.Dchg_1==+1:
        data.a_1    = tensorly.tensor(densities[m1]['a'   ][chg_i1,chg_j1])
        data.caa_1  = tensorly.tensor(densities[m1]['caa' ][chg_i1,chg_j1])
    if data.Dchg_1==-2:
        data.cc_1   = tensorly.tensor(densities[m1]['cc'  ][chg_i1,chg_j1])
        data.ccca_1 = tensorly.tensor(densities[m1]['ccca'][chg_i1,chg_j1])
    if data.Dchg_1==+2:
        data.aa_1   = tensorly.tensor(densities[m1]['aa'  ][chg_i1,chg_j1])
        data.caaa_1 = tensorly.tensor(densities[m1]['caaa'][chg_i1,chg_j1])
    data.S_01 = tensorly.tensor(overlaps[m0,m1])
    data.S_10 = tensorly.tensor(overlaps[m1,m0])
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
        # 1 * 1 * (0)<-(1)
        X = _parameters2(densities, integrals, subsystem, charges, permutation)
        if X.Dchg_0==-1 and X.Dchg_1==+1:
            prefactor = (-1)**(X.n_i1 + X.P)
            def diagram(i0,i1,j0,j1):
                #return prefactor * np.einsum("pq,p,q->", sig12, c1[i1][j1], a2[i2][j2])
                partial =          tendot(X.c_0[i0][j0], X.S_01,        axes=([0],[0]))
                return prefactor * tendot(partial,       X.a_1[i1][j1], axes=([0],[0]))
            return diagram, permutation
        else:
            return None, None

    @staticmethod
    def order2_CT0(densities, integrals, subsystem, charges):
        # 1/2! * 1 * (0)<-->(1)
        X = _parameters2(densities, integrals, subsystem, charges, permutation=(0,1))
        if X.Dchg_0==0 and X.Dchg_1==0:
            prefactor = -1
            def diagram(i0,i1,j0,j1):
                #return prefactor * np.einsum("qs,sq->", np.einsum("pq,ps->qs", sig12, ca1[i1][j1]), np.einsum("rs,rq->sq", sig21, ca2[i2][j2]))
                partial =          tendot(X.ca_0[i0][j0], X.S_01,         axes=([0],[0]))
                partial =          tendot(partial,        X.S_10,         axes=([0],[1]))
                return prefactor * tendot(partial,        X.ca_1[i1][j1], axes=([0,1],[1,0]))
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
        # 1/2! * 1 * (0)<-<-(1)
        X = _parameters2(densities, integrals, subsystem, charges, permutation)
        if X.Dchg_0==-2 and X.Dchg_1==+2:
            prefactor = 1/2.
            def diagram(i0,i1,j0,j1):
                #return prefactor * np.einsum("qr,rq->", np.einsum("pq,pr->qr", sig12, cc1[i1][j1]), np.einsum("rs,sq->rq", sig12, aa2[i2][j2]))
                partial =          tendot(X.cc_0[i0][j0], X.S_01,         axes=([0],[0]))
                partial =          tendot(partial,        X.S_01,         axes=([0],[0]))
                return prefactor * tendot(partial,        X.aa_1[i1][j1], axes=([0,1],[1,0]))
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
        # 1/3! * 3 * (0)<-<-->(1)
        X = _parameters2(densities, integrals, subsystem, charges, permutation)
        if X.Dchg_0==-1 and X.Dchg_1==+1:
            prefactor = (-1)**(X.n_i1 + X.P + 1) / 2.
            def diagram(i0,i1,j0,j1):
                partial =          tendot(X.cca_0[i0][j0], X.S_01,          axes=([0],[0]))
                partial =          tendot(partial,         X.S_01,          axes=([0],[0]))
                partial =          tendot(partial,         X.S_10,          axes=([0],[1]))
                return prefactor * tendot(partial,         X.caa_1[i1][j1], axes=([0,1,2],[2,1,0]))
            return diagram, permutation
        else:
            return None, None

    @staticmethod
    def order4_CT0(densities, integrals, subsystem, charges):
        # (1/4!) * 3 * (0)<-<-->->(1)
        X = _parameters2(densities, integrals, subsystem, charges, permutation=(0,1))
        if X.Dchg_0==0 and X.Dchg_1==0:
            prefactor = 1/4.
            def diagram(i0,i1,j0,j1):
                partial =          tendot(X.ccaa_0[i0][j0], X.S_01,           axes=([0],[0]))
                partial =          tendot(partial,          X.S_01,           axes=([0],[0]))
                partial =          tendot(partial,          X.S_10,           axes=([0],[1]))
                partial =          tendot(partial,          X.S_10,           axes=([0],[1]))
                return prefactor * tendot(partial,          X.ccaa_1[i1][j1], axes=([0,1,2,3],[3,2,1,0]))
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
        # 1/4! * 4 * (0)<-<-<-->(1)
        X = _parameters2(densities, integrals, subsystem, charges, permutation)
        if X.Dchg_0==-2 and X.Dchg_1==+2:
            prefactor = -1 / 6.
            def diagram(i0,i1,j0,j1):
                partial =          tendot(X.ccca_0[i0][j0], X.S_01,           axes=([0],[0]))
                partial =          tendot(partial,          X.S_01,           axes=([0],[0]))
                partial =          tendot(partial,          X.S_01,           axes=([0],[0]))
                partial =          tendot(partial,          X.S_10,           axes=([0],[1]))
                return prefactor * tendot(partial,          X.caaa_1[i1][j1], axes=([0,1,2,3],[3,2,1,0]))
            return diagram, permutation
        else:
            return None, None



##########
# A dictionary catalog.  the string association lets users specify active diagrams at the top level.
# would like to build automatically, but more difficult than expected to get function references correct
# e.g., does not work
#catalog[2] = {}
#for k,v in body_2.__dict__.items():
#    catalog[2][k] = v
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
