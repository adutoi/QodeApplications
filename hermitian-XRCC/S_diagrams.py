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

from tendot import tendot

class _empty(object):  pass    # Basically just a dictionary

def _parameters(densities, overlaps, subsystem, charges, permutation=(0,)):
    # helper functions to do repetitive manipulations of data passed from above
    # needs to be generalized (should not be hard) and have "2" removed from its name ... or maybe it is better this way
    densities = [densities[m] for m in subsystem]
    overlaps = {(m0_,m1_):overlaps[m0,m1]    for m1_,m1 in enumerate(subsystem) for m0_,m0 in enumerate(subsystem)}
    #
    data = _empty()
    data.P = 0 if permutation==(0,1) else 1    # This line of code is specific to two fragments (needs to be generalized for >=3).
    #
    Dchg_rhos = {+2:["aa", "caaa"], +1:["a","caa"], 0:["ca","ccaa"], -1:["c","cca"], -2:["cc", "ccca"]}
    n_i = 0
    n_i_label = ""
    for m0,m0_ in reversed(list(enumerate(permutation))):
        m0_str = str(m0)
        n_i_label = m0_str + n_i_label
        chg_i_m0 , chg_j_m0  = charges[m0]
        chg_i_m0_, chg_j_m0_ = charges[m0_]
        Dchg_m0_ = chg_i_m0_ - chg_j_m0_
        n_i += densities[m0]['n_elec'][chg_i_m0]    # this is not an error!
        data.__dict__["Dchg_"+m0_str] = Dchg_m0_
        data.__dict__["n_i"+n_i_label] = n_i%2
        for Dchg,rhos in Dchg_rhos.items():
            if Dchg==Dchg_m0_:
                for rho in rhos:
                    data.__dict__[rho+"_"+m0_str] = densities[m0_][rho][chg_i_m0_,chg_j_m0_]
        for m1,m1_ in enumerate(permutation):
            m01_str = m0_str + str(m1)
            data.__dict__["S_"+m01_str] = overlaps[m0_,m1_]
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
        X = _parameters(densities, integrals, subsystem, charges, permutation)
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
        X = _parameters(densities, integrals, subsystem, charges, permutation=(0,1))
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
        X = _parameters(densities, integrals, subsystem, charges, permutation)
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
        X = _parameters(densities, integrals, subsystem, charges, permutation)
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
        X = _parameters(densities, integrals, subsystem, charges, permutation=(0,1))
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
        X = _parameters(densities, integrals, subsystem, charges, permutation)
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
