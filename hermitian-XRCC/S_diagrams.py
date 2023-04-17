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

#import numpy as np
import tensorly as tl
#from orb_projection import orb_proj_density
from tendot_wrapper import tendot

def prune_integrals(integrals, group):
    # this must be defined so that higher level knows how to build context restricted to subsystems
    return {(m1_,m2_):integrals[m1,m2] for m2_,m2 in enumerate(group) for m1_,m1 in enumerate(group)}

def _parameters2(densities, integrals, charges, permutation):
    # helper functions to do repetitive manipulations of data passed from above
    # needs to be generalized (should not be hard) and have "2" removed from its name ... or maybe it is better this way
    m1, m2 = 0, 1
    (chg_i1,chg_j1), (chg_i2,chg_j2) = charges
    n_i2 = densities[m2]['n_elec'][chg_i2]
    p = 0
    if permutation==(1,0):
        m1, m2 = 1, 0
        (chg_i2,chg_j2), (chg_i1,chg_j1) = (chg_i1,chg_j1), (chg_i2,chg_j2)
        p = 1
    rho1 = densities[m1]
    rho2 = densities[m2]
    return rho1, rho2, integrals[m1,m2], integrals[m2,m1], n_i2%2, p, (chg_i1,chg_j1), (chg_i2,chg_j2)



##########
# Here are the implementations of the actual diagrams.
# The public @staticmethods must take the arguments (densities, integrals, charges), but after that, it is up to you.
# It should return a list of kernels that takes state indices (for specified fragment charges) their relevant permutations.
# Don't forget to update the "catalog" dictionary at the end.
##########

class body_0(object):

    @staticmethod
    def identity(densities, integrals, charges):
        # Identity
        def diagram():
            return 1
        return [(diagram, [])]



class body_2(object):

    @staticmethod
    def order1_CT1(densities, integrals, charges):
        result01 = body_2._order1_CT1(densities, integrals, charges, permutation=(0,1))
        result10 = body_2._order1_CT1(densities, integrals, charges, permutation=(1,0))
        return [result01, result10]
    @staticmethod
    def _order1_CT1(densities, integrals, charges, permutation):
        # 1 * 1 * (1)<-(2)
        rho1, rho2, sig12, sig21, n_i2, p, (chg_i1,chg_j1), (chg_i2,chg_j2) = _parameters2(densities, integrals, charges, permutation)
        prefactor = (-1)**(n_i2 + p)
        if chg_i1==chg_j1-1 and chg_i2==chg_j2+1:
            c1 = tl.tensor(rho1['c'][chg_i1,chg_j1])
            a2 = tl.tensor(rho2['a'][chg_i2,chg_j2])
            def diagram(i1,i2,j1,j2):
                #return prefactor * np.einsum("pq,p,q->", sig12, c1[i1][j1], a2[i2][j2])
                partial = tendot(c1[i1][j1], sig12, axes=([0],[0]))
                return prefactor * tendot(partial, a2[i2][j2], axes=([0],[0]))
            return diagram, permutation
        else:
            return None, None

    @staticmethod
    def order2_CT0(densities, integrals, charges):
        # 1/2! * 1 * (1)<-->(2)
        permutation=(0,1)
        rho1, rho2, sig12, sig21, n_i2, p, (chg_i1,chg_j1), (chg_i2,chg_j2) = _parameters2(densities, integrals, charges, permutation)
        prefactor = -1
        if chg_i1==chg_j1 and chg_i2==chg_j2:
            ca1 = tl.tensor(rho1['ca'][chg_i1,chg_j1])
            ca2 = tl.tensor(rho2['ca'][chg_i2,chg_j2])
            def diagram(i1,i2,j1,j2):
                #return prefactor * np.einsum("qs,sq->", np.einsum("pq,ps->qs", sig12, ca1[i1][j1]), np.einsum("rs,rq->sq", sig21, ca2[i2][j2]))
                partial = tendot(ca1[i1][j1], sig12, axes=([0],[0]))
                partial = tendot(partial,     sig21, axes=([0],[1]))
                return prefactor * tendot(partial, ca2[i2][j2], axes=([0,1],[1,0]))
            return [(diagram, permutation)]
        else:
            return [(None, None)]

    @staticmethod
    def order2_CT2(densities, integrals, charges):
        result01 = body_2._order2_CT2(densities, integrals, charges, permutation=(0,1))
        result10 = body_2._order2_CT2(densities, integrals, charges, permutation=(1,0))
        return [result01, result10]
    @staticmethod
    def _order2_CT2(densities, integrals, charges, permutation):
        # 1/2! * 1 * (1)<-<-(2)
        rho1, rho2, sig12, sig21, n_i2, p, (chg_i1,chg_j1), (chg_i2,chg_j2) = _parameters2(densities, integrals, charges, permutation)
        prefactor = 1/2.
        if chg_i1==chg_j1-2 and chg_i2==chg_j2+2:
            cc1 = tl.tensor(rho1['cc'][chg_i1,chg_j1])
            aa2 = tl.tensor(rho2['aa'][chg_i2,chg_j2])
            def diagram(i1,i2,j1,j2):
                #return prefactor * np.einsum("qr,rq->", np.einsum("pq,pr->qr", sig12, cc1[i1][j1]), np.einsum("rs,sq->rq", sig12, aa2[i2][j2]))
                partial = tendot(cc1[i1][j1], sig12, axes=([0],[0]))
                partial = tendot(partial,     sig12, axes=([0],[0]))
                return prefactor * tendot(partial, aa2[i2][j2], axes=([0,1],[1,0]))
            return diagram, permutation
        else:
            return None, None

    @staticmethod
    def order3_CT1(densities, integrals, charges):
        result01 = body_2._order3_CT1(densities, integrals, charges, permutation=(0,1))
        result10 = body_2._order3_CT1(densities, integrals, charges, permutation=(1,0))
        return [result01, result10]
    @staticmethod
    def _order3_CT1(densities, integrals, charges, permutation):
        # 1/3! * 3 * (1)<-<-->(2)
        rho1, rho2, sig12, sig21, n_i2, p, (chg_i1,chg_j1), (chg_i2,chg_j2) = _parameters2(densities, integrals, charges, permutation)
        prefactor = (-1)**(n_i2 + p + 1) / 2.
        if chg_i1==chg_j1-1 and chg_i2==chg_j2+1:
            cca1 = rho1['cca'][chg_i1,chg_j1]
            caa2 = rho2['caa'][chg_i2,chg_j2]
            def diagram(i1,i2,j1,j2):
                partial = tendot(cca1[i1][j1], sig12, axes=([0],[0]))
                partial = tendot(partial,      sig12, axes=([0],[0]))
                partial = tendot(partial,      sig21, axes=([0],[1]))
                return prefactor * tendot(partial, caa2[i2][j2], axes=([0,1,2],[2,1,0]))
            return diagram, permutation
        else:
            return None, None

    @staticmethod
    def order4_CT0(densities, integrals, charges):
        # (1/4!) * 6 * (1)<-<-->->(2)
        permutation=(0,1)
        rho1, rho2, sig12, sig21, n_i2, p, (chg_i1,chg_j1), (chg_i2,chg_j2) = _parameters2(densities, integrals, charges, permutation)
        prefactor = 1/4.
        if chg_i1==chg_j1 and chg_i2==chg_j2:
            ccaa1 = rho1['ccaa'][chg_i1,chg_j1]
            ccaa2 = rho2['ccaa'][chg_i2,chg_j2]
            def diagram(i1,i2,j1,j2):
                partial = tendot(ccaa1[i1][j1], sig12, axes=([0],[0]))
                partial = tendot(partial,       sig12, axes=([0],[0]))
                partial = tendot(partial,       sig21, axes=([0],[1]))
                partial = tendot(partial,       sig21, axes=([0],[1]))
                return prefactor * tendot(partial, ccaa2[i2][j2], axes=([0,1,2,3],[3,2,1,0]))
            return [(diagram, permutation)]
        else:
            return [(None, None)]

    @staticmethod
    def order4_CT0_approx(densities, integrals, charges):
        # (1/4!) * 6 * (1)<-<-->->(2)
        permutation=(0,1)
        rho1, rho2, sig12, sig21, n_i2, p, (chg_i1,chg_j1), (chg_i2,chg_j2) = _parameters2(densities, integrals, charges, permutation)
        prefactor = 1/2.
        if chg_i1==chg_j1 and chg_i2==chg_j2:
            ca1 = rho1['ca'][chg_i1,chg_j1]
            ca2 = rho2['ca'][chg_i2,chg_j2]
            def diagram(i1,i2,j1,j2):
                if i1==j1 and i2==j2:
                    partialA = tendot(ca1[i1][j1],  sig12, axes=([0],[0]))
                    partialA = tendot(partialA,     sig21, axes=([0],[1]))
                    term1    = tendot(partialA, ca2[i2][j2], axes=([0,1],[1,0]))
                    term1    = term1**2
                    partialB = tendot(partialA, ca2[i2][j2], axes=([0],[1]))
                    partialB = tendot(ca2[i2][j2], partialB, axes=([0],[0]))
                    term2    = tendot(partialA, partialB, axes=([0,1],[0,1]))
                    return prefactor * (term1 - term2)
                else:
                    return 0
            return [(diagram, permutation)]
        else:
            return [(None, None)]



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
"order4_CT0_approx": body_2.order4_CT0_approx
}

# e.g., does not work
#catalog[2] = {}
#for k,v in body_2.__dict__.items():
#    catalog[2][k] = v
