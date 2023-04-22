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

import numpy
import tensorly
from tendot_wrapper import tendot



class _empty(object):  pass    # Basically just a dictionary

def _parameters1(densities, integrals, subsystem, charges):
    # helper functions to do repetitive manipulations of data passed from above
    # needs to be generalized (should not be hard) and have "2" removed from its name ... or maybe it is better this way
    densities = [densities[m] for m in subsystem]
    S = {(m1_,m2_):integrals.S[m1,m2] for m2_,m2 in enumerate(subsystem) for m1_,m1 in enumerate(subsystem)}
    T = {(m1_,m2_):integrals.T[m1,m2] for m2_,m2 in enumerate(subsystem) for m1_,m1 in enumerate(subsystem)}
    U = {(m1_,m2_,m3_):integrals.U[m1,m2,m3] for m3_,m3 in enumerate(subsystem) for m2_,m2 in enumerate(subsystem) for m1_,m1 in enumerate(subsystem)}
    V = {(m1_,m2_,m3_,m4_):integrals.V[m1,m2,m3,m4] for m4_,m4 in enumerate(subsystem) for m3_,m3 in enumerate(subsystem) for m2_,m2 in enumerate(subsystem) for m1_,m1 in enumerate(subsystem)}
    m1 = 0
    (chg_i1,chg_j1), = charges
    data = _empty()
    data.Dchg_1 = chg_i1 - chg_j1
    if data.Dchg_1==0:
        data.ca_1    = tensorly.tensor(densities[m1]['ca'  ][chg_i1,chg_j1])
        data.ccaa_1  = tensorly.tensor(densities[m1]['ccaa'][chg_i1,chg_j1])
    if data.Dchg_1==-1:
        data.c_1     = tensorly.tensor(densities[m1]['c'   ][chg_i1,chg_j1])
        data.cca_1   = tensorly.tensor(densities[m1]['cca' ][chg_i1,chg_j1])
    if data.Dchg_1==+1:
        data.a_1     = tensorly.tensor(densities[m1]['a'   ][chg_i1,chg_j1])
        data.caa_1   = tensorly.tensor(densities[m1]['caa' ][chg_i1,chg_j1])
    if data.Dchg_1==-2:
        data.cc_1    = tensorly.tensor(densities[m1]['cc'  ][chg_i1,chg_j1])
    if data.Dchg_1==+2:
        data.aa_1    = tensorly.tensor(densities[m1]['aa'  ][chg_i1,chg_j1])
    data.T_11   = tensorly.tensor(T[m1,m1])
    data.U_1_11 = tensorly.tensor(U[m1,m1,m1])
    data.V_1111 = tensorly.tensor(V[m1,m1,m1,m1])
    return data

def _parameters2(densities, integrals, subsystem, charges, permutation, key_list):
    # helper functions to do repetitive manipulations of data passed from above
    # needs to be generalized (should not be hard) and have "2" removed from its name ... or maybe it is better this way
    densities = [densities[m] for m in subsystem]
    pruned = {}
    for key in integrals:
        #print(key)
        if key == "v":
            pruned[key] = {(m1_,m2_,m3_,m4_):integrals[key][m1,m2,m3,m4] for m4_,m4 in enumerate(subsystem)
                        for m3_,m3 in enumerate(subsystem) for m2_,m2 in enumerate(subsystem) for m1_,m1 in enumerate(subsystem)}
        else:
            pruned[key] = {(m1_,m2_):integrals[key][m1,m2] for m2_,m2 in enumerate(subsystem) for m1_,m1 in enumerate(subsystem)}
    integrals = pruned
    #
    m1, m2 = 0, 1
    (chg_i1,chg_j1), (chg_i2,chg_j2) = charges
    n_i2 = densities[m2]['n_elec'][chg_i2]
    p = 0
    if permutation==(1,0):
        m1, m2 = 1, 0
        (chg_i2,chg_j2), (chg_i1,chg_j1) = (chg_i1,chg_j1), (chg_i2,chg_j2)
        p = 1
    data = _empty()
    data.n_i2   = n_i2%2
    data.p      = p
    data.Dchg_1 = chg_i1 - chg_j1
    data.Dchg_2 = chg_i2 - chg_j2
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
    if data.Dchg_1==+2:
        data.aa_1   = tensorly.tensor(densities[m1]['aa'  ][chg_i1,chg_j1])
    if data.Dchg_2==0:
        data.ca_2   = tensorly.tensor(densities[m2]['ca'  ][chg_i2,chg_j2])
        data.ccaa_2 = tensorly.tensor(densities[m2]['ccaa'][chg_i2,chg_j2])
    if data.Dchg_2==-1:
        data.c_2    = tensorly.tensor(densities[m2]['c'   ][chg_i2,chg_j2])
        data.cca_2  = tensorly.tensor(densities[m2]['cca' ][chg_i2,chg_j2])
    if data.Dchg_2==+1:
        data.a_2    = tensorly.tensor(densities[m2]['a'   ][chg_i2,chg_j2])
        data.caa_2  = tensorly.tensor(densities[m2]['caa' ][chg_i2,chg_j2])
    if data.Dchg_2==-2:
        data.cc_2   = tensorly.tensor(densities[m2]['cc'  ][chg_i2,chg_j2])
    if data.Dchg_2==+2:
        data.aa_2   = tensorly.tensor(densities[m2]['aa'  ][chg_i2,chg_j2])
    #
    map_dict = {"0": m1, "1": m2}
    for elem in key_list:
        elems = [*elem]
        newelem = elem[0] + "_" + "".join(elems[1:])
        if elems[0] == "v" and len(elems) == 5:
            data.__dict__[newelem] = integrals[elems[0]][map_dict[elems[1]], map_dict[elems[2]], map_dict[elems[3]], map_dict[elems[4]]]
        elif elems[0] in ["h", "s"] and len(elems) == 3:
            data.__dict__[newelem] = integrals[elems[0]][map_dict[elems[1]], map_dict[elems[2]]]
        else:
            raise NotImplementedError(f"input string {elems} couldn't be formatted")
    return data

##########
# Here are the implementations of the actual diagrams.
# The public @staticmethods must take the arguments (densities, integrals, subsystem, charges), but after that, it is up to you.
# It should return a list of kernels that takes state indices (for specified fragment charges) their relevant permutations.
# Don't forget to update the "catalog" dictionary at the end.
##########

class body_1(object):

    @staticmethod
    def order1(densities, integrals, subsystem, charges):
        X = _parameters1(densities, integrals, subsystem, charges)
        if X.Dchg_1==0:
            prefactor = 1
            def diagram(i1,j1):
                return prefactor * tendot((X.T_11+X.U_1_11), X.ca_1[i1][j1], axes=([0,1],[0,1]))
            return [(diagram, (0,))]
        else:
            return [(None, None)]

    @staticmethod
    def order2(densities, integrals, subsystem, charges):
        X = _parameters1(densities, integrals, subsystem, charges)
        if X.Dchg_1==0:
            prefactor = 1
            def diagram(i1,j1):
                return prefactor * tendot(X.V_1111, X.ccaa_1[i1][j1], axes=([0,1,2,3],[0,1,3,2]))
            return [(diagram, (0,))]
        else:
            return [(None, None)]



class body_2(object):
    # Here is the thing with H1 and H2...they include pure one fragment contributions,
    # which get canceled by one-body terms, but still full H1 and H2 are important, if we
    # do the Taylor expansion of S^{-1}. Hence, we include them, and introduce
    # additional diagrams, which are the correction terms for pure H1 and H2.
    # We include them in this class for now, even though they are one-body terms

    @staticmethod
    def H1(densities, integrals, subsystem, charges):
        result00 = body_2._H1_one_body00(densities, integrals, subsystem, charges, permutation=(0,1))
        result01 = body_2._H1(           densities, integrals, subsystem, charges, permutation=(0,1))
        result10 = body_2._H1(           densities, integrals, subsystem, charges, permutation=(1,0))
        result11 = body_2._H1_one_body00(densities, integrals, subsystem, charges, permutation=(1,0))
        return [result00, result01, result10, result11]
    @staticmethod
    def _H1(densities, integrals, subsystem, charges, permutation):
        X = _parameters2(densities, integrals, subsystem, charges, permutation, ["h01"])
        if X.Dchg_1==-1 and X.Dchg_2==+1:
            prefactor = (-1)**(X.n_i2 + X.p)
            def diagram(i1,i2,j1,j2):
                #return prefactor * numpy.einsum("pq,p,q->", h01, X.c_1[i1][j1], X.a_2[i2][j2])
                partial =          tendot(X.h_01,  X.c_1[i1][j1], axes=([0], [0]))
                return prefactor * tendot(partial, X.a_2[i2][j2], axes=([0], [0]))
            return diagram, permutation
        else:
            return None, None
    @staticmethod
    def _H1_one_body00(densities, integrals, subsystem, charges, permutation):
        X = _parameters2(densities, integrals, subsystem, charges, permutation, ["h00"])
        if X.Dchg_1==0 and X.Dchg_2==0:
            prefactor = 1
            def diagram(i1,i2,j1,j2):
                if i2==j2:
                    #return numpy.einsum("pq,pq->", h00, X.ca_1[i1][j1])
                    return prefactor * tendot(X.h_00, X.ca_1[i1][j1], axes=([0, 1], [0, 1]))
                else:
                    return 0
            return diagram, permutation
        else:
            return None, None
        
    #@staticmethod
    # this is not true, since one would lack the contributions from the nuclear attraction
    # integrals between different fragments
    #def H1_pure_2_body(densities, integrals, subsystem, charges):
    #    result01 = body_2._H1(densities, integrals, subsystem, charges, permutation=(0,1))
    #    result10 = body_2._H1(densities, integrals, subsystem, charges, permutation=(1,0))
    #    return [result01, result10]
    
    @staticmethod
    def H2(densities, integrals, subsystem, charges):
        result0000 =       body_2._H2_one_body00(densities, integrals, subsystem, charges, permutation=(0,1))
        result0001_CT1 =   body_2._H2_0001_CT1(  densities, integrals, subsystem, charges, permutation=(0,1))
        result0111_CT1 =   body_2._H2_0001_CT1(  densities, integrals, subsystem, charges, permutation=(1,0))
        result0011_CT0 =   body_2._H2_0011_CT0(  densities, integrals, subsystem, charges, permutation=(0,1))
        result0011_CT2_1 = body_2._H2_0011_CT2(  densities, integrals, subsystem, charges, permutation=(0,1))
        result0011_CT2_2 = body_2._H2_0011_CT2(  densities, integrals, subsystem, charges, permutation=(1,0))
        result1111 =       body_2._H2_one_body00(densities, integrals, subsystem, charges, permutation=(1,0))
        return [result0000, result0001_CT1, result0011_CT0, result0011_CT2_1, result0011_CT2_2, result0111_CT1, result1111]
    @staticmethod
    def _H2_one_body00(densities, integrals, subsystem, charges, permutation):
        X = _parameters2(densities, integrals, subsystem, charges, permutation, ["v0000"])
        if X.Dchg_1==0 and X.Dchg_2==0:
            prefactor = 1
            def diagram(i1,i2,j1,j2):
                if i2==j2:
                    #return numpy.einsum("pqrs,pqsr->", v_0000, X.ccaa_1[i1][j1])
                    return prefactor * tendot(X.v_0000, X.ccaa_1[i1][j1], axes=([0, 1, 2, 3], [0, 1, 3, 2]))
                else:
                    return 0
            return diagram, permutation
        else:
            return None, None
    @staticmethod
    def _H2_0001_CT1(densities, integrals, subsystem, charges, permutation):
        X = _parameters2(densities, integrals, subsystem, charges, permutation, ["v0010", "v0100"])
        if X.Dchg_1==-1 and X.Dchg_2==+1:
            prefactor = 2 * (-1)**(X.n_i2 + X.p)
            def diagram(i1,i2,j1,j2):
                #return prefactor * numpy.einsum("pqsr,pqr,s->", v_0010, X.cca_1[i1][j1], X.a_2[i2][j2])
                #return prefactor * numpy.einsum("pqr,pqr->", numpy.einsum("pqsr,s->pqr", v_0010, X.a_2[i2][j2]), X.cca_1[i1][j1])
                partial =          tendot(X.v_0010, X.a_2[i2][j2],   axes=([2], [0]))
                return prefactor * tendot(partial,  X.cca_1[i1][j1], axes=([0, 1, 2], [0, 1, 2]))
            return diagram, permutation
        if X.Dchg_1==+1 and X.Dchg_2==-1:
            prefactor = 2 * (-1)**(X.n_i2 + X.p)
            def diagram(i1,i2,j1,j2):
                #return prefactor * numpy.einsum("psrq,pqr,s->", v_0100, X.caa_1[i1][j1], X.c_2[i2][j2])
                #return prefactor * numpy.einsum("prq,pqr->", numpy.einsum("psrq,s->prq", v_0100, X.c_2[i2][j2]), X.caa_1[i1][j1])
                partial =          tendot(X.v_0100, X.c_2[i2][j2],   axes=([1], [0]))
                return prefactor * tendot(partial,  X.caa_1[i1][j1], axes=([0, 1, 2], [0, 2, 1]))
            return diagram, permutation
        else:
            return None, None
    @staticmethod
    def _H2_0011_CT0(densities, integrals, subsystem, charges, permutation):
        X = _parameters2(densities, integrals, subsystem, charges, permutation, ["v0101"])
        if X.Dchg_1==0 and X.Dchg_2==0:
            prefactor = 4
            def diagram(i1,i2,j1,j2):
                #return 4 * numpy.einsum("prqs,pq,rs->", v_0101, X.ca_1[i1][j1], X.ca_2[i2][j2])
                #return 4 * numpy.einsum("rs,rs->", numpy.einsum("prqs,pq->rs", v_0101, X.ca_1[i1][j1]), X.ca_2[i2][j2])
                partial =          tendot(X.v_0101, X.ca_1[i1][j1], axes=([0, 2], [0, 1]))
                return prefactor * tendot(partial,  X.ca_2[i2][j2], axes=([0, 1], [0, 1]))
            return diagram, permutation
        else:
            return None, None
    @staticmethod
    def _H2_0011_CT2(densities, integrals, subsystem, charges, permutation):
        X = _parameters2(densities, integrals, subsystem, charges, permutation, ["v0011"])
        if X.Dchg_1==-2 and X.Dchg_2==+2:
            prefactor = 1
            def diagram(i1,i2,j1,j2):
                #return numpy.einsum("pqsr,pq,rs->", v_0011, X.cc_1[i1][j1], X.aa_2[i2][j2])
                #return numpy.einsum("sr,rs->", numpy.einsum("pqsr,pq->sr", v_0011, X.cc_1[i1][j1]), X.aa_2[i2][j2])
                partial =          tendot(X.v_0011, X.cc_1[i1][j1], axes=([0, 1], [0, 1]))
                return prefactor * tendot(partial,  X.aa_2[i2][j2], axes=([0, 1], [1, 0]))
            return diagram, permutation
        else:
            return None, None
        
    #@staticmethod
    #def H2_pure_2_body(densities, integrals, subsystem, charges):
    #    result0001_CT1 = body_2._H2_0001_CT1(densities, integrals, subsystem, charges, permutation=(0,1))
    #    result0111_CT1 = body_2._H2_0001_CT1(densities, integrals, subsystem, charges, permutation=(1,0))
    #    result0011_CT0 = body_2._H2_0011_CT0(densities, integrals, subsystem, charges, permutation=(0,1))
    #    result0011_CT2_1 = body_2._H2_0011_CT2(densities, integrals, subsystem, charges, permutation=(0,1))
    #    result0011_CT2_2 = body_2._H2_0011_CT2(densities, integrals, subsystem, charges, permutation=(1,0))
    #    return [result0001_CT1, result0011_CT0, result0011_CT2_1, result0011_CT2_2, result0111_CT1]
    
    @staticmethod
    def S1H1(densities, integrals, subsystem, charges):
        result0001_CT1 =   body_2._S1H1_0001_CT1(densities, integrals, subsystem, charges, permutation=(0,1))
        result0111_CT1 =   body_2._S1H1_0001_CT1(densities, integrals, subsystem, charges, permutation=(1,0))
        result0011_CT0_1 = body_2._S1H1_0011_CT0(densities, integrals, subsystem, charges, permutation=(0,1))
        result0011_CT0_2 = body_2._S1H1_0011_CT0(densities, integrals, subsystem, charges, permutation=(1,0))
        result0011_CT2_1 = body_2._S1H1_0011_CT2(densities, integrals, subsystem, charges, permutation=(0,1))
        result0011_CT2_2 = body_2._S1H1_0011_CT2(densities, integrals, subsystem, charges, permutation=(1,0))
        return [result0001_CT1, result0011_CT0_1, result0011_CT0_2, result0011_CT2_1, result0011_CT2_2, result0111_CT1]
    @staticmethod
    def _S1H1_0001_CT1(densities, integrals, subsystem, charges, permutation):
        X = _parameters2(densities, integrals, subsystem, charges, permutation, ["s01", "s10", "h00"])
        if X.Dchg_1==-1 and X.Dchg_2==+1:
            prefactor = (-1)**(X.n_i2 + X.p)
            def diagram(i1,i2,j1,j2):
                #return numpy.einsum("pq,rs,prs,q->", s01, h00, X.cca_1[i1][j1], X.a_2[i2][j2])
                #return prefactor * numpy.einsum("pq,p,q->", s01, numpy.einsum("rs,prs->p", h00, X.cca_1[i1][j1]), X.a_2[i2][j2])
                partial =          tendot(X.h_00,  X.cca_1[i1][j1], axes=([0, 1], [1, 2]))
                partial =          tendot(X.s_01,  partial,         axes=([0], [0]))
                return prefactor * tendot(partial, X.a_2[i2][j2],   axes=([0], [0]))
            return diagram, permutation
        if X.Dchg_1==+1 and X.Dchg_2==-1:
            prefactor = (-1)**(X.n_i2 + X.p)
            def diagram(i1,i2,j1,j2):
                #return numpy.einsum("pq,rs,rqs,p->", s10, h00, X.caa_1[i1][j1], X.c_2[i2][j2])
                #return prefactor * numpy.einsum("pq,q,p->", s10, numpy.einsum("rs,rqs->q", h00, X.caa_1[i1][j1]), X.c_2[i2][j2])
                partial =          tendot(X.h_00,   X.caa_1[i1][j1], axes=([0, 1], [0, 2]))
                partial =          tendot(X.s_10,   partial,         axes=([1], [0]))
                return prefactor * tendot(partial,  X.c_2[i2][j2],   axes=([0], [0]))
            return diagram, permutation
        else:
            return None, None
    @staticmethod
    def _S1H1_0011_CT0(densities, integrals, subsystem, charges, permutation):
        X = _parameters2(densities, integrals, subsystem, charges, permutation, ["s10", "h01"])
        if X.Dchg_1==0 and X.Dchg_2==0:
            prefactor = -1
            def diagram(i1,i2,j1,j2):
                #return - numpy.einsum("pq,rs,rq,ps->", s10, h01, X.ca_1[i1][j1], X.ca_2[i2][j2])
                #return prefactor * numpy.einsum("pr,rp->", numpy.einsum("pq,rq->pr", s10, X.ca_1[i1][j1]), numpy.einsum("rs,ps->rp", h01, X.ca_2[i2][j2]))
                partial  =         tendot(X.s_10,   X.ca_1[i1][j1], axes=([1], [1]))
                partial2 =         tendot(X.h_01,   X.ca_2[i2][j2], axes=([1], [1]))
                return prefactor * tendot(partial,  partial2,       axes=([0, 1], [1, 0]))
            return diagram, permutation
        else:
            return None, None
    @staticmethod
    def _S1H1_0011_CT2(densities, integrals, subsystem, charges, permutation):
        X = _parameters2(densities, integrals, subsystem, charges, permutation, ["s01", "h01"])
        if X.Dchg_1==-2 and X.Dchg_2==+2:
            prefactor = 1
            def diagram(i1,i2,j1,j2):
                #return numpy.einsum("pq,rs,rp,qs->", s01, h01, X.cc_1[i1][j1], X.aa_2[i2][j2])
                #return numpy.einsum("qr,rq->", numpy.einsum("pq,rp->qr", s01, X.cc_1[i1][j1]), numpy.einsum("rs,qs->rq", h01, X.aa_2[i2][j2]))
                partial  =         tendot(X.s_01,  X.cc_1[i1][j1], axes=([0], [1]))
                partial2 =         tendot(X.h_01,  X.aa_2[i2][j2], axes=([1], [1]))
                return prefactor * tendot(partial, partial2,       axes=([0, 1], [1, 0]))
            return diagram, permutation
        else:
            return None, None
        
    @staticmethod
    def S1H2(densities, integrals, subsystem, charges):
        #ret_000011_CT2 = body_2._S1H2_000011_CT2(densities, integrals, subsystem, charges, permutation=(0,1))
        #ret_110000_CT2 = body_2._S1H2_000011_CT2(densities, integrals, subsystem, charges, permutation=(1,0))
        ret_000011_CT0 = body_2._S1H2_000011_CT0(densities, integrals, subsystem, charges, permutation=(0,1))
        ret_110000_CT0 = body_2._S1H2_000011_CT0(densities, integrals, subsystem, charges, permutation=(1,0))
        #ret_000111_CT3 = body_2._S1H2_000111_CT3(densities, integrals, subsystem, charges, permutation=(0,1))
        #ret_111000_CT3 = body_2._S1H2_000111_CT3(densities, integrals, subsystem, charges, permutation=(1,0))
        ret_000111_CT1 = body_2._S1H2_000111_CT1(densities, integrals, subsystem, charges, permutation=(0,1))
        ret_111000_CT1 = body_2._S1H2_000111_CT1(densities, integrals, subsystem, charges, permutation=(1,0))
        #ret_000001_CT1 = body_2._S1H2_000001_CT1(densities, integrals, subsystem, charges, permutation=(0,1))
        #ret_111110_CT1 = body_2._S1H2_000001_CT1(densities, integrals, subsystem, charges, permutation=(1,0))
        return [ret_000011_CT0, ret_110000_CT0, ret_000111_CT1, ret_111000_CT1]
    @staticmethod
    def _S1H2_000011_CT2(densities, integrals, subsystem, charges, permutation):
        X = _parameters2(densities, integrals, subsystem, charges, permutation, ["s01", "s10", "v0010", "v0100"])
        if X.Dchg_1==-2 and X.Dchg_2==+2:
            prefactor = 2
            def diagram(i1,i2,j1,j2):
                return prefactor * (numpy.einsum(  "ij,pqsr,qpir,js->", X.s_01, X.v_0010, X.ccca_1[i1][j1], X.aa_2[i2][j2])
                                    + numpy.einsum("ij,psrq,pjqr,si->", X.s_10, X.v_0100, X.caaa_1[i1][j1], X.cc_2[i2][j2]))
            return diagram, permutation
        else:
            return None, None
    @staticmethod
    def _S1H2_000011_CT0(densities, integrals, subsystem, charges, permutation):
        X = _parameters2(densities, integrals, subsystem, charges, permutation, ["s01", "s10", "v0010", "v0100"])
        if X.Dchg_1==0 and X.Dchg_2==0:
            prefactor = 2
            def diagram(i1,i2,j1,j2):
                #return 2 * (numpy.einsum("ij,pqsr,pqjr,is->", s10, v_0010, X.ccaa_1[i1][j1], X.ca_2[i2][j2])
                #            + numpy.einsum("ij,psrq,pirq,sj->", s01, v_0100, X.ccaa_1[i1][j1], X.ca_2[i2][j2]))
                #return 2 * (numpy.einsum("pqsr,pqsr->", v_0010, numpy.einsum("pqjr,js->pqsr", X.ccaa_1[i1][j1], numpy.einsum("ij,is->js", s10, X.ca_2[i2][j2])))
                #            + numpy.einsum("psrq,psrq->", v_0100, numpy.einsum("pirq,is->psrq", X.ccaa_1[i1][j1], numpy.einsum("ij,sj->is", s01, X.ca_2[i2][j2]))))
                partial  = tendot(X.s_10,           X.ca_2[i2][j2], axes=([0], [0]))
                partial  = tendot(X.ccaa_1[i1][j1], partial,        axes=([2], [0]))
                partial2 = tendot(X.s_01,           X.ca_2[i2][j2], axes=([1], [1]))
                partial2 = tendot(X.ccaa_1[i1][j1], partial2,       axes=([1], [0]))
                return prefactor * (tendot(X.v_0010, partial, axes=([0, 1, 2, 3], [0, 1, 2, 3])) + tendot(X.v_0100, partial2, axes=([0, 1, 2, 3], [0, 1, 2, 3])))
            return diagram, permutation
        else:
            return None, None
    @staticmethod
    def _S1H2_000111_CT3(densities, integrals, subsystem, charges, permutation):
        X = _parameters2(densities, integrals, subsystem, charges, permutation, ["s01", "v0011"])
        if X.Dchg_1==-3 and X.Dchg_2==+3:
            prefactor = (-1)**(X.n_i2 + X.p)
            def diagram(i1,i2,j1,j2):
                return prefactor * numpy.einsum("ij,pqsr,pqi,jrs->", X.s_01, X.v_0011, X.ccc_1[i1][j1], X.aaa_2[i2][j2])
            return diagram, permutation
        else:
            return None, None
    @staticmethod
    def _S1H2_000111_CT1(densities, integrals, subsystem, charges, permutation):
        X = _parameters2(densities, integrals, subsystem, charges, permutation, ["s01", "s10", "v0101", "v0011"])
        if X.Dchg_1==-1 and X.Dchg_2==+1:
            prefactor = (-1)**(X.n_i2 + X.p)
            def diagram(i1,i2,j1,j2):
                #return prefactor * (4 * numpy.einsum("ij,prqs,piq,rjs->", s01, v_0101, X.cca_1[i1][j1], X.caa_2[i2][j2])
                #                    + numpy.einsum("ij,pqsr,qpj,irs->", s10, v_0011, X.cca_1[i1][j1], X.caa_2[i2][j2]))
                #return prefactor * (4 * numpy.einsum("ij,ji->", s01, numpy.einsum("rjs,rsi->ji", X.caa_2[i2][j2], numpy.einsum("prqs,piq->rsi", v_0101, X.cca_1[i1][j1])))
                #                    + numpy.einsum("ij,ij->", s10, numpy.einsum("irs,srj->ij", X.caa_2[i2][j2], numpy.einsum("pqsr,qpj->srj", v_0011, X.cca_1[i1][j1]))))
                partial  = tendot(X.v_0101,        X.cca_1[i1][j1], axes=([0, 2], [0, 2]))
                partial  = tendot(X.caa_2[i2][j2], partial,         axes=([0, 2], [0, 1]))
                partial2 = tendot(X.v_0011,        X.cca_1[i1][j1], axes=([0, 1], [1, 0]))
                partial2 = tendot(X.caa_2[i2][j2], partial2,        axes=([1, 2], [1, 0]))
                return prefactor * (4 * tendot(X.s_01, partial, axes=([0, 1], [1, 0])) + tendot(X.s_10, partial2, axes=([0, 1], [0, 1])))
            return diagram, permutation
        else:
            return None, None
    @staticmethod
    def _S1H2_000001_CT1(densities, integrals, subsystem, charges, permutation):
        X = _parameters2(densities, integrals, subsystem, charges, permutation, ["s01", "s10", "v0000"])
        if X.Dchg_1==-1 and X.Dchg_2==+1:
            prefactor = (-1)**(X.n_i2 + X.p)
            def diagram(i1,i2,j1,j2):
                return prefactor * (numpy.einsum(  "ij,pqrs,pqisr,j->", X.s_01, X.v_0000, X.cccaa_1[i1][j1], X.a_2[i2][j2])
                                    + numpy.einsum("ij,pqrs,pqjrs,i->", X.s_10, X.v_0000, X.ccaaa_1[i1][j1], X.c_2[i2][j2]))
            return diagram, permutation
        else:
            return None, None
        
    @staticmethod
    def S2H1(densities, integrals, subsystem, charges):
        ret_000011_CT2 = body_2._S2H1_000011_CT2(densities, integrals, subsystem, charges, permutation=(0,1))
        ret_110000_CT2 = body_2._S2H1_000011_CT2(densities, integrals, subsystem, charges, permutation=(1,0))
        ret_000011_CT0 = body_2._S2H1_000011_CT0(densities, integrals, subsystem, charges, permutation=(0,1))
        ret_110000_CT0 = body_2._S2H1_000011_CT0(densities, integrals, subsystem, charges, permutation=(1,0))
        ret_000111_CT1 = body_2._S2H1_000111_CT1(densities, integrals, subsystem, charges, permutation=(0,1))
        ret_111000_CT1 = body_2._S2H1_000111_CT1(densities, integrals, subsystem, charges, permutation=(1,0))
        ret_000111_CT3 = body_2._S2H1_000111_CT3(densities, integrals, subsystem, charges, permutation=(0,1))
        ret_111000_CT3 = body_2._S2H1_000111_CT3(densities, integrals, subsystem, charges, permutation=(1,0))
        return [ret_000011_CT2, ret_110000_CT2, ret_000011_CT0, ret_110000_CT0, ret_000111_CT1, ret_111000_CT1, ret_000111_CT3, ret_111000_CT3]
    @staticmethod
    def _S2H1_000011_CT2(densities, integrals, subsystem, charges, permutation):
        X = _parameters2(densities, integrals, subsystem, charges, permutation, ["s01", "h00"])
        if X.Dchg_1==-2 and X.Dchg_2==+2:
            prefactor = 1/2.
            def diagram(i1,i2,j1,j2):
                #return 0.5 * numpy.einsum("pq,rs,ij,iprj,sq->", s01, s01, h00, X.ccca_1[i1][j1], X.aa_2[i2][j2])
                return prefactor * numpy.einsum("qr,rq->", numpy.einsum("pq,pr->qr", X.s_01, numpy.einsum("ij,iprj->pr", X.h_00, X.ccca_1[i1][j1])), numpy.einsum("rs,sq->rq", X.s_01, X.aa_2[i2][j2]))
            return diagram, permutation
        if X.Dchg_1==+2 and X.Dchg_2==-2:
            prefactor = 1/2.
            def diagram(i1,i2,j1,j2):
                #return 0.5 * numpy.einsum("pq,rs,ij,isqj,pr->", s10, s10, h00, X.caaa_1[i1][j1], X.cc_2[i2][j2])
                return prefactor * numpy.einsum("rq,qr->", numpy.einsum("rs,sq->rq", X.s_10, numpy.einsum("ij,isqj->sq", X.h_00, X.caaa_1[i1][j1])), numpy.einsum("pq,pr->qr", X.s_10, X.cc_2[i2][j2]))
            return diagram, permutation
        else:
            return None, None
    @staticmethod
    def _S2H1_000011_CT0(densities, integrals, subsystem, charges, permutation):
        X = _parameters2(densities, integrals, subsystem, charges, permutation, ["s01", "s10", "h00"])
        if X.Dchg_1==0 and X.Dchg_2==0:
            prefactor = -1/2.
            def diagram(i1,i2,j1,j2):
                #return - 0.5 * (numpy.einsum("pq,rs,ij,ipsj,rq->", s01, s10, h00, X.ccaa_1[i1][j1], X.ca_2[i2][j2])
                #                + numpy.einsum("pq,rs,ij,irqj,ps->", s10, s01, h00, X.ccaa_1[i1][j1], X.ca_2[i2][j2]))
                return prefactor * (numpy.einsum(  "qs,sq->", numpy.einsum("pq,ps->qs", X.s_01, numpy.einsum("ij,ipsj->ps", X.h_00, X.ccaa_1[i1][j1])), numpy.einsum("rs,rq->sq", X.s_10, X.ca_2[i2][j2]))
                                    + numpy.einsum("pr,pr->", numpy.einsum("pq,rq->pr", X.s_10, numpy.einsum("ij,irqj->rq", X.h_00, X.ccaa_1[i1][j1])), numpy.einsum("rs,ps->pr", X.s_01, X.ca_2[i2][j2])))
            return diagram, permutation
        else:
            return None, None
    @staticmethod
    def _S2H1_000111_CT3(densities, integrals, subsystem, charges, permutation):
        X = _parameters2(densities, integrals, subsystem, charges, permutation, ["s01", "h01"])
        if X.Dchg_1==-3 and X.Dchg_2==+3:
            prefactor = (-1)**(X.n_i2 + X.p) / 2.
            def diagram(i1,i2,j1,j2):
                #return prefactor * numpy.einsum("pq,rs,ij,ipr,sqj->", s01, s01, h01, X.ccc_1[i1][j1], X.aaa_2[i2][j2])
                return prefactor * numpy.einsum("jqr,rqj->", numpy.einsum("pq,jpr->jqr", X.s_01, numpy.einsum("ij,ipr->jpr", X.h_01, X.ccc_1[i1][j1])), numpy.einsum("rs,sqj->rqj", X.s_01, X.aaa_2[i2][j2]))
            return diagram, permutation
        else:
            return None, None
    @staticmethod
    def _S2H1_000111_CT1(densities, integrals, subsystem, charges, permutation):
        X = _parameters2(densities, integrals, subsystem, charges, permutation, ["s01", "s10", "h01", "h10"])
        if X.Dchg_1==-3 and X.Dchg_2==+3:
            prefactor = (-1)**(X.n_i2 + X.p + 1) / 2.
            def diagram(i1,i2,j1,j2):
                #return prefactor * (numpy.einsum("pq,rs,ij,prj,isq->", s01, s01, h10, X.cca_1[i1][j1], X.caa_2[i2][j2])
                #                      + numpy.einsum("pq,rs,ij,ips,rqj->", s01, s10, h01, X.cca_1[i1][j1], X.caa_2[i2][j2])
                #                      + numpy.einsum("pq,rs,ij,irq,psj->", s10, s01, h01, X.cca_1[i1][j1], X.caa_2[i2][j2]))
                return prefactor * (numpy.einsum(  "isq,qsi->", X.caa_2[i2][j2], numpy.einsum("pq,psi->qsi", X.s_01, numpy.einsum("rs,pri->psi", X.s_01, numpy.einsum("ij,prj->pri", X.h_10, X.cca_1[i1][j1]))))
                                    + numpy.einsum("rqj,jqr->", X.caa_2[i2][j2], numpy.einsum("pq,jpr->jqr", X.s_01, numpy.einsum("rs,jps->jpr", X.s_10, numpy.einsum("ij,ips->jps", X.h_01, X.cca_1[i1][j1]))))
                                    + numpy.einsum("psj,jsp->", X.caa_2[i2][j2], numpy.einsum("pq,jsq->jsp", X.s_10, numpy.einsum("rs,jrq->jsq", X.s_01, numpy.einsum("ij,irq->jrq", X.h_01, X.cca_1[i1][j1])))))
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

catalog[1] = {
    "order1": body_1.order1,
    "order2": body_1.order2
}

catalog[2] = {
    "H1": body_2.H1,
    #"H1_pure_2_body": body_2.H1_pure_2_body,
    "H2": body_2.H2,
    #"H2_pure_2_body": body_2.H2_pure_2_body,
    "S1H1": body_2.S1H1,
    "S1H2": body_2.S1H2,
    "S2H1": body_2.S2H1
}
