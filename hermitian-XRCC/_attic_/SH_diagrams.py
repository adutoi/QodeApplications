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

def _parameters(densities, integrals, subsystem, charges, permutation=(0,)):
    # helper function to do repetitive manipulations of data passed from above
    densities = [densities[m] for m in subsystem]
    S = {(m0_,m1_):integrals.S[m0,m1] for m1_,m1 in enumerate(subsystem) for m0_,m0 in enumerate(subsystem)}
    T = {(m0_,m1_):integrals.T[m0,m1] for m1_,m1 in enumerate(subsystem) for m0_,m0 in enumerate(subsystem)}
    h = {(m0_,m1_):integrals.h[m0,m1] for m1_,m1 in enumerate(subsystem) for m0_,m0 in enumerate(subsystem)}
    U = {(m0_,m1_,m2_):integrals.U[m0,m1,m2] for m2_,m2 in enumerate(subsystem) for m1_,m1 in enumerate(subsystem) for m0_,m0 in enumerate(subsystem)}
    V = {(m0_,m1_,m2_,m3_):integrals.V[m0,m1,m2,m3] for m3_,m3 in enumerate(subsystem)
         for m2_,m2 in enumerate(subsystem) for m1_,m1 in enumerate(subsystem) for m0_,m0 in enumerate(subsystem)}
    #
    data = _empty()
    data.P = 0 if permutation==(0,1) else 1    # This line of code is specific to two fragments (needs to be generalized for >=3).
    #
    Dchg_rhos = {+2:["aa"], +1:["a","caa"], 0:["ca","ccaa"], -1:["c","cca"], -2:["cc"]}
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
            data.__dict__["S_"+m01_str] = S[m0_,m1_]
            data.__dict__["T_"+m01_str] = T[m0_,m1_]
            data.__dict__["h_"+m01_str] = h[m0_,m1_]
            for m2,m2_ in enumerate(permutation):
                m012_str  = m01_str + str(m2)
                m2_01_str = str(m2) + "_" + m01_str
                data.__dict__["U_"+m2_01_str] = U[m2_,m0_,m1_]
                for m3,m3_ in enumerate(permutation):
                    m0123_str = m012_str + str(m3)
                    data.__dict__["V_"+m0123_str] = V[m0_,m1_,m2_,m3_]
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
        X = _parameters(densities, integrals, subsystem, charges)
        if X.Dchg_0==0:
            prefactor = 1
            def diagram(i0,j0):
                return prefactor * tendot((X.T_00+X.U_0_00), X.ca_0[i0][j0], axes=([0,1],[0,1]))
            return [(diagram, (0,))]
        else:
            return [(None, None)]

    @staticmethod
    def order2(densities, integrals, subsystem, charges):
        X = _parameters(densities, integrals, subsystem, charges)
        if X.Dchg_0==0:
            prefactor = 1
            def diagram(i0,j0):
                return prefactor * tendot(X.V_0000, X.ccaa_0[i0][j0], axes=([0,1,2,3],[0,1,3,2]))
            return [(diagram, (0,))]
        else:
            return [(None, None)]



class body_2(object):

    # pq,pq-> :  h_00  ca_0
    @staticmethod
    def H1_one_body00(densities, integrals, subsystem, charges):
        result01 = body_2._H1_one_body00(densities, integrals, subsystem, charges, permutation=(0,1))
        result10 = body_2._H1_one_body00(densities, integrals, subsystem, charges, permutation=(1,0))
        return [result01, result10]
    @staticmethod
    def _H1_one_body00(densities, integrals, subsystem, charges, permutation):
        X = _parameters(densities, integrals, subsystem, charges, permutation)
        if X.Dchg_0==0 and X.Dchg_1==0:
            prefactor = 1
            def diagram(i0,i1,j0,j1):
                if i1==j1:
                    #return prefactor * numpy.einsum("pq,pq->", X.h_00, X.ca_0[i0][j0])
                    return prefactor * tendot(X.h_00, X.ca_0[i0][j0], axes=([0, 1], [0, 1]))
                else:
                    return 0
            return diagram, permutation
        else:
            return None, None

    # pqrs,pqsr-> :  V_0000  ccaa_0
    @staticmethod
    def H2_one_body00(densities, integrals, subsystem, charges):
        result01 = body_2._H2_one_body00(densities, integrals, subsystem, charges, permutation=(0,1))
        result10 = body_2._H2_one_body00(densities, integrals, subsystem, charges, permutation=(1,0))
        return [result01, result10]
    @staticmethod
    def _H2_one_body00(densities, integrals, subsystem, charges, permutation):
        X = _parameters(densities, integrals, subsystem, charges, permutation)
        if X.Dchg_0==0 and X.Dchg_1==0:
            prefactor = 1
            def diagram(i0,i1,j0,j1):
                if i1==j1:
                    #return prefactor * numpy.einsum("pqrs,pqsr->", X.V_0000, X.ccaa_0[i0][j0])
                    return prefactor * tendot(X.V_0000, X.ccaa_0[i0][j0], axes=([0, 1, 2, 3], [0, 1, 3, 2]))
                else:
                    return 0
            return diagram, permutation
        else:
            return None, None

    # pq,p,q-> :  h_01  c_0  a_1
    @staticmethod
    def H1(densities, integrals, subsystem, charges):
        result01 = body_2._H1(densities, integrals, subsystem, charges, permutation=(0,1))
        result10 = body_2._H1(densities, integrals, subsystem, charges, permutation=(1,0))
        return [result01, result10]
    @staticmethod
    def _H1(densities, integrals, subsystem, charges, permutation):
        X = _parameters(densities, integrals, subsystem, charges, permutation)
        if X.Dchg_0==-1 and X.Dchg_1==+1:
            prefactor = (-1)**(X.n_i1 + X.P)
            def diagram(i0,i1,j0,j1):
                #return prefactor * numpy.einsum("pq,p,q->", X.h_01, X.c_0[i0][j0], X.a_1[i1][j1])
                partial =          tendot(X.h_01,  X.c_0[i0][j0], axes=([0], [0]))
                return prefactor * tendot(partial, X.a_1[i1][j1], axes=([0], [0]))
            return diagram, permutation
        else:
            return None, None

    # prqs,pq,rs-> :  V_0101  ca_0  ca_1
    @staticmethod
    def H2_0011_CT0(densities, integrals, subsystem, charges):
        X = _parameters(densities, integrals, subsystem, charges, permutation=(0,1))
        if X.Dchg_0==0 and X.Dchg_1==0:
            prefactor = 4
            def diagram(i0,i1,j0,j1):
                #return prefactor * numpy.einsum("prqs,pq,rs->", X.V_0101, X.ca_0[i0][j0], X.ca_1[i1][j1])
                partial =          tendot(X.V_0101, X.ca_0[i0][j0], axes=([0, 2], [0, 1]))
                return prefactor * tendot(partial,  X.ca_1[i1][j1], axes=([0, 1], [0, 1]))
            return [(diagram, (0,1))]
        else:
            return [(None, None)]

    # pqsr,pqr,s-> :  V_0010  cca_0  a_1
    # psrq,pqr,s-> :  V_0100  caa_0  c_1
    @staticmethod
    def H2_0001_CT1(densities, integrals, subsystem, charges):
        result01 = body_2._H2_0001_CT1(  densities, integrals, subsystem, charges, permutation=(0,1))
        result10 = body_2._H2_0001_CT1(  densities, integrals, subsystem, charges, permutation=(1,0))
        return [result01, result10]
    @staticmethod
    def _H2_0001_CT1(densities, integrals, subsystem, charges, permutation):
        X = _parameters(densities, integrals, subsystem, charges, permutation)
        if X.Dchg_0==-1 and X.Dchg_1==+1:
            prefactor = 2 * (-1)**(X.n_i1 + X.P)
            def diagram(i0,i1,j0,j1):
                #return prefactor * numpy.einsum("pqsr,pqr,s->", X.V_0010, X.cca_0[i0][j0], X.a_1[i1][j1])
                partial =          tendot(X.V_0010, X.a_1[i1][j1],   axes=([2], [0]))
                return prefactor * tendot(partial,  X.cca_0[i0][j0], axes=([0, 1, 2], [0, 1, 2]))
            return diagram, permutation
        if X.Dchg_0==+1 and X.Dchg_1==-1:
            prefactor = 2 * (-1)**(X.n_i1 + X.P)
            def diagram(i0,i1,j0,j1):
                #return prefactor * numpy.einsum("psrq,pqr,s->", X.V_0100, X.caa_0[i0][j0], X.c_1[i1][j1])
                partial =          tendot(X.V_0100, X.c_1[i1][j1],   axes=([1], [0]))
                return prefactor * tendot(partial,  X.caa_0[i0][j0], axes=([0, 1, 2], [0, 2, 1]))
            return diagram, permutation
        else:
            return None, None

    # pqsr,pq,rs-> :  V_0011  cc_0  aa_1
    @staticmethod
    def H2_0011_CT2(densities, integrals, subsystem, charges):
        result01 = body_2._H2_0011_CT2(  densities, integrals, subsystem, charges, permutation=(0,1))
        result10 = body_2._H2_0011_CT2(  densities, integrals, subsystem, charges, permutation=(1,0))
        return [result01, result10]
    @staticmethod
    def _H2_0011_CT2(densities, integrals, subsystem, charges, permutation):
        X = _parameters(densities, integrals, subsystem, charges, permutation)
        if X.Dchg_0==-2 and X.Dchg_1==+2:
            prefactor = 1
            def diagram(i0,i1,j0,j1):
                #return prefactor * numpy.einsum("pqsr,pq,rs->", X.V_0011, X.cc_0[i0][j0], aa_1[i1][j1])
                partial =          tendot(X.V_0011, X.cc_0[i0][j0], axes=([0, 1], [0, 1]))
                return prefactor * tendot(partial,  X.aa_1[i1][j1], axes=([0, 1], [1, 0]))
            return diagram, permutation
        else:
            return None, None
        
    # pq,rs,rq,ps-> :  S_10  h_01  ca_0  ca_1
    @staticmethod
    def S1H1_0011_CT0(densities, integrals, subsystem, charges):
        result01 = body_2._S1H1_0011_CT0(densities, integrals, subsystem, charges, permutation=(0,1))
        result10 = body_2._S1H1_0011_CT0(densities, integrals, subsystem, charges, permutation=(1,0))
        return [result01, result10]
    @staticmethod
    def _S1H1_0011_CT0(densities, integrals, subsystem, charges, permutation):
        X = _parameters(densities, integrals, subsystem, charges, permutation)
        if X.Dchg_0==0 and X.Dchg_1==0:
            prefactor = -1
            def diagram(i0,i1,j0,j1):
                #return prefactor * numpy.einsum("pq,rs,rq,ps->", X.S_10, X.h_01, X.ca_0[i0][j0], X.ca_1[i1][j1])
                partial  =         tendot(X.S_10,   X.ca_0[i0][j0], axes=([1], [1]))
                partial2 =         tendot(X.h_01,   X.ca_1[i1][j1], axes=([1], [1]))
                return prefactor * tendot(partial,  partial2,       axes=([0, 1], [1, 0]))
            return diagram, permutation
        else:
            return None, None

    # pq,rs,prs,q-> :  S_01  h_00  cca_0  a_1
    # pq,rs,rqs,p-> :  S_10  h_00  caa_0  c_1
    @staticmethod
    def S1H1_0001_CT1(densities, integrals, subsystem, charges):
        result01 = body_2._S1H1_0001_CT1(densities, integrals, subsystem, charges, permutation=(0,1))
        result10 = body_2._S1H1_0001_CT1(densities, integrals, subsystem, charges, permutation=(1,0))
        return [result01, result10]
    @staticmethod
    def _S1H1_0001_CT1(densities, integrals, subsystem, charges, permutation):
        X = _parameters(densities, integrals, subsystem, charges, permutation)
        if X.Dchg_0==-1 and X.Dchg_1==+1:
            prefactor = (-1)**(X.n_i1 + X.P)
            def diagram(i0,i1,j0,j1):
                #return prefactor * numpy.einsum("pq,rs,prs,q->", X.S_01, X.h_00, X.cca_0[i0][j0], X.a_1[i1][j1])
                partial =          tendot(X.h_00,  X.cca_0[i0][j0], axes=([0, 1], [1, 2]))
                partial =          tendot(X.S_01,  partial,         axes=([0], [0]))
                return prefactor * tendot(partial, X.a_1[i1][j1],   axes=([0], [0]))
            return diagram, permutation
        if X.Dchg_0==+1 and X.Dchg_1==-1:
            prefactor = (-1)**(X.n_i1 + X.P)
            def diagram(i0,i1,j0,j1):
                #return prefactor * numpy.einsum("pq,rs,rqs,p->", X.S_10, X.h_00, X.caa_0[i0][j0], X.c_1[i1][j1])
                partial =          tendot(X.h_00,   X.caa_0[i0][j0], axes=([0, 1], [0, 2]))
                partial =          tendot(X.S_10,   partial,         axes=([1], [0]))
                return prefactor * tendot(partial,  X.c_1[i1][j1],   axes=([0], [0]))
            return diagram, permutation
        else:
            return None, None

    # pq,rs,rp,qs-> :  S_01  h_01  cc_0  aa_1
    @staticmethod
    def S1H1_0011_CT2(densities, integrals, subsystem, charges):
        result01 = body_2._S1H1_0011_CT2(densities, integrals, subsystem, charges, permutation=(0,1))
        result10 = body_2._S1H1_0011_CT2(densities, integrals, subsystem, charges, permutation=(1,0))
        return [result01, result10]
    @staticmethod
    def _S1H1_0011_CT2(densities, integrals, subsystem, charges, permutation):
        X = _parameters(densities, integrals, subsystem, charges, permutation)
        if X.Dchg_0==-2 and X.Dchg_1==+2:
            prefactor = 1
            def diagram(i0,i1,j0,j1):
                #return prefactor * numpy.einsum("pq,rs,rp,qs->", X.S_01, X.h_01, X.cc_0[i0][j0], X.aa_1[i1][j1])
                partial  =         tendot(X.S_01,  X.cc_0[i0][j0], axes=([0], [1]))
                partial2 =         tendot(X.h_01,  X.aa_1[i1][j1], axes=([1], [1]))
                return prefactor * tendot(partial, partial2,       axes=([0, 1], [1, 0]))
            return diagram, permutation
        else:
            return None, None

    # ij,pqsr,pqjr,is-> :  S_10  V_0010  ccaa_0  ca_1   +
    # ij,psrq,pirq,sj-> :  S_01  V_0100  ccaa_0  ca_1
    @staticmethod
    def S1H2_000011_CT0(densities, integrals, subsystem, charges):
        result01 = body_2._S1H2_000011_CT0(densities, integrals, subsystem, charges, permutation=(0,1))
        result10 = body_2._S1H2_000011_CT0(densities, integrals, subsystem, charges, permutation=(1,0))
        return [result01, result10]
    @staticmethod
    def _S1H2_000011_CT0(densities, integrals, subsystem, charges, permutation):
        X = _parameters(densities, integrals, subsystem, charges, permutation)
        if X.Dchg_0==0 and X.Dchg_1==0:
            prefactor = 2
            def diagram(i0,i1,j0,j1):
                #return prefactor * numpy.einsum("ij,pqsr,pqjr,is->", X.S_10, X.V_0010, X.ccaa_0[i0][j0], X.ca_1[i1][j1]) + numpy.einsum("ij,psrq,pirq,sj->", X.S_01, X.V_0100, X.ccaa_0[i0][j0], X.ca_1[i1][j1])
                partial  = tendot(X.S_10,           X.ca_1[i1][j1], axes=([0], [0]))
                partial  = tendot(X.ccaa_0[i0][j0], partial,        axes=([2], [0]))
                partial2 = tendot(X.S_01,           X.ca_1[i1][j1], axes=([1], [1]))
                partial2 = tendot(X.ccaa_0[i0][j0], partial2,       axes=([1], [0]))
                return prefactor * (tendot(X.V_0010, partial, axes=([0, 1, 2, 3], [0, 1, 3, 2])) + tendot(X.V_0100, partial2, axes=([0, 2, 3, 1], [0, 1, 2, 3])))
            return diagram, permutation
        else:
            return None, None

    # (4 * ij,prqs,piq,rjs-> :  S_01  V_0101  cca_0  caa_1)   +
    #      ij,pqsr,qpj,irs-> :  S_10  V_0011  cca_0  caa_1
    @staticmethod
    def S1H2_000111_CT1(densities, integrals, subsystem, charges):
        result01 = body_2._S1H2_000111_CT1(densities, integrals, subsystem, charges, permutation=(0,1))
        result10 = body_2._S1H2_000111_CT1(densities, integrals, subsystem, charges, permutation=(1,0))
        return [result01, result10]
    @staticmethod
    def _S1H2_000111_CT1(densities, integrals, subsystem, charges, permutation):
        X = _parameters(densities, integrals, subsystem, charges, permutation)
        if X.Dchg_0==-1 and X.Dchg_1==+1:
            prefactor = (-1)**(X.n_i1 + X.P)
            def diagram(i0,i1,j0,j1):
                #return prefactor * (4*numpy.einsum("ij,prqs,piq,rjs->", X.S_01, X.V_0101, X.cca_0[i0][j0], X.caa_1[i1][j1])) + numpy.einsum("ij,pqsr,qpj,irs->", X.S_10, X.V_0011, X.cca_0[i0][j0], X.caa_1[i1][j1])
                partial  = tendot(X.V_0101,        X.cca_0[i0][j0], axes=([0, 2], [0, 2]))
                partial  = tendot(X.caa_1[i1][j1], partial,         axes=([0, 2], [0, 1]))
                partial2 = tendot(X.V_0011,        X.cca_0[i0][j0], axes=([0, 1], [1, 0]))
                partial2 = tendot(X.caa_1[i1][j1], partial2,        axes=([1, 2], [1, 0]))
                return prefactor * (4 * tendot(X.S_01, partial, axes=([0, 1], [1, 0])) + tendot(X.S_10, partial2, axes=([0, 1], [0, 1])))
            return diagram, permutation
        else:
            return None, None
"""
    # ij,pqrs,pqisr,j-> :  S_01  V_0000  cccaa_0  a_1   +
    # ij,pqrs,pqjrs,i-> :  S_10  V_0000  ccaaa_0  c_1
    @staticmethod
    def S1H2_000001_CT1(densities, integrals, subsystem, charges):
        result01 = body_2._S1H2_000001_CT1(densities, integrals, subsystem, charges, permutation=(0,1))
        result10 = body_2._S1H2_000001_CT1(densities, integrals, subsystem, charges, permutation=(1,0))
        return [result01, result10]
    @staticmethod
    def _S1H2_000001_CT1(densities, integrals, subsystem, charges, permutation):
        X = _parameters(densities, integrals, subsystem, charges, permutation)
        if X.Dchg_0==-1 and X.Dchg_1==+1:
            prefactor = (-1)**(X.n_i1 + X.P)
            def diagram(i0,i1,j0,j1):
                return prefactor * (numpy.einsum(  "ij,pqrs,pqisr,j->", X.S_01, X.V_0000, X.cccaa_0[i0][j0], X.a_1[i1][j1])
                                    + numpy.einsum("ij,pqrs,pqjrs,i->", X.S_10, X.V_0000, X.ccaaa_0[i0][j0], X.c_1[i1][j1]))
            return diagram, permutation
        else:
            return None, None
        
    # ij,pqsr,qpir,js-> :  S_01  V_0010  ccca_0  aa_1   +
    # ij,psrq,pjqr,si-> :  S_10  V_0100  caaa_0  cc_1
    @staticmethod
    def S1H2_000011_CT2(densities, integrals, subsystem, charges):
        result01 = body_2._S1H2_000011_CT2(densities, integrals, subsystem, charges, permutation=(0,1))
        result10 = body_2._S1H2_000011_CT2(densities, integrals, subsystem, charges, permutation=(1,0))
        return [result01, result10]
    @staticmethod
    def _S1H2_000011_CT2(densities, integrals, subsystem, charges, permutation):
        X = _parameters(densities, integrals, subsystem, charges, permutation)
        if X.Dchg_0==-2 and X.Dchg_1==+2:
            prefactor = 2
            def diagram(i0,i1,j0,j1):
                return prefactor * (numpy.einsum(  "ij,pqsr,qpir,js->", X.S_01, X.V_0010, X.ccca_0[i0][j0], X.aa_1[i1][j1])
                                    + numpy.einsum("ij,psrq,pjqr,si->", X.S_10, X.V_0100, X.caaa_0[i0][j0], X.cc_1[i1][j1]))
            return diagram, permutation
        else:
            return None, None

    # ij,pqsr,pqi,jrs-> :  S_01  V_0011  ccc_0  aaa_1
    @staticmethod
    def S1H2_000111_CT3(densities, integrals, subsystem, charges):
        result01 = body_2._S1H2_000111_CT3(densities, integrals, subsystem, charges, permutation=(0,1))
        result10 = body_2._S1H2_000111_CT3(densities, integrals, subsystem, charges, permutation=(1,0))
        return [result01, result10]
    @staticmethod
    def _S1H2_000111_CT3(densities, integrals, subsystem, charges, permutation):
        X = _parameters(densities, integrals, subsystem, charges, permutation)
        if X.Dchg_0==-3 and X.Dchg_1==+3:
            prefactor = (-1)**(X.n_i1 + X.P)
            def diagram(i0,i1,j0,j1):
                return prefactor * numpy.einsum("ij,pqsr,pqi,jrs->", X.S_01, X.V_0011, X.ccc_0[i0][j0], X.aaa_1[i1][j1])
            return diagram, permutation
        else:
            return None, None

    # pq,rs,ij,ipsj,rq-> :  S_01  S_10  h_00  ccaa_0  ca_1   +
    # pq,rs,ij,irqj,ps-> :  S_10  S_01  h_00  ccaa_0  ca_1
    @staticmethod
    def S2H1_000011_CT0(densities, integrals, subsystem, charges):
        result01 = body_2._S2H1_000011_CT0(densities, integrals, subsystem, charges, permutation=(0,1))
        result10 = body_2._S2H1_000011_CT0(densities, integrals, subsystem, charges, permutation=(1,0))
        return [result01, result10]
    @staticmethod
    def _S2H1_000011_CT0(densities, integrals, subsystem, charges, permutation):
        X = _parameters(densities, integrals, subsystem, charges, permutation)
        if X.Dchg_0==0 and X.Dchg_1==0:
            prefactor = -1/2.
            def diagram(i0,i1,j0,j1):
                return prefactor * (numpy.einsum(  "qs,sq->", numpy.einsum("pq,ps->qs", X.S_01, numpy.einsum("ij,ipsj->ps", X.h_00, X.ccaa_0[i0][j0])), numpy.einsum("rs,rq->sq", X.S_10, X.ca_1[i1][j1]))
                                    + numpy.einsum("pr,pr->", numpy.einsum("pq,rq->pr", X.S_10, numpy.einsum("ij,irqj->rq", X.h_00, X.ccaa_0[i0][j0])), numpy.einsum("rs,ps->pr", X.S_01, X.ca_1[i1][j1])))
            return diagram, permutation
        else:
            return None, None

    # pq,rs,ij,prj,isq-> :  S_01  S_01  h_10  cca_0  caa_1   +
    # pq,rs,ij,ips,rqj-> :  S_01  S_10  h_01  cca_0  caa_1   +
    # pq,rs,ij,irq,psj-> :  S_10  S_01  h_01  cca_0  caa_1
    @staticmethod
    def S2H1_000111_CT1(densities, integrals, subsystem, charges):
        result01 = body_2._S2H1_000111_CT1(densities, integrals, subsystem, charges, permutation=(0,1))
        result10 = body_2._S2H1_000111_CT1(densities, integrals, subsystem, charges, permutation=(1,0))
        return [result01, result10]
    @staticmethod
    def _S2H1_000111_CT1(densities, integrals, subsystem, charges, permutation):
        X = _parameters(densities, integrals, subsystem, charges, permutation)
        if X.Dchg_0==-3 and X.Dchg_1==+3:
            prefactor = (-1)**(X.n_i1 + X.P + 1) / 2.
            def diagram(i0,i1,j0,j1):
                return prefactor * (numpy.einsum(  "isq,qsi->", X.caa_1[i1][j1], numpy.einsum("pq,psi->qsi", X.S_01, numpy.einsum("rs,pri->psi", X.S_01, numpy.einsum("ij,prj->pri", X.h_10, X.cca_0[i0][j0]))))
                                    + numpy.einsum("rqj,jqr->", X.caa_1[i1][j1], numpy.einsum("pq,jpr->jqr", X.S_01, numpy.einsum("rs,jps->jpr", X.S_10, numpy.einsum("ij,ips->jps", X.h_01, X.cca_0[i0][j0]))))
                                    + numpy.einsum("psj,jsp->", X.caa_1[i1][j1], numpy.einsum("pq,jsq->jsp", X.S_10, numpy.einsum("rs,jrq->jsq", X.S_01, numpy.einsum("ij,irq->jrq", X.h_01, X.cca_0[i0][j0])))))
            return diagram, permutation
        else:
            return None, None

    # pq,rs,ij,iprj,sq-> :  S_01  S_01  h_00  ccca_0  aa_1
    # pq,rs,ij,isqj,pr-> :  S_10  S_10  h_00  caaa_0  cc_1
    @staticmethod
    def S2H1_000011_CT2(densities, integrals, subsystem, charges):
        result01 = body_2._S2H1_000011_CT2(densities, integrals, subsystem, charges, permutation=(0,1))
        result10 = body_2._S2H1_000011_CT2(densities, integrals, subsystem, charges, permutation=(1,0))
        return [result01, result10]
    @staticmethod
    def _S2H1_000011_CT2(densities, integrals, subsystem, charges, permutation):
        X = _parameters(densities, integrals, subsystem, charges, permutation)
        if X.Dchg_0==-2 and X.Dchg_1==+2:
            prefactor = 1/2.
            def diagram(i0,i1,j0,j1):
                return prefactor * numpy.einsum("qr,rq->", numpy.einsum("pq,pr->qr", X.S_01, numpy.einsum("ij,iprj->pr", X.h_00, X.ccca_0[i0][j0])), numpy.einsum("rs,sq->rq", X.S_01, X.aa_1[i1][j1]))
            return diagram, permutation
        if X.Dchg_0==+2 and X.Dchg_1==-2:
            prefactor = 1/2.
            def diagram(i0,i1,j0,j1):
                return prefactor * numpy.einsum("rq,qr->", numpy.einsum("rs,sq->rq", X.S_10, numpy.einsum("ij,isqj->sq", X.h_00, X.caaa_0[i0][j0])), numpy.einsum("pq,pr->qr", X.S_10, X.cc_1[i1][j1]))
            return diagram, permutation
        else:
            return None, None

    # pq,rs,ij,ipr,sqj-> :  S_01  S_01  h_01  ccc_0  aaa_1
    @staticmethod
    def S2H1_000111_CT3(densities, integrals, subsystem, charges):
        result01 = body_2._S2H1_000111_CT3(densities, integrals, subsystem, charges, permutation=(0,1))
        result10 = body_2._S2H1_000111_CT3(densities, integrals, subsystem, charges, permutation=(1,0))
        return [result01, result10]
    @staticmethod
    def _S2H1_000111_CT3(densities, integrals, subsystem, charges, permutation):
        X = _parameters(densities, integrals, subsystem, charges, permutation)
        if X.Dchg_0==-3 and X.Dchg_1==+3:
            prefactor = (-1)**(X.n_i1 + X.P) / 2.
            def diagram(i0,i1,j0,j1):
                return prefactor * numpy.einsum("jqr,rqj->", numpy.einsum("pq,jpr->jqr", X.S_01, numpy.einsum("ij,ipr->jpr", X.h_01, X.ccc_0[i0][j0])), numpy.einsum("rs,sqj->rqj", X.S_01, X.aaa_1[i1][j1]))
            return diagram, permutation
        else:
            return None, None
"""


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
    "H1_one_body00":   body_2.H1_one_body00,
    "H2_one_body00":   body_2.H2_one_body00,
    "H1":              body_2.H1,
    "H2_0011_CT0":     body_2.H2_0011_CT0,
    "H2_0001_CT1":     body_2.H2_0001_CT1,
    "H2_0011_CT2":     body_2.H2_0011_CT2,
    "S1H1_0011_CT0":   body_2.S1H1_0011_CT0,
    "S1H1_0001_CT1":   body_2.S1H1_0001_CT1,
    "S1H1_0011_CT2":   body_2.S1H1_0011_CT2,
    "S1H2_000011_CT0": body_2.S1H2_000011_CT0,
    "S1H2_000111_CT1": body_2.S1H2_000111_CT1
#   "S1H2_000001_CT1": body_2.S1H2_000001_CT1,
#   "S1H2_000011_CT2": body_2.S1H2_000011_CT2,
#   "S1H2_000111_CT3": body_2.S1H2_000111_CT3,
#   "S2H1_000011_CT0": body_2.S2H1_000011_CT0,
#   "S2H1_000111_CT1": body_2.S2H1_000111_CT1,
#   "S2H1_000011_CT2": body_2.S2H1_000011_CT2,
#   "S2H1_000111_CT3": body_2.S2H1_000111_CT3
}
