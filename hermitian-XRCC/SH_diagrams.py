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

import numpy as np
import tensorly as tl
from tendot_wrapper import tendot

def prune_integrals(integrals, group):
    ret = {}
    for key in integrals:
        #print(key)
        if key == "v":
            ret[key] = {(m1_,m2_,m3_,m4_):integrals[key][m1,m2,m3,m4] for m4_,m4 in enumerate(group)
                        for m3_,m3 in enumerate(group) for m2_,m2 in enumerate(group) for m1_,m1 in enumerate(group)}
        else:
            ret[key] = {(m1_,m2_):integrals[key][m1,m2] for m2_,m2 in enumerate(group) for m1_,m1 in enumerate(group)}
    return ret

def _parameters2(densities, charges, permutation):
    # helper functions to do repetitive manipulations of data passed from above
    # needs to be generalized (should not be hard) and have "2" removed from its name ... or maybe it is better this way
    m1, m2 = 0, 1  # perm (0, 0) and (1, 1) can be handled like this as well
    (chg_i1,chg_j1), (chg_i2,chg_j2) = charges
    n_i2 = densities[m2]['n_elec'][chg_i2]
    p = 0
    if permutation==(1,0):
        m1, m2 = 1, 0
        (chg_i2,chg_j2), (chg_i1,chg_j1) = (chg_i1,chg_j1), (chg_i2,chg_j2)
        p = 1
    rho1 = densities[m1]
    rho2 = densities[m2]
    return rho1, rho2, n_i2%2, p, (chg_i1,chg_j1), (chg_i2,chg_j2)

def _ints2(permutation, integrals, key_list):
    m1, m2 = 0, 1  # perm (0, 0) and (1, 1) can be handled like this as well
    if permutation == (1, 0):
        m1, m2 = 1, 0
    map_dict = {"0": m1, "1": m2}
    ret = []
    for elem in key_list:
        elems = [*elem]
        if elems[0] == "v" and len(elems) == 5:
            ret.append(integrals[elems[0]][map_dict[elems[1]], map_dict[elems[2]], map_dict[elems[3]], map_dict[elems[4]]])
        elif elems[0] in ["h", "s"] and len(elems) == 3:
            ret.append(integrals[elems[0]][map_dict[elems[1]], map_dict[elems[2]]])
        else:
            raise NotImplementedError(f"input string {elems} couldn't be formatted")
    if len(ret) == 1:
        ret = ret[0]
    return ret

##########
# Here are the implementations of the actual diagrams.
# The public @staticmethods must take the arguments (densities, integrals, charges), but after that, it is up to you.
# It should return a list of kernels that takes state indices (for specified fragment charges) their relevant permutations.
# Don't forget to update the "catalog" dictionary at the end.
##########


# One body terms are essentially the correlated eigenenergies of the single fragments.
# I would probably add them together at the highest computational layer.
# However, this is not the most general case, for which we have to include the body_1 class here!


class body_2(object):
    # Here is the thing with H1 and H2...they include pure one fragment contributions,
    # which get canceled by one-body terms, but still full H1 and H2 are important, if we
    # do the Taylor expansion of S^{-1}. Hence, we include them, and introduce
    # additional diagrams, which are the correction terms for pure H1 and H2.
    # We include them in this class for now, even though they are one-body terms

    @staticmethod
    def H1(densities, integrals, charges):
        result00 = body_2._H1_one_body00(densities, integrals, charges, permutation=(0,1))
        result01 = body_2._H1(densities, integrals, charges, permutation=(0,1))
        result10 = body_2._H1(densities, integrals, charges, permutation=(1,0))
        result11 = body_2._H1_one_body00(densities, integrals, charges, permutation=(1,0))
        return [result00, result01, result10, result11]
    @staticmethod
    def _H1(densities, integrals, charges, permutation):
        rho1, rho2, n_i2, p, (chg_i1,chg_j1), (chg_i2,chg_j2) = _parameters2(densities, charges, permutation)
        h01 = _ints2(permutation, integrals, ["h01"])
        prefactor = (-1)**(n_i2 + p)
        if chg_i1==chg_j1-1 and chg_i2==chg_j2+1:
            c1 = rho1['c'][chg_i1,chg_j1]
            a2 = rho2['a'][chg_i2,chg_j2]
            def diagram(i1,i2,j1,j2):
                #return prefactor * np.einsum("pq,p,q->", h01, c1[i1][j1], a2[i2][j2])
                partial = tendot(h01, tl.tensor(c1[i1][j1]), axes=([0], [0]))
                return prefactor * tendot(partial, tl.tensor(a2[i2][j2]), axes=([0], [0]))
            return diagram, permutation
        else:
            return None, None
    @staticmethod
    def _H1_one_body00(densities, integrals, charges, permutation):
        rho1, rho2, n_i2, p, (chg_i1,chg_j1), (chg_i2,chg_j2) = _parameters2(densities, charges, permutation)
        if chg_i1==chg_j1 and chg_i2==chg_j2:
            h00 = _ints2(permutation, integrals, ["h00"])
            ca1 = rho1["ca"][chg_i1,chg_j1]
            def diagram(i1,i2,j1,j2):
                if i2==j2:
                    #return np.einsum("pq,pq->", h00, ca1[i1][j1])
                    return tendot(h00, tl.tensor(ca1[i1][j1]), axes=([0, 1], [0, 1]))
                else:
                    return 0
            return diagram, permutation
        else:
            return None, None
        
    #@staticmethod
    # this is not true, since one would lack the contributions from the nuclear attraction
    # integrals between different fragments
    #def H1_pure_2_body(densities, integrals, charges):
    #    result01 = body_2._H1(densities, integrals, charges, permutation=(0,1))
    #    result10 = body_2._H1(densities, integrals, charges, permutation=(1,0))
    #    return [result01, result10]
    
    @staticmethod
    def H2(densities, integrals, charges):
        result0000 = body_2._H2_one_body00(densities, integrals, charges, permutation=(0,1))
        result0001_CT1 = body_2._H2_0001_CT1(densities, integrals, charges, permutation=(0,1))
        result0111_CT1 = body_2._H2_0001_CT1(densities, integrals, charges, permutation=(1,0))
        result0011_CT0 = body_2._H2_0011_CT0(densities, integrals, charges, permutation=(0,1))
        result0011_CT2_1 = body_2._H2_0011_CT2(densities, integrals, charges, permutation=(0,1))
        result0011_CT2_2 = body_2._H2_0011_CT2(densities, integrals, charges, permutation=(1,0))
        result1111 = body_2._H2_one_body00(densities, integrals, charges, permutation=(1,0))
        return [result0000, result0001_CT1, result0011_CT0, result0011_CT2_1, result0011_CT2_2, result0111_CT1, result1111]
    @staticmethod
    def _H2_one_body00(densities, integrals, charges, permutation):
        rho1, rho2, n_i2, p, (chg_i1,chg_j1), (chg_i2,chg_j2) = _parameters2(densities, charges, permutation)
        if chg_i1==chg_j1 and chg_i2==chg_j2:
            v0000 = _ints2(permutation, integrals, ["v0000"])
            ccaa1 = rho1["ccaa"][chg_i1,chg_j1]
            def diagram(i1,i2,j1,j2):
                if i2==j2:
                    #return np.einsum("pqrs,pqsr->", v0000, ccaa1[i1][j1])
                    return tendot(v0000, tl.tensor(ccaa1[i1][j1]), axes=([0, 1, 2, 3], [0, 1, 3, 2]))
                else:
                    return 0
            return diagram, permutation
        else:
            return None, None
    @staticmethod
    def _H2_0001_CT1(densities, integrals, charges, permutation):
        rho1, rho2, n_i2, p, (chg_i1,chg_j1), (chg_i2,chg_j2) = _parameters2(densities, charges, permutation)
        prefactor = 2 * (-1)**(n_i2 + p)
        v0010, v0100 = _ints2(permutation, integrals, ["v0010", "v0100"])
        if chg_i1==chg_j1-1 and chg_i2==chg_j2+1:
            cca1 = rho1["cca"][chg_i1,chg_j1]
            a2 = rho2["a"][chg_i2,chg_j2]
            def diagram(i1,i2,j1,j2):
                #return prefactor * np.einsum("pqsr,pqr,s->", v0010, cca1[i1][j1], a2[i2][j2])
                #return prefactor * np.einsum("pqr,pqr->", np.einsum("pqsr,s->pqr", v0010, a2[i2][j2]), cca1[i1][j1])
                partial = tendot(v0010, tl.tensor(a2[i2][j2]), axes=([2], [0]))
                return prefactor * tendot(partial, tl.tensor(cca1[i1][j1]), axes=([0, 1, 2], [0, 1, 2]))
            return diagram, permutation
        elif chg_i1==chg_j1+1 and chg_i2==chg_j2-1:
            caa1 = rho1["caa"][chg_i1,chg_j1]
            c2 = rho2["c"][chg_i2,chg_j2]
            def diagram(i1,i2,j1,j2):
                #return prefactor * np.einsum("psrq,pqr,s->", v0100, caa1[i1][j1], c2[i2][j2])
                #return prefactor * np.einsum("prq,pqr->", np.einsum("psrq,s->prq", v0100, c2[i2][j2]), caa1[i1][j1])
                partial = tendot(v0100, tl.tensor(c2[i2][j2]), axes=([1], [0]))
                return prefactor * tendot(partial, tl.tensor(caa1[i1][j1]), axes=([0, 1, 2], [0, 2, 1]))
            return diagram, permutation
        else:
            return None, None
    @staticmethod
    def _H2_0011_CT0(densities, integrals, charges, permutation):
        rho1, rho2, n_i2, p, (chg_i1,chg_j1), (chg_i2,chg_j2) = _parameters2(densities, charges, permutation)
        if chg_i1==chg_j1 and chg_i2==chg_j2:
            v0101 = _ints2(permutation, integrals, ["v0101"])
            ca1 = rho1["ca"][chg_i1,chg_j1]
            ca2 = rho2["ca"][chg_i2,chg_j2]
            def diagram(i1,i2,j1,j2):
                #return 4 * np.einsum("prqs,pq,rs->", v0101, ca1[i1][j1], ca2[i2][j2])
                #return 4 * np.einsum("rs,rs->", np.einsum("prqs,pq->rs", v0101, ca1[i1][j1]), ca2[i2][j2])
                partial = tendot(v0101, tl.tensor(ca1[i1][j1]), axes=([0, 2], [0, 1]))
                return 4 * tendot(partial, tl.tensor(ca2[i2][j2]), axes=([0, 1], [0, 1]))
            return diagram, permutation
        else:
            return None, None
    @staticmethod
    def _H2_0011_CT2(densities, integrals, charges, permutation):
        rho1, rho2, n_i2, p, (chg_i1,chg_j1), (chg_i2,chg_j2) = _parameters2(densities, charges, permutation)
        if chg_i1==chg_j1-2 and chg_i2==chg_j2+2:
            v0011 = _ints2(permutation, integrals, ["v0011"])
            cc1 = rho1["cc"][chg_i1,chg_j1]
            aa2 = rho2["aa"][chg_i2,chg_j2]
            def diagram(i1,i2,j1,j2):
                #return np.einsum("pqsr,pq,rs->", v0011, cc1[i1][j1], aa2[i2][j2])
                #return np.einsum("sr,rs->", np.einsum("pqsr,pq->sr", v0011, cc1[i1][j1]), aa2[i2][j2])
                partial = tendot(v0011, tl.tensor(cc1[i1][j1]), axes=([0, 1], [0, 1]))
                return tendot(partial, tl.tensor(aa2[i2][j2]), axes=([0, 1], [1, 0]))
            return diagram, permutation
        else:
            return None, None
        
    #@staticmethod
    #def H2_pure_2_body(densities, integrals, charges):
    #    result0001_CT1 = body_2._H2_0001_CT1(densities, integrals, charges, permutation=(0,1))
    #    result0111_CT1 = body_2._H2_0001_CT1(densities, integrals, charges, permutation=(1,0))
    #    result0011_CT0 = body_2._H2_0011_CT0(densities, integrals, charges, permutation=(0,1))
    #    result0011_CT2_1 = body_2._H2_0011_CT2(densities, integrals, charges, permutation=(0,1))
    #    result0011_CT2_2 = body_2._H2_0011_CT2(densities, integrals, charges, permutation=(1,0))
    #    return [result0001_CT1, result0011_CT0, result0011_CT2_1, result0011_CT2_2, result0111_CT1]
    
    @staticmethod
    def S1H1(densities, integrals, charges):
        result0001_CT1 = body_2._S1H1_0001_CT1(densities, integrals, charges, permutation=(0,1))
        result0111_CT1 = body_2._S1H1_0001_CT1(densities, integrals, charges, permutation=(1,0))
        result0011_CT0_1 = body_2._S1H1_0011_CT0(densities, integrals, charges, permutation=(0,1))
        result0011_CT0_2 = body_2._S1H1_0011_CT0(densities, integrals, charges, permutation=(1,0))
        result0011_CT2_1 = body_2._S1H1_0011_CT2(densities, integrals, charges, permutation=(0,1))
        result0011_CT2_2 = body_2._S1H1_0011_CT2(densities, integrals, charges, permutation=(1,0))
        return [result0001_CT1, result0011_CT0_1, result0011_CT0_2, result0011_CT2_1, result0011_CT2_2, result0111_CT1]
    @staticmethod
    def _S1H1_0001_CT1(densities, integrals, charges, permutation):
        rho1, rho2, n_i2, p, (chg_i1,chg_j1), (chg_i2,chg_j2) = _parameters2(densities, charges, permutation)
        prefactor = (-1)**(n_i2 + p)
        if chg_i1==chg_j1-1 and chg_i2==chg_j2+1:
            s01, h00 = _ints2(permutation, integrals, ["s01", "h00"])
            cca1 = rho1["cca"][chg_i1,chg_j1]
            a2 = rho2["a"][chg_i2,chg_j2]
            def diagram(i1,i2,j1,j2):
                #return np.einsum("pq,rs,prs,q->", s01, h00, cca1[i1][j1], a2[i2][j2])
                #return prefactor * np.einsum("pq,p,q->", s01, np.einsum("rs,prs->p", h00, cca1[i1][j1]), a2[i2][j2])
                partial = tendot(h00, tl.tensor(cca1[i1][j1]), axes=([0, 1], [1, 2]))
                partial = tendot(s01, partial, axes=([0], [0]))
                return prefactor * tendot(partial, tl.tensor(a2[i2][j2]), axes=([0], [0]))
            return diagram, permutation
        elif chg_i1==chg_j1+1 and chg_i2==chg_j2-1:
            s10, h00 = _ints2(permutation, integrals, ["s10", "h00"])
            caa1 = rho1["caa"][chg_i1,chg_j1]
            c2 = rho2["c"][chg_i2,chg_j2]
            def diagram(i1,i2,j1,j2):
                #return np.einsum("pq,rs,rqs,p->", s10, h00, caa1[i1][j1], c2[i2][j2])
                #return prefactor * np.einsum("pq,q,p->", s10, np.einsum("rs,rqs->q", h00, caa1[i1][j1]), c2[i2][j2])
                partial = tendot(h00, tl.tensor(caa1[i1][j1]), axes=([0, 1], [0, 2]))
                partial = tendot(s10, partial, axes=([1], [0]))
                return prefactor * tendot(partial, tl.tensor(c2[i2][j2]), axes=([0], [0]))
            return diagram, permutation
        else:
            return None, None
    @staticmethod
    def _S1H1_0011_CT0(densities, integrals, charges, permutation):
        rho1, rho2, n_i2, p, (chg_i1,chg_j1), (chg_i2,chg_j2) = _parameters2(densities, charges, permutation)
        prefactor = -1
        if chg_i1==chg_j1 and chg_i2==chg_j2:
            s10, h01 = _ints2(permutation, integrals, ["s10", "h01"])
            ca1 = rho1["ca"][chg_i1,chg_j1]
            ca2 = rho2["ca"][chg_i2,chg_j2]
            def diagram(i1,i2,j1,j2):
                #return - np.einsum("pq,rs,rq,ps->", s10, h01, ca1[i1][j1], ca2[i2][j2])
                #return prefactor * np.einsum("pr,rp->", np.einsum("pq,rq->pr", s10, ca1[i1][j1]), np.einsum("rs,ps->rp", h01, ca2[i2][j2]))
                partial = tendot(s10, tl.tensor(ca1[i1][j1]), axes=([1], [1]))
                partial2 = tendot(h01, tl.tensor(ca2[i2][j2]), axes=([1], [1]))
                return prefactor * tendot(partial, partial2, axes=([0, 1], [1, 0]))
            return diagram, permutation
        else:
            return None, None
    @staticmethod
    def _S1H1_0011_CT2(densities, integrals, charges, permutation):
        rho1, rho2, n_i2, p, (chg_i1,chg_j1), (chg_i2,chg_j2) = _parameters2(densities, charges, permutation)
        if chg_i1==chg_j1-2 and chg_i2==chg_j2+2:
            s01, h01 = _ints2(permutation, integrals, ["s01", "h01"])
            cc1 = rho1["cc"][chg_i1,chg_j1]
            aa2 = rho2["aa"][chg_i2,chg_j2]
            def diagram(i1,i2,j1,j2):
                #return np.einsum("pq,rs,rp,qs->", s01, h01, cc1[i1][j1], aa2[i2][j2])
                #return np.einsum("qr,rq->", np.einsum("pq,rp->qr", s01, cc1[i1][j1]), np.einsum("rs,qs->rq", h01, aa2[i2][j2]))
                partial = tendot(s01, tl.tensor(cc1[i1][j1]), axes=([0], [1]))
                partial2 = tendot(h01, tl.tensor(aa2[i2][j2]), axes=([1], [1]))
                return tendot(partial, partial2, axes=([0, 1], [1, 0]))
            return diagram, permutation
        else:
            return None, None
        
    @staticmethod
    def S1H2(densities, integrals, charges):
        #ret_000011_CT2 = body_2._S1H2_000011_CT2(densities, integrals, charges, permutation=(0,1))
        #ret_110000_CT2 = body_2._S1H2_000011_CT2(densities, integrals, charges, permutation=(1,0))
        ret_000011_CT0 = body_2._S1H2_000011_CT0(densities, integrals, charges, permutation=(0,1))
        ret_110000_CT0 = body_2._S1H2_000011_CT0(densities, integrals, charges, permutation=(1,0))
        #ret_000111_CT3 = body_2._S1H2_000111_CT3(densities, integrals, charges, permutation=(0,1))
        #ret_111000_CT3 = body_2._S1H2_000111_CT3(densities, integrals, charges, permutation=(1,0))
        ret_000111_CT1 = body_2._S1H2_000111_CT1(densities, integrals, charges, permutation=(0,1))
        ret_111000_CT1 = body_2._S1H2_000111_CT1(densities, integrals, charges, permutation=(1,0))
        #ret_000001_CT1 = body_2._S1H2_000001_CT1(densities, integrals, charges, permutation=(0,1))
        #ret_111110_CT1 = body_2._S1H2_000001_CT1(densities, integrals, charges, permutation=(1,0))
        return [ret_000011_CT0, ret_110000_CT0, ret_000111_CT1, ret_111000_CT1]
    @staticmethod
    def _S1H2_000011_CT2(densities, integrals, charges, permutation):
        rho1, rho2, n_i2, p, (chg_i1,chg_j1), (chg_i2,chg_j2) = _parameters2(densities, charges, permutation)
        if chg_i1==chg_j1-2 and chg_i2==chg_j2+2:
            s01, s10, v0010, v0100 = _ints2(permutation, integrals, ["s01", "s10", "v0010", "v0100"])
            ccca1 = rho1["ccca"][chg_i1,chg_j1]
            caaa1 = rho1["caaa"][chg_i1,chg_j1]
            aa2 = rho2["aa"][chg_i2,chg_j2]
            cc2 = rho2["cc"][chg_i2,chg_j2]
            def diagram(i1,i2,j1,j2):
                return 2 * (np.einsum("ij,pqsr,qpir,js->", s01, v0010, ccca1[i1][j1], aa2[i2][j2])
                            + np.einsum("ij,psrq,pjqr,si->", s10, v0100, caaa1[i1][j1], cc2[i2][j2]))
            return diagram, permutation
        else:
            return None, None
    @staticmethod
    def _S1H2_000011_CT0(densities, integrals, charges, permutation):
        rho1, rho2, n_i2, p, (chg_i1,chg_j1), (chg_i2,chg_j2) = _parameters2(densities, charges, permutation)
        if chg_i1==chg_j1 and chg_i2==chg_j2:
            s01, s10, v0010, v0100 = _ints2(permutation, integrals, ["s01", "s10", "v0010", "v0100"])
            ccaa1 = rho1["ccaa"][chg_i1,chg_j1]
            ca2 = rho2["ca"][chg_i2,chg_j2]
            def diagram(i1,i2,j1,j2):
                #return 2 * (np.einsum("ij,pqsr,pqjr,is->", s10, v0010, ccaa1[i1][j1], ca2[i2][j2])
                #            + np.einsum("ij,psrq,pirq,sj->", s01, v0100, ccaa1[i1][j1], ca2[i2][j2]))
                #return 2 * (np.einsum("pqsr,pqsr->", v0010, np.einsum("pqjr,js->pqsr", ccaa1[i1][j1], np.einsum("ij,is->js", s10, ca2[i2][j2])))
                #            + np.einsum("psrq,psrq->", v0100, np.einsum("pirq,is->psrq", ccaa1[i1][j1], np.einsum("ij,sj->is", s01, ca2[i2][j2]))))
                partial = tendot(s10, tl.tensor(ca2[i2][j2]), axes=([0], [0]))
                partial = tendot(tl.tensor(ccaa1[i1][j1]), partial, axes=([2], [0]))
                partial2 = tendot(s01, tl.tensor(ca2[i2][j2]), axes=([1], [1]))
                partial2 = tendot(tl.tensor(ccaa1[i1][j1]), partial2, axes=([1], [0]))
                return 2 * (tendot(v0010, partial, axes=([0, 1, 2, 3], [0, 1, 2, 3])) + tendot(v0100, partial2, axes=([0, 1, 2, 3], [0, 1, 2, 3])))
            return diagram, permutation
        else:
            return None, None
    @staticmethod
    def _S1H2_000111_CT3(densities, integrals, charges, permutation):
        rho1, rho2, n_i2, p, (chg_i1,chg_j1), (chg_i2,chg_j2) = _parameters2(densities, charges, permutation)
        prefactor = (-1)**(n_i2 + p)
        if chg_i1==chg_j1-3 and chg_i2==chg_j2+3:
            s01, v0011 = _ints2(permutation, integrals, ["s01", "v0011"])
            ccc1 = rho1["ccc"][chg_i1,chg_j1]
            aaa2 = rho2["aaa"][chg_i2,chg_j2]
            def diagram(i1,i2,j1,j2):
                return prefactor * np.einsum("ij,pqsr,pqi,jrs->", s01, v0011, ccc1[i1][j1], aaa2[i2][j2])
            return diagram, permutation
        else:
            return None, None
    @staticmethod
    def _S1H2_000111_CT1(densities, integrals, charges, permutation):
        rho1, rho2, n_i2, p, (chg_i1,chg_j1), (chg_i2,chg_j2) = _parameters2(densities, charges, permutation)
        prefactor = (-1)**(n_i2 + p)
        if chg_i1==chg_j1-1 and chg_i2==chg_j2+1:
            s01, s10, v0101, v0011 = _ints2(permutation, integrals, ["s01", "s10", "v0101", "v0011"])
            cca1 = rho1["cca"][chg_i1,chg_j1]
            caa2 = rho2["caa"][chg_i2,chg_j2]
            def diagram(i1,i2,j1,j2):
                #return prefactor * (4 * np.einsum("ij,prqs,piq,rjs->", s01, v0101, cca1[i1][j1], caa2[i2][j2])
                #                    + np.einsum("ij,pqsr,qpj,irs->", s10, v0011, cca1[i1][j1], caa2[i2][j2]))
                #return prefactor * (4 * np.einsum("ij,ji->", s01, np.einsum("rjs,rsi->ji", caa2[i2][j2], np.einsum("prqs,piq->rsi", v0101, cca1[i1][j1])))
                #                    + np.einsum("ij,ij->", s10, np.einsum("irs,srj->ij", caa2[i2][j2], np.einsum("pqsr,qpj->srj", v0011, cca1[i1][j1]))))
                partial = tendot(v0101, tl.tensor(cca1[i1][j1]), axes=([0, 2], [0, 2]))
                partial = tendot(tl.tensor(caa2[i2][j2]), partial, axes=([0, 2], [0, 1]))
                partial2 = tendot(v0011, tl.tensor(cca1[i1][j1]), axes=([0, 1], [1, 0]))
                partial2 = tendot(tl.tensor(caa2[i2][j2]), partial2, axes=([1, 2], [1, 0]))
                return prefactor * (4 * tendot(s01, partial, axes=([0, 1], [1, 0])) + tendot(s10, partial2, axes=([0, 1], [0, 1])))
            return diagram, permutation
        else:
            return None, None
    @staticmethod
    def _S1H2_000001_CT1(densities, integrals, charges, permutation):
        rho1, rho2, n_i2, p, (chg_i1,chg_j1), (chg_i2,chg_j2) = _parameters2(densities, charges, permutation)
        prefactor = (-1)**(n_i2 + p)
        if chg_i1==chg_j1-1 and chg_i2==chg_j2+1:
            s01, s10, v0000 = _ints2(permutation, integrals, ["s01", "s10", "v0000"])
            cccaa1 = rho1["cccaa"][chg_i1,chg_j1]
            ccaaa1 = rho1["ccaaa"][chg_i1,chg_j1]
            a2 = rho2["a"][chg_i2,chg_j2]
            c2 = rho2["c"][chg_i2,chg_j2]
            def diagram(i1,i2,j1,j2):
                return prefactor * (np.einsum("ij,pqrs,pqisr,j->", s01, v0000, cccaa1[i1][j1], a2[i2][j2])
                        + np.einsum("ij,pqrs,pqjrs,i->", s10, v0000, ccaaa1[i1][j1], c2[i2][j2]))
            return diagram, permutation
        else:
            return None, None
        
    @staticmethod
    def S2H1(densities, integrals, charges):
        ret_000011_CT2 = body_2._S2H1_000011_CT2(densities, integrals, charges, permutation=(0,1))
        ret_110000_CT2 = body_2._S2H1_000011_CT2(densities, integrals, charges, permutation=(1,0))
        ret_000011_CT0 = body_2._S2H1_000011_CT0(densities, integrals, charges, permutation=(0,1))
        ret_110000_CT0 = body_2._S2H1_000011_CT0(densities, integrals, charges, permutation=(1,0))
        ret_000111_CT1 = body_2._S2H1_000111_CT1(densities, integrals, charges, permutation=(0,1))
        ret_111000_CT1 = body_2._S2H1_000111_CT1(densities, integrals, charges, permutation=(1,0))
        ret_000111_CT3 = body_2._S2H1_000111_CT3(densities, integrals, charges, permutation=(0,1))
        ret_111000_CT3 = body_2._S2H1_000111_CT3(densities, integrals, charges, permutation=(1,0))
        return [ret_000011_CT2, ret_110000_CT2, ret_000011_CT0, ret_110000_CT0, ret_000111_CT1, ret_111000_CT1, ret_000111_CT3, ret_111000_CT3]
    @staticmethod
    def _S2H1_000011_CT2(densities, integrals, charges, permutation):
        rho1, rho2, n_i2, p, (chg_i1,chg_j1), (chg_i2,chg_j2) = _parameters2(densities, charges, permutation)
        if chg_i1==chg_i2-2 and chg_i2==chg_j2+2:
            s01, h00 = _ints2(permutation, integrals, ["s01", "h00"])
            ccca1 = rho1["ccca"][chg_i1,chg_j1]
            aa2 = rho2["aa"][chg_i2,chg_j2]
            def diagram(i1,i2,j1,j2):
                #return 0.5 * np.einsum("pq,rs,ij,iprj,sq->", s01, s01, h00, ccca1[i1][j1], aa2[i2][j2])
                return 0.5 * np.einsum("qr,rq->", np.einsum("pq,pr->qr", s01, np.einsum("ij,iprj->pr", h00, ccca1[i1][j1])), np.einsum("rs,sq->rq", s01, aa2[i2][j2]))
            return diagram, permutation
        elif chg_i1==chg_j1+2 and chg_i2==chg_j2-2:
            s10, h00 = _ints2(permutation, integrals, ["s10", "h00"])
            caaa1 = rho1["caaa"][chg_i1,chg_j1]
            cc2 = rho2["cc"][chg_i2,chg_j2]
            def diagram(i1,i2,j1,j2):
                #return 0.5 * np.einsum("pq,rs,ij,isqj,pr->", s10, s10, h00, caaa1[i1][j1], cc2[i2][j2])
                return 0.5 * np.einsum("rq,qr->", np.einsum("rs,sq->rq", s10, np.einsum("ij,isqj->sq", h00, caaa1[i1][j1])), np.einsum("pq,pr->qr", s10, cc2[i2][j2]))
            return diagram, permutation
        else:
            return None, None
    @staticmethod
    def _S2H1_000011_CT0(densities, integrals, charges, permutation):
        rho1, rho2, n_i2, p, (chg_i1,chg_j1), (chg_i2,chg_j2) = _parameters2(densities, charges, permutation)
        if chg_i1==chg_i2 and chg_i2==chg_j2:
            s01, s10, h00 = _ints2(permutation, integrals, ["s01", "s10", "h00"])
            ccaa1 = rho1["ccaa"][chg_i1,chg_j1]
            ca2 = rho2["ca"][chg_i2,chg_j2]
            def diagram(i1,i2,j1,j2):
                #return - 0.5 * (np.einsum("pq,rs,ij,ipsj,rq->", s01, s10, h00, ccaa1[i1][j1], ca2[i2][j2])
                #                + np.einsum("pq,rs,ij,irqj,ps->", s10, s01, h00, ccaa1[i1][j1], ca2[i2][j2]))
                return - 0.5 * (np.einsum("qs,sq->", np.einsum("pq,ps->qs", s01, np.einsum("ij,ipsj->ps", h00, ccaa1[i1][j1])), np.einsum("rs,rq->sq", s10, ca2[i2][j2]))
                                + np.einsum("pr,pr->", np.einsum("pq,rq->pr", s10, np.einsum("ij,irqj->rq", h00, ccaa1[i1][j1])), np.einsum("rs,ps->pr", s01, ca2[i2][j2])))
            return diagram, permutation
        else:
            return None, None
    @staticmethod
    def _S2H1_000111_CT3(densities, integrals, charges, permutation):
        rho1, rho2, n_i2, p, (chg_i1,chg_j1), (chg_i2,chg_j2) = _parameters2(densities, charges, permutation)
        prefactor = 0.5 * (-1)**(n_i2 + p)
        if chg_i1==chg_i2-3 and chg_i2==chg_j2+3:
            s01, h01 = _ints2(permutation, integrals, ["s01", "h01"])
            ccc1 = rho1["ccc"][chg_i1,chg_j1]
            aaa2 = rho2["aaa"][chg_i2,chg_j2]
            def diagram(i1,i2,j1,j2):
                #return prefactor * np.einsum("pq,rs,ij,ipr,sqj->", s01, s01, h01, ccc1[i1][j1], aaa2[i2][j2])
                return prefactor * np.einsum("jqr,rqj->", np.einsum("pq,jpr->jqr", s01, np.einsum("ij,ipr->jpr", h01, ccc1[i1][j1])), np.einsum("rs,sqj->rqj", s01, aaa2[i2][j2]))
            return diagram, permutation
        else:
            return None, None
    @staticmethod
    def _S2H1_000111_CT1(densities, integrals, charges, permutation):
        rho1, rho2, n_i2, p, (chg_i1,chg_j1), (chg_i2,chg_j2) = _parameters2(densities, charges, permutation)
        prefactor = 0.5 * (-1)**(n_i2 + p)
        if chg_i1==chg_i2-3 and chg_i2==chg_j2+3:
            s01, s10, h01, h10 = _ints2(permutation, integrals, ["s01", "s10", "h01", "h10"])
            cca1 = rho1["cca"][chg_i1,chg_j1]
            caa2 = rho2["caa"][chg_i2,chg_j2]
            def diagram(i1,i2,j1,j2):
                #return - prefactor * (np.einsum("pq,rs,ij,prj,isq->", s01, s01, h10, cca1[i1][j1], caa2[i2][j2])
                #                      + np.einsum("pq,rs,ij,ips,rqj->", s01, s10, h01, cca1[i1][j1], caa2[i2][j2])
                #                      + np.einsum("pq,rs,ij,irq,psj->", s10, s01, h01, cca1[i1][j1], caa2[i2][j2]))
                return - prefactor * (np.einsum("isq,qsi->", caa2[i2][j2], np.einsum("pq,psi->qsi", s01, np.einsum("rs,pri->psi", s01, np.einsum("ij,prj->pri", h10, cca1[i1][j1]))))
                                      + np.einsum("rqj,jqr->", caa2[i2][j2], np.einsum("pq,jpr->jqr", s01, np.einsum("rs,jps->jpr", s10, np.einsum("ij,ips->jps", h01, cca1[i1][j1]))))
                                      + np.einsum("psj,jsp->", caa2[i2][j2], np.einsum("pq,jsq->jsp", s10, np.einsum("rs,jrq->jsq", s01, np.einsum("ij,irq->jrq", h01, cca1[i1][j1])))))
            return diagram, permutation
        else:
            return None, None


##########
# A dictionary catalog.  the string association lets users specify active diagrams at the top level.
# would like to build automatically, but more difficult than expected to get function references correct
##########

catalog = {}

catalog[2] = {
    "H1": body_2.H1,
    #"H1_pure_2_body": body_2.H1_pure_2_body,
    "H2": body_2.H2,
    #"H2_pure_2_body": body_2.H2_pure_2_body,
    "S1H1": body_2.S1H1,
    "S1H2": body_2.S1H2,
    "S2H1": body_2.S2H1
}

# e.g., does not work
#catalog[2] = {}
#for k,v in body_2.__dict__.items():
#    catalog[2][k] = v
