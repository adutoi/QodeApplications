#    (C) Copyright 2024 Anthony D. Dutoi
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
from qode.util.dynamic_array import dynamic_array, cached
from qode.math.tensornet     import evaluate



def precontract(densities, integrals):
    p,q,r,s = "pqrs"
    precontractions = {}

    def ccaaMpqsr_Vpqrs(m):
        V = integrals.V[m,m,m,m]
        densities_m = densities[m]
        n_states = densities_m["n_states"]
        def ccaaMpqsr_Vpqrs_m(chg_i,chg_j):
            def ccaaMpqsr_Vpqrs_m_charges(i,j):
                ccaa = densities_m["ccaa"][chg_i,chg_j][i,j]
                return evaluate(ccaa(p,q,s,r) @ V(p,q,r,s))
            if chg_i-chg_j==0:
                return dynamic_array(cached(ccaaMpqsr_Vpqrs_m_charges), [range(n_states[chg_i]), range(n_states[chg_j])])
            else:
                return None
        return dynamic_array(cached(ccaaMpqsr_Vpqrs_m), [n_states.keys()]*2)
    precontractions["ccaa#pqsr_Vpqrs"] = dynamic_array(cached(ccaaMpqsr_Vpqrs), [range(len(densities))])

    def caMpr_VpMrM(m0,m1,m2):
        V = integrals.V[m0,m1,m0,m2]
        densities_m = densities[m0]
        n_states = densities_m["n_states"]
        def caMpr_VpMrM_m(chg_i,chg_j):
            def caMpr_VpMrM_m_charges(i,j):
                ca = densities_m["ca"][chg_i,chg_j][i,j]
                return evaluate(ca(p,r) @ V(p,0,r,1))
            if chg_i-chg_j==0:
                return dynamic_array(cached(caMpr_VpMrM_m_charges), [range(n_states[chg_i]), range(n_states[chg_j])])
            else:
                return None
        return dynamic_array(cached(caMpr_VpMrM_m), [n_states.keys()]*2)
    precontractions["ca#pr_Vp#r#"] = dynamic_array(cached(caMpr_VpMrM), [range(len(densities))])

    def ccaMpqs_VpqMs(m0,m1):
        V = integrals.V[m0,m0,m1,m0]
        densities_m = densities[m0]
        n_states = densities_m["n_states"]
        def ccaMpqs_VpqMs_m(chg_i,chg_j):
            def ccaMpqs_VpqMs_m_charges(i,j):
                cca = densities_m["cca"][chg_i,chg_j][i,j]
                return evaluate(cca(p,q,s) @ V(p,q,0,s))
            if chg_i-chg_j==-1:
                return dynamic_array(cached(ccaMpqs_VpqMs_m_charges), [range(n_states[chg_i]), range(n_states[chg_j])])
            else:
                return None
        return dynamic_array(cached(ccaMpqs_VpqMs_m), [n_states.keys()]*2)
    precontractions["cca#pqs_Vpq#s"] = dynamic_array(cached(ccaMpqs_VpqMs), [range(len(densities))])

    def caaMqsr_VMqrs(m0,m1):
        V = integrals.V[m1,m0,m0,m0]
        densities_m = densities[m0]
        n_states = densities_m["n_states"]
        def caaMqsr_VMqrs_m(chg_i,chg_j):
            def caaMqsr_VMqrs_m_charges(i,j):
                caa = densities_m["caa"][chg_i,chg_j][i,j]
                return evaluate(caa(q,s,r) @ V(0,q,r,s))
            if chg_i-chg_j==+1:
                return dynamic_array(cached(caaMqsr_VMqrs_m_charges), [range(n_states[chg_i]), range(n_states[chg_j])])
            else:
                return None
        return dynamic_array(cached(caaMqsr_VMqrs_m), [n_states.keys()]*2)
    precontractions["caa#qsr_V#qrs"] = dynamic_array(cached(caaMqsr_VMqrs), [range(len(densities))])

    def ccMpq_VpqMM(m0,m1,m2):
        V = integrals.V[m0,m0,m1,m2]
        densities_m = densities[m0]
        n_states = densities_m["n_states"]
        def ccMpq_VpqMM_m(chg_i,chg_j):
            def ccMpq_VpqMM_m_charges(i,j):
                cc = densities_m["cc"][chg_i,chg_j][i,j]
                return evaluate(cc(p,q) @ V(p,q,0,1))
            if chg_i-chg_j==-2:
                return dynamic_array(cached(ccMpq_VpqMM_m_charges), [range(n_states[chg_i]), range(n_states[chg_j])])
            else:
                return None
        return dynamic_array(cached(ccMpq_VpqMM_m), [n_states.keys()]*2)
    precontractions["cc#pq_Vpq##"] = dynamic_array(cached(ccMpq_VpqMM), [range(len(densities))])

    def ccaaMpqXs_VpqMs(m0,m1):
        V = integrals.V[m0,m0,m1,m0]
        densities_m = densities[m0]
        n_states = densities_m["n_states"]
        def ccaaMpqXs_VpqMs_m(chg_i,chg_j):
            def ccaaMpqXs_VpqMs_m_charges(i,j):
                ccaa = densities_m["ccaa"][chg_i,chg_j][i,j]
                return evaluate(ccaa(p,q,0,s) @ V(p,q,1,s))
            if chg_i-chg_j==0:
                return dynamic_array(cached(ccaaMpqXs_VpqMs_m_charges), [range(n_states[chg_i]), range(n_states[chg_j])])
            else:
                return None
        return dynamic_array(cached(ccaaMpqXs_VpqMs_m), [n_states.keys()]*2)
    precontractions["ccaa#pqXs_Vpq#s"] = dynamic_array(cached(ccaaMpqXs_VpqMs), [range(len(densities))])

    def ccaaMqXsr_VMqrs(m0,m1):
        V = integrals.V[m1,m0,m0,m0]
        densities_m = densities[m0]
        n_states = densities_m["n_states"]
        def ccaaMqXsr_VMqrs_m(chg_i,chg_j):
            def ccaaMqXsr_VMqrs_m_charges(i,j):
                ccaa = densities_m["ccaa"][chg_i,chg_j][i,j]
                return evaluate(ccaa(q,0,s,r) @ V(1,q,r,s))
            if chg_i-chg_j==0:
                return dynamic_array(cached(ccaaMqXsr_VMqrs_m_charges), [range(n_states[chg_i]), range(n_states[chg_j])])
            else:
                return None
        return dynamic_array(cached(ccaaMqXsr_VMqrs_m), [n_states.keys()]*2)
    precontractions["ccaa#qXsr_V#qrs"] = dynamic_array(cached(ccaaMqXsr_VMqrs), [range(len(densities))])

    def cccaaMpqXsr_Vpqrs(m):
        V = integrals.V[m,m,m,m]
        densities_m = densities[m]
        n_states = densities_m["n_states"]
        def cccaaMpqXsr_Vpqrs_m(chg_i,chg_j):
            def cccaaMpqXsr_Vpqrs_m_charges(i,j):
                cccaa = densities_m["cccaa"][chg_i,chg_j][i,j]
                return evaluate(cccaa(p,q,0,s,r) @ V(p,q,r,s))
            if chg_i-chg_j==-1:
                return dynamic_array(cached(cccaaMpqXsr_Vpqrs_m_charges), [range(n_states[chg_i]), range(n_states[chg_j])])
            else:
                return None
        return dynamic_array(cached(cccaaMpqXsr_Vpqrs_m), [n_states.keys()]*2)
    precontractions["cccaa#pqXsr_Vpqrs"] = dynamic_array(cached(cccaaMpqXsr_Vpqrs), [range(len(densities))])

    def ccaaaMpqXsr_Vpqrs(m):
        V = integrals.V[m,m,m,m]
        densities_m = densities[m]
        n_states = densities_m["n_states"]
        def ccaaaMpqXsr_Vpqrs_m(chg_i,chg_j):
            def ccaaaMpqXsr_Vpqrs_m_charges(i,j):
                ccaaa = densities_m["ccaaa"][chg_i,chg_j][i,j]
                return evaluate(ccaaa(p,q,0,s,r) @ V(p,q,r,s))
            if chg_i-chg_j==+1:
                return dynamic_array(cached(ccaaaMpqXsr_Vpqrs_m_charges), [range(n_states[chg_i]), range(n_states[chg_j])])
            else:
                return None
        return dynamic_array(cached(ccaaaMpqXsr_Vpqrs_m), [n_states.keys()]*2)
    precontractions["ccaaa#pqXsr_Vpqrs"] = dynamic_array(cached(ccaaaMpqXsr_Vpqrs), [range(len(densities))])

    def ccaMpXr_VpMrM(m0,m1,m2):
        V = integrals.V[m0,m1,m0,m2]
        densities_m = densities[m0]
        n_states = densities_m["n_states"]
        def ccaMpXr_VpMrM_m(chg_i,chg_j):
            def ccaMpXr_VpMrM_m_charges(i,j):
                cca = densities_m["cca"][chg_i,chg_j][i,j]
                return evaluate(cca(p,0,r) @ V(p,1,r,2))
            if chg_i-chg_j==-1:
                return dynamic_array(cached(ccaMpXr_VpMrM_m_charges), [range(n_states[chg_i]), range(n_states[chg_j])])
            else:
                return None
        return dynamic_array(cached(ccaMpXr_VpMrM_m), [n_states.keys()]*2)
    precontractions["cca#pXr_Vp#r#"] = dynamic_array(cached(ccaMpXr_VpMrM), [range(len(densities))])

    def ccaMpqX_VpqMM(m0,m1,m2):
        V = integrals.V[m0,m0,m1,m2]
        densities_m = densities[m0]
        n_states = densities_m["n_states"]
        def ccaMpqX_VpqMM_m(chg_i,chg_j):
            def ccaMpqX_VpqMM_m_charges(i,j):
                cca = densities_m["cca"][chg_i,chg_j][i,j]
                return evaluate(cca(p,q,0) @ V(p,q,1,2))
            if chg_i-chg_j==-1:
                return dynamic_array(cached(ccaMpqX_VpqMM_m_charges), [range(n_states[chg_i]), range(n_states[chg_j])])
            else:
                return None
        return dynamic_array(cached(ccaMpqX_VpqMM_m), [n_states.keys()]*2)
    precontractions["cca#pqX_Vpq##"] = dynamic_array(cached(ccaMpqX_VpqMM), [range(len(densities))])

    def cccaMpqXs_VpqMs(m0,m1):
        V = integrals.V[m0,m0,m1,m0]
        densities_m = densities[m0]
        n_states = densities_m["n_states"]
        def cccaMpqXs_VpqMs_m(chg_i,chg_j):
            def cccaMpqXs_VpqMs_m_charges(i,j):
                ccca = densities_m["ccca"][chg_i,chg_j][i,j]
                return evaluate(ccca(p,q,0,s) @ V(p,q,1,s))
            if chg_i-chg_j==-2:
                return dynamic_array(cached(cccaMpqXs_VpqMs_m_charges), [range(n_states[chg_i]), range(n_states[chg_j])])
            else:
                return None
        return dynamic_array(cached(cccaMpqXs_VpqMs_m), [n_states.keys()]*2)
    precontractions["ccca#pqXs_Vpq#s"] = dynamic_array(cached(cccaMpqXs_VpqMs), [range(len(densities))])

    def caaaMqXsr_VMqrs(m0,m1):
        V = integrals.V[m1,m0,m0,m0]
        densities_m = densities[m0]
        n_states = densities_m["n_states"]
        def caaaMqXsr_VMqrs_m(chg_i,chg_j):
            def caaaMqXsr_VMqrs_m_charges(i,j):
                caaa = densities_m["caaa"][chg_i,chg_j][i,j]
                return evaluate(caaa(q,0,s,r) @ V(1,q,r,s))
            if chg_i-chg_j==+2:
                return dynamic_array(cached(caaaMqXsr_VMqrs_m_charges), [range(n_states[chg_i]), range(n_states[chg_j])])
            else:
                return None
        return dynamic_array(cached(caaaMqXsr_VMqrs_m), [n_states.keys()]*2)
    precontractions["caaa#qXsr_V#qrs"] = dynamic_array(cached(caaaMqXsr_VMqrs), [range(len(densities))])

    return precontractions
