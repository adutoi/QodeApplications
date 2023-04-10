#    (C) Copyright 2018, 2019 Anthony D. Dutoi
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
from qode.util.PyC import import_C

contract = import_C("H_contractions", flags="-O2")
contract.monomer_1e.return_type(float)
contract.monomer_extPot.return_type(float)
contract.dimer_2min2pls.return_type(float)
contract.dimer_1min1pls_1e.return_type(float)
contract.dimer_1min1pls_2e.return_type(float)
contract.dimer_ExEx.return_type(float)
contract.trimer_2min1pls1pls.return_type(float)
contract.trimer_2pls1min1min.return_type(float)
contract.trimer_Ex1min1pls.return_type(float)




# This class manages the many different tensor contractions of primitive integrals and fragment density tensors,
# which are needed to build the renormalized supersystem Hamiltonian matrix.  Its member functions are called
# from inside loops over fragments and states of those fragments, building matrix elements one at a time.
class build_matrix_elements(object):
    def __init__(self, supersystem, integrals, nuc_repulsion):
        n_elec = [fragment.n_elec_ref for fragment in supersystem]    # n_elec refers to the fragments in their reference states
        rho    = [fragment.rho        for fragment in supersystem]
        self.data = rho, integrals.T, integrals.U, integrals.V, nuc_repulsion, n_elec
    def monomer(self, fragment, I, J):
        rho, T, U, V, nuc_repulsion, n_elec = self.data
        m = fragment
        i_m_chg, i_m_idx = I
        j_m_chg, j_m_idx = J
        if i_m_chg == j_m_chg:
            n_orb = T[m,m].shape[0]
            Rca   = rho[m]["ca"  ][(i_m_chg, j_m_chg)][i_m_idx][j_m_idx]
            Rccaa = rho[m]["ccaa"][(i_m_chg, j_m_chg)][i_m_idx][j_m_idx]	# replaced with single number b/c tensor precontracted with ints during monomer build
            #return contract.monomer(n_orb, Rca, Rccaa, T[m,m]+U[m,m,m], V[m,m,m,m])
            Enuc = nuc_repulsion[m,m] if i_m_idx==j_m_idx else 0
            return Enuc + Rccaa + contract.monomer_1e(n_orb, Rca, T[m,m]+U[m,m,m])
        else:
            return 0
    def dimer(self, fragments, I, J):
        rho, T, U, V, nuc_repulsion, n_elec = self.data
        m1, m2 = fragments
        i_m1, i_m2 = I
        j_m1, j_m2 = J
        i_m1_chg, i_m1_idx = i_m1
        i_m2_chg, i_m2_idx = i_m2
        j_m1_chg, j_m1_idx = j_m1
        j_m2_chg, j_m2_idx = j_m2
        if (i_m1_chg+i_m2_chg == j_m1_chg+j_m2_chg) and (i_m1_chg >= j_m1_chg-2) and (i_m1_chg <= j_m1_chg+2):
            n_orb1 = T[m1,m1].shape[0]
            n_orb2 = T[m2,m2].shape[0]
            sign2 = (-1)**((n_elec[m2]-i_m2_chg) % 2)
            if i_m1_chg == j_m1_chg-2:
                Rcc1 = rho[m1]["cc"][(i_m1_chg, j_m1_chg)][i_m1_idx][j_m1_idx]
                Raa2 = rho[m2]["aa"][(i_m2_chg, j_m2_chg)][i_m2_idx][j_m2_idx]
                return contract.dimer_2min2pls(n_orb1, n_orb2, Rcc1, Raa2, V[m1,m1,m2,m2])
            if i_m1_chg == j_m1_chg-1:
                Rc1   = rho[m1]["c"  ][(i_m1_chg, j_m1_chg)][i_m1_idx][j_m1_idx]
                Rcca1 = rho[m1]["cca"][(i_m1_chg, j_m1_chg)][i_m1_idx][j_m1_idx]
                Ra2   = rho[m2]["a"  ][(i_m2_chg, j_m2_chg)][i_m2_idx][j_m2_idx]
                Rcaa2 = rho[m2]["caa"][(i_m2_chg, j_m2_chg)][i_m2_idx][j_m2_idx]
                tmp  = contract.dimer_1min1pls_1e(n_orb1, n_orb2, Rc1, Ra2, T[m1,m2]+U[m1,m1,m2]+U[m2,m1,m2])
                tmp += contract.dimer_1min1pls_2e(n_orb1, n_orb2, Rc1, Rcca1, Ra2, Rcaa2, V[m1,m1,m1,m2], V[m1,m2,m2,m2])
                return +sign2 * tmp
            if i_m1_chg == j_m1_chg:
                Rca1 = rho[m1]["ca"][(i_m1_chg, j_m1_chg)][i_m1_idx][j_m1_idx]
                Rca2 = rho[m2]["ca"][(i_m2_chg, j_m2_chg)][i_m2_idx][j_m2_idx]
                tmp = contract.dimer_ExEx(n_orb1, n_orb2, Rca1, Rca2, V[m1,m2,m1,m2])
                if i_m1_idx==j_m1_idx:  tmp += contract.monomer_extPot(n_orb2, Rca2, U[m1,m2,m2])
                if i_m2_idx==j_m2_idx:  tmp += contract.monomer_extPot(n_orb1, Rca1, U[m2,m1,m1])
                if i_m1_idx==j_m1_idx and i_m2_idx==j_m2_idx:  tmp += nuc_repulsion[m1,m2]
                return tmp
            if i_m1_chg == j_m1_chg+1:
                Ra1   = rho[m1]["a"  ][(i_m1_chg, j_m1_chg)][i_m1_idx][j_m1_idx]
                Rcaa1 = rho[m1]["caa"][(i_m1_chg, j_m1_chg)][i_m1_idx][j_m1_idx]
                Rc2   = rho[m2]["c"  ][(i_m2_chg, j_m2_chg)][i_m2_idx][j_m2_idx]
                Rcca2 = rho[m2]["cca"][(i_m2_chg, j_m2_chg)][i_m2_idx][j_m2_idx]
                tmp  = contract.dimer_1min1pls_1e(n_orb2, n_orb1, Rc2, Ra1, T[m2,m1]+U[m1,m2,m1]+U[m2,m2,m1])
                tmp += contract.dimer_1min1pls_2e(n_orb2, n_orb1, Rc2, Rcca2, Ra1, Rcaa1, V[m2,m2,m2,m1], V[m2,m1,m1,m1])
                return -sign2 * tmp
            if i_m1_chg == j_m1_chg+2:
                Raa1 = rho[m1]["aa"][(i_m1_chg, j_m1_chg)][i_m1_idx][j_m1_idx]
                Rcc2 = rho[m2]["cc"][(i_m2_chg, j_m2_chg)][i_m2_idx][j_m2_idx]
                return contract.dimer_2min2pls(n_orb2, n_orb1, Rcc2, Raa1, V[m2,m2,m1,m1])
        else:
                return 0
    def trimer(self, fragments, I, J):
        rho, T, U, V, nuc_repulsion, n_elec = self.data
        m1, m2, m3 = fragments
        (i_m1_chg,i_m1_idx), (i_m2_chg,i_m2_idx), (i_m3_chg,i_m3_idx) = I
        (j_m1_chg,j_m1_idx), (j_m2_chg,j_m2_idx), (j_m3_chg,j_m3_idx) = J
        chg_diffs = sorted([i_m1_chg-j_m1_chg, i_m2_chg-j_m2_chg, i_m3_chg-j_m3_chg])
        if chg_diffs==[-2,+1,+1] or chg_diffs==[-1,0,+1] or chg_diffs==[-1,-1,+2]:
            n_orb1 = T[m1,m1].shape[0]
            n_orb2 = T[m2,m2].shape[0]
            n_orb3 = T[m3,m3].shape[0]
            sign2 = (-1)**((n_elec[m2]-i_m2_chg) % 2)
            sign3 = (-1)**((n_elec[m3]-i_m3_chg) % 2)
            if i_m1_chg==j_m1_chg-2:
                Rcc1 = rho[m1]["cc"][i_m1_chg,j_m1_chg][i_m1_idx][j_m1_idx]
                Ra2  = rho[m2]["a" ][i_m2_chg,j_m2_chg][i_m2_idx][j_m2_idx]
                Ra3  = rho[m3]["a" ][i_m3_chg,j_m3_chg][i_m3_idx][j_m3_idx]
                return sign3 * contract.trimer_2min1pls1pls(n_orb1, n_orb2, n_orb3, Rcc1, Ra2, Ra3, V[m1,m1,m2,m3])
            if i_m2_chg==j_m2_chg-2:
                Rcc2 = rho[m2]["cc"][i_m2_chg,j_m2_chg][i_m2_idx][j_m2_idx]
                Ra1  = rho[m1]["a" ][i_m1_chg,j_m1_chg][i_m1_idx][j_m1_idx]
                Ra3  = rho[m3]["a" ][i_m3_chg,j_m3_chg][i_m3_idx][j_m3_idx]
                return sign2*sign3 * contract.trimer_2min1pls1pls(n_orb2, n_orb1, n_orb3, Rcc2, Ra1, Ra3, V[m2,m2,m1,m3])
            if i_m3_chg==j_m3_chg-2:
                Rcc3 = rho[m3]["cc"][i_m3_chg,j_m3_chg][i_m3_idx][j_m3_idx]
                Ra1  = rho[m1]["a" ][i_m1_chg,j_m1_chg][i_m1_idx][j_m1_idx]
                Ra2  = rho[m2]["a" ][i_m2_chg,j_m2_chg][i_m2_idx][j_m2_idx]
                return sign2 * contract.trimer_2min1pls1pls(n_orb3, n_orb1, n_orb2, Rcc3, Ra1, Ra2, V[m3,m3,m1,m2])
            if i_m1_chg==j_m1_chg+2:
                Raa1 = rho[m1]["aa"][i_m1_chg,j_m1_chg][i_m1_idx][j_m1_idx]
                Rc2  = rho[m2]["c" ][i_m2_chg,j_m2_chg][i_m2_idx][j_m2_idx]
                Rc3  = rho[m3]["c" ][i_m3_chg,j_m3_chg][i_m3_idx][j_m3_idx]
                return sign3 * contract.trimer_2pls1min1min(n_orb1, n_orb2, n_orb3, Raa1, Rc2, Rc3, V[m2,m3,m1,m1])
            if i_m2_chg==j_m2_chg+2:
                Raa2 = rho[m2]["aa"][i_m2_chg,j_m2_chg][i_m2_idx][j_m2_idx]
                Rc1  = rho[m1]["c" ][i_m1_chg,j_m1_chg][i_m1_idx][j_m1_idx]
                Rc3  = rho[m3]["c" ][i_m3_chg,j_m3_chg][i_m3_idx][j_m3_idx]
                return sign2*sign3 * contract.trimer_2pls1min1min(n_orb2, n_orb1, n_orb3, Raa2, Rc1, Rc3, V[m1,m3,m2,m2])
            if i_m3_chg==j_m3_chg+2:
                Raa3 = rho[m3]["aa"][i_m3_chg,j_m3_chg][i_m3_idx][j_m3_idx]
                Rc1  = rho[m1]["c" ][i_m1_chg,j_m1_chg][i_m1_idx][j_m1_idx]
                Rc2  = rho[m2]["c" ][i_m2_chg,j_m2_chg][i_m2_idx][j_m2_idx]
                return sign2 * contract.trimer_2pls1min1min(n_orb3, n_orb1, n_orb2, Raa3, Rc1, Rc2, V[m1,m2,m3,m3])
            if i_m1_chg==j_m1_chg:
                Rca1 = rho[m1]["ca"][i_m1_chg,j_m1_chg][i_m1_idx][j_m1_idx]
                if i_m2_chg==j_m2_chg-1:
                    Rc2 = rho[m2]["c"][i_m2_chg,j_m2_chg][i_m2_idx][j_m2_idx]
                    Ra3 = rho[m3]["a"][i_m3_chg,j_m3_chg][i_m3_idx][j_m3_idx]
                    tmp = contract.trimer_Ex1min1pls(n_orb1, n_orb2, n_orb3, Rca1, Rc2, Ra3, V[m1,m2,m1,m3])
                    if i_m1_idx==j_m1_idx:  tmp += contract.dimer_1min1pls_1e(n_orb2, n_orb3, Rc2, Ra3, U[m1,m2,m3])
                    return +sign3 * tmp
                if i_m3_chg==j_m3_chg-1:
                    Rc3 = rho[m3]["c"][i_m3_chg,j_m3_chg][i_m3_idx][j_m3_idx]
                    Ra2 = rho[m2]["a"][i_m2_chg,j_m2_chg][i_m2_idx][j_m2_idx]
                    tmp = contract.trimer_Ex1min1pls(n_orb1, n_orb3, n_orb2, Rca1, Rc3, Ra2, V[m1,m3,m1,m2])
                    if i_m1_idx==j_m1_idx:  tmp += contract.dimer_1min1pls_1e(n_orb3, n_orb2, Rc3, Ra2, U[m1,m3,m2])
                    return -sign3 * tmp
            if i_m2_chg==j_m2_chg:
                Rca2 = rho[m2]["ca"][i_m2_chg,j_m2_chg][i_m2_idx][j_m2_idx]
                if i_m1_chg==j_m1_chg-1:
                    Rc1 = rho[m1]["c"][i_m1_chg,j_m1_chg][i_m1_idx][j_m1_idx]
                    Ra3 = rho[m3]["a"][i_m3_chg,j_m3_chg][i_m3_idx][j_m3_idx]
                    tmp = contract.trimer_Ex1min1pls(n_orb2, n_orb1, n_orb3, Rca2, Rc1, Ra3, V[m2,m1,m2,m3])
                    if i_m2_idx==j_m2_idx:  tmp += contract.dimer_1min1pls_1e(n_orb1, n_orb3, Rc1, Ra3, U[m2,m1,m3])
                    return +sign2*sign3 * tmp
                if i_m3_chg==j_m3_chg-1:
                    Rc3 = rho[m3]["c"][i_m3_chg,j_m3_chg][i_m3_idx][j_m3_idx]
                    Ra1 = rho[m1]["a"][i_m1_chg,j_m1_chg][i_m1_idx][j_m1_idx]
                    tmp = contract.trimer_Ex1min1pls(n_orb2, n_orb3, n_orb1, Rca2, Rc3, Ra1, V[m2,m3,m2,m1])
                    if i_m2_idx==j_m2_idx:  tmp += contract.dimer_1min1pls_1e(n_orb3, n_orb1, Rc3, Ra1, U[m2,m3,m1])
                    return -sign2*sign3 * tmp
            if i_m3_chg==j_m3_chg:
                Rca3 = rho[m3]["ca"][i_m3_chg,j_m3_chg][i_m3_idx][j_m3_idx]
                if i_m1_chg==j_m1_chg-1:
                    Rc1 = rho[m1]["c"][i_m1_chg,j_m1_chg][i_m1_idx][j_m1_idx]
                    Ra2 = rho[m2]["a"][i_m2_chg,j_m2_chg][i_m2_idx][j_m2_idx]
                    tmp = contract.trimer_Ex1min1pls(n_orb3, n_orb1, n_orb2, Rca3, Rc1, Ra2, V[m3,m1,m3,m2])
                    if i_m3_idx==j_m3_idx:  tmp += contract.dimer_1min1pls_1e(n_orb1, n_orb2, Rc1, Ra2, U[m3,m1,m2])
                    return +sign2 * tmp
                if i_m2_chg==j_m2_chg-1:
                    Rc2 = rho[m2]["c"][i_m2_chg,j_m2_chg][i_m2_idx][j_m2_idx]
                    Ra1 = rho[m1]["a"][i_m1_chg,j_m1_chg][i_m1_idx][j_m1_idx]
                    tmp = contract.trimer_Ex1min1pls(n_orb3, n_orb2, n_orb1, Rca3, Rc2, Ra1, V[m3,m2,m3,m1])
                    if i_m3_idx==j_m3_idx:  tmp += contract.dimer_1min1pls_1e(n_orb2, n_orb1, Rc2, Ra1, U[m3,m2,m1])
                    return -sign2 * tmp
        else:
                return 0
