#    (C) Copyright 2026 Marco Bauer
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

# Exact tests of the spin-adapted XR[0] Hamiltonian.
#
# Two minimal-basis H2 fragments have complete fragment Fock spaces (16 states each), so with the
# symmetric integrals the XR[0] matrix in the spin-coupled product basis must agree element by
# element with the second-quantized Hamiltonian (with neglected inter-fragment overlap), for every
# total charge and every total multiplicity.  The fragment densities come from an exact Fock-space
# FCI (ref_spin_densities.py) and pass through the same wrapping/reversal code as hummr's densities.
#
# Usage (PYTHONPATH as for the mains, plus this directory):
#     pytest -q tests/spin_adapted

import os, sys, itertools, functools
import numpy as np
import pytest

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)

from ref_spin_densities import Fragment, ReferenceBackend, jordan_wigner, second_quantized_H, spin_operators, high_spin_eigenstates, cg, hummr_types

ReferenceTypes = hummr_types(2)

charges = [0, 1, -1, 2, -2]
n_elec_ref = 2
distance = 1.6

class empty(object):  pass

@functools.lru_cache(maxsize=None)
def system():
    from dens_provider import DensBackend
    from get_ints import get_ints
    from get_xr_result_spin_adapt import build_frag_densities
    from qode.util import timer
    from qode.math.tensornet import raw
    hummr = DensBackend("hummr")
    roots = {0: {1: 3, 3: 1}, 1: {2: 2}, -1: {2: 2}, 2: {1: 1}, -2: {1: 1}}    # complete H2/STO-3G Fock space
    mo, hummr_states = hummr.init_backend([["H2", os.path.join(here, "h2_sto3g.inp"), roots, 2]])[0]   # densities up to XR[2]
    frags = []
    for m in range(2):
        f = empty()
        f.atoms = [["H", [0, 0, m * distance]], ["H", [0, 0, m * distance + 0.74]]]
        f.n_elec_ref = n_elec_ref
        f.basis = empty()
        f.basis.n_spatial_orb, f.basis.AOcode, f.basis.core, f.basis.MOcoeffs = 2, "STO-3G", [], np.array(mo)
        frags.append(f)
    symm, bior, nuc = get_ints(frags, True, timer(), spin_ints=False, backend="lible")
    ref_frag = Fragment(raw(symm.T[0, 0]) + raw(symm.U[0, 0, 0]), raw(symm.V[0, 0, 0, 0]), 2, n_elec_ref)
    backend = ReferenceBackend(ref_frag, charges)
    dens = [build_frag_densities(backend, "H2", backend.placeholder_states(), 0, n_elec_ref) for _ in range(2)]
    # full supersystem integrals (orbitals 0,1 on fragment 0 and 2,3 on fragment 1)
    def full(blocks, rank, nuclei=None):
        out = np.zeros([4] * rank)
        for key, b in blocks.items():
            if nuclei is not None:
                idx = key[1:]
            else:
                idx = key
            out[tuple(slice(2 * i, 2 * i + 2) for i in idx)] += b
        return out
    T = {(i, j): raw(symm.T[i, j]) for i in range(2) for j in range(2)}
    U = {(n, i, j): raw(symm.U[n, i, j]) for n in range(2) for i in range(2) for j in range(2)}
    V = {k: raw(symm.V[k]) for k in itertools.product(range(2), repeat=4)}
    # as in the XR[0] workflow: all terms (including the fragment-local ones) with biorthogonal integrals, the
    # zeroth order of S H with the biorthogonal Hamiltonian
    Tb = {k: raw(bior.T[k]) for k in T}
    Ub = {k: raw(bior.U[k]) for k in U}
    Vb = {k: raw(bior.V[k]) for k in V}
    return struct_(frags=frags, symm=symm, bior=bior, nuc=nuc, ref_frag=ref_frag, dens=dens, hummr=hummr, hummr_states=hummr_states,
                   h=full(T, 2) + full(U, 2, nuclei=True), W=full(V, 4),
                   h_bior=full(Tb, 2) + full(Ub, 2, nuclei=True), W_bior=full(Vb, 4))

class struct_(object):
    def __init__(self, **kw): self.__dict__.update(kw)

@functools.lru_cache(maxsize=None)
def hummr_dens():
    """hummr densities of the same complete H2 Fock space (state basis from hummr's RI integrals)"""
    from get_xr_result_spin_adapt import build_frag_densities
    s = system()
    return [build_frag_densities(s.hummr, "H2", s.hummr_states, 0, n_elec_ref) for _ in range(2)]

def xr_H(tm, dens=None, bior=False):
    from get_xr_result_spin_adapt import get_xr_H
    from qode.util import struct
    s = system()
    # symmetric integrals in place of the biorthogonal ones -> exact second-quantized Hamiltonian with s.h, s.W
    # biorthogonal ones (as in the workflow, monomer and dimer terms) -> with s.h_bior, s.W_bior
    dimer_ints = s.bior if bior else struct(T=s.symm.T, U=s.symm.U, V=s.symm.V)
    return get_xr_H((s.symm, dimer_ints, s.nuc), s.dens if dens is None else dens, 0, [charges, charges], tm)


@functools.lru_cache(maxsize=None)
def reference_densities(xr_level):
    """reference densities of the H2 fragment up to xr_level, passed through the density provider (incl. reversal)"""
    from get_xr_result_spin_adapt import build_frag_densities
    s = system()
    backend = ReferenceBackend(s.ref_frag, charges)
    return build_frag_densities(backend, "H2", backend.placeholder_states(), xr_level, n_elec_ref)

def test_reversal_rules():
    """
    the reversed (adjoint) densities of the provider (all types and paths up to XR[2], and the mult_bra > mult_ket
    channels of the self-adjoint types) agree with directly computed ones
    """
    from qode.math.tensornet import raw
    s = system()
    rho = reference_densities(2)
    direct = s.ref_frag.raising_densities(charges, 2)
    for name, restr in ReferenceTypes:
        if name.count("c") == name.count("a"):
            direct.setdefault(name, {})
            for cp, chans in s.ref_frag.densities(charges, [(name, restr)])[name].items():
                for ch, v in chans.items():
                    if ch[0] > ch[1]:
                        direct[name].setdefault(cp, {})[ch] = v
    n_checked = 0
    for name, by_chg in direct.items():
        for cp, chans in by_chg.items():
            for ch, ref in chans.items():
                np.testing.assert_allclose(raw(rho[name][cp][ch]), ref, atol=1e-12, err_msg=f"{name} {cp} {ch}")
                n_checked += 1
    assert n_checked >= 85


def test_reversal_rules_4_orbitals():
    """
    the same for a fragment with 4 orbitals and 2 electrons (charges -2 .. 2, up to 4 open shells, random integrals),
    where all types and paths up to XR[2] occur with several channels
    """
    from get_xr_result_spin_adapt import build_frag_densities
    from qode.math.tensornet import raw
    rng = np.random.default_rng(11)
    n, n_ref, chgs = 4, 2, [0, 1, -1, 2, -2]
    h = rng.normal(size=(n, n)); h = h + h.T
    W = rng.normal(size=(n, n, n, n)) / 10
    W = W + W.transpose(1, 0, 3, 2); W = W + W.transpose(2, 1, 0, 3); W = W + W.transpose(0, 3, 2, 1)
    frag = Fragment(h, W, n, n_ref)
    backend = ReferenceBackend(frag, chgs)
    rho = build_frag_densities(backend, "X", backend.placeholder_states(), 2, n_ref)
    direct = frag.raising_densities(chgs, 2)
    for name, restr in ReferenceTypes:
        if name.count("c") == name.count("a"):
            for cp, chans in frag.densities(chgs, [(name, restr)])[name].items():
                for ch, v in chans.items():
                    if ch[0] > ch[1]:
                        direct.setdefault(name, {}).setdefault(cp, {})[ch] = v
    n_checked = 0
    for name, by_chg in direct.items():
        for cp, chans in by_chg.items():
            for ch, ref in chans.items():
                np.testing.assert_allclose(raw(rho[name][cp][ch]), ref, atol=1e-11, err_msg=f"{name} {cp} {ch}")
                n_checked += 1
    assert n_checked >= 181


@pytest.mark.parametrize("bior", [False, True])
@pytest.mark.parametrize("tm", [1, 2, 3, 4, 5])
def test_xr0_matrix_elements(tm, bior):
    """
    XR[0] matrix elements between spin-coupled product states are exact, also for the non-hermitian
    Hamiltonian with biorthogonal dimer integrals (which lack the v_pqrs = v_rqps symmetry)
    """
    import excitonic
    from get_xr_result_spin_adapt import local_sectors, multiplicities_couple
    s = system()
    H1, H2 = xr_H(tm, bior=bior)
    Hxr = excitonic._build_fci_matrix(H1, [[None, H2], [None, None]])
    frag_states = [(c, m, i) for c, m, sl in local_sectors(s.dens[0]['n_states'], charges) for i in range(sl.stop - sl.start)]
    a8 = jordan_wigner(8)
    c8 = [x.T for x in a8]
    Hex = second_quantized_H(s.h_bior if bior else s.h, s.W_bior if bior else s.W, 4, a8, c8)
    m_states = {st: s.ref_frag.m_states(*st) for st in frag_states}
    basis, vecs = [], []
    Stot = (tm - 1) / 2
    for i0, s0 in enumerate(frag_states):
        for i1, s1 in enumerate(frag_states):
            if not multiplicities_couple(s0[1], s1[1], tm):
                continue
            S0, S1 = (s0[1] - 1) / 2, (s1[1] - 1) / 2
            v = np.zeros(256)
            for M0, v0 in m_states[s0].items():
                M1 = Stot - M0
                if abs(M1) <= S1 + 1e-9:
                    v += cg(S0, M0, S1, M1, Stot, Stot) * np.kron(v0, m_states[s1][round(M1, 1)])
            basis.append(i0 * len(frag_states) + i1)
            vecs.append(v)
    vecs = np.array(vecs).T
    np.testing.assert_allclose(vecs.T @ vecs, np.eye(len(basis)), atol=1e-12)
    np.testing.assert_allclose(Hxr[np.ix_(basis, basis)], vecs.T @ Hex @ vecs, atol=1e-10)


def check_xr0_spectra(dens):
    from get_xr_result_spin_adapt import get_xr_states_from_H, product_basis_mask
    s = system()
    a8 = jordan_wigner(8)
    c8 = [x.T for x in a8]
    Hex = second_quantized_H(s.h, s.W, 4, a8, c8)
    N, Sz, Sp = spin_operators(4, a8, c8)
    for tm in [1, 2, 3, 4, 5]:
        H1, H2 = xr_H(tm, dens)
        for n in range(9):
            ref = high_spin_eigenstates(Hex, N, Sz, Sp, n).get((n, tm))
            if ref is None:
                continue
            mask = product_basis_mask(dens, [charges, charges], 2 * n_elec_ref - n, tm)
            E, _ = get_xr_states_from_H(H1, H2, target_state=0, basis_mask=mask)
            assert abs(E - ref[0][0]) < 1e-10, (n, tm, E, ref[0][0])

def test_xr0_spectra():
    """lowest XR[0] eigenvalue for every total charge and multiplicity (product basis restricted by get_xr_states)"""
    check_xr0_spectra(system().dens)

def test_xr0_spectra_hummr():
    """
    the same with hummr's densities: for complete fragment Fock spaces the XR[0] spectrum does not
    depend on the fragment state basis, so this is an end-to-end test of hummr + density wrapping
    """
    check_xr0_spectra(hummr_dens())


@functools.lru_cache(maxsize=None)
def spin_orbital_system():
    """spin-orbital densities (all M components) and integrals (not antisymmetrized) of the H2 dimer"""
    from ref_spin_densities import so_densities
    from get_ints import get_ints
    from qode.util import timer
    s = system()
    symm, bior, nuc = get_ints(s.frags, True, timer(), spin_ints=True, backend="lible", antisymmetrize=False)
    rho, states = so_densities(s.ref_frag, charges, n_elec_ref)
    return struct_(symm=symm, bior=bior, nuc=nuc, rho=rho, states=states)

@pytest.mark.parametrize("bior", [False, True])
def test_xr0_spin_orbital(bior):
    """
    the spin-orbital evaluation of the diagrams (the original formalism, used as reference for XR[1]/XR[2] below)
    is exact at XR[0], with spin-orbital densities and integrals (not antisymmetrized)
    """
    import excitonic
    from get_xr_result_spin_adapt import get_xr_H
    from qode.util import struct
    s, so = system(), spin_orbital_system()
    dimer = so.bior if bior else struct(T=so.symm.T, U=so.symm.U, V=so.symm.V)
    H1, H2 = get_xr_H((so.symm, dimer, so.nuc), [so.rho, so.rho], 0, [charges] * 2, None)
    Hxr = excitonic._build_fci_matrix(H1, [[None, H2], [None, None]])
    a8 = jordan_wigner(8)
    c8 = [x.T for x in a8]
    Hex = second_quantized_H(s.h_bior if bior else s.h, s.W_bior if bior else s.W, 4, a8, c8)
    vecs = [v for chg in charges for *_, v in so.states[chg]]
    P = np.array([np.kron(v0, v1) for v0 in vecs for v1 in vecs]).T
    np.testing.assert_allclose(Hxr, P.T @ Hex @ P, atol=1e-10)

@pytest.mark.parametrize("xr_order,tms", [(1, (1, 2, 3, 4, 5)), (2, (1, 2, 3))])
def test_xr_spin_adapted_vs_spin_orbital(xr_order, tms):
    """
    The spin-adapted XR[1] and XR[2] matrices (biorthogonal integrals, overlap between the H2 fragments) agree with
    the spin-orbital evaluation of the same diagrams, transformed into the spin-coupled product basis.  (The
    regenerated V diagrams with spatial V were in turn checked against the original ones, derived with
    antisymmetrized V, which agree to 1e-14 with V_diff = 2 V_half - V, see get_ints.py.)
    """
    import excitonic
    from get_xr_result_spin_adapt import get_xr_H
    s, so = system(), spin_orbital_system()
    H1, H2 = get_xr_H((so.symm, so.bior, so.nuc), [so.rho, so.rho], xr_order, [charges] * 2, None)
    Hso = excitonic._build_fci_matrix(H1, [[None, H2], [None, None]])
    rho = reference_densities(2)
    for tm in tms:
        H1, H2 = get_xr_H((s.symm, s.bior, s.nuc), [rho, rho], xr_order, [charges] * 2, tm)
        Hsa = excitonic._build_fci_matrix(H1, [[None, H2], [None, None]])
        U, basis = spin_coupling(rho, so.states, tm)
        np.testing.assert_allclose(Hsa[np.ix_(basis, basis)], U.T @ Hso @ U, atol=1e-10, err_msg=f"multiplicity {tm}")

def spin_coupling(rho, states, tm):
    """
    the product states (a0, a1) coupled to the multiplicity tm (M = S) in terms of the spin-orbital product basis
    (all M components, see so_states): (U, basis), basis: their indices in the spin-adapted product basis
    """
    from get_xr_result_spin_adapt import local_sectors, multiplicities_couple
    index = {}
    for chg in charges:
        for mult, i, M, v in states[chg]:
            index[(chg, mult, i, round(2 * M))] = len(index)
    n_so = len(index)
    frag_states = [(c, m, i) for c, m, sl in local_sectors(rho["n_states"], charges) for i in range(sl.stop - sl.start)]
    cols, basis = [], []
    for a0, (c0, m0, i0) in enumerate(frag_states):
        for a1, (c1, m1, i1) in enumerate(frag_states):
            if not multiplicities_couple(m0, m1, tm):
                continue
            u = np.zeros(n_so * n_so)
            for M02 in range(-(m0 - 1), m0, 2):
                M12 = tm - 1 - M02
                if abs(M12) <= m1 - 1:
                    u[index[(c0, m0, i0, M02)] * n_so + index[(c1, m1, i1, M12)]] += cg((m0 - 1) / 2, M02 / 2, (m1 - 1) / 2, M12 / 2, (tm - 1) / 2, (tm - 1) / 2)
            cols.append(u)
            basis.append(a0 * len(frag_states) + a1)
    return np.array(cols).T, basis

def S2_product_matrix(rho, S, terms, tm):
    """dimer overlap with the 2-fragment S diagrams terms (plus the identity), in the tensor-product ordering of get_xr_H"""
    import diagrammatic_expansion, XR_term
    import diagram_lists as D
    from diagrams import S_diagrams
    from precontract import precontract
    from get_xr_result_spin_adapt import local_sectors
    from qode.util import timer
    t = timer()
    blocks = diagrammatic_expansion.blocks(densities=rho, integrals=S, diagrams=S_diagrams, contract_cache=precontract(rho, S, t),
                                           timings=t, precon_timings=t, target_multiplicity=tm)
    frag_sectors = [[(c, m) for c, m, _ in local_sectors(r["n_states"], charges)] for r in rho]
    dimer_sectors = [(a, b) for a in frag_sectors[0] for b in frag_sectors[1]]
    S2 = XR_term.dimer_matrix(blocks, {0: D.S0[0], 2: terms}, (0, 1), dimer_sectors, t)
    dimer_index = {}
    for (c0, m0), (c1, m1) in dimer_sectors:
        for i0 in range(rho[0]["n_states"][c0][m0]):
            for i1 in range(rho[1]["n_states"][c1][m1]):
                dimer_index[(c0, m0, i0, c1, m1, i1)] = len(dimer_index)
    fs = [[(c, m, i) for c, m in frag_sectors[f] for i in range(rho[f]["n_states"][c][m])] for f in range(2)]
    order = [dimer_index[k0 + k1] for k0 in fs[0] for k1 in fs[1]]
    return S2[np.ix_(order, order)]

def test_S2_orders_spin_adapted_vs_spin_orbital():
    """the spin-adapted S2 diagrams of all orders (1-4) agree with their spin-orbital evaluation"""
    import diagram_lists as D
    s, so = system(), spin_orbital_system()
    rho = reference_densities(2)
    n = sum(len(v) for v in so.states.values())
    for name in sum((D.S2[k] for k in range(1, 5)), []):
        Sso = S2_product_matrix([so.rho, so.rho], so.symm.S, [name], None) - np.eye(n * n)
        for tm in (1, 3, 5):
            U, basis = spin_coupling(rho, so.states, tm)
            Ssa = S2_product_matrix([rho, rho], s.symm.S, [name], tm)[np.ix_(basis, basis)] - np.eye(len(basis))
            np.testing.assert_allclose(Ssa, U.T @ Sso @ U, atol=1e-12, err_msg=f"{name}, multiplicity {tm}")

def test_S2_orders_exact():
    """
    order by order in sigma = s - 1, the S2 diagrams reproduce the exact overlap of the antisymmetrized product states
    (determinants in the non-orthogonal fragment orbitals), obtained from a polynomial fit in s(lambda) = 1 + lambda sigma
    (H2/STO-3G, complete Fock spaces: orders 5-8 only appear in the sectors with more than 4 electrons)
    """
    import scipy.linalg
    import diagram_lists as D
    from ref_spin_densities import so_densities, SO_TYPES
    from qode.math.tensornet import raw
    s, so = system(), spin_orbital_system()
    rho, states = so_densities(s.ref_frag, charges, n_elec_ref, types=SO_TYPES + ["ccc", "aaa", "cccc", "aaaa"])
    sigma = np.zeros((8, 8))    # modes 4 f + 2 p + spin
    for p, q, spin in itertools.product(range(2), range(2), range(2)):
        sigma[2 * p + spin, 4 + 2 * q + spin] = sigma[4 + 2 * q + spin, 2 * p + spin] = raw(s.symm.S[0, 1])[p, q]
    a8 = [x.toarray() for x in jordan_wigner(8)]
    c8 = [x.T for x in a8]
    vecs = [v for chg in charges for *_, v in states[chg]]
    P = np.array([np.kron(v0, v1) for v0 in vecs for v1 in vecs]).T
    def exact(lam):
        # c^+(chi_i) = sum_j L_ji c^+(o_j) with orthonormal o, L^T L = s(lambda); JW basis vector n = ascending creators
        L = np.real(scipy.linalg.sqrtm(np.eye(8) + lam * sigma))
        c_chi = [sum(L[j, i] * c8[j] for j in range(8)) for i in range(8)]
        M = np.zeros((256, 256))
        for n in range(256):
            w = np.zeros(256); w[0] = 1.
            for i in reversed([i for i in range(8) if (n >> (7 - i)) & 1]):
                w = c_chi[i] @ w
            M[:, n] = w
        return (M @ P).T @ (M @ P)
    lams = np.linspace(0, 1, 9)
    coef = np.linalg.solve(np.vander(lams, 9, increasing=True), np.array([exact(l) for l in lams]).reshape(9, -1))
    lists = {1: D.S2[1], 2: D.S2[2], 3: D.S2[3] + ["s01s01s01"], 4: D.S2[4] + ["s01s01s01s01"]}
    for k, terms in lists.items():
        Sk = S2_product_matrix([rho, rho], so.symm.S, terms, None) - np.eye(len(P[0]))
        np.testing.assert_allclose(coef[k].reshape(Sk.shape), Sk, atol=1e-8, err_msg=f"order {k}")


def ccaa_vsym(x):
    """
    part of ccaa (layout p r s t = c_p c_r a_s a_t, pairs (p,s), (r,t)) that is symmetric under the symmetries
    of real symmetric V.  Only this part enters the (monomer) V contractions and hummr only returns this part.
    """
    perms = [(0,1,2,3), (1,0,3,2), (2,1,0,3), (0,3,2,1), (3,2,1,0), (2,3,0,1), (1,2,3,0), (3,0,1,2)]
    return sum(x.transpose((0, 1) + tuple(2 + i for i in p)) for p in perms) / len(perms)

def compare_densities(hummr, ref, skip=lambda name, ch: False):
    """
    The state bases differ (hummr uses RI integrals), but for complete sector sets the Gram matrices
    sum_IJ rho_IJ,x rho_IJ,y are basis invariant.  Returns the deviating channels.
    """
    from qode.math.tensornet import raw
    rng = np.random.default_rng(3)
    def G(x):
        X = x.reshape(x.shape[0] * x.shape[1], -1)
        if X.shape[1] <= 1000:
            return X.T @ X
        # large orbital spaces: the Gram matrix applied to a few random vectors (same invariance)
        V = np.random.default_rng(3).normal(size=(X.shape[1], 4))
        return X.T @ (X @ V)
    bad = []
    for name in ref:
        if not set(name) <= set("ca") or name == "":
            continue
        for cp, chans in ref[name].items():
            for ch, r in chans.items():
                if skip(name, ch):
                    continue
                h = hummr.get(name, {}).get(cp, {}).get(ch)
                if h is None:
                    bad.append((name, cp, ch, "missing"))
                    continue
                r, h = raw(r), raw(h)
                if np.linalg.norm(G(h) - G(r)) > 1e-8 * max(1, np.linalg.norm(G(r))):
                    bad.append((name, cp, ch, f"|rho_hummr| = {np.linalg.norm(h):.4f}, |rho_ref| = {np.linalg.norm(r):.4f}"))
        for cp, chans in hummr.get(name, {}).items():
            for ch in chans:
                if not skip(name, ch) and ch not in ref[name].get(cp, {}):
                    bad.append((name, cp, ch, "only in hummr"))
    return bad

@pytest.mark.parametrize("xr_level", [0, 1, 2])
def test_hummr_densities(xr_level):
    """hummr densities of H2/STO-3G (at most 2 open shells) agree with the reference ones (all types and paths)"""
    s = system()
    bad = compare_densities(s.hummr.build_densities("H2", xr_level, s.hummr_states, s.hummr_states), reference_densities(xr_level))
    assert not bad, "\n".join(str(b) for b in bad)


def unit_vectors(states):
    """single CSFs (unit vectors) of every sector"""
    return {chg: {mult: list(np.eye(len(vecs[0]))) for mult, vecs in mult_dict.items()} for chg, mult_dict in states.items()}

def compare_rectangular_densities(dens_builder, frag_name, states, n_elec_ref):
    """
    <CSF|...|state> densities against the square <state|...|state> ones transformed with the CSF coefficients
    (requires complete sectors, so that every CSF is a linear combination of the states). Returns deviations.
    """
    from get_xr_result_spin_adapt import build_frag_densities
    from qode.math.tensornet import raw
    square = build_frag_densities(dens_builder, frag_name, states, 0, n_elec_ref)
    rect = build_frag_densities(dens_builder, frag_name, states, 0, n_elec_ref, bra_states=unit_vectors(states))
    bad, n_checked = [], 0
    for name in ["a", "c", "aa", "cc", "ca", "caa", "cca", "ccaa"]:
        for cp, chans in square[name].items():
            for ch, sq in chans.items():
                C = np.array(states[cp[0]][ch[0]])          # states x CSFs of the bra sector
                expected = np.einsum("kp,kj...->pj...", C, raw(sq))
                r = rect.get(name, {}).get(cp, {}).get(ch)
                if r is None:
                    bad.append((name, cp, ch, "missing"))
                elif np.abs(raw(r) - expected).max() > 1e-10:
                    bad.append((name, cp, ch, f"max dev {np.abs(raw(r) - expected).max():.1e}"))
                n_checked += 1
    assert n_checked > 20
    return bad

def test_rectangular_densities():
    """densities between different bra (single CSFs) and ket (states) sets, as needed for the state gradients"""
    s = system()
    bad = compare_rectangular_densities(s.hummr, "H2", s.hummr_states, n_elec_ref)
    assert not bad, "\n".join(str(b) for b in bad)

def test_rectangular_xr0():
    """rectangular XR[0] Hamiltonian (single CSFs as bra on fragment 0) against the transformed square one"""
    from get_xr_result_spin_adapt import build_frag_densities, get_xr_H, local_sectors
    s = system()
    states = s.hummr_states
    square = [build_frag_densities(s.hummr, "H2", states, 0, n_elec_ref) for _ in range(2)]
    rect = [build_frag_densities(s.hummr, "H2", states, 0, n_elec_ref, bra_states=unit_vectors(states)), square[1]]
    # U[p, K] = <CSF p|state K>, block diagonal over the sectors in the local ordering
    secs = local_sectors(square[0]["n_states"], charges)
    U = np.zeros((sum(len(states[c][m][0]) for c, m, _ in secs), sum(sl.stop - sl.start for _, _, sl in secs)))
    row = 0
    for chg, mult, sl in secs:
        C = np.array(states[chg][mult])
        U[row:row + C.shape[1], sl] = C.T
        row += C.shape[1]
    for tm in [1, 3]:
        for bior in [False, True]:
            from qode.util import struct
            dimer_ints = s.bior if bior else struct(T=s.symm.T, U=s.symm.U, V=s.symm.V)
            H1, H2 = get_xr_H((s.symm, dimer_ints, s.nuc), square, 0, [charges, charges], tm)
            H1r, H2r = get_xr_H((s.symm, dimer_ints, s.nuc), rect, 0, [charges, charges], tm)
            np.testing.assert_allclose(H1r[0], U @ H1[0], atol=1e-10)
            np.testing.assert_allclose(H1r[1], H1[1], atol=1e-10)
            np.testing.assert_allclose(H2r, np.kron(U, np.eye(H1[1].shape[0])) @ H2, atol=1e-10)


@pytest.mark.parametrize("bior,grad_level", [(False, "herm"), (False, "full"), (True, "full")])
def test_state_gradients(bior, grad_level):
    """
    state gradients (StateSpaceOptimizer/state_gradients_csf.py) against finite differences of the XR[0] energy for
    incomplete H2 state spaces: rotating state K into its normalized gradient changes the energy by |g_K| ("full",
    exact for the non-hermitian Hamiltonian with biorthogonal integrals) or 2|g_K| ("herm", hermitian H only)
    """
    import copy, contextlib, io
    import excitonic
    from get_xr_result_spin_adapt import build_frag_densities, get_xr_H, product_basis_mask
    from state_gradients_csf import state_gradients
    from qode.util import struct, timer
    s = system()
    ints = (s.symm, s.bior if bior else struct(T=s.symm.T, U=s.symm.U, V=s.symm.V), s.nuc)
    mc = [charges] * 2
    base = {c: {m: [np.array(v) for v in vs[:(2 if (c, m) == (0, 1) else 1)]] for m, vs in md.items()} for c, md in s.hummr_states.items()}
    def build_dens(frag, st, bra_states=None):
        return build_frag_densities(s.hummr, "H2", st, 0, n_elec_ref, bra_states=bra_states)
    def solve(states):
        with contextlib.redirect_stdout(io.StringIO()):
            dens = [build_dens(f, states[f]) for f in range(2)]
            H1, H2 = get_xr_H(ints, dens, 0, mc, 1)
            E, vl, vr = excitonic.fci((H1, [[None, H2], [None, None]]), struct(log=lambda *a: None), timer(), target_state=0,
                                      get_left_and_right=True, basis_mask=product_basis_mask(dens, mc, 0, 1))
        real = lambda v: np.real(v * np.conj(v[np.argmax(np.abs(v))]) / np.abs(v[np.argmax(np.abs(v))]))
        shape = (H1[0].shape[0], H1[1].shape[0])
        vr, vl = real(vr), real(vl)
        return np.real(E), (vr / np.linalg.norm(vr)).reshape(shape), vl.reshape(shape), dens, H1
    states = [copy.deepcopy(base), copy.deepcopy(base)]
    E0, d, dl, dens, H1 = solve(states)
    factor = 1 if grad_level == "full" else 2
    n_checked = 0
    for frag in [0, 1]:
        with contextlib.redirect_stdout(io.StringIO()):
            g = state_gradients(frag, states, dens, E0, d, H1, ints, build_dens, mc, 1, grad_level=grad_level, d_left=dl)
        for (chg, mult) in [(0, 1), (1, 2), (-1, 2)]:
            C = np.array(states[frag][chg][mult])
            for K, gK in enumerate(g[chg][mult]):
                n = np.linalg.norm(gK)
                assert n > 1e-6 and np.abs(C @ gK).max() < 1e-10 * max(1, n)
                eps, E = 1e-4, []
                for sgn in (+1, -1):
                    st = copy.deepcopy(states)
                    v = C[K] + sgn * eps * gK / n
                    st[frag][chg][mult][K] = v / np.linalg.norm(v)
                    E.append(solve(st)[0])
                assert abs((E[0] - E[1]) / (2 * eps) / (factor * n) - 1) < 1e-5, (frag, chg, mult, K)
                n_checked += 1
    assert n_checked == 8


def be_system(frozen_core=False, xr_level=0):
    """
    Be/STO-3G with complete (charge 0, +1, -1) sectors: up to 5 open shells.
    frozen_core: 1s frozen, i.e. hummr computes (2,4) active spaces and embeds the densities into the full space
    """
    from math import comb
    from dens_provider import DensBackend
    from get_ints import get_ints
    from get_xr_result_spin_adapt import build_frag_densities
    from qode.util import timer
    from qode.math.tensornet import raw
    n_orb, n_ref, be_charges = 5, 4, [0, 1, -1]
    n_core = 1 if frozen_core else 0
    def n_csf(N, S2):
        n = n_orb - n_core
        k = (N - S2) // 2
        return (S2 + 1) * comb(n + 1, k) * comb(n + 1, N - k + 1) // (n + 1)
    roots = {chg: {S2 + 1: n_csf(n_ref - 2 * n_core - chg, S2) for S2 in range((n_ref - chg) % 2, n_ref - 2 * n_core - chg + 1, 2)}
             for chg in be_charges}
    hummr = DensBackend("hummr")
    inp = "be_sto3g_fcore.inp" if frozen_core else "be_sto3g.inp"
    mo, states = hummr.init_backend([["Be", os.path.join(here, inp), roots, xr_level]])[0]
    assert hummr.frozen_core("Be") == list(range(n_core))
    frags = []
    for m in range(2):
        f = empty()
        f.atoms = [["Be", [0, 0, 10.0 * m]]]
        f.n_elec_ref = n_ref
        f.basis = empty()
        f.basis.n_spatial_orb, f.basis.AOcode, f.basis.core, f.basis.MOcoeffs = n_orb, "STO-3G", [], np.array(mo)
        frags.append(f)
    symm, bior, nuc = get_ints(frags, True, timer(), spin_ints=False, backend="lible")
    frag = Fragment(raw(symm.T[0, 0]) + raw(symm.U[0, 0, 0]), raw(symm.V[0, 0, 0, 0]), n_orb, n_ref, frozen_core=n_core)
    backend = ReferenceBackend(frag, be_charges)
    ref = build_frag_densities(backend, "Be", backend.placeholder_states(), xr_level, n_ref)
    return (build_frag_densities(hummr, "Be", states, xr_level, n_ref), ref,
            compare_rectangular_densities(hummr, "Be", states, n_ref) if xr_level == 0 else [])

rank1_ca_transition = lambda name, ch: name == "ca" and ch[2] == 2 and ch[0] != ch[1]

@functools.lru_cache(maxsize=None)
def be_comparison(frozen_core=False):
    """
    hummr cannot set up a second fragment in the same process (it keeps the geometry of the first one),
    so the Be comparison runs in a subprocess: {"other": [...], "rank1_ca_transitions": [...]}
    """
    import json, subprocess
    args = ["be_comparison"] + (["frozen_core"] if frozen_core else [])
    out = subprocess.run([sys.executable, os.path.abspath(__file__)] + args, capture_output=True, text=True,
                         env=dict(os.environ, PYTHONPATH=os.pathsep.join([here] + sys.path)))
    lines = [l for l in out.stdout.splitlines() if l.startswith("BE_COMPARISON ")]
    assert lines, out.stdout[-3000:] + out.stderr[-3000:]
    return json.loads(lines[-1][len("BE_COMPARISON "):])

def test_hummr_densities_be():
    """the same for Be/STO-3G (up to 5 open shells), except for the rank-1 ca transitions (see below)"""
    bad = be_comparison()["other"]
    assert not bad, "\n".join(str(b) for b in bad)

def test_hummr_rank1_ca_transitions_be():
    """rank-1 ca between different multiplicities, which only showed errors with more than 2 open shells"""
    bad = be_comparison()["rank1_ca_transitions"]
    assert not bad, "\n".join(str(b) for b in bad)

def test_rectangular_densities_be():
    """<CSF|...|state> densities for Be/STO-3G, all-electron and with frozen core (overlaps in the embedding)"""
    bad = be_comparison()["rectangular"] + be_comparison(frozen_core=True)["rectangular"]
    assert not bad, "\n".join(str(b) for b in bad)

def test_hummr_densities_be_frozen_core():
    """hummr's (2,4) active-space densities with frozen 1s, embedded into the full space, against the exact ones"""
    result = be_comparison(frozen_core=True)
    bad = result["other"] + result["rank1_ca_transitions"]
    assert not bad, "\n".join(str(b) for b in bad)

def test_frozen_core_embedding():
    """
    Embedding of active-space densities (as returned by hummr for a frozen core) into the full space against
    the exact full-space densities of the same states (core doubly occupied), element by element for every
    density type up to XR[2] and path (including the reversed ones), charge and multiplicity.  Random integrals
    suffice for this.
    """
    from get_xr_result_spin_adapt import build_frag_densities
    from qode.math.tensornet import raw
    rng = np.random.default_rng(7)
    n, n_core, n_ref = 4, 1, 4
    h = rng.normal(size=(n, n)); h = h + h.T
    W = rng.normal(size=(n, n, n, n)) / 10
    W = W + W.transpose(1, 0, 3, 2); W = W + W.transpose(2, 1, 0, 3); W = W + W.transpose(0, 3, 2, 1)   # real V symmetries
    frag = Fragment(h, W, n, n_ref, frozen_core=n_core)
    fc_charges = [0, 1, -1, 2, -2]
    full = ReferenceBackend(frag, fc_charges)
    embedded = ReferenceBackend(frag, fc_charges, active_only=True)
    rho_full = build_frag_densities(full, "X", full.placeholder_states(), 2, n_ref)
    rho_emb = build_frag_densities(embedded, "X", embedded.placeholder_states(), 2, n_ref)
    n_checked = 0
    for name in [x for x in rho_full if set(x) <= set("ca") and x]:
        assert rho_emb[name].keys() == rho_full[name].keys(), name
        for cp, chans in rho_full[name].items():
            assert rho_emb[name][cp].keys() == chans.keys(), (name, cp)
            for ch, ref in chans.items():
                np.testing.assert_allclose(raw(rho_emb[name][cp][ch]), raw(ref), atol=1e-12, err_msg=f"{name} {cp} {ch}")
                n_checked += 1
    assert n_checked > 200

def h2_631g_system():
    """H2/6-31G (4 orbitals, up to 4 open shells) with complete sectors for charges 0, +-1, +-2, densities up to XR[2]"""
    from math import comb
    from dens_provider import DensBackend
    from get_ints import get_ints
    from get_xr_result_spin_adapt import build_frag_densities
    from qode.util import timer
    from qode.math.tensornet import raw
    n_orb, n_ref, chgs = 4, 2, [0, 1, -1, 2, -2]
    def n_csf(N, S2):
        k = (N - S2) // 2
        return (S2 + 1) * comb(n_orb + 1, k) * comb(n_orb + 1, N - k + 1) // (n_orb + 1)
    roots = {c: {S2 + 1: n_csf(n_ref - c, S2) for S2 in range((n_ref - c) % 2, min(n_ref - c, 2 * n_orb - n_ref + c) + 1, 2)} for c in chgs}
    hummr = DensBackend("hummr")
    mo, states = hummr.init_backend([["H2", os.path.join(here, "h2_631g.inp"), roots, 2]])[0]
    frags = []
    for m in range(2):
        f = empty()
        f.atoms = [["H", [0, 0, 10.0 * m]], ["H", [0, 0, 10.0 * m + 0.74]]]
        f.n_elec_ref = n_ref
        f.basis = empty()
        f.basis.n_spatial_orb, f.basis.AOcode, f.basis.core, f.basis.MOcoeffs = n_orb, "6-31G", [], np.array(mo)
        frags.append(f)
    symm, bior, nuc = get_ints(frags, True, timer(), spin_ints=False, backend="lible")
    frag = Fragment(raw(symm.T[0, 0]) + raw(symm.U[0, 0, 0]), raw(symm.V[0, 0, 0, 0]), n_orb, n_ref)
    backend = ReferenceBackend(frag, chgs)
    return build_frag_densities(hummr, "H2", states, 2, n_ref), build_frag_densities(backend, "H2", backend.placeholder_states(), 2, n_ref)

@functools.lru_cache(maxsize=None)
def subprocess_comparison(system):
    """runs compare_densities for the given system in a subprocess (hummr cannot set up a second fragment)"""
    import json, subprocess
    out = subprocess.run([sys.executable, os.path.abspath(__file__), "comparison", system], capture_output=True, text=True,
                         env=dict(os.environ, PYTHONPATH=os.pathsep.join([here] + sys.path)))
    lines = [l for l in out.stdout.splitlines() if l.startswith("COMPARISON ")]
    assert lines, out.stdout[-3000:] + out.stderr[-3000:]
    return json.loads(lines[-1][len("COMPARISON "):])

@pytest.mark.parametrize("system", ["h2_631g_xr2", "be_frozen_core_xr2"])
def test_hummr_densities_xr2(system):
    """hummr densities up to XR[2] (all types and paths, incl. the reversed ones and the frozen-core embedding)"""
    result = subprocess_comparison(system)
    assert result["n_channels"] > 200
    assert not result["bad"], "\n".join(result["bad"])

if __name__ == "__main__" and sys.argv[1:2] == ["comparison"]:
    import json
    if sys.argv[2] == "h2_631g_xr2":
        hd, ref = h2_631g_system()
    else:
        hd, ref, _ = be_system(frozen_core=True, xr_level=2)
    n = sum(len(ch) for name, by in ref.items() if set(name) <= set("ca") and name for ch in by.values())
    print("COMPARISON " + json.dumps({"n_channels": n, "bad": [str(b) for b in compare_densities(hd, ref)]}))

if __name__ == "__main__" and sys.argv[1:2] == ["be_comparison"]:
    import json
    *dens, rect = be_system(frozen_core=sys.argv[2:] == ["frozen_core"])
    print("BE_COMPARISON " + json.dumps({
        "rectangular":          [str(b) for b in rect],
        "other":                [str(b) for b in compare_densities(*dens, skip=rank1_ca_transition)],
        "rank1_ca_transitions": [str(b) for b in compare_densities(*dens, skip=lambda name, ch: not rank1_ca_transition(name, ch))]}))


def test_xr2_exact_for_two_electrons():
    """
    H2/6-31G as two H-atom fragments with truncated state spaces (charges 0, +1, -1): with two electrons in total the
    S expansion terminates at order 2, so S^[<=2] and (S H)^[<=2] are the exact overlap and Hamiltonian of the
    antisymmetrized product states, and XR[2] equals the exact projection (generalized eigenproblem H~ c = E S~ c in
    the product space), while XR[0] and XR[1] do not.  This checks the integral types (symmetric, biorthogonal,
    2 V_half - V) of every term up to XR[2] against the working equations (Bauer, Dreuw, Dutoi, arXiv:2409.07628).
    """
    import scipy.linalg
    import get_xr_result_spin_adapt as G
    from get_ints import get_ints
    from qode.util import timer
    from qode.math.tensornet import raw
    n_orb, d, chgs = 2, 1.5, [0, 1, -1]
    def frags(dist, mo):
        out = []
        for m in range(2):
            f = empty(); f.atoms = [["H", [0, 0, m * dist]]]; f.n_elec_ref = 1
            f.basis = empty(); f.basis.n_spatial_orb, f.basis.AOcode, f.basis.core, f.basis.MOcoeffs = n_orb, "6-31G", [], mo
            out.append(f)
        return out
    ao, _, _ = get_ints(frags(30.0, np.eye(n_orb)), True, timer(), spin_ints=False, backend="lible")
    w, U = np.linalg.eigh(raw(ao.S[0, 0]))
    mo = U @ np.diag(w ** -0.5) @ U.T
    far, _, _ = get_ints(frags(30.0, mo), True, timer(), spin_ints=False, backend="lible")
    frag = Fragment(raw(far.T[0, 0]) + raw(far.U[0, 0, 0]), raw(far.V[0, 0, 0, 0]), n_orb, 1)
    keep = {0: {2: 1}, 1: {1: 1}, -1: {1: 2, 3: 1}}
    frag.states = {(c, m): frag.states[(c, m)][:, :k] for c in chgs for m, k in keep[c].items()}
    backend = ReferenceBackend(frag, chgs)
    rho = G.build_frag_densities(backend, "H", backend.placeholder_states(), 2, 1)
    # exact: singlet-coupled product states of total charge 0, built with c^+(chi_i) = sum_j L_ji c^+(o_j) (L^2 = s)
    fs = [(c, m, i) for c, m, sl in G.local_sectors(rho["n_states"], chgs) for i in range(sl.stop - sl.start)]
    V = np.array([sum(cg((m0 - 1) / 2, M0, (m1 - 1) / 2, -M0, 0, 0) * np.kron(v0, frag.m_states(c1, m1, i1)[round(-M0, 1)])
                      for M0, v0 in frag.m_states(c0, m0, i0).items() if round(-M0, 1) in frag.m_states(c1, m1, i1))
                  for (c0, m0, i0), (c1, m1, i1) in itertools.product(fs, fs)
                  if c0 + c1 == 0 and G.multiplicities_couple(m0, m1, 1)]).T
    ints = get_ints(frags(d, mo), True, timer(), spin_ints=False, backend="lible")
    symm = ints[0]
    blk = lambda i: slice(n_orb * i, n_orb * (i + 1))
    s, h, W = np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4,) * 4)
    for i, j in itertools.product(range(2), repeat=2):
        s[blk(i), blk(j)] = raw(symm.S[i, j])
        h[blk(i), blk(j)] = raw(symm.T[i, j]) + sum(raw(symm.U[k, i, j]) for k in range(2))
    for key in itertools.product(range(2), repeat=4):
        W[tuple(blk(i) for i in key)] = raw(symm.V[key])
    L = np.real(scipy.linalg.sqrtm(s))
    X = np.linalg.inv(L)
    a8 = [x.toarray() for x in jordan_wigner(8)]
    c8 = [x.T for x in a8]
    Ho = second_quantized_H(X @ h @ X, np.einsum("pqrs,pa,qb,rc,sd->abcd", W, X, X, X, X), 4, a8, c8)
    sm = np.kron(L, np.eye(2))
    c_chi = [sum(sm[j, i] * c8[j] for j in range(8)) for i in range(8)]
    M = np.zeros((256, 256))
    for n in range(256):
        v = np.zeros(256); v[0] = 1.
        for i in reversed([i for i in range(8) if (n >> (7 - i)) & 1]):
            v = c_chi[i] @ v
        M[:, n] = v
    MV = M @ V
    E_exact = np.min(np.real(scipy.linalg.eigvals(MV.T @ Ho @ MV, MV.T @ MV)))
    E = {o: np.real(G.get_xr_states(ints, [rho, rho], o, [chgs, chgs], 1, target_charge=0)[0]) for o in (0, 1, 2)}
    assert abs(E[2] - E_exact) < 1e-10, (E[2], E_exact)
    assert abs(E[0] - E_exact) > 1e-6 and abs(E[1] - E_exact) > 1e-6, (E, E_exact)


# ----------------------------------------------------------------------------------------------------------------------
# frozen core folded into the one-electron operator (get_ints fold_core), RI integrals (integrals="ri"), basis check
# ----------------------------------------------------------------------------------------------------------------------

def _be2_frags(mo, dist, n_core=1, basis="STO-3G"):
    frags = []
    for m in range(2):
        f = empty()
        f.atoms = [["Be", [0, 0, m * dist]]]
        f.n_elec_ref = 4
        f.basis = empty()
        f.basis.n_spatial_orb, f.basis.AOcode, f.basis.core, f.basis.MOcoeffs = mo.shape[1], basis, list(range(n_core)), mo
        frags.append(f)
    return frags

def _lowdin_mos(basis="STO-3G"):
    """orthonormal orbitals of a Be atom (Loewdin orthonormalized AOs, the first one 1s-like)"""
    sys.path.insert(0, os.path.join(os.path.dirname(__import__("qode").__file__), "atoms", "integrals", "external_engines"))
    import pylible_ints as pli
    S, _ = pli.overlap_and_kinetic([("Be", 0.0, 0.0, 0.0, False)], basis)
    w, v = np.linalg.eigh(S)
    return (v / np.sqrt(w)) @ v.T

def _supermatrices(S, T, U, V, n, nuc):
    """raw supersystem S, h = T + sum U, <pq|rs> = 2 V and the total nuclear (+ core) energy of fragment-blocked integrals"""
    from qode.math.tensornet import raw
    r = lambda x: np.asarray(raw(x)) if hasattr(x, "_raw") else np.asarray(x)
    blk = lambda f: np.block([[f(a, b) for b in range(2)] for a in range(2)])
    s = blk(lambda a, b: r(S[a, b]))
    h = blk(lambda a, b: r(T[a, b]) + sum(r(U[m, a, b]) for m in range(2)))
    G = np.zeros((2 * n,) * 4)
    for k in itertools.product(range(2), repeat=4):
        G[tuple(slice(i * n, (i + 1) * n) for i in k)] = 2 * r(V[k])
    return s, h, G, sum(nuc[m1, m2] for m1 in range(2) for m2 in range(m1 + 1))

def _closed_shell_energy(s, h, G, occ, frozen=()):
    """energy of the determinant of the doubly occupied orbitals frozen + occ, the others Loewdin orthonormalized"""
    n = s.shape[0]
    rest = [i for i in range(n) if i not in frozen]
    w, v = np.linalg.eigh(s[np.ix_(rest, rest)])
    X = np.eye(n)
    X[np.ix_(rest, rest)] = (v / np.sqrt(w)) @ v.T
    h, G = X.T @ h @ X, np.einsum("pqrs,pa,qb,rc,sd->abcd", G, X, X, X, X, optimize=True)
    o = list(frozen) + list(occ)
    return sum(2 * h[i, i] for i in o) + sum(2 * G[i, j, i, j] - G[i, j, j, i] for i in o for j in o)

def test_fold_core_determinant_energies():
    """
    Be2/STO-3G at 2.0 A (1s frozen): for closed-shell determinants with the (jointly orthonormalized) cores doubly
    occupied, the energy from the integrals of all orbitals equals the one from the folded integrals of the active
    orbitals (core potentials in U[m], core energies in the nuclear repulsion matrix), get_ints.fold_frozen_core
    """
    from get_ints import direct_CoreProj, fold_frozen_core
    from qode.atoms.integrals.fragments import AO_integrals, fragMO_integrals, bra_transformed, ket_transformed, Nuc_repulsion
    mo = _lowdin_mos()
    frags = _be2_frags(mo, 2.0)
    n = mo.shape[1]
    ints = fragMO_integrals(AO_integrals(frags, backend="lible"), [f.basis.MOcoeffs for f in frags], cache=True)
    Left, Right = direct_CoreProj(frags, ints.S, True, orthonormalize_core=True)
    ints = ket_transformed(Right, bra_transformed(Left, ints, cache=True), cache=True)
    nuc = Nuc_repulsion(frags).matrix
    full = _supermatrices(ints.S, ints.T, ints.U, ints.V, n, nuc)
    cores = [0, n]
    assert np.abs(full[0][np.ix_(cores, cores)] - np.eye(2)).max() < 1e-12
    _, folded, nuc_f = fold_frozen_core(frags, ints, nuc)
    fold = _supermatrices(folded.S, folded.T, folded.U, folded.V, n - 1, nuc_f)
    for a, b in [(0, 0), (0, 1), (2, 3), (1, 1)]:
        E_full = _closed_shell_energy(*full[:3], [1 + a, n + 1 + b], frozen=cores) + full[3]
        E_fold = _closed_shell_energy(*fold[:3], [a, n - 1 + b]) + fold[3]
        assert abs(E_full - E_fold) < 1e-10, ((a, b), E_full, E_fold)

def test_ri_integrals():
    """
    RI (lible RIEngine, Qode pybind_lible_ints.RI_factors) through get_ints integrals="ri" (with fold_core): the
    one-electron integrals are the exact ones, the two-electron integrals agree to the fitting error, which shrinks
    with a larger auxiliary basis
    """
    from get_ints import get_ints
    from qode.util import timer
    from qode.math.tensornet import raw
    mo = _lowdin_mos()
    exact = get_ints(_be2_frags(mo, 2.0), True, timer(), spin_ints=False, backend="lible", fold_core=True)
    errors = []
    for aux in ("def2-universal-jkfit", "cc-pv5z-rifit"):
        ri = get_ints(_be2_frags(mo, 2.0), True, timer(), spin_ints=False, backend="lible", fold_core=True,
                      integrals="ri", aux_basis=aux)
        for k in itertools.product(range(2), repeat=2):
            assert np.abs(np.asarray(raw(ri[0].S[k])) - np.asarray(raw(exact[0].S[k]))).max() < 1e-12
            assert np.abs(np.asarray(raw(ri[0].T[k])) - np.asarray(raw(exact[0].T[k]))).max() < 1e-12
        errors.append(max(np.abs(np.asarray(raw(ri[0].V[k])) - np.asarray(raw(exact[0].V[k]))).max()
                          for k in itertools.product(range(2), repeat=4)))
    assert errors[0] < 2e-3 and errors[1] < errors[0] / 3, errors

def test_basis_check():
    """check_basis_consistency: the XR basis (and auxiliary basis for RI) has to be the one of the hummr input"""
    from hummr_dens.pybind_interface import check_basis_consistency, hummr_basis_info
    inp = os.path.join(here, "be_sto3g_fcore.inp")    # Basis STO-3G, Ints RI, AuxBasis def2-JK
    info = hummr_basis_info(inp)
    assert (info["basis"], info["aux_basis"], info["ints"]) == ("sto-3g", "def2-universal-jkfit", "RI"), info
    mo = np.eye(5)
    check_basis_consistency(_be2_frags(mo, 2.0), [inp, inp], integrals="ri", aux_basis="def2-JK", printout=lambda *a: None)
    check_basis_consistency(_be2_frags(mo, 2.0), [inp, inp], printout=lambda *a: None)    # exact in XR: reported only
    with pytest.raises(ValueError, match="auxiliary basis"):
        check_basis_consistency(_be2_frags(mo, 2.0), [inp, inp], integrals="ri", aux_basis="cc-pvdz-rifit", printout=lambda *a: None)
    frags = _be2_frags(mo, 2.0)
    frags[1].basis.AOcode = "6-31G"
    with pytest.raises(ValueError, match="fragment 1: XR basis"):
        check_basis_consistency(frags, [inp, inp], printout=lambda *a: None)

def test_fold_core_xr():
    """
    Be2/STO-3G at 2.5 A, 1s frozen, complete sectors (charges 0, +1, -1), the same hummr states: XR with the core
    embedded into the densities vs XR with the core folded into the integrals (active-space densities).  XR[0] is
    invariant (the biorthogonal core terms are exact at order 0), XR[1] and XR[2] differ only by the core-core overlap
    terms the embedded S expansion truncates (decreasing with the order)
    """
    result = subprocess_fold_core()
    E_emb, E_fold = result["embedded"], result["folded"]
    assert abs(E_emb["0"] - E_fold["0"]) < 1e-9, result
    assert abs(E_emb["2"] - E_fold["2"]) < 1e-5, result
    assert abs(E_emb["2"] - E_fold["2"]) <= abs(E_emb["1"] - E_fold["1"]) + 1e-9, result

@functools.lru_cache(maxsize=None)
def subprocess_fold_core():
    import json, subprocess
    out = subprocess.run([sys.executable, os.path.abspath(__file__), "fold_core"], capture_output=True, text=True,
                         env=dict(os.environ, PYTHONPATH=os.pathsep.join([here] + sys.path)))
    lines = [l for l in out.stdout.splitlines() if l.startswith("FOLD_CORE ")]
    assert lines, out.stdout[-3000:] + out.stderr[-3000:]
    return json.loads(lines[-1][len("FOLD_CORE "):])

if __name__ == "__main__" and sys.argv[1:2] == ["fold_core"]:
    import json
    from math import comb
    from dens_provider import DensBackend
    from get_ints import get_ints
    import get_xr_result_spin_adapt as G
    from qode.util import timer
    n_orb, n_core, n_ref, chgs = 5, 1, 4, [0, 1, -1]
    def n_csf(N, S2):
        n = n_orb - n_core
        k = (N - S2) // 2
        return (S2 + 1) * comb(n + 1, k) * comb(n + 1, N - k + 1) // (n + 1)
    roots = {c: {S2 + 1: n_csf(n_ref - 2 * n_core - c, S2) for S2 in range((n_ref - c) % 2, n_ref - 2 * n_core - c + 1, 2)}
             for c in chgs}
    hummr = DensBackend("hummr")
    mo, states = hummr.init_backend([["Be", os.path.join(here, "be_sto3g_fcore.inp"), roots, 2]])[0]
    frag = hummr.engine.fragments["Be"]
    out = {}
    for label, fold in (("embedded", False), ("folded", True)):
        frag.fold_core = fold
        ints = get_ints(_be2_frags(np.array(mo), 2.5), True, timer(), spin_ints=False, backend="lible", fold_core=fold)
        rho = G.build_frag_densities(hummr, "Be", states, 2, n_ref)
        E_nuc = sum(ints[2][m1, m2] for m1 in range(2) for m2 in range(m1 + 1))
        out[label] = {str(o): float(np.real(G.get_xr_states(ints, [rho, rho], o, [chgs, chgs], 1, target_charge=0)[0]) + E_nuc)
                      for o in (0, 1, 2)}
    print("FOLD_CORE " + json.dumps(out))
