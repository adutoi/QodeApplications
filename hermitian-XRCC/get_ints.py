#    (C) Copyright 2018, 2019 Anthony D. Dutoi
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
from qode.util               import struct
from qode.util.PyC           import Double
from qode.util.dynamic_array import dynamic_array, wrap, cached
from qode.math               import precise_numpy_inverse, linear_inner_product_space, iterative_biorthog, biorthog_iteration, svd_decomposition
from qode.math.tensornet     import tl_tensor
from qode.atoms.integrals.fragments import AO_integrals, fragMO_integrals, bra_transformed, ket_transformed, spin_orb_integrals, Nuc_repulsion, as_raw_mat, as_frag_blocked_mat, zeros2, Id, mat_as_rows, mat_as_columns, space_traits, add, subtract, mat_mul



def tens_wrap(tensor):
    return tl_tensor.init(tensorly.tensor(tensor, dtype=Double.tensorly))

def tensorly_wrapper(timings):
    def wrapper(rule):
        def wrap_it(*indices):
            timings.start()
            Z = svd_decomposition(rule(*indices), [0,1], wrapper=tens_wrap)
            timings.record("1e integrals SVD")
            return Z
        return wrap_it
    return wrapper

def tensorly_wrapper2(timings):
    def wrapper(rule):
        def wrap_it(*indices):
            print(indices)
            free_indices = [[],[]]
            for i,m in enumerate(indices):
                free_indices[m] += [i]
            timings.start()
            if False and len(free_indices[0])>0 and len(free_indices[1])>0:
                Z = svd_decomposition(rule(*indices), free_indices[0], free_indices[1], wrapper=tens_wrap)
            else:
                Z = svd_decomposition(rule(*indices), [0,1,2,3], wrapper=tens_wrap)
            timings.record("2e integrals SVD")
            return Z
        return wrap_it
    return wrapper

def electron_swap_symmetrized(V):
    """
    1/2 (V_pqrs + V_qpsr): the spin-summed operator sum V_pqrs c_ps c_qt a_st a_rs only depends on this part of V,
    and the diagrams were derived using this symmetry, which the half-transformed V_half does not have
    """
    def rule(m1, m2, m3, m4):
        return 0.5 * (V[m1, m2, m3, m4] + numpy.transpose(V[m2, m1, m4, m3], (1, 0, 3, 2)))
    return rule

def tens_diff(A, B, factor_A=1):
    def rule(*indices):
        return factor_A * A[indices] - B[indices]
    return rule

# V_diff, which enters the S expansion of XR[1] and XR[2] (get_xr_result.py), is the first-order (in S) expansion of the
# fully biorthogonalized V, i.e. one correction per electron:  2 V_half - V,  with V_half transformed for one electron.
# With the antisymmetrized spin-orbital integrals of spin_orb_integrals this is obtained as V_half - V, because the
# antisymmetrizer of V_half (_spinAnti4half_rule, over both index pairs) carries twice the weight of the one of V
# (_spinAnti4_rule) for integrals with the electron-swap symmetry.  For plain (spatial or not antisymmetrized
# spin-orbital) integrals the factor 2 has to be included explicitly.
V_HALF_FACTOR_PLAIN = 2

# TODO: move the antisymmetrization somewhere else
def _spinAnti4_rule(V):
    def spinAnti4(m1,m2,m3,m4):
        tmp1 = V[m1,m2,m3,m4]
        tmp2 = V[m1,m2,m4,m3]
        return (1/4.) * (tmp1 - numpy.swapaxes(tmp2,2,3))
    return spinAnti4

def _spinAnti4half_rule(V):
    def spinAnti4half(m1,m2,m3,m4):
        tmp1 = V[m1,m2,m3,m4]
        tmp2 = V[m1,m2,m4,m3]
        tmp3 = V[m2,m1,m3,m4]
        tmp4 = V[m2,m1,m4,m3]
        return (1/4.) * ((tmp1-numpy.swapaxes(tmp2,2,3)) - numpy.swapaxes(tmp3-numpy.swapaxes(tmp4,2,3),0,1))
    return spinAnti4half

def pair_factors(S, thresh):
    """
    {(i, j): (L, R)} for the inter-fragment blocks of S: the corresponding-orbital pairs (singular value decomposition
    of S_ij) with a singular value above thresh, S_ij ~ L R^T with L = U_K sigma_K (n_i x K) and R = V_K (n_j x K);
    the factors of S_ji are the swapped ones of S_ij (i < j).  Without any pair above thresh, one pair with a zero
    singular value is kept (the block vanishes).
    """
    def rule(i, j):
        if i > j:
            L, R = factors[j, i]
            return R, L
        U, s, Vt = numpy.linalg.svd(numpy.array(S[i, j]), full_matrices=False)
        keep = s > thresh
        if not keep.any():
            return numpy.zeros((U.shape[0], 1)), Vt[:1].T
        return U[:, keep] * s[keep], Vt[keep].T
    factors = dynamic_array([cached, rule], S.ranges)
    return factors

def pair_compressed(S, factors):
    """
    S with every inter-fragment block S_ij replaced by L R^T of its pair factors (pair_factors); the fragment-diagonal
    blocks are kept.  As long as the compressed S is used consistently, i.e. also for the biorthogonalization (S^-1,
    see get_ints), the XR terms are evaluated exactly for an overlap that only couples the leading pairs (Be2 cc-pVDZ:
    with a compressed S only in the explicit S factors of the diagrams, the order-2 increment of the leakage is off by
    1e2 to 1e3 uEh, see get_xr_result_spin_adapt.get_leakage_metric_H)
    """
    def rule(i, j):
        if i == j:
            return numpy.array(S[i, j])
        L, R = factors[i, j]
        return L @ R.T
    return dynamic_array([cached, rule], S.ranges)

def factorized_wrapper(factors, timings):
    """
    as tensorly_wrapper for the compressed S (pair_compressed), but the inter-fragment blocks are given as the rank-K
    products of their pair factors (as qode.math.svd_decomposition does), so that the contractions of the diagrams
    with explicit S factors run over the K pairs instead of the orbitals (ordered by the tensornet heuristic)
    """
    def wrapper(rule):
        def wrap_it(i, j):
            timings.start()
            if i == j:
                Z = tens_wrap(rule(i, j))
            else:
                L, R = factors[i, j]
                Z = tens_wrap(numpy.ascontiguousarray(L.T))("k", 0) @ tens_wrap(numpy.ascontiguousarray(R.T))("k", 1)
            timings.record("1e integrals SVD")
            return Z
        return wrap_it
    return wrapper

def direct_Sinv(fragments, S):
    S = as_raw_mat(S, fragments)
    Sinv = precise_numpy_inverse(S)
    Sinv = as_frag_blocked_mat(Sinv, fragments)
    return Sinv

def direct_CoreProj(fragments, S, orthonormalize_valence=True, orthonormalize_core=False, return_raw=False):
    # orthonormalize_core: the core orbitals of all fragments are in addition Loewdin orthonormalized jointly,
    # c' = sum_c c (S_cc^-1/2), the prerequisite of folding the core into the one-electron operator (fold_frozen_core);
    # the projected valence orbitals are the same (the cores span the same space), and core orbital c' stays in the
    # block of its parent fragment (Loewdin: the closest orthonormal orbitals to the original ones)
    # Projects the core orbitals of all fragments out of every valence orbital, v' = v - sum_c c (S_cc^-1 S_cv), so that
    # core and valence are exactly orthogonal (between and within the fragments).  The projection alone does NOT
    # retain the orthonormality within a fragment: the projected valence orbitals of one fragment overlap as
    # 1 - S_vc S_cc^-1 S_cv, a deviation of second order in the core-valence overlap with the other fragment (Be2 6-31G,
    # 1s core: largest deviation 6.6e-3 at 2.5 Angstrom), which XR neglects, since it assumes the identity within the
    # fragments.  With orthonormalize_valence, the projected valence orbitals are therefore Loewdin orthonormalized
    # within each fragment (still orthogonal to all cores, as linear combinations of them); for Be2 this changes XR[2]
    # by -3 to -4 uEh at 4.0 (6-31G) / 4.5 Angstrom (cc-pVDZ), about 25 % of the remaining error there.  Either way the
    # projected valence orbitals contain the other fragment's core functions, i.e. they are not absolutely localized
    # anymore.

    # Identify absolute indices of core orbitals and dimensions
    if_core = []
    for frag in fragments:
        core = [False]*frag.basis.n_spatial_orb
        for i in frag.basis.core:  core[i] = True
        if_core += core
    n_orb = len(if_core)				# total number of spatial orbitals
    core = [i for i,TF in enumerate(if_core) if TF]	# list of core-orbital indices
    n_core = len(core)					# total number of core spatial orbitals

    # Work with absolute indices of overlap matix
    S = as_raw_mat(S, fragments)

    # Extract blocks of the S matrix partitioned as core (c) and valence (v)
    Scc = numpy.zeros((n_core,n_core))
    Scv = numpy.zeros((n_core,n_orb))
    for i in range(n_core):
        jc = 0
        jv = 0
        for j in range(n_orb):
            if j in core:
                Scc[i,jc] = S[core[i],j]
                jc += 1
            else:
                Scv[i,jv] = S[core[i],j]
                jv += 1

    # Build the cv block of the projection transformation
    Scc_inv = precise_numpy_inverse(Scc)
    Tcv = -Scc_inv @ Scv

    # Build the right-hand tranformation
    Right = numpy.identity(n_orb)
    for i in range(n_core):
        jv = 0
        for j in range(n_orb):
            if j not in core:
                Right[core[i],j] = Tcv[i,jv]
                jv += 1

    # joint Loewdin orthonormalization of the cores
    if orthonormalize_core and n_core > 0:
        evals, evecs = numpy.linalg.eigh(Scc)
        Right[numpy.ix_(core, core)] = (evecs / numpy.sqrt(evals)) @ evecs.T

    # Loewdin orthonormalization of the projected valence orbitals within each fragment
    if orthonormalize_valence:
        S_proj = Right.T @ S @ Right
        offset = 0
        for frag in fragments:
            valence = [offset + i for i in range(frag.basis.n_spatial_orb) if i not in frag.basis.core]
            evals, evecs = numpy.linalg.eigh(S_proj[numpy.ix_(valence, valence)])
            Right[:, valence] = Right[:, valence] @ (evecs / numpy.sqrt(evals)) @ evecs.T
            offset += frag.basis.n_spatial_orb

    # Move back to the fragment-blocked representation
    Right_raw = Right
    Left  = as_frag_blocked_mat(Right.T,  fragments)
    Right = as_frag_blocked_mat(Right,    fragments)

    if return_raw:    # (also the transformation of all fragment orbitals as one matrix, e.g. for RI_factors)
        return Left, Right, Right_raw
    return Left, Right

def active_orbitals(frag):
    """the active (valence) MO indices of a fragment: basis.active if given, otherwise all non-core MOs"""
    active = getattr(frag.basis, "active", None)
    if active is None:
        active = [i for i in range(frag.basis.n_spatial_orb) if i not in frag.basis.core]
    return list(active)

def valence_fragment(frag):
    """a copy of the fragment that only carries its active orbitals (no core): the fragment of the folded integrals"""
    import copy
    val = copy.copy(frag)
    val.basis = copy.copy(frag.basis)
    active = active_orbitals(frag)
    val.basis.MOcoeffs = numpy.asarray(frag.basis.MOcoeffs)[:, active]
    val.basis.n_spatial_orb = len(active)
    val.basis.core = []
    val.basis.active = list(range(len(active)))
    return val

def fold_frozen_core(fragments, ints, nuc):
    """
    The integrals of the active orbitals with the frozen core folded into the one-electron operator, for core orbitals
    that are orthonormal (jointly over all fragments) and orthogonal to all active orbitals, i.e. the integrals after
    direct_CoreProj(..., orthonormalize_core=True).  For the states with all core orbitals doubly occupied the
    Hamiltonian is then exactly
        H = E_core + sum_pq (T + sum_m U[m] + sum_m G[m])_pq a+_p a_q + V (active orbitals only),
        G[m]_pq = sum_{c in core(m)} [2 (pq|cc) - (pc|cq)] = sum_c [4 V_pcqc - 2 V_pccq]          (V = 1/2 <pq|rs>)
        E_core  = sum_c 2 (T + sum_m U[m])_cc + sum_cd [2 (cc|dd) - (cd|dc)],
    independent of the overlap of the active orbitals (also between the fragments).  The core potential G[m] of the
    core of fragment m (each orthonormalized core orbital belongs to the fragment of the core orbital it comes from) is
    added to the nuclear attraction U[m] of that fragment, so that the fragment-additive structure of the diagrams
    (T, U[m], V) is kept: the core of fragment m acts like an additional potential of the nuclei of m.  E_core is split
    into the fragment terms of the nuclear repulsion matrix: [m, m] the core of m with T, its own nuclei and itself,
    [m, n] the cores of m and n with the nuclei of the other fragment and with each other.
    ints: fragment-blocked S, T, U, V of all MOs (core + active [+ virtual]); nuc: the nuclear repulsion matrix
    returns (the valence fragments, struct(fragments, S, T, U, V) of the active orbitals, nuclear + core energies as a
    numpy matrix [m1, m2] with the same summation convention as Nuc_repulsion.matrix)
    """
    frag_ids = list(range(len(fragments)))
    core = [list(f.basis.core) for f in fragments]
    act = [active_orbitals(f) for f in fragments]
    def sub(X, idx, lists):
        if X is ints.V and hasattr(ints, "V_sub"):
            return ints.V_sub(idx, lists)    # (RI: only the requested orbitals, see ri_fragment_integrals)
        return numpy.asarray(X[idx])[numpy.ix_(*lists)]
    # check the prerequisites: orthonormal cores, orthogonal to the active orbitals
    dev = 0.0
    for m1 in frag_ids:
        for m2 in frag_ids:
            S = numpy.asarray(ints.S[m1, m2])
            if core[m1] and core[m2]:
                dev = max(dev, numpy.abs(S[numpy.ix_(core[m1], core[m2])] - (numpy.eye(len(core[m1])) if m1 == m2 else 0)).max())
            if core[m1]:
                dev = max(dev, numpy.abs(S[numpy.ix_(core[m1], act[m2])]).max())
    if dev > 1e-10:
        raise ValueError(f"fold_frozen_core: the core orbitals are not orthonormal and orthogonal to the active ones "
                         f"(deviation {dev:.1e}; direct_CoreProj with orthonormalize_core=True)")
    def G(m, m1, m2):
        """core potential of the core of fragment m in the active blocks (m1, m2)"""
        c = core[m]
        if not c:
            return numpy.zeros((len(act[m1]), len(act[m2])))
        J = numpy.einsum("pcqc->pq", sub(ints.V, (m1, m, m2, m), [act[m1], c, act[m2], c]))
        K = numpy.einsum("pccq->pq", sub(ints.V, (m1, m, m, m2), [act[m1], c, c, act[m2]]))
        return 4 * J - 2 * K
    # core energies, split into the fragment terms
    n_frag = len(frag_ids)
    E = numpy.zeros((n_frag, n_frag))
    for m in frag_ids:
        c = core[m]
        if not c:
            continue
        E[m, m] += 2 * numpy.trace(sub(ints.T, (m, m), [c, c]))
        for n in frag_ids:
            E[max(m, n), min(m, n)] += 2 * numpy.trace(sub(ints.U, (n, m, m), [c, c]))
            d = core[n]
            if d:
                J = numpy.einsum("cdcd->", sub(ints.V, (m, n, m, n), [c, d, c, d]))
                K = numpy.einsum("cddc->", sub(ints.V, (m, n, n, m), [c, d, d, c]))
                E[max(m, n), min(m, n)] += 4 * J - 2 * K
    nuc_folded = numpy.array([[nuc[m1, m2] for m2 in frag_ids] for m1 in frag_ids], dtype=float)
    for m1 in frag_ids:
        for m2 in range(m1 + 1):
            nuc_folded[m1, m2] += E[m1, m2]
            nuc_folded[m2, m1] = nuc_folded[m1, m2]
    # the integrals of the active orbitals
    S = dynamic_array([cached, lambda m1, m2: sub(ints.S, (m1, m2), [act[m1], act[m2]])], ranges=[frag_ids] * 2)
    T = dynamic_array([cached, lambda m1, m2: sub(ints.T, (m1, m2), [act[m1], act[m2]])], ranges=[frag_ids] * 2)
    U = dynamic_array([cached, lambda m, m1, m2: sub(ints.U, (m, m1, m2), [act[m1], act[m2]]) + G(m, m1, m2)],
                      ranges=[frag_ids] * 3)
    V = dynamic_array([cached, lambda m1, m2, m3, m4: sub(ints.V, (m1, m2, m3, m4), [act[m1], act[m2], act[m3], act[m4]])],
                      ranges=[frag_ids] * 4)
    val_fragments = [valence_fragment(f) for f in fragments]
    return val_fragments, struct(fragments=val_fragments, S=S, T=T, U=U, V=V), nuc_folded

def ri_fragment_integrals(fragments, aux_basis, project_core, orthonormalize_valence, fold_core, max_batch_bytes=5e8):
    """
    The fragment-blocked S, T, U, V of the (core-projected) fragment orbitals with the two-electron integrals from RI
    (lible, Qode's pybind_lible_ints.RI_factors), without the AO four-index tensor (get_ints integrals="ri"):
        - S, T, U from the one-electron AO integrals (with the core projection of direct_CoreProj, as for the exact ones),
        - the orbitals of the RI transformation: the fragment MOs, padded with zeros on the AOs of the other fragments,
          transformed with the core projection (C' = C R), and only the core + active ones with fold_core,
        - V[m1, m2, m3, m4][p, q, r, s] = 1/2 <pq|rs> = 1/2 sum_P B[P, p, r] B[P, q, s], evaluated on demand per block,
          and V_sub for the sub-blocks the core folding needs.
    returns (the fragments of these orbitals, struct(fragments, S, T, U, V, V_sub)); the fragments carry core and active
    as positions within the kept orbitals (with fold_core all MOs that are neither core nor active are dropped)
    """
    from qode.atoms.integrals.external_engines import pybind_lible_ints
    basis = fragments[0].basis.AOcode
    if any(frag.basis.AOcode != basis for frag in fragments):
        raise NotImplementedError("RI integrals: all fragments must have the same basis (as for the exact ones)")
    frag_ids = list(range(len(fragments)))
    AO_ints = AO_integrals(fragments, backend="lible", two_electron=False)
    FragMO_ints = fragMO_integrals(AO_ints, [frag.basis.MOcoeffs for frag in fragments], cache=True)
    n_mo = [frag.basis.n_spatial_orb for frag in fragments]
    R = numpy.identity(sum(n_mo))
    if project_core:
        Left, Right, R = direct_CoreProj(fragments, FragMO_ints.S, orthonormalize_valence, orthonormalize_core=fold_core,
                                         return_raw=True)
        FragMO_ints = ket_transformed(Right, bra_transformed(Left, FragMO_ints, cache=True), cache=True)
    # the kept orbitals per fragment (positions within the fragment's MOs)
    keep = [list(frag.basis.core) + active_orbitals(frag) if fold_core else list(range(n)) for frag, n in zip(fragments, n_mo)]
    # the transformed fragment orbitals as one AO matrix (block-diagonal C, the AOs of the fragments in order)
    C = numpy.zeros((sum(numpy.asarray(f.basis.MOcoeffs).shape[0] for f in fragments), sum(n_mo)))
    r = c = 0
    for frag in fragments:
        Cm = numpy.asarray(frag.basis.MOcoeffs)
        C[r:r + Cm.shape[0], c:c + Cm.shape[1]] = Cm
        r, c = r + Cm.shape[0], c + Cm.shape[1]
    C = C @ R
    offsets = numpy.cumsum([0] + n_mo[:-1])
    columns = [offsets[m] + numpy.array(keep[m], dtype=int) for m in frag_ids]
    geometry = "".join(AO_integrals._add_fragment(frag.atoms) for frag in fragments)
    B = pybind_lible_ints.RI_factors(geometry, basis, aux_basis, C[:, numpy.concatenate(columns)],
                                     max_batch_bytes=max_batch_bytes)
    starts = numpy.cumsum([0] + [len(k) for k in keep])
    def orbs(m, local=None):
        """positions of fragment m's kept orbitals (or of the given local positions among them) in B"""
        return numpy.arange(starts[m], starts[m + 1]) if local is None else starts[m] + numpy.asarray(local, dtype=int)
    def V_sub(idx, lists):
        m1, m2, m3, m4 = idx
        o1, o2, o3, o4 = (orbs(m, l) for m, l in zip(idx, lists))
        return 0.5 * numpy.einsum("Ppr,Pqs->pqrs", B[:, o1][:, :, o3], B[:, o2][:, :, o4], optimize=True)
    def V_block(m1, m2, m3, m4):
        return V_sub((m1, m2, m3, m4), [None] * 4)
    def one_el(X, nidx):
        if nidx == 2:
            return dynamic_array([cached, lambda m1, m2: numpy.asarray(X[m1, m2])[numpy.ix_(keep[m1], keep[m2])]],
                                 ranges=[frag_ids] * 2)
        return dynamic_array([cached, lambda n, m1, m2: numpy.asarray(X[n, m1, m2])[numpy.ix_(keep[m1], keep[m2])]],
                             ranges=[frag_ids] * 3)
    import copy
    kept_fragments = []
    for frag, k in zip(fragments, keep):
        f = copy.copy(frag)
        f.basis = copy.copy(frag.basis)
        f.basis.MOcoeffs = numpy.asarray(frag.basis.MOcoeffs)[:, k]
        f.basis.n_spatial_orb = len(k)
        pos = {i: j for j, i in enumerate(k)}
        f.basis.core = [pos[i] for i in frag.basis.core]
        f.basis.active = [pos[i] for i in active_orbitals(frag) if i in pos]
        kept_fragments.append(f)
    ints = struct(fragments=kept_fragments, S=one_el(FragMO_ints.S, 2), T=one_el(FragMO_ints.T, 2),
                  U=one_el(FragMO_ints.U, 3), V=dynamic_array([cached, V_block], ranges=[frag_ids] * 4), V_sub=V_sub)
    return kept_fragments, ints

def projected_overlap(fragments, timings, backend="lible", orthonormalize_valence=True):
    """only the core-projected overlap of get_ints (as get_ints(...)[0].S, returned as (struct(S=S),)), from the
    one-electron AO integrals (without any two-electron integrals for lible), e.g. for the corresponding orbitals"""
    AO_ints = AO_integrals(fragments, backend=backend, two_electron="lible" not in backend)
    FragMO_ints = fragMO_integrals(AO_ints, [frag.basis.MOcoeffs for frag in fragments], cache=True)
    Left, Right = direct_CoreProj(fragments, FragMO_ints.S, orthonormalize_valence)
    FragMO_ints = ket_transformed(Right, bra_transformed(Left, FragMO_ints, cache=True), cache=True)
    return (struct(S=wrap(FragMO_ints.S, [cached, tensorly_wrapper(timings)])),)

def get_ints(fragments, project_core=True, timings=None, spin_ints=True, backend="psi4", antisymmetrize=True,
             orthonormalize_valence=True, pair_thresh=None, fold_core=False, integrals="exact", aux_basis=None,
             ri_batch_bytes=5e8):    # spin_ints=False does not appear to be used ... prefer to remove option
    # antisymmetrize: only for spin_ints, antisymmetrized spin-orbital V (as required by diagrams derived with V
    #                 antisymmetry) or the plain one (as the spatial V of spin-adapted calculations)
    # orthonormalize_valence: only with project_core, Loewdin orthonormalization of the projected valence orbitals
    #                 within each fragment (see direct_CoreProj); False reproduces the plain projection
    # pair_thresh:    if given, the inter-fragment overlap (after the core projection) is restricted to the
    #                 corresponding-orbital pairs with a singular value above pair_thresh (pair_compressed), both in the
    #                 returned symmetric S and in the S^-1 of the biorthogonalization (T, U, V are not compressed); the
    #                 inter-fragment blocks of the symmetric S are then rank-K products (factorized_wrapper), and the
    #                 pair factors are returned with it as S_factors (precontract_pairs.py, only for spin_ints=False)
    # fold_core:     the core orbitals of all fragments are orthonormalized jointly (direct_CoreProj with
    #                 orthonormalize_core, requires project_core) and folded into the one-electron operator
    #                 (fold_frozen_core): all integrals are then those of the active orbitals (basis.active, default all
    #                 non-core MOs) only, the core potential of fragment m is part of U[m], the core energies part of
    #                 the returned nuclear repulsion matrix, and the densities must not contain the core (the fragment
    #                 densities of the active space, without frozen-core embedding)
    # integrals:     "exact" (the AO four-index integrals of the backend, transformed to the fragment orbitals) or "ri":
    #                 the two-electron integrals from RI (density fitting) with the auxiliary basis aux_basis, built by
    #                 lible directly in the (core-projected) fragment orbitals in batches of ri_batch_bytes
    #                 (ri_fragment_integrals; backend lible only), so that no AO four- or three-index tensor is held
    # More needs to be done regarding the basis to prevent mismatches with the fragment states (see the basis check of
    # workflow_spin_adapt.run_xr, hummr_dens.pybind_interface.hummr_basis_info)
    if fold_core and not project_core:
        raise ValueError("fold_core requires project_core (the core has to be orthogonal to the active orbitals)")
    if integrals not in ("exact", "ri"):
        raise ValueError(f"unknown integrals {integrals!r}")
    nuc = Nuc_repulsion(fragments).matrix
    if integrals == "ri":
        if "lible" not in backend or aux_basis is None:
            raise ValueError('integrals="ri" requires backend="lible" and an aux_basis')
        fragments, FragMO_ints = ri_fragment_integrals(fragments, aux_basis, project_core, orthonormalize_valence, fold_core,
                                                       max_batch_bytes=ri_batch_bytes)
    else:
        AO_ints     = AO_integrals(fragments, backend=backend)
        FragMO_ints = fragMO_integrals(AO_ints, [frag.basis.MOcoeffs for frag in fragments], cache=True)     # Cache because multiple calls to each block during biorthogonalization

        ### project the core out of the valence (and restore the orthonormality within the fragments, see direct_CoreProj)
        if project_core:
            Left, Right = direct_CoreProj(fragments, FragMO_ints.S, orthonormalize_valence, orthonormalize_core=fold_core)
            FragMO_ints_tmp = bra_transformed(Left,  FragMO_ints,     cache=True)
            FragMO_ints     = ket_transformed(Right, FragMO_ints_tmp, cache=True)

    ### fold the (orthonormalized) core into the one-electron operator, keep the active orbitals only
    if fold_core:
        fragments, FragMO_ints, nuc = fold_frozen_core(fragments, FragMO_ints, nuc)

    ### restrict the inter-fragment overlap to the leading orbital pairs
    if pair_thresh is not None:
        S_factors = pair_factors(FragMO_ints.S, pair_thresh)
        FragMO_ints = struct(fragments=FragMO_ints.fragments, S=pair_compressed(FragMO_ints.S, S_factors),
                             T=FragMO_ints.T, U=FragMO_ints.U, V=FragMO_ints.V, full=FragMO_ints)

    ### biorthogonalize everything without regard to core or valence
    Sinv = direct_Sinv(fragments, FragMO_ints.S)
    BiFragMO_ints = bra_transformed(Sinv, FragMO_ints.full if pair_thresh is not None else FragMO_ints, cache=True)
    # the half-transformed V_half, symmetrized under the electron swap (see electron_swap_symmetrized)
    BiFragMO_ints = struct(fragments=BiFragMO_ints.fragments, S=BiFragMO_ints.S, T=BiFragMO_ints.T, U=BiFragMO_ints.U, V=BiFragMO_ints.V,
                           V_half=dynamic_array([cached, electron_swap_symmetrized(BiFragMO_ints.V_half)], BiFragMO_ints.V.ranges))

    if spin_ints:
        #-#-# FragMO_spin_ints   = spin_orb_integrals(  FragMO_ints, rule_wrappers=[tensorly_wrapper], cache=True)     # no need to cache? because each block only called once by contraction code
        #-#-# BiFragMO_spin_ints = spin_orb_integrals(BiFragMO_ints, rule_wrappers=[tensorly_wrapper], cache=True)     # no need to cache? because each block only called once by contraction code
        FragMO_spin_ints_raw   = spin_orb_integrals(  FragMO_ints, "blocked", antisymmetrize=antisymmetrize)     # no need to cache? because each block only called once by contraction code
        BiFragMO_spin_ints_raw = spin_orb_integrals(BiFragMO_ints, "blocked", antisymmetrize=antisymmetrize)     # no need to cache? because each block only called once by contraction code
        FragMO_spin_ints = struct(
            S = wrap(FragMO_spin_ints_raw.S,   [cached, tensorly_wrapper(timings)]),
            T = wrap(FragMO_spin_ints_raw.T,   [cached, tensorly_wrapper(timings)]),
            U = wrap(FragMO_spin_ints_raw.U,   [cached, tensorly_wrapper(timings)]),
            V = wrap(FragMO_spin_ints_raw.V,   [cached, tensorly_wrapper2(timings)])
        )
        BiFragMO_spin_ints = struct(
            S      = wrap(BiFragMO_spin_ints_raw.S, [cached, tensorly_wrapper(timings)]),
            T      = wrap(BiFragMO_spin_ints_raw.T, [cached, tensorly_wrapper(timings)]),
            U      = wrap(BiFragMO_spin_ints_raw.U, [cached, tensorly_wrapper(timings)]),
            V      = wrap(BiFragMO_spin_ints_raw.V, [cached, tensorly_wrapper2(timings)]),
            V_half = wrap(BiFragMO_spin_ints_raw.V_half, [cached, tensorly_wrapper2(timings)]),
            V_diff = dynamic_array([cached, tensorly_wrapper2(timings), tens_diff(BiFragMO_spin_ints_raw.V_half, BiFragMO_spin_ints_raw.V, 1 if antisymmetrize else V_HALF_FACTOR_PLAIN)], BiFragMO_spin_ints_raw.V.ranges),
            #V_half1 = wrap(BiFragMO_spin_ints_raw.V_half1, [cached, tensorly_wrapper2(timings)]),
            #V_half2 = wrap(BiFragMO_spin_ints_raw.V_half2, [cached, tensorly_wrapper2(timings)])
        )

        return FragMO_spin_ints, BiFragMO_spin_ints, nuc
    else:
        FragMO_ints_wrapped = struct(
            S = wrap(FragMO_ints.S,   [cached, tensorly_wrapper(timings)]),
            T = wrap(FragMO_ints.T,   [cached, tensorly_wrapper(timings)]),
            U = wrap(FragMO_ints.U,   [cached, tensorly_wrapper(timings)]),
            V = wrap(FragMO_ints.V,   [cached, tensorly_wrapper2(timings)])
        )
        if pair_thresh is not None:
            FragMO_ints_wrapped.S = wrap(FragMO_ints.S, [cached, factorized_wrapper(S_factors, timings)])
            FragMO_ints_wrapped.S_factors = S_factors
        BiFragMO_ints_wrapped = struct(
            S      = wrap(BiFragMO_ints.S, [cached, tensorly_wrapper(timings)]),
            T      = wrap(BiFragMO_ints.T, [cached, tensorly_wrapper(timings)]),
            U      = wrap(BiFragMO_ints.U, [cached, tensorly_wrapper(timings)]),
            V      = wrap(BiFragMO_ints.V, [cached, tensorly_wrapper2(timings)]),  # eri without antisymmetry
            #V = dynamic_array([cached, tensorly_wrapper2(timings), _spinAnti4_rule(BiFragMO_ints.V)], ranges=BiFragMO_ints.V.ranges),  # antisymmetrized eri
            V_half = wrap(BiFragMO_ints.V_half, [cached, tensorly_wrapper2(timings)]),  # without antisymmetry, but electron-swap symmetric
            #V_half = dynamic_array([cached, tensorly_wrapper2(timings), _spinAnti4half_rule(BiFragMO_ints.V_half)], ranges=BiFragMO_ints.V.ranges),  # antisymmetrized
            V_diff = dynamic_array([cached, tensorly_wrapper2(timings), tens_diff(BiFragMO_ints.V_half, BiFragMO_ints.V, V_HALF_FACTOR_PLAIN)], BiFragMO_ints.V.ranges)  # without antisymmetry
            #V_diff = dynamic_array([cached, tensorly_wrapper2(timings), tens_diff(_spinAnti4half_rule(BiFragMO_ints.V_half), _spinAnti4_rule(BiFragMO_ints.V))], BiFragMO_ints.V.ranges)  # antisymmetrized
        )
        return FragMO_ints_wrapped, BiFragMO_ints_wrapped, nuc
