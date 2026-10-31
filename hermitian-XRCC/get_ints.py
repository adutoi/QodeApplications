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

def direct_CoreProj(fragments, S, orthonormalize_valence=True):
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
    Left  = as_frag_blocked_mat(Right.T,  fragments)
    Right = as_frag_blocked_mat(Right,    fragments)

    return Left, Right

def get_ints(fragments, project_core=True, timings=None, spin_ints=True, backend="psi4", antisymmetrize=True,
             orthonormalize_valence=True, pair_thresh=None):    # spin_ints=False does not appear to be used ... prefer to remove option
    # antisymmetrize: only for spin_ints, antisymmetrized spin-orbital V (as required by diagrams derived with V
    #                 antisymmetry) or the plain one (as the spatial V of spin-adapted calculations)
    # orthonormalize_valence: only with project_core, Loewdin orthonormalization of the projected valence orbitals
    #                 within each fragment (see direct_CoreProj); False reproduces the plain projection
    # pair_thresh:    if given, the inter-fragment overlap (after the core projection) is restricted to the
    #                 corresponding-orbital pairs with a singular value above pair_thresh (pair_compressed), both in the
    #                 returned symmetric S and in the S^-1 of the biorthogonalization (T, U, V are not compressed); the
    #                 inter-fragment blocks of the symmetric S are then rank-K products (factorized_wrapper), and the
    #                 pair factors are returned with it as S_factors (precontract_pairs.py, only for spin_ints=False)
    # More needs to be done regarding the basis to prevent mismatches with the fragment states
    AO_ints     = AO_integrals(fragments, backend=backend)
    FragMO_ints = fragMO_integrals(AO_ints, [frag.basis.MOcoeffs for frag in fragments], cache=True)     # Cache because multiple calls to each block during biorthogonalization

    ### project the core out of the valence (and restore the orthonormality within the fragments, see direct_CoreProj)
    if project_core:
        Left, Right = direct_CoreProj(fragments, FragMO_ints.S, orthonormalize_valence)
        FragMO_ints_tmp = bra_transformed(Left,  FragMO_ints,     cache=True)
        FragMO_ints     = ket_transformed(Right, FragMO_ints_tmp, cache=True)

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

        return FragMO_spin_ints, BiFragMO_spin_ints, Nuc_repulsion(fragments).matrix
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
        return FragMO_ints_wrapped, BiFragMO_ints_wrapped, Nuc_repulsion(fragments).matrix
