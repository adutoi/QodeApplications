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
    return tl_tensor(tensorly.tensor(tensor, dtype=Double.tensorly))

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

def tens_diff(A, B):
    def rule(*indices):
        return A[indices] - B[indices]
    return rule



def direct_Sinv(fragments, S):
    S = as_raw_mat(S, fragments)
    Sinv = precise_numpy_inverse(S)
    Sinv = as_frag_blocked_mat(Sinv, fragments)
    return Sinv

def direct_CoreProj(fragments, S):

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

    # Move back to the fragment-blocked representation
    Left  = as_frag_blocked_mat(Right.T,  fragments)
    Right = as_frag_blocked_mat(Right,    fragments)

    return Left, Right

def get_ints(fragments, project_core=True, timings=None, spin_ints=True):    # spin_ints=False does not appear to be used ... prefer to remove option
    # More needs to be done regarding the basis to prevent mismatches with the fragment states
    AO_ints     = AO_integrals(fragments)
    FragMO_ints = fragMO_integrals(AO_ints, [frag.basis.MOcoeffs for frag in fragments], cache=True)     # Cache because multiple calls to each block during biorthogonalization

    ### project the core out of the valence
    if project_core:
        Left, Right = direct_CoreProj(fragments, FragMO_ints.S)
        FragMO_ints_tmp = bra_transformed(Left,  FragMO_ints,     cache=True)
        FragMO_ints     = ket_transformed(Right, FragMO_ints_tmp, cache=True)

    ### biorthogonalize everything without regard to core or valence
    Sinv = direct_Sinv(fragments, FragMO_ints.S)
    BiFragMO_ints = bra_transformed(Sinv, FragMO_ints, cache=True)

    if spin_ints:
        #-#-# FragMO_spin_ints   = spin_orb_integrals(  FragMO_ints, rule_wrappers=[tensorly_wrapper], cache=True)     # no need to cache? because each block only called once by contraction code
        #-#-# BiFragMO_spin_ints = spin_orb_integrals(BiFragMO_ints, rule_wrappers=[tensorly_wrapper], cache=True)     # no need to cache? because each block only called once by contraction code
        FragMO_spin_ints_raw   = spin_orb_integrals(  FragMO_ints, "blocked")     # no need to cache? because each block only called once by contraction code
        BiFragMO_spin_ints_raw = spin_orb_integrals(BiFragMO_ints, "blocked")     # no need to cache? because each block only called once by contraction code
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
            V_diff = dynamic_array([cached, tensorly_wrapper2(timings), tens_diff(BiFragMO_spin_ints_raw.V_half, BiFragMO_spin_ints_raw.V)], BiFragMO_spin_ints_raw.V.ranges),
            #V_half1 = wrap(BiFragMO_spin_ints_raw.V_half1, [cached, tensorly_wrapper2(timings)]),
            #V_half2 = wrap(BiFragMO_spin_ints_raw.V_half2, [cached, tensorly_wrapper2(timings)])
        )

        return FragMO_spin_ints, BiFragMO_spin_ints, Nuc_repulsion(fragments).matrix
    else:
        return FragMO_ints, BiFragMO_ints, Nuc_repulsion(fragments).matrix
