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
#import tensorly
from qode.math import precise_numpy_inverse, linear_inner_product_space, iterative_biorthog, biorthog_iteration
from qode.util import indented, no_print
from qode.util.PyC import Double
from qode.atoms.integrals.fragments import AO_integrals, fragMO_integrals, bra_transformed, ket_transformed, spin_orb_integrals, Nuc_repulsion, as_raw_mat, as_frag_blocked_mat, zeros2, Id, mat_as_rows, mat_as_columns, space_traits, add, subtract, mat_mul, cached


def tensor_wrapper(rule):
    def wrap_it(*indices):
        #return tensorly.tensor(rule(*indices), dtype=tensorly.float64)
        return rule(*indices)
    return wrap_it

def iterative_Sinv(fragments, S):
    Sinv = zeros2(fragments)
    space = linear_inner_product_space(space_traits(float))
    S_as_cols    = mat_as_columns(S)
    Sinv_as_rows = mat_as_rows(Sinv)    
    S_cols    = []
    Sinv_rows = []
    for m,frag in enumerate(fragments):
        for i in range(frag.basis.n_spatial_orb):
            S_col      =  space.member(   S_as_cols(m,i))
            Sinv_row   =  space.member(Sinv_as_rows(m,i))
            Sinv_row  +=  S_col	# Sinv_row starts populated with zeros
            S_cols    += [S_col]
            Sinv_rows += [Sinv_row]
    iterative_biorthog(S_cols, in_place=Sinv_rows)
    return Sinv

def direct_Sinv(fragments, S):
    S = as_raw_mat(S, fragments)
    Sinv = precise_numpy_inverse(S)
    Sinv = as_frag_blocked_mat(Sinv, fragments)
    return Sinv

def direct_CoreSinv(fragments, S):

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

    # Build Sinv in core-only space
    core_S = numpy.zeros((n_core,n_core))
    for i in range(n_core):
        for j in range(n_core):
            core_S[i,j] = S[core[i],core[j]]
    core_Sinv = precise_numpy_inverse(core_S)

    # Build the left-hand integral tranformation
    Left  = precise_numpy_inverse(S)
    for i in range(n_core):
        for j in range(n_orb):
            Left[core[i],j] = 0
        for j in range(n_core):
            Left[core[i],core[j]] = core_Sinv[i,j]

    # Build the right-hand integral tranformation
    tmpS = Left @ S
    Right = numpy.identity(n_orb)
    for i in range(n_orb):
        if i not in core:
            for j in core:
                Right[j,i] = -tmpS[j,i]

    # Move back to the fragment-blocked representation
    Left  = as_frag_blocked_mat(Left,  fragments)
    Right = as_frag_blocked_mat(Right, fragments)

    return Left, Right

def iterative_CoreSinv(fragments, S):

    # Space for the inverse with the proper fragment-blocked structure
    ID   = Id(fragments)
    Left = zeros2(fragments)

    # Parse S and Left as rows and columns living in a vector space
    space = linear_inner_product_space(space_traits(float))
    ID_as_rows   = mat_as_rows(ID)
    Left_as_rows = mat_as_rows(Left)    
    S_as_cols    = mat_as_columns(S)
    ID_rows       = []
    Left_rows     = []
    S_cols        = []
    core          = []
    running = 0
    for m,frag in enumerate(fragments):
        for i in range(frag.basis.n_spatial_orb):
            ID_rows   += [space.member(  ID_as_rows(m,i))]
            Left_rows += [space.member(Left_as_rows(m,i))]
            S_cols    += [space.member(   S_as_cols(m,i))]
            if i in frag.basis.core:  core += [running]
            running += 1

    enum_Left_rows = []
    for k in range(running):
        ID_row, Left_row, S_col = ID_rows[k], Left_rows[k], S_cols[k]
        Left_row += ID_row	# Left_row starts populated with zeros
        Left_row /= (Left_row|S_col)
        enum_Left_rows += [(k,Left_row)]

    # Compute the full inverse
    diff = float("inf")
    while diff>1e-12:
        overlaps = biorthog_iteration(enum_Left_rows, S_cols, Vc=Left_rows)
        diff = numpy.linalg.norm( overlaps - numpy.identity(len(S_cols)) ) / len(S_cols)
        print(diff)

    # Reset rows corresponding to the core orbitals
    coreS_cols         = []
    coreSinv_rows      = []
    enum_coreSinv_rows = []
    for k,c in enumerate(core):
        ID_row, Left_row, S_col = ID_rows[c], Left_rows[c], S_cols[c]
        Left_row *= 0
        Left_row += ID_row
        Left_row /= (Left_row|S_col)
        coreS_cols         += [S_col]
        coreSinv_rows      += [Left_row]
        enum_coreSinv_rows += [(k,Left_row)]

    # Compute the inverse in the core space only (replaces rows of Sinv directly)
    diff = float("inf")
    while diff>1e-12:
        overlaps = biorthog_iteration(enum_coreSinv_rows, coreS_cols, Vc=coreSinv_rows)
        diff = numpy.linalg.norm( overlaps - numpy.identity(len(coreS_cols)) ) / len(coreS_cols)
        print(diff)

    # A cute trick to get the right-hand transform without having to go element-wise
    tmp = mat_mul(Left, S)
    Right = subtract(add(ID,ID), tmp, rules=[cached])

    return Left, Right


def get_ints(fragments, spin_ints=True, printout=print):
    # More needs to be done regarding the basis to prevent mismatches with the fragment states
    AO_ints     = AO_integrals(fragments, printout=printout)
    FragMO_ints = fragMO_integrals(AO_ints, [frag.basis.MOcoeffs for frag in fragments], cache=True)     # Cache because multiple calls to each block during biorthogonalization

    if len(fragments)>1:
        ### project the core out of the valence bf biorthogonalizing
        #Left, Right = direct_CoreSinv(fragments, FragMO_ints.S)
        #BiFragMO_ints_half = bra_transformed(Left,  FragMO_ints, cache=True)
        #BiFragMO_ints      = ket_transformed(Right, BiFragMO_ints_half)	# no need to cache because each block only called once
        ### biorthogonalize everything without regard to core or valence
        Sinv = direct_Sinv(fragments, FragMO_ints.S)
        BiFragMO_ints = bra_transformed(Sinv, FragMO_ints)    # no need to cache because each block only called once
    else:
        BiFragMO_ints = FragMO_ints

    if spin_ints:
        FragMO_ints   = spin_orb_integrals(  FragMO_ints, "blocked", rule_wrappers=[tensor_wrapper], cache=True)     # no need to cache? because each block only called once by contraction code
        BiFragMO_ints = spin_orb_integrals(BiFragMO_ints, "blocked", rule_wrappers=[tensor_wrapper], cache=True)     # no need to cache? because each block only called once by contraction code

    return FragMO_ints, BiFragMO_ints, Nuc_repulsion(fragments).matrix
