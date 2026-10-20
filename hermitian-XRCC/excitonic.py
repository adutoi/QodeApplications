#    (C) Copyright 2018 Anthony D. Dutoi
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
import time
import numpy
import scipy
import itertools
import qode
from   qode.util import sort_eigen
from   qode.many_body import CCSD
from   qode.many_body.hierarchical_fluctuations.excitonic import SD_operator, H_012_operator, inv_diagE, BakerCampbellHausdorff, space_traits
from   hamiltonian import braket_loops

import scipy.sparse as sp




def ccsd(H, out, resources, diis_start=0):
	monomer_Hamiltonians, dimer_Couplings = H
	states_per_frag = [h.shape[0] for h in monomer_Hamiltonians]

	out.log("Digesting H ...")
	E0, H = H_012_operator(states_per_frag, monomer_Hamiltonians, dimer_Couplings)

	out.log("Setting up fragment-energy preconditioner ...")
	invH0 = inv_diagE([numpy.diag(F) for F in monomer_Hamiltonians], states_per_frag)

	out.log("Initializing T ...")
	T = SD_operator(states_per_frag)		# Initialized to zero

	BCH = BakerCampbellHausdorff(states_per_frag, resources)
	out.log("Baker-Campbell-Hausdorff module initialized.")

	out.log("Performing coupled-cluster iterations ...")
	t_ccsd_start = time.time()
	dE, T = CCSD(H, T, invH0, BCH, space_traits, out.log.sublog(), thresh=1e-10, diis_start=diis_start)
	E = E0 + dE
	t_ccsd_end = time.time()

	out.log("CCSD Time {}".format(t_ccsd_end - t_ccsd_start))
	out.log("CCSD Energy =", E, 'Hartree')
	return E, T



def fci_old(H, out, target_state=0):
	#monomer_Hamiltonians, _, _ = H
	monomer_Hamiltonians, _ = H
	N_frag = len(monomer_Hamiltonians)
	states_per_frag = monomer_Hamiltonians[0].shape[0]
	for h in monomer_Hamiltonians:
		if h.shape[0]!=states_per_frag:  raise AssertionError	# Only for homogeneous systems right now

	out.log("Building ...")
	Hmat = numpy.zeros((states_per_frag**N_frag, states_per_frag**N_frag))
	braket_loops(Hmat, N_frag, states_per_frag, H)

	out.log("Diagonalizing ...")
	dim = Hmat.shape[0]
	if True:	# Because Lanczos does not yet handle non-hermitian matrices
		#for i, row in enumerate(Hmat):
		#	for j, elem in enumerate(row):
		#		if abs(elem) > 1e-2:
		#			print((i,j), elem)
		print(Hmat[:7, :7])
		vals, vecs = sort_eigen(numpy.linalg.eig(Hmat))
		E = vals[target_state]
	else:
		traits = qode.math.numpy_space.real_traits(dim)
		S = qode.math.linear_inner_product_space(traits)
		H = S.lin_op(Hmat)
		v = S.member(qode.util.basis_vector(0,dim))
		(Eval,Evec), = qode.math.lanczos.lowest_eigen(H, [v], thresh=1e-8)
		print("... Done.  \n\nE_gs = {}\n".format(Eval))
		E = Eval

	out.log("FCI Energy  =", E)
	return E, vecs[target_state]


def _build_fci_matrix_old(monomer_Hamiltonians, dimer_Couplings):
    """
    Build the Hamiltonian in the tensor-product basis.

    monomer_Hamiltonians[m]:
        square matrix for fragment m

    dimer_Couplings[m][n]:
        square matrix in the product basis of fragments m and n

    The local fragment dimensions need not be equal.
    """

    N_frag = len(monomer_Hamiltonians)

    dims = [
        H.shape[0]
        for H in monomer_Hamiltonians
    ]

    for m, H in enumerate(monomer_Hamiltonians):
        if H.shape != (dims[m], dims[m]):
            raise ValueError(
                f"Monomer Hamiltonian {m} is not square: {H.shape}"
            )

    dim_total = numpy.prod(dims)

    Hmat = numpy.zeros(
        (dim_total, dim_total),
        dtype=numpy.result_type(
            *monomer_Hamiltonians,
            *[
                dimer_Couplings[m][n]
                for m in range(N_frag)
                for n in range(m + 1, N_frag)
                if dimer_Couplings[m][n] is not None
            ],
        ),
    )

    def flat_index(indices):
        index = 0
        for m, i in enumerate(indices):
            index = index * dims[m] + i
        return index

    for bra in itertools.product(*[range(dim) for dim in dims]):
        I = flat_index(bra)

        for ket in itertools.product(*[range(dim) for dim in dims]):
            J = flat_index(ket)

            value = 0.0

            # -------------------------------------------------------------
            # Monomer terms:
            #
            # H_m acts on fragment m and identity on every other
            # fragment. Therefore all other bra/ket indices must agree.
            # -------------------------------------------------------------
            for m in range(N_frag):

                if all(
                    bra[n] == ket[n]
                    for n in range(N_frag)
                    if n != m
                ):
                    value += monomer_Hamiltonians[m][
                        bra[m], ket[m]
                    ]

            # -------------------------------------------------------------
            # Dimer terms:
            #
            # H_mn acts on fragments m and n and identity on all
            # remaining fragments.
            # -------------------------------------------------------------
            for m in range(N_frag):
                for n in range(m + 1, N_frag):

                    coupling = dimer_Couplings[m][n]

                    if coupling is None:
                        continue

                    if not all(
                        bra[k] == ket[k]
                        for k in range(N_frag)
                        if k != m and k != n
                    ):
                        continue

                    i = bra[m] * dims[n] + bra[n]
                    j = ket[m] * dims[n] + ket[n]

                    value += coupling[i, j]

            Hmat[I, J] = value

    return Hmat


def _embed_two_body(coupling, m, n, dims):
    """Embed a coupling in the (m,n) product basis into the full space (identity elsewhere)."""
    N = len(dims)
    rest = [k for k in range(N) if k not in (m, n)]
    order = [m, n] + rest
    d_order = [dims[k] for k in order]
    d_rest = int(numpy.prod([dims[k] for k in rest])) if rest else 1

    K = sp.kron(sp.csr_matrix(coupling), sp.identity(d_rest, format="csr"), format="csr")
    if not rest:                       # 2 fragments: already in the right order
        return K

    # permute tensor axes from (m,n,rest...) back to (0,1,...,N-1)
    inv = numpy.argsort(order)
    D = int(numpy.prod(dims))
    K = K.toarray().reshape(d_order + d_order)          # dense only if N>2
    K = K.transpose(list(inv) + [N + i for i in inv]).reshape(D, D)
    return sp.csr_matrix(K)


def _build_fci_matrix(monomer_Hamiltonians, dimer_Couplings, sparse=False):
    N = len(monomer_Hamiltonians)
    dims = [H.shape[0] for H in monomer_Hamiltonians]
    for m, H in enumerate(monomer_Hamiltonians):
        if H.shape != (dims[m], dims[m]):
            raise ValueError(f"Monomer Hamiltonian {m} is not square: {H.shape}")
    D = int(numpy.prod(dims))

    # monomer terms: I x ... x H_m x ... x I
    Hmat = sp.csr_matrix((D, D), dtype=numpy.result_type(
        *monomer_Hamiltonians,
        *[dimer_Couplings[m][n] for m in range(N) for n in range(m+1, N)
          if dimer_Couplings[m][n] is not None]))
    for m in range(N):
        left  = sp.identity(int(numpy.prod(dims[:m])),   format="csr")
        right = sp.identity(int(numpy.prod(dims[m+1:])), format="csr")
        Hmat = Hmat + sp.kron(sp.kron(left, sp.csr_matrix(monomer_Hamiltonians[m])), right, format="csr")

    # dimer terms
    for m in range(N):
        for n in range(m+1, N):
            c = dimer_Couplings[m][n]
            if c is not None:
                Hmat = Hmat + _embed_two_body(c, m, n, dims)

    return Hmat if sparse else Hmat.toarray()


def fci(H, out, timings, target_state=0, get_left_and_right=False, dense_cutoff=1000, basis_mask=None):
    """
    basis_mask: optional boolean array (or index list) over the tensor-product basis selecting the
                states to diagonalize in (e.g. only a given total charge and the fragment spin pairs,
                which can couple to the target multiplicity). The returned eigenvectors are given in
                the full tensor-product basis, with zeros for the states outside of the mask.
    """

    monomer_Hamiltonians, dimer_Couplings = H

    out.log("Building ...")

    timings.start()

    Hmat = _build_fci_matrix(
        monomer_Hamiltonians,
        dimer_Couplings,
        sparse=True,
    )

    timings.record("Build global H from H1 and H2")

    out.log("Diagonalizing ...")

    timings.start()

    D_full = Hmat.shape[0]
    if basis_mask is None:
        sel = numpy.arange(D_full)
    else:
        sel = numpy.flatnonzero(basis_mask) if numpy.asarray(basis_mask).dtype == bool else numpy.asarray(basis_mask)
    Hblock = Hmat[sel][:, sel]
    D = Hblock.shape[0]

    # Ensure target_state is a numpy array if it's a list/tuple for safe 2D slicing
    is_sequence = isinstance(target_state, (list, tuple, numpy.ndarray))
    if is_sequence:
        target_state_idx = numpy.asarray(target_state)
    else:
        target_state_idx = target_state

    vecs_l = None
    if D <= dense_cutoff or get_left_and_right or is_sequence:
        A = Hblock.toarray()
        if get_left_and_right:
            vals, vecs_l, vecs = scipy.linalg.eig(A, left=True, right=True, check_finite=False)
        else:
            vals, vecs = scipy.linalg.eig(A, check_finite=False, overwrite_a=True)
        idx = numpy.argsort(vals.real)
        vals, vecs = vals[idx], vecs[:, idx]
        if vecs_l is not None:
            vecs_l = vecs_l[:, idx]
    else:
        k = target_state + 1
        vals, vecs = sp.linalg.eigs(Hblock, k=k, which="SR", ncv=min(max(2*k+1, 40), D-1))
        idx = numpy.argsort(vals.real)
        vals, vecs = vals[idx], vecs[:, idx]

    timings.record("Diagonalization")

    E = vals[target_state_idx]

    out.log("FCI Energy  =", E)

    def embed(v):
        full = numpy.zeros((D_full,) + v.shape[1:], dtype=v.dtype)
        full[sel] = v
        return full

    if get_left_and_right:
        return E, embed(vecs_l[:, target_state_idx]), embed(vecs[:, target_state_idx])
    else:
        return E, embed(vecs[:, target_state_idx])

