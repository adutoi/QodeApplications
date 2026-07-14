import numpy as np
import itertools
from collections import defaultdict
from sympy.physics.quantum.cg import CG
from sympy import S as symS


###############################################################################
#
# PURPOSE OF THIS CODE
#
# ---------------------------------------------------------------------------
#
# This code constructs the SPIN-ADAPTED XR BASIS TRANSFORMATION
#
# starting from:
#
#   fragment eigenstates
#
# NOT fragment CSFs.
#
# ---------------------------------------------------------------------------
#
# INPUT:
#
# Fragment states:
#
#   |A_i, S_i, M_i>
#   |B_j, S_j, M_j>
#
# where:
#
#   i,j = fragment eigenstate indices
#
# and S_i,M_i are spin quantum numbers.
#
# ---------------------------------------------------------------------------
#
# OUTPUT:
#
# Linear combination coefficients:
#
#   |Psi_alpha^(S,M)>
#
# =
#
#   sum C_alpha(I,J)
#
#   |A_i,S_i,M_i> |B_j,S_j,M_j>
#
# which define:
#
#   the spin-adapted XR basis
#
# for a target total spin S.
#
# ---------------------------------------------------------------------------
#
# THIS IS THE MISSING BRIDGE:
#
# It tells you exactly how to linearly combine
#
# XR fragment product states
#
# into
#
# total-spin-adapted supersystem states.
#
###############################################################################


###############################################################################
# FRAGMENT STATE OBJECT
###############################################################################

class FragmentState:

    def __init__(
        self,
        frag,
        state_index,
        S,
        M,
        energy=0.0
    ):

        self.frag = frag
        self.idx = state_index

        self.S = float(S)
        self.M = float(M)

        self.energy = energy

    def label(self):

        return (
            f"{self.frag}{self.idx}"
            f"(S={self.S},M={self.M})"
        )


###############################################################################
# BUILD ALL M COMPONENTS OF A FRAGMENT MULTIPLET
###############################################################################

def build_spin_multiplet(
    frag_label,
    state_index,
    S,
    energy=0.0
):

    states = []

    for M in np.arange(-S, S + 1, 1):

        states.append(
            FragmentState(
                frag=frag_label,
                state_index=state_index,
                S=S,
                M=M,
                energy=energy
            )
        )

    return states


###############################################################################
# BUILD XR PRODUCT BASIS
###############################################################################

def build_product_basis(
    fragA_states,
    fragB_states
):

    basis = []

    for a in fragA_states:
        for b in fragB_states:

            basis.append((a, b))

    return basis


###############################################################################
# CLEBSCH-GORDAN COUPLED BASIS
###############################################################################

def build_spin_adapted_basis(
    fragA_multiplets,
    fragB_multiplets,
    S_total,
    M_total=None
):

    """
    Build spin-adapted XR basis.

    Parameters
    ----------
    fragA_multiplets:
        list of fragment multiplets

    fragB_multiplets:
        list of fragment multiplets

    S_total:
        target total spin

    M_total:
        target M value

    Returns
    -------
    basis_vectors:
        list of coefficient dictionaries

    Each dictionary contains:
        key   = XR product basis index
        value = coefficient
    """

    if M_total is None:
        M_total = S_total

    basis_vectors = []

    #
    # Loop over fragment state multiplets
    #
    for multA in fragA_multiplets:

        SA = multA[0].S

        for multB in fragB_multiplets:

            SB = multB[0].S

            #
            # Triangle condition
            #
            if not (
                abs(SA - SB)
                <= S_total
                <= SA + SB
            ):
                continue

            coeffs = {}

            #
            # Build coupled state
            #
            for a in multA:

                MA = a.M

                for b in multB:

                    MB = b.M

                    if abs(MA + MB - M_total) > 1e-12:
                        continue

                    cg = CG(
                        symS(SA), symS(MA),
                        symS(SB), symS(MB),
                        symS(S_total), symS(M_total)
                    ).doit()

                    if cg is None:
                        continue

                    cg = float(cg)

                    if abs(cg) < 1e-12:
                        continue

                    coeffs[(a, b)] = cg

            #
            # normalize
            #
            norm = np.sqrt(
                sum(c*c for c in coeffs.values())
            )

            for k in coeffs:
                coeffs[k] /= norm

            basis_vectors.append({
                "SA": SA,
                "SB": SB,
                "coeffs": coeffs
            })

    return basis_vectors


###############################################################################
# PRINTING
###############################################################################

def print_spin_adapted_basis(
    basis_vectors,
    S_total,
    M_total
):

    print("\n")
    print("=" * 80)
    print("SPIN-ADAPTED XR BASIS")
    print("=" * 80)

    print(
        f"\nTarget spin:"
        f" S={S_total}, M={M_total}"
    )

    for i, vec in enumerate(basis_vectors):

        print("\n")
        print("-" * 80)

        print(
            f"Spin-adapted basis vector {i}"
        )

        print(
            f"Fragment spins:"
            f" SA={vec['SA']},"
            f" SB={vec['SB']}"
        )

        print("\nExpansion:\n")

        coeffs = vec["coeffs"]

        for (a, b), c in coeffs.items():

            print(
                f"{c:+.6f}   "
                f"|{a.label()}> "
                f"|{b.label()}>"
            )


###############################################################################
# BUILD TRANSFORMATION MATRIX
###############################################################################

def build_transformation_matrix(
    basis_vectors,
    xr_product_basis
):

    """
    Construct matrix U

    columns:
        spin-adapted basis vectors

    rows:
        XR product basis states
    """

    idx = {
        state: i
        for i, state in enumerate(xr_product_basis)
    }

    U = np.zeros((
        len(xr_product_basis),
        len(basis_vectors)
    ))

    for j, vec in enumerate(basis_vectors):

        for state_pair, coeff in vec["coeffs"].items():

            i = idx[state_pair]

            U[i, j] = coeff

    return U


###############################################################################
# PROJECT XR HAMILTONIAN
###############################################################################

def project_xr_hamiltonian(
    H_xr,
    U
):

    """
    H_spin = U^dagger H_xr U
    """

    return U.T @ H_xr @ U


###############################################################################
# SPIN LEAKAGE ANALYSIS
###############################################################################

def analyze_spin_leakage(
    eigvecs_spin,
    U
):

    """
    Measures how much of each projected state
    lies outside the spin-adapted subspace.
    """

    print("\n")
    print("=" * 80)
    print("SPIN LEAKAGE ANALYSIS")
    print("=" * 80)

    for n in range(eigvecs_spin.shape[1]):

        v = eigvecs_spin[:, n]

        #
        # back-transform to XR basis
        #
        full = U @ v

        norm = np.dot(full, full)

        print(
            f"State {n}:"
            f" projected norm = {norm:.12f}"
        )


###############################################################################
# EXAMPLE
###############################################################################

if __name__ == "__main__":

    #
    # -----------------------------------------------------------------------
    # Example:
    #
    # two fragments
    #
    # each fragment has:
    #
    #   one singlet state
    #   one triplet state
    #
    # We construct:
    #
    #   total singlet XR basis
    #
    # -----------------------------------------------------------------------
    #

    #
    # Fragment A
    #
    A_singlet = build_spin_multiplet(
        frag_label="A",
        state_index=0,
        S=0
    )

    A_triplet = build_spin_multiplet(
        frag_label="A",
        state_index=1,
        S=1
    )

    #
    # Fragment B
    #
    B_singlet = build_spin_multiplet(
        frag_label="B",
        state_index=0,
        S=0
    )

    B_triplet = build_spin_multiplet(
        frag_label="B",
        state_index=1,
        S=1
    )

    #
    # Collect multiplets
    #
    fragA_multiplets = [
        A_singlet,
        A_triplet
    ]

    fragB_multiplets = [
        B_singlet,
        B_triplet
    ]

    #
    # Build ordinary XR product basis
    #
    xr_product_basis = build_product_basis(
        list(itertools.chain(*fragA_multiplets)),
        list(itertools.chain(*fragB_multiplets))
    )

    #
    # Build spin-adapted XR basis
    #
    basis_vectors = build_spin_adapted_basis(
        fragA_multiplets,
        fragB_multiplets,
        S_total=0,
        M_total=0
    )

    #
    # Print basis
    #
    print_spin_adapted_basis(
        basis_vectors,
        S_total=0,
        M_total=0
    )

    #
    # Transformation matrix
    #
    U = build_transformation_matrix(
        basis_vectors,
        xr_product_basis
    )

    print("\n")
    print("=" * 80)
    print("TRANSFORMATION MATRIX U")
    print("=" * 80)

    print(U)

    #
    # Example XR Hamiltonian
    #
    np.random.seed(1)

    H_xr = np.random.rand(
        len(xr_product_basis),
        len(xr_product_basis)
    )

    H_xr = 0.5 * (H_xr + H_xr.T)

    #
    # Spin-projected Hamiltonian
    #
    H_spin = project_xr_hamiltonian(
        H_xr,
        U
    )

    print("\n")
    print("=" * 80)
    print("SPIN-PROJECTED XR HAMILTONIAN")
    print("=" * 80)

    print(H_spin)

    #
    # Diagonalize
    #
    eigvals, eigvecs = np.linalg.eigh(H_spin)

    print("\n")
    print("=" * 80)
    print("SPIN-ADAPTED XR EIGENVALUES")
    print("=" * 80)

    print(eigvals)

    #
    # Leakage analysis
    #
    analyze_spin_leakage(
        eigvecs,
        U
    )
