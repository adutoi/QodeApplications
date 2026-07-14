import itertools
import math
import numpy as np
from collections import defaultdict
from sympy.physics.quantum.cg import CG
from sympy import S as symS


###############################################################################
# SPIN LABELS
###############################################################################

def spin_label_to_S(label):

    mapping = {
        "singlet": 0,
        "doublet": 0.5,
        "triplet": 1,
        "quartet": 1.5,
        "quintet": 2,
        "sextet": 2.5,
        "septet": 3,
    }

    if isinstance(label, str):
        return float(mapping[label.lower()])

    return float(label)


###############################################################################
# PRIMITIVE BASIS
###############################################################################

def primitive_basis(n, M):

    n_alpha = int(round(n/2 + M))

    basis = []

    for occ in itertools.combinations(range(n), n_alpha):

        s = ["β"] * n

        for i in occ:
            s[i] = "α"

        basis.append(tuple(s))

    return basis


def basis_index_map(basis):

    return {b: i for i, b in enumerate(basis)}


###############################################################################
# EXCHANGE OPERATOR / S^2
###############################################################################

def apply_exchange(state, i, j):

    s = list(state)

    s[i], s[j] = s[j], s[i]

    return tuple(s)


def build_S2_matrix(n, M):

    basis = primitive_basis(n, M)

    dim = len(basis)

    idx = basis_index_map(basis)

    S2 = np.zeros((dim, dim))

    #
    # Dirac identity:
    #
    # S^2 = -N(N-4)/4 + sum(P_ij)
    #
    const = -n * (n - 4) / 4

    for a, st in enumerate(basis):

        S2[a, a] += const

        for i in range(n):
            for j in range(i + 1, n):

                ex = apply_exchange(st, i, j)

                b = idx[ex]

                S2[b, a] += 1.0

    return basis, S2


###############################################################################
# M VALUE
###############################################################################

def compute_M_from_basis(basis):

    st = basis[0]

    na = st.count("α")
    nb = st.count("β")

    return 0.5 * (na - nb)


###############################################################################
# LOWERING OPERATOR
###############################################################################

def apply_S_minus(vec, basis):

    n = len(basis[0])

    M_old = compute_M_from_basis(basis)

    target_basis = primitive_basis(n, M_old - 1)

    target_idx = basis_index_map(target_basis)

    out = np.zeros(len(target_basis))

    for i, st in enumerate(basis):

        c = vec[i]

        if abs(c) < 1e-14:
            continue

        for p in range(n):

            if st[p] == "α":

                ns = list(st)
                ns[p] = "β"
                ns = tuple(ns)

                j = target_idx[ns]

                out[j] += c

    return target_basis, out


###############################################################################
# BUILD HIGHEST-WEIGHT GENEALOGICAL CSFS
###############################################################################

def highest_weight_csfs(n, S_target):

    M = S_target

    basis, S2 = build_S2_matrix(n, M)

    eigvals, eigvecs = np.linalg.eigh(S2)

    target = S_target * (S_target + 1)

    csfs = []

    for k, val in enumerate(eigvals):

        if abs(val - target) < 1e-10:

            vec = eigvecs[:, k]

            vec /= np.linalg.norm(vec)

            csfs.append(vec)

    return basis, csfs


###############################################################################
# BUILD FULL SU(2) MULTIPLETS
###############################################################################

def build_full_multiplets(n, S_target):

    top_basis, top_csfs = highest_weight_csfs(n, S_target)

    multiplets = []

    for top_vec in top_csfs:

        multiplet = {}

        current_basis = top_basis
        current_vec = top_vec.copy()

        M = S_target

        multiplet[M] = (
            current_basis,
            current_vec.copy()
        )

        while M > -S_target:

            next_basis, next_vec = apply_S_minus(
                current_vec,
                current_basis
            )

            factor = math.sqrt(
                (S_target + M) *
                (S_target - M + 1)
            )

            next_vec /= factor

            next_vec /= np.linalg.norm(next_vec)

            M_next = M - 1

            multiplet[M_next] = (
                next_basis,
                next_vec.copy()
            )

            current_basis = next_basis
            current_vec = next_vec

            M = M_next

        multiplets.append(multiplet)

    return multiplets


###############################################################################
# FRAGMENT MULTIPLETS
###############################################################################

def fragment_multiplets(n_fragment, max_spin):

    max_spin = spin_label_to_S(max_spin)

    Smax = n_fragment / 2

    out = {}

    S = 0 if n_fragment % 2 == 0 else 0.5

    while S <= min(Smax, max_spin):

        out[S] = build_full_multiplets(
            n_fragment,
            S
        )

        S += 1

    return out


###############################################################################
# TENSOR PRODUCTS
###############################################################################

def tensor_product_state(stA, stB):

    return tuple(list(stA) + list(stB))


def tensor_product_vectors(
    vecA,
    basisA,
    vecB,
    basisB
):

    coeffs = defaultdict(float)

    for i, stA in enumerate(basisA):

        cA = vecA[i]

        if abs(cA) < 1e-14:
            continue

        for j, stB in enumerate(basisB):

            cB = vecB[j]

            if abs(cB) < 1e-14:
                continue

            full_state = tensor_product_state(
                stA,
                stB
            )

            coeffs[full_state] += cA * cB

    return coeffs


###############################################################################
# GLOBAL GENEALOGICAL BASIS
###############################################################################

def global_genealogical_basis(n, S):

    return build_full_multiplets(n, S)


###############################################################################
# BUILD FRAGMENT-COUPLED STATE
###############################################################################

def build_fragment_coupled_state(
    multA,
    multB,
    SA,
    SB,
    S_total,
    M_total,
    global_basis
):

    global_idx = basis_index_map(global_basis)

    out = np.zeros(len(global_basis))

    #
    # Sum over all MA MB
    #
    for MA in np.arange(-SA, SA + 1, 1):

        MB = M_total - MA

        if abs(MB) > SB:
            continue

        cg = CG(
            symS(SA), symS(MA),
            symS(SB), symS(MB),
            symS(S_total), symS(M_total)
        ).doit()

        if cg is None:
            continue

        cg = float(cg)

        if abs(cg) < 1e-14:
            continue

        basisA, vecA = multA[MA]

        basisB, vecB = multB[MB]

        coeffs = tensor_product_vectors(
            vecA,
            basisA,
            vecB,
            basisB
        )

        for st, coeff in coeffs.items():

            if st in global_idx:

                idx = global_idx[st]

                out[idx] += cg * coeff

    #
    # normalize
    #
    norm = np.linalg.norm(out)

    if norm > 1e-14:
        out /= norm

    return out


###############################################################################
# PROJECTIONS ONTO GLOBAL GENEALOGICAL CSFS
###############################################################################

def projection_matrix(
    fragment_vectors,
    global_vectors
):

    P = np.zeros((
        len(fragment_vectors),
        len(global_vectors)
    ))

    for i, fv in enumerate(fragment_vectors):

        for j, gv in enumerate(global_vectors):

            P[i, j] = np.dot(fv, gv)

    return P


###############################################################################
# MAIN DRIVER
###############################################################################

def analyze_fragment_couplings(
    nA,
    nB,
    SA_select,
    SB_select,
    S_total
):

    #
    # fragment irreps
    #
    fragA = fragment_multiplets(
        nA,
        SA_select
    )

    fragB = fragment_multiplets(
        nB,
        SB_select
    )

    SA = spin_label_to_S(SA_select)
    SB = spin_label_to_S(SB_select)

    S_total = spin_label_to_S(S_total)

    #
    # global genealogical basis
    #
    global_mults = global_genealogical_basis(
        nA + nB,
        S_total
    )

    #
    # choose M_total = S_total
    #
    M_total = S_total

    global_basis = global_mults[0][M_total][0]

    #
    # collect global genealogical vectors
    #
    global_vectors = []

    for mu, mult in enumerate(global_mults):

        basis, vec = mult[M_total]

        global_vectors.append(vec)

    #
    # build fragment-coupled vectors
    #
    fragment_vectors = []

    labels = []

    for muA, multA in enumerate(fragA[SA]):

        for muB, multB in enumerate(fragB[SB]):

            fv = build_fragment_coupled_state(
                multA,
                multB,
                SA,
                SB,
                S_total,
                M_total,
                global_basis
            )

            fragment_vectors.append(fv)

            labels.append(
                (muA, muB)
            )

    #
    # projection matrix
    #
    P = projection_matrix(
        fragment_vectors,
        global_vectors
    )

    return {
        "primitive_basis": global_basis,
        "global_vectors": global_vectors,
        "fragment_vectors": fragment_vectors,
        "projection_matrix": P,
        "labels": labels,
        "SA": SA,
        "SB": SB,
        "S_total": S_total
    }


###############################################################################
# PRINTING
###############################################################################

def print_basis(basis):

    print("\nPrimitive basis:\n")

    for i, st in enumerate(basis):

        print(
            f"{i:3d}  |{''.join(st)}>"
        )


def print_global_csfs(result):

    print("\n")
    print("=" * 80)
    print("GLOBAL GENEALOGICAL CSFS")
    print("=" * 80)

    basis = result["primitive_basis"]

    for k, vec in enumerate(result["global_vectors"]):

        print(f"\nTheta_{k+1}:\n")

        for i, c in enumerate(vec):

            if abs(c) > 1e-10:

                print(
                    f"{c:+.6f}  "
                    f"|{''.join(basis[i])}>"
                )


def print_projection_analysis(result):

    print("\n")
    print("=" * 80)
    print("FRAGMENT TRIPLET/TENSOR PROJECTIONS")
    print("=" * 80)

    P = result["projection_matrix"]

    for i, label in enumerate(result["labels"]):

        muA, muB = label

        print("\n")
        print(
            f"Fragment combination "
            f"(muA={muA}, muB={muB})"
        )

        for j in range(P.shape[1]):

            coeff = P[i, j]

            print(
                f"Projection onto Theta_{j+1}: "
                f"{coeff:+.6f}"
            )


###############################################################################
# EXPLICIT CG EXPANSION
###############################################################################

def print_explicit_coupling_formula(
    SA,
    SB,
    S_total,
    M_total
):

    print("\n")
    print("=" * 80)
    print("EXPLICIT CLEBSCH-GORDAN COUPLING")
    print("=" * 80)

    print(
        f"\n|S={S_total},M={M_total}> ="
    )

    for MA in np.arange(-SA, SA + 1, 1):

        MB = M_total - MA

        if abs(MB) > SB:
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

        print(
            f"{cg:+.6f} "
            f"|{SA},{MA}>_A "
            f"|{SB},{MB}>_B"
        )


###############################################################################
# EXAMPLE: 2x2 -> SINGLET
###############################################################################

if __name__ == "__main__":

    #
    # THIS IS THE IMPORTANT TEST:
    #
    # 2 spins on A
    # 2 spins on B
    #
    # triplet x triplet
    #
    # coupled to singlet
    #
    # should produce only ONE of the two
    # global singlet genealogical CSFs.
    #

    result = analyze_fragment_couplings(
        nA=2,
        nB=2,
        SA_select="triplet",
        SB_select="triplet",
        S_total="singlet"
    )

    print_basis(
        result["primitive_basis"]
    )

    print_global_csfs(result)

    #print_projection_analysis(result)

    print_explicit_coupling_formula(
        SA=1,
        SB=1,
        S_total=0,
        M_total=0
    )

    print_explicit_coupling_formula(
        SA=0,
        SB=0, 
        S_total=0, 
        M_total=0
    )
