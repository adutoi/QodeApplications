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
import re
from sympy import sqrt, S as Sym
from sympy.physics.wigner import wigner_6j


def mult_to_spin(mult):
    """Convert multiplicity (2S+1) to spin."""
    return Sym(mult - 1) / 2


def _triangle_allowed(j1, j2, j3):
    """
    Return True if j1, j2, j3 satisfy the angular-momentum
    triangle conditions.
    """
    return (
        abs(j1 - j2) <= j3 <= j1 + j2
        and
        (j1 + j2 + j3).is_integer
    )


def recoupling_factor(
    bra_mult1,
    bra_mult2,
    ket_mult1,
    ket_mult2,
    total_mult,
    tensor_mult,
):
    """
    Spin recoupling coefficient for two spin-adapted monomer
    transition densities.

    The coefficient is the reduction of

        { S1'  S2'  S
          k    k    0
          S1   S2   S }

    from the corresponding 9-j symbol to a 6-j symbol.

    Parameters
    ----------
    bra_mult1, bra_mult2
        Multiplicities of the monomer states in the bra.

    ket_mult1, ket_mult2
        Multiplicities of the monomer states in the ket.

    total_mult
        Total multiplicity of the supersystem state.

    tensor_mult
        Multiplicity of the tensor operator:

            1 : k = 0
            2 : k = 1/2
            3 : k = 1
            ...

    Returns
    -------
    float
        Spin recoupling coefficient.
    """

    S1p = mult_to_spin(bra_mult1)
    S2p = mult_to_spin(bra_mult2)

    S1 = mult_to_spin(ket_mult1)
    S2 = mult_to_spin(ket_mult2)

    Stot = mult_to_spin(total_mult)
    k = mult_to_spin(tensor_mult)

    #
    # The tensor of rank k must be able to connect the bra and ket
    # spin on each fragment separately, otherwise sympy won't return
    # zero, but an error instead.
    #
    if not _triangle_allowed(S1p, S1, k):
        return 0.0

    if not _triangle_allowed(S2p, S2, k):
        return 0.0

    #
    # Reduction of
    #
    #     { S1' S2' S
    #       k   k   0
    #       S1  S2  S }
    #
    # to a 6-j symbol:
    #
    #     (-1)^(S2' + S + k + S1)
    #     -------------------------------- { S1' S2' S
    #             sqrt(2k+1)                S2  S1  k }
    #
    # The factor sqrt(2S+1) from the original 9-j expression
    # cancels against the corresponding zero-entry reduction.
    #
    phase_exponent = S2p + Stot + k + S1

    phase = (-1) ** int(phase_exponent)

    sixj = wigner_6j(
        S1p, S2p, Stot,
        S2,  S1,  k,
    )

    prefactor = phase / sqrt(2 * k + 1)

    return float(prefactor * sixj)


def tensor_multiplicities_from_label(label):
    """
    Determine the allowed tensor multiplicities from the operator
    topology encoded in the diagram label.

    The returned multiplicities describe the tensor rank k:

        1 -> k = 0
        2 -> k = 1/2
        3 -> k = 1
        ...

    Local c-a pairs belonging to the same operator are internally
    rank restricted.  Their number is therefore tracked separately
    through ``restr_ca`` and does not enter the tensor multiplicity
    calculation.
    """

    free_legs = [0, 0]

    for tensor, digits in re.findall(r"([stuv])([01]+)", label):

        if tensor == "s":
            ops = [
                ("c", digits[0]),
                ("a", digits[1]),
            ]

        elif tensor == "t":
            ops = [
                ("c", digits[0]),
                ("a", digits[1]),
            ]

        elif tensor == "u":
            ops = [
                ("c", digits[1]),
                ("a", digits[2]),
            ]

        elif tensor == "v":
            ops = [
                ("c", digits[0]),
                ("c", digits[1]),
                ("a", digits[2]),
                ("a", digits[3]),
            ]

        else:
            raise RuntimeError(f"Unknown tensor '{tensor}'")

        fragment_ops = {}

        for op, frag in ops:
            fragment_ops.setdefault(frag, []).append(op)

        for frag, ops_here in fragment_ops.items():

            nc = ops_here.count("c")
            na = ops_here.count("a")

            paired = min(nc, na)

            free_legs[int(frag)] += nc - paired
            free_legs[int(frag)] += na - paired

    # the rank for both fragments needs to be equal, when put together to the
    # total Hamiltonian, which has rank 0, so pick the smallest maximum rank possible.
    return tuple(range(1 + min(free_legs) % 2, min(free_legs) + 2, 2))


def allowed_total_multiplicities(mults):
    """
    Return all multiplicities obtainable by successively coupling
    the supplied monomer multiplicities.
    """

    allowed = {mults[0]}

    for mult in mults[1:]:

        new_allowed = set()

        for total in allowed:

            S1 = (total - 1) / 2
            S2 = (mult - 1) / 2

            Smin = abs(S1 - S2)
            Smax = S1 + S2

            S = Smin

            while S <= Smax:
                new_allowed.add(int(2 * S + 1))
                S += 1

        allowed = new_allowed

    return allowed


def tensor_restr_ca_from_label(label):
    """
    Determine how many c-a pairs are locally restricted to rank k=0.

    A c-a pair is restricted when the creator and annihilator belonging
    to the same operator occur on the same fragment.

    The operator labels use creator-first / annihilator-second ordering,
    so the relevant local pairs can be identified directly from the
    fragment indices.

    Returns
    -------
    int
        Number of locally rank-0-restricted c-a pairs.
    """

    restr_ca = [0, 0]

    for tensor, digits in re.findall(r"([stuv])([01]+)", label):

        if tensor == "s":
            creator_frag = digits[0]
            annihilator_frag = digits[1]

            if creator_frag == annihilator_frag:
                restr_ca[int(digits[0])] += 1

        elif tensor == "t":
            creator_frag = digits[0]
            annihilator_frag = digits[1]

            if creator_frag == annihilator_frag:
                restr_ca[int(digits[0])] += 1

        elif tensor == "u":
            creator_frag = digits[1]
            annihilator_frag = digits[2]

            if creator_frag == annihilator_frag:
                restr_ca[int(digits[1])] += 1

        elif tensor == "v":
            #
            # v has creators first and annihilators second:
            #
            #   c(d0) c(d1) a(d2) a(d3)
            #
            # The two operator-local ca pairings are therefore
            #
            #   d0 <-> d2
            #   d1 <-> d3
            #
            if digits[0] == digits[2]:
                restr_ca[int(digits[0])] += 1

            if digits[1] == digits[3]:
                restr_ca[int(digits[1])] += 1

        else:
            raise RuntimeError(f"Unknown tensor '{tensor}'")

    return restr_ca


def diagram_couplings(supersys_info, X, Dchgs, tensor_mults):
    """
    Return all spin-recoupling contributions for this diagram.

    Each returned entry contains

        (prefactor, Dmult, rank2)

    where

        prefactor
            Spin recoupling coefficient.

        Dmult
            Required bra-ket multiplicity difference of each
            fragment density.

        rank2
            Twice the tensor rank k.  Thus

                rank2 = 0  -> k = 0
                rank2 = 1  -> k = 1/2
                rank2 = 2  -> k = 1
                ...

    The restriction in the number of rank-0 ca pairs is diagram
    specific and is therefore not part of this return value.
    """

    #
    # Non-spin-adapted calculations.
    #
    if X.mult[0][0] is None:
        return [(1.0, None, None)]

    target_mult = supersys_info.target_multiplicity

    #
    # Spin recoupling is currently implemented for dimer diagrams.
    #
    if len(X.mult) != 2 and len(X.mult) != 1:
        raise NotImplementedError(
            "Spin recoupling currently implemented only for monomer and dimer diagrams."
        )

    if len(X.mult) == 1:
        bra_mult, ket_mult = X.mult[0]

        if bra_mult != target_mult or ket_mult != target_mult:
            return []

        return [
            (
                1.0,       # prefactor
                (0, 0),    # Dmult
                0,       # rank2 = 2*k
            )
        ]

    #
    # Check whether the requested supersystem multiplicity can be
    # formed from the bra and ket fragment multiplicities.
    #
    bra_allowed = allowed_total_multiplicities(
        [X.mult[i][0] for i in range(len(X.mult))] #[mult_i for mult_i, _ in X.mult]
    )

    ket_allowed = allowed_total_multiplicities(
        [X.mult[i][1] for i in range(len(X.mult))] #[mult_j for _, mult_j in X.mult]
    )

    if target_mult not in bra_allowed:
        return []

    if target_mult not in ket_allowed:
        return []

    mult1 = X.mult[0]
    mult2 = X.mult[1]

    bra_mult1, ket_mult1 = mult1
    bra_mult2, ket_mult2 = mult2

    couplings = []

    #
    # tensor_mult = 2*k + 1
    #
    for tensor_mult in tensor_mults:

        prefactor = recoupling_factor(
            bra_mult1,
            bra_mult2,
            ket_mult1,
            ket_mult2,
            target_mult,
            tensor_mult,
        )

        if abs(prefactor) < 1e-10:
            continue

        # returning Dmults here is not a unique definition, but that
        # is also not necessary, because only one specific transition of
        # subsystem-transitions is requested.
        Dmult = (
            bra_mult1 - ket_mult1,
            bra_mult2 - ket_mult2,
        )

        rank2 = tensor_mult - 1

        couplings.append(
            (
                prefactor,
                Dmult,
                rank2,
            )
        )

    return couplings
