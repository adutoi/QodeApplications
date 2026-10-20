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

    # These two triads should already be checked for by the allowed mults
    #if not _triangle_allowed(S2, S1, Stot):
    #    return 0.0

    #if not _triangle_allowed(S1p, S2p, Stot):
    #    return 0.0

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

    phase = (-1) ** round(phase_exponent)

    sixj = wigner_6j(
        S1p, S2p, Stot,
        S2,  S1,  k,
    )

    prefactor = phase / sqrt(2 * k + 1)
    prefactor *= sqrt(bra_mult1 * bra_mult2)  # Edmonds to CG transition

    return float(prefactor * sixj)


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


def diagram_couplings(supersys_info, X, label, n_frag):
    """
    Spin-recoupling contributions of a diagram for spin-adapted fragment states.

    Returns a list of

        (prefactor, channels)

    where channels[f] = (mult_bra, mult_ket, rank2, restr_ca, path, views) selects the density of fragment f
    (in the fragment order of the diagram) and views (a tuple of (attr template, (positions, sign))) maps
    its canonical layout onto the index order of the accesses in the diagram code (see spin_structure.py).
    The prefactor contains the spin weight of the diagram for these tensors and the recoupling factor of
    the spin-coupled product states.  Non-spin-adapted calculations return [(1.0, None)].
    """
    from .spin_structure import diagram_structure

    if X is None or X.mult[0][0] is None:
        return [(1.0, None)]

    if len(X.mult) not in (1, 2):
        raise NotImplementedError("Spin recoupling currently implemented only for monomer and dimer diagrams.")

    frags, accesses, weights = diagram_structure(label, n_frag)

    def channels(rank2, paths):
        return tuple((X.mult[f][0], X.mult[f][1], rank2, frags[f].restr, paths[f], tuple(sorted(accesses[f].items())))
                     for f in range(len(frags)))

    # ------------------------------------------------------------------
    # Monomer operator (spin scalar)
    # ------------------------------------------------------------------
    if len(X.mult) == 1:
        bra_mult, ket_mult = X.mult[0]
        if bra_mult != ket_mult:
            return []
        return [(w, channels(0, [P0])) for (rank2, P0), w in weights.items()]

    # ------------------------------------------------------------------
    # Dimer operator
    # ------------------------------------------------------------------
    target_mult = supersys_info.target_multiplicity

    if target_mult not in allowed_total_multiplicities([X.mult[0][0], X.mult[1][0]]):
        return []
    if target_mult not in allowed_total_multiplicities([X.mult[0][1], X.mult[1][1]]):
        return []

    #
    # X.mult is in the (permuted) fragment order of the diagram, but the
    # supersystem states are coupled in the absolute fragment order
    # |(S_0 S_1) S>.  The recoupling is therefore evaluated in absolute order,
    # where the operator [T_B x T_A]^0 of a permuted diagram becomes
    # (-1)^(2k) [T_A x T_B]^0.
    #
    permutation = tuple(X._permutation)
    if permutation not in ((0, 1), (1, 0)):
        raise NotImplementedError(f"unexpected fragment permutation {permutation}")
    swapped = permutation == (1, 0)
    mult1, mult2 = (X.mult[1], X.mult[0]) if swapped else (X.mult[0], X.mult[1])

    couplings = []
    for (rank2, P0, P1), w in weights.items():
        prefactor = recoupling_factor(mult1[0], mult2[0], mult1[1], mult2[1], target_mult, rank2 + 1)
        if swapped:
            prefactor *= (-1) ** rank2
        prefactor *= w
        if abs(prefactor) < 1e-12:
            continue
        couplings.append((prefactor, channels(rank2, [P0, P1])))
    return couplings
