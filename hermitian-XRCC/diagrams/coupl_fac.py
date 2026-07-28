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
from sympy.physics.wigner import wigner_9j


def mult_to_spin(mult):
    """Convert multiplicity (2S+1) to spin."""
    return Sym(mult - 1) / 2


def recoupling_factor(
    bra_mult1,
    bra_mult2,
    ket_mult1,
    ket_mult2,
    total_mult,
    tensor_mult,
):
    """
    Spin recoupling coefficient for two spin-adapted monomer transition densities.

    Parameters
    ----------
    bra_mult1, bra_mult2
        Multiplicities of the monomer states in the bra.

    ket_mult1, ket_mult2
        Multiplicities of the monomer states in the ket.

    total_mult
        Total multiplicity of the dimer state (identical for bra and ket).

    tensor_mult
        Multiplicity of the tensor operator:

            1 : singlet pair density (k = 0)
            2 : single creation/annihilation (k = 1/2)
            3 : spin density (k = 1)

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

    ninej = wigner_9j(
        S1p, S2p, Stot,
        k,   k,   Sym(0),
        S1,  S2,  Stot, prec=64
    )
    prefactor = sqrt(2*Stot + 1)   # Delta_0(S_tot, S_tot)
    return float(prefactor * ninej)

def tensor_multiplicities_from_label(label):
    """
    Determine the allowed tensor multiplicities from the
    operator topology encoded in the diagram label.

    The diagram labels encode how every field operator is
    distributed over the fragments.

        s : ca
        t : ca
        u : core,c,a
        v : ccaa

    Local c-a pairs belonging to the same tensor are already
    internally spin coupled and therefore do not contribute
    an independent spin-1/2 object.

    Returns
    -------
    tuple[int]

        Allowed tensor multiplicities

            1 -> k = 0
            2 -> k = 1/2
            3 -> k = 1
            ...
    """

    free_legs = 0

    #
    # parse every tensor appearing in the label
    #
    for tensor, digits in re.findall(r"([stuv])([01]+)", label):

        #
        # assign operators to the fragment indices encoded
        # in the label
        #
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

            #
            # first digit = core
            #
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

        #
        # collect operators by fragment
        #
        fragment_ops = {}

        for op, frag in ops:

            fragment_ops.setdefault(frag, []).append(op)

        #
        # remove local c-a pairs
        #
        for ops_here in fragment_ops.values():

            nc = ops_here.count("c")
            na = ops_here.count("a")

            paired = min(nc, na)

            free_legs += (nc - paired)
            free_legs += (na - paired)

    #
    # every two uncoupled spin-1/2 objects define
    # one unit of tensor rank
    #
    kmax = free_legs // 2

    #
    # convert
    #
    # k = 0     -> mult 1
    # k = 1/2   -> mult 2
    # k = 1     -> mult 3
    #
    return tuple(range(1, 2 * kmax + 2))

def allowed_total_multiplicities(mults):
    """
    Parameters
    ----------
    mults
        Iterable of monomer multiplicities.

    Returns
    -------
    set[int]
        Allowed coupled multiplicities.
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

                new_allowed.add(
                    int(2 * S + 1)
                )

                S += 1

        allowed = new_allowed

    return allowed

def tensor_Dmult(tensor_mult, Dchgs):
    """
    Map a tensor multiplicity onto the corresponding
    multiplicity change.
    """

    if tensor_mult == 1:
        return (0, 0)

    if tensor_mult == 2:
        return Dchgs

    step = tensor_mult - 1

    Dmult = []

    for dchg in Dchgs:

        if dchg > 0:
            Dmult.append(step)

        elif dchg < 0:
            Dmult.append(-step)

        else:
            Dmult.append(0)

    return tuple(Dmult)

def diagram_couplings(supersys_info, X, Dchgs, tensor_mults):
    """
    Return all spin-recoupling contributions for this diagram.

    Parameters
    ----------
    supersys_info
        Contains the requested total multiplicity.

    X
        Fragment resolver for one particular permutation.

    Dchgs
        Charge changes of the active fragments.

    tensor_mults
        Tensor multiplicities allowed by the contraction topology.

    Returns
    -------
    list[(prefactor, Dmult)]

        Empty list
            Diagram forbidden.

        Otherwise
            Each tuple contains one recoupling prefactor together with
            the multiplicity differences required from the transition
            densities.
    """

    #
    # Non-spin-adapted calculations.
    #
    if X.mult[0][0] is None:
        return [(1.0, None)]

    target_mult = supersys_info.target_multiplicity

    #
    # The complete subsystem represented by X already consists only
    # of the fragments participating in this diagram.
    #
    if len(X.mult) != 2:
        raise NotImplementedError(
            "Spin recoupling currently implemented only for dimer diagrams."
        )

    #
    # Check whether the requested supersystem multiplicity can be
    # formed from the bra and ket fragment multiplicities.
    #
    bra_allowed = allowed_total_multiplicities(
        [mult_i for mult_i, _ in X.mult]
    )

    ket_allowed = allowed_total_multiplicities(
        [mult_j for _, mult_j in X.mult]
    )

    if target_mult not in bra_allowed:
        return []

    if target_mult not in ket_allowed:
        return []

    (bra_mult1, ket_mult1), (bra_mult2, ket_mult2) = X.mult

    couplings = []

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

        Dmult = tensor_Dmult(Dchgs, tensor_mult)

        couplings.append((prefactor, Dmult))

    return couplings
