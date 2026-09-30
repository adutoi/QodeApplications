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

import numpy as np
from densities import _tens_wrap


def multiplicity_iterator(second_layer):
    """
    Normalize the multiplicity/channel hierarchy.

    Spin-adapted input:
        (multiplicity_bra, multiplicity_ket, k, restr_ca) -> states

    Non-spin-adapted input:
        states

    Returns
    -------
    iterator over
        (channel, state_group)

    where channel is

        (mult_bra, mult_ket, k, restr_ca)

    for spin-adapted densities and

        (None, None, None, None)

    for non-spin-adapted densities.
    """

    #
    # State dictionaries always contain the keys
    #
    #     dims
    #     data
    #
    first_value = next(iter(second_layer.values()))

    if isinstance(first_value, dict) and "dims" in first_value:
        yield (None, None, None, None), second_layer

    else:
        for channel, state_group in second_layer.items():
            #channel = tuple(map(int, channel.split("_")))

            if len(channel) != 4:
                raise ValueError(
                    "Expected density channel "
                    "(mult_bra, mult_ket, k, restr_ca), "
                    f"got {channel!r}"
                )

            yield channel, state_group


def build_reverse_density_spin_adapt(
    densities,
    dens_name,
    data,
    charge_pair,
    channel,
):
    """
    Construct the Hermitian/anti-Hermitian partner density.

    Parameters
    ----------
    channel
        (mult_bra, mult_ket, rank2, restr_ca)
    """

    chg_a, chg_b = charge_pair

    mult_a, mult_b, rank2, restr_ca = channel

    rev_op_string = (
        dens_name[::-1]
        .replace("c", "x")
        .replace("a", "c")
        .replace("x", "a")
    )

    if rev_op_string not in densities:
        densities[rev_op_string] = {}

    #
    # Don't regenerate self-adjoint operators.
    #
    if (
        rev_op_string == dens_name
        and mult_a == mult_b
    ):
        return

    indices = tuple(p + 2 for p in range(len(dens_name)))
    rev_indices = tuple(reversed(indices))

    #
    # Non-spin-adapted tensors.
    #
    if mult_a is None:
        parity_factor = 1

    #
    # Spin-adapted tensors.
    #
    else:
        #
        # rank2 = 2*k.
        #
        k = rank2 / 2.0

        #
        # q = Delta S = S_bra - S_ket.
        #
        q = (mult_b - mult_a) / 2.0

        exponent = k - q

        if not np.isclose(exponent, round(exponent)):
            raise ValueError(
                "Invalid tensor-channel parity exponent: "
                f"k={k}, q={q}, exponent={exponent}."
            )

        parity_factor = (-1) ** int(round(exponent))

    reverse_channel = (
        mult_b,
        mult_a,
        rank2,
        restr_ca,
    )

    #print(rev_op_string, (chg_b, chg_a), reverse_channel)

    # since np.transpose is a view and _tens_wrap builds a tensornet
    # tensor, the tensor is not copied this way. Is this true, or is
    # it still copied during the _tens_wrap?
    # TODO: Check the above assumption.
    densities[rev_op_string].setdefault(
            (chg_b, chg_a),
            {}
        )[reverse_channel] = parity_factor * _tens_wrap(
            np.transpose(
                data,
                (1, 0, *rev_indices),
            )
        )
