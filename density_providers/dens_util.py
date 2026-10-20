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
from sympy import Rational
from sympy.physics.quantum.cg import CG
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

        (mult_bra, mult_ket, rank2, restr_ca, path)

    for spin-adapted densities (path: spin coupling path, see spin_tensors.py) and

        (None, None, None, None, None)

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
        yield (None, None, None, None, None), second_layer

    else:
        for channel, state_group in second_layer.items():
            #channel = tuple(map(int, channel.split("_")))

            if len(channel) != 5:
                raise ValueError(
                    "Expected density channel "
                    "(mult_bra, mult_ket, rank2, restr_ca, path), "
                    f"got {channel!r}"
                )

            yield channel, state_group


def adjoint_name(dens_name):
    """name of the adjoint operator string, e.g. caa -> cca"""
    return dens_name[::-1].replace("c", "x").replace("a", "c").replace("x", "a")

def reversed_densities_spin_adapt(dens):
    """
    Adjoint partners of spin-adapted densities in the CG convention,

        <I S_I S_I| T^K_q |J S_J S_J> = <S_J S_J K q|S_I S_I> rho_IJ,

    dens[name][(chg_bra, chg_ket)][(mult_bra, mult_ket, rank2, restr_ca, path)] = ndarray(n_bra, n_ket, orbitals...)
    with the tensors of spin_tensors.py.  For the density type X, the adjoint of type Y,

        X_q(P)^dagger = sum_P' c(P,P') Y_{-q}(P')   (legs permuted, spin_tensors.reversal)

    gives  rho^X_BA(P) = sum_P' c(P,P') <S_B S_B K -q|S_A S_A> / <S_A S_A K q|S_B S_B> rho^Y_AB(P').
    Generated are all channels of the adjoint types of non-self-adjoint types, and the mult_bra > mult_ket
    channels of the self-adjoint ones (the others are computed directly).  Returns a dict of the same structure.
    """
    from spin_tensors import paths, reversal, cg2
    out = {}
    for name_y, by_chg in dens.items():
        name_x = adjoint_name(name_y)
        n_c_x, n_a_x = name_x.count("c"), name_x.count("a")
        self_adjoint = name_x == name_y
        for (chg_a, chg_b), by_channel in by_chg.items():
            groups = {}
            for (mult_a, mult_b, rank2, restr, path), data in by_channel.items():
                if self_adjoint and mult_a >= mult_b:
                    continue
                groups.setdefault((mult_a, mult_b, rank2, restr), {})[path] = data
            for (mult_a, mult_b, rank2, restr), by_path in groups.items():
                Sa2, Sb2 = mult_a - 1, mult_b - 1
                q2 = Sb2 - Sa2                     # of X between bra B and ket A
                denom = cg2(Sa2, Sa2, rank2, q2, Sb2, Sb2)
                if abs(denom) < 1e-12:
                    continue
                ratio = cg2(Sb2, Sb2, rank2, -q2, Sa2, Sa2) / denom
                for path_x in paths(n_c_x, n_a_x, rank2, restr):
                    perm, coefs = reversal(n_c_x, n_a_x, restr, path_x, q2)
                    axes = (1, 0) + tuple(2 + int(y) for y in np.argsort(perm))
                    result = None
                    for path_y, c in coefs.items():
                        if path_y not in by_path:
                            raise KeyError(f"reversal of {name_y} {(chg_a, chg_b)} {(mult_a, mult_b, rank2, restr)} "
                                           f"requires the missing path {path_y}")
                        term = c * ratio * np.transpose(by_path[path_y], axes)
                        result = term if result is None else result + term
                    if result is not None:
                        out.setdefault(name_x, {}).setdefault((chg_b, chg_a), {})[(mult_b, mult_a, rank2, restr, path_x)] = result
    return out

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
        # The densities are stored in the CG convention for proper
        # spherical tensors, i.e. for the high-spin components
        #
        #     <I S_I S_I| T^k_q |J S_J S_J> = <S_J S_J k q|S_I S_I> rho_IJ
        #
        # with q = S_I - S_J (the annihilator tensor is at_{+1/2} = a_beta,
        # at_{-1/2} = -a_alpha). The partner density of the adjoint operator
        # string then is
        #
        #     rho'_JI = eta (-1)^(k-q) <S_J S_J k q|S_I S_I> / <S_I S_I k -q|S_J S_J> rho_IJ
        #
        # where eta accounts for the reversed coupling order of the operator
        # string.  This was verified against reference densities from an exact
        # Fock-space FCI (machine precision for all channels of an H2 fragment).
        #
        S_I = Rational(mult_a - 1, 2)
        S_J = Rational(mult_b - 1, 2)
        k = Rational(rank2, 2)
        q = S_I - S_J

        if dens_name in ("a", "caa"):
            eta = 1
        elif dens_name == "aa":
            eta = (-1) ** int(1 - k)
        elif dens_name == "ca":
            eta = (-1) ** int(k)
        else:
            raise NotImplementedError(
                f"the adjoint of the spin-adapted density {dens_name} (channel {channel}) "
                "has not been derived/validated yet"
            )

        parity_factor = float(
            eta * (-1) ** int(k - q)
            * CG(S_J, S_J, k, q, S_I, S_I).doit()
            / CG(S_I, S_I, k, -q, S_J, S_J).doit()
        )

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
