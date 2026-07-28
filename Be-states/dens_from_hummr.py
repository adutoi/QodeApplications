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

import json
import numpy as np
import struct
from pathlib import Path
from densities import _tens_wrap
import tensorly as tl

# lible (which is the hummr integral backend) and psi4
# use the same definitions for the 6-31g basis set, but
# order it differently!!!


def _load_tensor(obj):
    """
    Reconstruct a numpy array from the { "dims": [...], "data": [...] } dict
    written by export_tensors.cpp.  All DArrayND types are stored row-major
    (C order) by hummr, so a plain C-order reshape is correct.
    """
    dims = obj["dims"]
    data = np.array(obj["data"], dtype=np.float64)
    return data.reshape(dims, order="C")

# original function
"""
def load_densities_json(filename):
    densities = {}
    # the -2 factor comes from the different definition of the hummr program,
    # which pulls the -1/2 factor of the ERI into the 2p density, since the
    # CAS energy evaluation only contracts those two and adds the 1p and 1 el int
    # contraction to obtain the energy.
    print("apply -2 definition on ccaa in json load")

    with open(filename, "r") as f:
        root = json.load(f)

    for dens_name, dens_group in root.items():  # ca, ccaa, ...
        top_map = {}

        for chg_key, mult_group in dens_group.items():  # "0_1", "1_0", ...
            chg_a, chg_b = map(int, chg_key.split("_"))
            chg_tuple = (chg_a, chg_b)

            mult_map = {}
            for mult_key, state_group in mult_group.items():
                mult_a, mult_b = map(int, mult_key.split("_"))
                mult_tuple = (mult_a, mult_b)

                states_indices = []
                for state_key in state_group.keys():
                    state_a, state_b = map(int, state_key.split("_"))
                    states_indices.append((state_a, state_b))

                state_a_dim = max(a for a, _ in states_indices) + 1
                state_b_dim = max(b for _, b in states_indices) + 1

                # Determine the shape of a single state tensor from the "0_0" entry.
                ref_tensor = _load_tensor(state_group["0_0"])
                data = np.zeros(
                    (state_a_dim, state_b_dim, *ref_tensor.shape),
                    dtype=np.float64
                )

                for state_key, tensor_obj in state_group.items():
                    state_a, state_b = map(int, state_key.split("_"))
                    data[state_a, state_b] = _load_tensor(tensor_obj)

                    if dens_name == "ccaa":
                        data[state_a, state_b] *= -2

                # build remaining densities as transposes
                #TODO: This should happen in a lazy fashion (tensornet) and then be used as such in the
                # final diagram evaluation. This would save a factor of almost 2 in memory.
                rev_op_string = dens_name[::-1].replace("c", "x").replace("a", "c").replace("x", "a")
                if rev_op_string not in densities.keys():
                    densities[rev_op_string] = {}
                if rev_op_string != dens_name or (rev_op_string == dens_name and mult_a != mult_b):
                    indices = tuple(p + 2 for p in range(len(dens_name)))
                    rev_indices = tuple(reversed(indices))
                    # The spin-adapted (rank 0) charge-conserving spin-flip operators,
                    # i.e. T+1 and T-1 are anti-hermitian (T+1.t = -T-1) while the standard
                    # a and c operators are hermitian conjugates instead of anti-hermitian.
                    # The aa operators are also spin adapted, but are hermitian to their
                    # cc counterparts. a and c are not spin-adapted, but obviously also hermitian.
                    # The general rule is that the path is chosen which keeps the amount of
                    # ca spin-flips as low as possible. Keeping this is mind the following
                    # computes the parity factor.
                    #TODO: Check whether the sign flip is correct with the definition of a and c
                    # or if they also need a sign flip.
                    n_excess_a = dens_name.count("a") - dens_name.count("c")
                    ca_mult_change = abs(mult_a - mult_b) - n_excess_a
                    if ca_mult_change <= 0:
                        parity_factor = 1
                    else:
                        if not np.isclose(ca_mult_change / 2, ca_mult_change // 2):
                            raise ValueError("ca_mult_change is not divisible by 2")
                        parity_factor = (-1)**(ca_mult_change // 2)
                    densities[rev_op_string][(chg_b, chg_a)][(mult_b, mult_a)] = _tens_wrap(
                        parity_factor * np.transpose(data, (1, 0, *rev_indices))
                    )

                mult_map[mult_tuple] = _tens_wrap(data)

            top_map[chg_tuple] = mult_map

        densities[dens_name] = top_map

    return densities
"""

def _multiplicity_iterator(second_layer):
    """
    Normalize the multiplicity hierarchy.

    Spin-adapted input:
        multiplicity -> states

    Non-spin-adapted input:
        states

    Returns an iterator over

        (multiplicity_pair, state_group)

    where multiplicity_pair is (None,None) in the non-spin-adapted case.
    """

    #
    # State dictionaries always contain the keys
    #
    #     dims
    #     data
    #
    first_value = next(iter(second_layer.values()))

    if isinstance(first_value, dict) and "dims" in first_value:
        yield (None, None), second_layer
    else:
        for mult_key, state_group in second_layer.items():
            yield tuple(map(int, mult_key.split("_"))), state_group

def _build_reverse_density(
    densities,
    dens_name,
    data,
    charge_pair,
    mult_pair,
):
    """
    Construct the Hermitian (or anti-Hermitian) partner density.
    """

    chg_a, chg_b = charge_pair
    mult_a, mult_b = mult_pair

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
    if rev_op_string == dens_name and mult_a == mult_b:
        return

    indices = tuple(p + 2 for p in range(len(dens_name)))
    rev_indices = tuple(reversed(indices))

    #
    # Non-spin-adapted tensors
    #
    if mult_a is None:
        parity_factor = 1

    #
    # Spin-adapted tensors
    #
    else:
        n_excess_a = dens_name.count("a") - dens_name.count("c")
        ca_mult_change = abs(mult_a - mult_b) - n_excess_a

        if ca_mult_change <= 0:
            parity_factor = 1
        else:
            if ca_mult_change % 2 != 0:
                raise ValueError(
                    "ca_mult_change is not divisible by 2."
                )

            parity_factor = (-1) ** (ca_mult_change // 2)

    densities[rev_op_string].setdefault(
        (chg_b, chg_a),
        {}
    )[(mult_b, mult_a)] = _tens_wrap(
        parity_factor
        * np.transpose(
            data,
            (1, 0, *rev_indices),
        )
    )

def load_densities_json(filename, spin_adapted=None):

    densities = {}

    print("apply -2 definition on ccaa in json load")

    with open(filename, "r") as f:
        root = json.load(f)

    #
    # Decide which hierarchy we expect.
    #
    if spin_adapted is None:

        first_op = next(iter(root.values()))
        first_charge_group = next(iter(first_op.values()))
        first_child = next(iter(first_charge_group.values()))

        spin_adapted = "dims" not in first_child

    densities["spin_adapted"] = spin_adapted

    for dens_name, dens_group in root.items():

        top_map = {}

        for chg_key, second_layer in dens_group.items():

            charge_pair = tuple(map(int, chg_key.split("_")))

            #
            # If the user explicitly requested a hierarchy,
            # verify that the file actually has that hierarchy.
            #
            if spin_adapted:

                first_child = next(iter(second_layer.values()))

                if isinstance(first_child, dict) and "dims" in first_child:
                    raise ValueError(
                        "Expected spin-adapted densities, "
                        "but multiplicity layer was not found."
                    )

            else:

                first_child = next(iter(second_layer.values()))

                if not (isinstance(first_child, dict) and "dims" in first_child):
                    raise ValueError(
                        "Expected non-spin-adapted densities, "
                        "but multiplicity layer was found."
                    )

            mult_map = {}

            for mult_pair, state_group in _multiplicity_iterator(second_layer):

                states = [
                    tuple(map(int, key.split("_")))
                    for key in state_group.keys()
                ]

                state_a_dim = max(i for i, _ in states) + 1
                state_b_dim = max(j for _, j in states) + 1

                ref_tensor = _load_tensor(state_group["0_0"])

                data = np.zeros(
                    (
                        state_a_dim,
                        state_b_dim,
                        *ref_tensor.shape,
                    ),
                    dtype=np.float64,
                )

                for state_key, tensor_obj in state_group.items():

                    i, j = map(int, state_key.split("_"))

                    data[i, j] = _load_tensor(tensor_obj)

                    if dens_name == "ccaa":
                        data[i, j] *= -2

                mult_map[mult_pair] = _tens_wrap(data)

                _build_reverse_density(
                    densities=densities,
                    dens_name=dens_name,
                    data=data,
                    charge_pair=charge_pair,
                    mult_pair=mult_pair,
                )

            top_map[charge_pair] = mult_map

        densities[dens_name] = top_map

    return densities

def save_mos_to_hummr_readable(fname, array):
    array = np.asarray(array, dtype=np.float64)

    if array.ndim != 2:
        raise ValueError("Only 2D arrays are supported")

    n_rows, n_cols = array.shape

    # Ensure row-major (C-order) storage
    data = np.ascontiguousarray(array, dtype=np.float64)

    with open(fname, "wb") as f:
        # Write dimensions (int32)
        f.write(struct.pack("ii", n_rows, n_cols))
        # Write data in row-major order
        f.write(data.tobytes(order="C"))


def load_mos_from_hummr(fname):
    fname = Path(fname)
    if not fname.exists():
        raise FileNotFoundError(f"Matrix file not found: {fname}")

    with open(fname, "rb") as f:
        # Read dimensions
        n_rows, n_cols = struct.unpack("ii", f.read(8))

        # Read matrix data
        data = np.frombuffer(
            f.read(n_rows * n_cols * 8),
            dtype=np.float64
        )

    # Reshape using row-major order
    return data.reshape((n_rows, n_cols), order="C")
