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
from ..dens_util import build_reverse_density_spin_adapt, multiplicity_iterator


def _load_tensor(obj):
    """
    Reconstruct a numpy array from the { "dims": [...], "data": [...] } dict
    written by export_tensors.cpp.  All DArrayND types are stored row-major
    (C order) by hummr, so a plain C-order reshape is correct.
    """
    dims = obj["dims"]
    data = np.array(obj["data"], dtype=np.float64)
    return data.reshape(dims, order="C")

def load_densities_json(filename, spin_adapted=None):
    """
    Load transition densities exported by hummr.

    Internal hierarchy
    ------------------
    densities[density_name][charge_pair][channel_key]

    where

        charge_pair =
            (charge_bra, charge_ket)

        channel_key =
            (mult_bra, mult_ket, rank2, restr_ca)

    and rank2 = 2*k.

    For non-spin-adapted densities the channel key is

        (None, None, None, 0)

    so that downstream code can always expect the same four-part
    channel description.
    """

    densities = {}

    print("apply -2 definition on ccaa in json load, note that this definition" \
    "is only in line with the internal hummr definition, but not with the" \
    "ccaa build of the density builder from the XR module.")

    with open(filename, "r") as f:
        root = json.load(f)

    #
    # Determine the hierarchy automatically unless explicitly specified.
    #
    if spin_adapted is None:

        first_op = next(iter(root.values()))
        first_charge_group = next(iter(first_op.values()))
        first_child = next(iter(first_charge_group.values()))

        spin_adapted = not (
            isinstance(first_child, dict)
            and "dims" in first_child
        )

    densities["spin_adapted"] = spin_adapted

    for dens_name, dens_group in root.items():

        #
        # Do not treat our own metadata as a density.
        #
        if dens_name == "spin_adapted":
            continue

        top_map = {}

        for chg_key, second_layer in dens_group.items():

            charge_pair = tuple(map(int, chg_key.split("_")))

            #
            # Verify that the supplied hierarchy agrees with the
            # requested mode.
            #
            first_child = next(iter(second_layer.values()))

            child_is_tensor = (
                isinstance(first_child, dict)
                and "dims" in first_child
            )

            if spin_adapted and child_is_tensor:
                raise ValueError(
                    "Expected spin-adapted densities, "
                    "but multiplicity/tensor-channel layer "
                    "was not found."
                )

            if not spin_adapted and not child_is_tensor:
                raise ValueError(
                    "Expected non-spin-adapted densities, "
                    "but multiplicity/tensor-channel layer "
                    "was found."
                )

            channel_map = {}

            for channel, state_group in multiplicity_iterator(
                second_layer
            ):

                states = [
                    tuple(map(int, key.split("_")))
                    for key in state_group.keys()
                ]

                if not states:
                    raise ValueError(
                        f"Density '{dens_name}' contains an empty "
                        f"state group for channel {channel}."
                    )

                state_a_dim = max(
                    i for i, _ in states
                ) + 1

                state_b_dim = max(
                    j for _, j in states
                ) + 1

                #
                # All state tensors belonging to one channel must
                # have the same orbital tensor shape.
                #
                ref_tensor = _load_tensor(
                    state_group["0_0"]
                )

                data = np.zeros(
                    (
                        state_a_dim,
                        state_b_dim,
                        *ref_tensor.shape,
                    ),
                    dtype=np.float64,
                )

                for state_key, tensor_obj in state_group.items():

                    i, j = map(
                        int,
                        state_key.split("_"),
                    )

                    data[i, j] = _load_tensor(
                        tensor_obj
                    )

                    #
                    # hummr's ccaa convention differs by -1/2.
                    #
                    if dens_name == "ccaa":
                        data[i, j] *= -2

                channel_map[channel] = _tens_wrap(data)

                #
                # Generate the reversed density immediately.
                #
                build_reverse_density_spin_adapt(
                    densities=densities,
                    dens_name=dens_name,
                    data=data,
                    charge_pair=charge_pair,
                    channel=channel,
                )

            top_map[charge_pair] = channel_map

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