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

        for chg_key, state_group in dens_group.items():  # "0_1", "1_0", ...
            chg_a, chg_b = map(int, chg_key.split("_"))
            chg_tuple = (chg_a, chg_b)

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
            rev_op_string = dens_name[::-1].replace("c", "x").replace("a", "c").replace("x", "a")
            if rev_op_string not in densities.keys():
                densities[rev_op_string] = {}
            if rev_op_string != dens_name:
                indices = tuple(p + 2 for p in range(len(dens_name)))
                rev_indices = tuple(reversed(indices))
                densities[rev_op_string][(chg_b, chg_a)] = _tens_wrap(
                    np.transpose(data, (1, 0, *rev_indices))
                )

            top_map[chg_tuple] = _tens_wrap(data)

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
