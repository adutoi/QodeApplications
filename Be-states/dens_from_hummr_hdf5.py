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

import h5py
import numpy as np
import struct
from pathlib import Path
from densities import _tens_wrap
import tensorly as tl

# lible (which is the hummr integral backend) and psi4
# use the same definitions for the 6-31g basis set, but
# order it differently!!!

def load_densities_hdf5(filename):
    densities = {}
    # the -2 factor comes from the different definition of the hummr program,
    # which pulls the -1/2 factor of the ERI into the 2p density, since the
    # CAS energy evaluation only contracts those two and adds the 1p and 1 el int
    # contraction to obtain the energy.
    print("apply -2 definition on ccaa in hdf5 load")
    #print("double check the convention for the higher than one particle components, e.g. 2p dens pqsr to pqrs, as currently the pqrs 2p density is swapped to pqsr")

    with h5py.File(filename, "r") as f:
        for dens_name in f.keys():  # ca, ccaa, ...
            dens_group = f[dens_name]
            top_map = {}

            for chg_key in dens_group.keys():  # "0_1", "1_0", ...
                chg_a, chg_b = map(int, chg_key.split("_"))
                chg_tuple = (chg_a, chg_b)

                state_group = dens_group[chg_key]
                states_indices = []

                for state_key in state_group.keys():
                    state_a, state_b = map(int, state_key.split("_"))
                    states_indices.append((state_a, state_b))

                state_a_dim = max(a for a, _ in states_indices) + 1
                state_b_dim = max(b for _, b in states_indices) + 1

                # this would be correct, but only if the core is separated
                # in the xr calculation, which is not yet implemented

                data = np.zeros(
                    (state_a_dim, state_b_dim, *state_group["0_0"].shape),
                    dtype=np.float64
                )

                for state_key in state_group.keys():
                    state_a, state_b = map(int, state_key.split("_"))
                    # make sure to apply the same ordering for the MOs as when
                    # loading the MOs
                    data[state_a, state_b] = np.array(
                        state_group[state_key]
                    )

                    #if dens_name == "ca" and (chg_a == chg_b and state_a == state_b):
                    #    data[state_a, state_a] = data[state_a, state_a]

                    if dens_name == "ccaa":
                        #print("apply -1 definition on ccaa in hdf5 load")
                        data[state_a, state_b] *= -2
                        #pass
                        # ccaa has the wrong global sign and has only been corrected for the
                        # 1p contribution (p, q, q, r), with delta(q, q) * 1p(p, r), but that
                        # makes it far from symmetric and naively symmetrizing would mess things up
                        #data[state_a, state_b] *= -2  # correct for sign and division by 2 in hummr
                        # hummr symmetrizes like this, because it uses a different convention
                        #data[state_a, state_b] = 0.5 * (
                        #      np.transpose(data[state_a, state_b], (0,2,1,3))
                        #    + np.transpose(data[state_a, state_b], (1,2,0,3))
                        #)

                        # hummr seems to be using a pqrs, instead of pqsr convention
                        # it might make more sense to write down contractions as pqrs,pqrs and not make use of antisymmetry
                        # to have positive signs for the terms, because with the hummr tensors the antisymmetry is not
                        # provided for the densities! Hence, this needs to be written down in a consistent manner,
                        # without applying antisymmetry after the evaluation to working equations.
                        #data[state_a, state_b] = np.swapaxes(data[state_a, state_b], 2, 3)

                        # reintroducing 1p correction from hummr??????????????????????????????????????????????????????
                        # 1p contributions are subtracted later, but I still think this is wrong
                        #from qode.math.tensornet import raw
                        #ca_dens = raw(densities["ca"][(chg_a, chg_b)][state_a, state_b, :, :])
                        #for i in range((data.shape)[2]): 
                        #    data[state_a, state_b][:, i, i, :] += 0.5 * ca_dens
                        #    data[state_a, state_b][:, i, i, :] *= 2

                        #from qode.math.tensornet import raw
                        #ca_dens = raw(densities["ca"][(chg_a, chg_b)][state_a, state_b, :, :])
                        #ca_eigvals, ca_eigvecs = np.linalg.eigh(ca_dens)
                        #for i in range((data.shape)[2]):
                        #    data[state_a, state_b][:, i, :, i] += 0.5 * ca_eigvals[-(i+1)] * ca_dens

                        #from qode.many_body.fermion_field.field_op import antisymmetrize
                        #antisymmetrize("ccaa", data[state_a, state_b])
                        #data[state_a, state_b] *= 1/4

                        #tmp = data[state_a, state_b]
                        #data[state_a, state_b] = (1/4.) * ((tmp-np.swapaxes(tmp,2,3)) - np.swapaxes(tmp-np.swapaxes(tmp,2,3),0,1))


                """
                tmp_ten_dims = state_group["0_0"].shape
                n_orb = tmp_ten_dims[-1] + 1  # spatial orbs

                # here we not only have to reintroduce the trivial frozen core
                # part, but also set up the density as from an unrestricted reference
                data = np.zeros(
                    (state_a_dim, state_b_dim, *[2 * (i+1) for i in tmp_ten_dims]),
                    dtype=np.float64
                )

                for state_key in state_group.keys():
                    state_a, state_b = map(int, state_key.split("_"))
                    #data[state_a, state_b] = np.array(
                    # occupy correct blocks
                    print("reshuffling of the basis is also required!!!!!! see reshuffled MOs")
                    data[(state_a, state_b) + (slice(1, n_orb),) * len(tmp_ten_dims)] = np.array(
                        state_group[state_key]
                    )
                    print("this is only correct for ca!!! even for ccaa this is missing e.g. c_a c_b a_a a_b")
                    data[(state_a, state_b) + (slice(n_orb + 1, 2 * n_orb),) * len(tmp_ten_dims)] = np.array(
                        state_group[state_key]
                    )
                    # include trivial part for frozen core orbitals
                    if dens_name == "ca":
                        # the standard hummr density builder has this factor of two in here,
                        # but e.g. for ccaa this factor is taken care of
                        data[state_a, state_b] *= 0.5
                        for i in range(state_a_dim):
                            data[i, i, 0, 0] = 1
                            data[i, i, n_orb, n_orb] = 1
                    elif dens_name == "ccaa":
                        for i in range(state_a_dim):
                            # this only works if ca has already been taken care of!
                            one_p_part = densities["ca"][chg_tuple][i, i]
                            for fo in [0, n_orb]:
                                data[i, i, fo, :, fo, :] = -one_p_part
                                data[i, i, fo, :, :, fo] = one_p_part
                                data[i, i, :, fo, :, fo] = -one_p_part
                                data[i, i, :, fo, fo, :] = one_p_part
                """
                                
                # build remaining densities as transposes
                #print("antisymmetry is not provided, but what about hermiticity? If not, the herm conj build of the missing densities is wrong! -> this should be fulfilled")
                rev_op_string = dens_name[::-1].replace("c","x").replace("a","c").replace("x","a")
                if rev_op_string not in densities.keys():
                    densities[rev_op_string] = {}
                if rev_op_string != dens_name:
                    indices = tuple(p+2 for p in range(len(dens_name)))
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
