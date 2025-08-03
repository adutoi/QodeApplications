#    (C) Copyright 2023 Marco Bauer
# 
#    This file is part of Qode.
# 
#    Qode is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
# 
#    Qode is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
# 
#    You should have received a copy of the GNU General Public License
#    along with Qode.  If not, see <http://www.gnu.org/licenses/>.
#

import tensorly as tl
#import ray
#import numpy as np

# Maybe try this with tensorly.tenalg.tensordot with
# tensorly.tenalg.set_backend('einsum'), which only does the dispatching different(?),
# so it works, even though the actual backend is set to e.g. pytorch,

# Keeping the backend agnostic, while also feeding tensordot either full or
# decomposed tensors, requires us to write a wrapper! Tensorly can deal with
# decomposed tensors as well as being backend agnostic.

# Example: Tucker (a) with full tensor (b) -> einsum("ijab,ikac->jbkc") = np.tensordot(a, b, axes=([0, 2], [0, 2]))
# Step 1: one by one contract indices of full tensor with decomposed tensors
# ikac -> r_i kac -> r_a r_i kc
# Step 2: contract with core tensor
# r_i r_j r_a r_b, r_a r_i kc -> r_j r_b kc
# Hence, it makes sense to return a decomposed tensor object, with the resulting tensor as core tensor,
# and the remaining decomposed matrices, while the missing decomposed matrices are filled with identities for now

"""
@ray.remote
def tendot_ray(a, b, axes):
    import numpy as np
    return np.tensordot(a, b, axes=axes)

def tendot(a, b, axes):
    #a, b = ray.put(a), ray.put(b)
    #print(type(a), type(b))
    if not isinstance(a, ray.ObjectRef):
        a = ray.put(a)
    if not isinstance(b, ray.ObjectRef):
        b = ray.put(b)
    res = ray.get(tendot_ray.remote(a, b, axes))
    ret = np.copy(res)
    del res
    return ret
"""

def tendot(a, b, axes):
    type_a, type_b = check_decomposed(a), check_decomposed(b)
    if type_a == "full" and type_b == "full":
        return tl.tensordot(a, b, axes=axes)
    elif type_a != "full" and type_b == "full":
        # get ascending axes order for b
        sorted_axes = sorted([i for i in zip(axes[1], axes[0])])
        axes = ([i[1] for i in sorted_axes], [i[0] for i in sorted_axes])
        if type_a == "CP" or type_a == "CP_normalized":
            new_b = tl.copy(b)
            for i, b_axis in enumerate(axes[1]):
                new_b = tl.tensordot(a[1][axes[0][i]], new_b, axes=([0], [b_axis]))
            if type_a == "CP":
                new_b = tl.sum(new_b, axis=tuple([i for i in range(len(axes[0]))]))
            else:
                raise NotImplementedError("try parafac without the normalization, or tucker")
            if tl.ndim(new_b) == 0:
                if len(axes[0]) == len(a[1]):
                    return new_b
            else:
                decomp_mats = [vec for i, vec in enumerate(a[1]) if i not in axes[0]]
                # the last entry is the indexes, which still belong to the CP decomposition
                return (a[0], decomp_mats.append(new_b), "CP_custom", [i for i in range(len(decomp_mats))])
        elif type_a == "CP_custom":
            raise NotImplementedError("tendot for CP_custom has not been implemented! Try Tucker or decompose all tensors with CP!")
        elif type_a == "tucker":
            new_core = tl.copy(b)
            for i, b_axis in enumerate(axes[1]):
                new_core = tl.tensordot(a[1][axes[0][i]], new_core, axes=([0], [b_axis]))  # b is new indices reversed at first and then original b indices
                # TODO: think of something to avoid a contraction with the identity later
            new_core = tl.tensordot(a[0], new_core, axes=(axes[0], [x for x in reversed(range(len(axes[1])))]))
            if tl.ndim(new_core) == 0:
                return new_core
            else:
                decomp_mats = [vec for i, vec in enumerate(a[1]) if i not in axes[0]]
                n_identities = tl.ndim(new_core) - len(decomp_mats)
                identities = [tl.eye(tl.shape(new_core)[i + len(decomp_mats)], dtype=tl.float64) for i in range(n_identities)]
                return (new_core, decomp_mats + identities, "tucker")
        else:
            raise TypeError(f"tensor of type {type_a} passed")
    elif type_a == "full" and type_b != "full":
        #return tendot(b, a, axes=(axes[1], axes[0]))  # this is wrong, due to the different index ordering of the resulting tensor, and a transpose requires unnecessary cpu time
        # get ascending axes order for a
        sorted_axes = sorted([i for i in zip(axes[0], axes[1])])
        axes = ([i[0] for i in sorted_axes], [i[1] for i in sorted_axes])
        if type_b == "CP" or type_b == "CP_normalized":
            new_a = tl.copy(a)
            for i, a_axis in enumerate(axes[0]):
                new_a = tl.tensordot(new_a, b[1][axes[1][i]], axes=([a_axis - i], [0]))
            if type_a == "CP":
                new_a = tl.sum(new_a, axis=tuple([i + (tl.ndim(new_a) - len(axes[0])) for i in range(len(axes[0]))]))
            else:
                raise NotImplementedError("try parafac without the normalization, or tucker")
            if tl.ndim(new_a) == 0:
                if len(axes[1]) == len(b[1]):
                    return new_a
            else:
                decomp_mats = [vec for i, vec in enumerate(b[1]) if i not in axes[1]]
                # the last entry is the indexes, which still belong to the CP decomposition
                return (b[0], [new_a] + decomp_mats, "CP_custom", [i + (tl.ndim(new_a) - len(axes[0])) for i in range(len(decomp_mats))])
        elif type_a == "CP_custom":
            raise NotImplementedError("tendot for CP_custom has not been implemented! Try Tucker or decompose all tensors with CP!")
        elif type_b == "tucker":
            new_core = tl.copy(a)
            for i, a_axis in enumerate(axes[0]):
                new_core = tl.tensordot(new_core, b[1][axes[1][i]], axes=([a_axis - i], [0]))  # a is original a indices and then new b indices in correct order
                # TODO: think of something to avoid a contraction with the identity later
            new_core = tl.tensordot(new_core, b[0], axes=([x + (tl.ndim(new_core) - len(axes[0])) for x in range(len(axes[0]))], axes[1]))
            if tl.ndim(new_core) == 0:
                return new_core
            else:
                decomp_mats = [vec for i, vec in enumerate(b[1]) if i not in axes[1]]
                n_identities = tl.ndim(new_core) - len(decomp_mats)
                identities = [tl.eye(tl.shape(new_core)[i], dtype=tl.float64) for i in range(n_identities)]
                return (new_core, identities + decomp_mats, "tucker")
        else:
            raise TypeError(f"tensor of type {type_b} passed")
    else:
        sorted_axes = sorted([i for i in zip(axes[1], axes[0])])
        axes = ([i[1] for i in sorted_axes], [i[0] for i in sorted_axes])
        #try:
        transforms = [tl.tensordot(a[1][axes[0][i]], b[1][axes[1][i]], axes=([0], [0])) for i in range(len(axes[0]))]
        #except RuntimeError:
        #    for i in range(len(axes[0])):
        #        print(tl.shape(a[1][axes[0][i]]), tl.shape(b[1][axes[1][i]]))
        #        print(a[1][axes[0][i]])
        #        print(b[1][axes[1][i]])
        #    transforms = [tl.tensordot(a[1][axes[0][i]], b[1][axes[1][i]], axes=([0], [0])) for i in range(len(axes[0]))]
        if type_a == "CP" and type_b == "CP":
            factor = tl.prod([tl.sum(i) for i in transforms])  # for CP_normalized with CP_normalized just contract transforms with corresponding core
            mats_a = [a[1][i] for i in range(len(a[1])) if i not in axes[0]]
            mats_b = [b[1][i] for i in range(len(b[1])) if i not in axes[1]]
            mats = mats_a + mats_b
            if len(mats_a) + len(mats_b) == 0:
                return factor
            else:
                return (a[0], [factor * i for i in mats], "CP")
        if type_a == "tucker" and type_b == "tucker":
            b_core = tl.copy(b[0])
            for i, mat in enumerate(transforms):
                b_core = tl.tensordot(mat, b_core, axes=([1], [axes[1][i]]))  # b is new indices reversed at first and then original b indices
            core = tl.tensordot(a[0], b_core, axes=(axes[0], [x for x in reversed(range(len(axes[1])))]))
            if tl.ndim(core) == 0:
                return core
            else:
                mats_a = [a[1][i] for i in range(len(a[1])) if i not in axes[0]]
                mats_b = [b[1][i] for i in range(len(b[1])) if i not in axes[1]]
                return (core, mats_a + mats_b, "tucker")
        else:
            raise NotImplementedError(f"tendot for {type_a} with {type_b} has not been implemented! Try Tucker or decompose all tensors with CP!")


# Checking through the decomposed tensors via class instances would be nice, but it seems that the tl.decomposed classes
# can't be populated from decomposed tensors, so we would need to either build a custom class for the returned tensor objects,
# or build an ugly check_decomposed function, since we only need to deal with tendot to evaluate the diagrams and preprocess
# the tensors

# If we also want CP, we need a custom partial_decomposed class

def check_decomposed(tensor):
    if isinstance(tensor, tl.tucker_tensor.TuckerTensor):
         return "tucker"
    elif isinstance(tensor, tl.cp_tensor.CPTensor):
        if tensor[0][0] == 1.:
            return "CP"
        else:
            return "CP_normalized"
    elif type(tensor) == tuple:  # these are the custom (partially) decomposed objects
        if len(tensor) != 3 and len(tensor) != 4:
            raise ValueError(f"tensor of len {len(tensor)} given, but needs to be of len 3 or 4 for the custom classes")
        return tensor[2]
        #try:
        #    if tl.ndim(tensor[0]) == 1 and tensor[0][0] == 1.:
        #        return "CP"
        #    else:
        #        return "tucker"
        #except ValueError:  # tl.tensor type does not exist (the type is the backend tensor type)
        #    return TypeError(f"tendot received tuple with tensor of type {type(tensor)}, instead of tl.tensor")
    else:  # get backend here and ask if tensor is that type, to make it is full tensor...me might not cover all decomp schemes
        return "full"



