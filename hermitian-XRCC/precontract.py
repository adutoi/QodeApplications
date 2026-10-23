#    (C) Copyright 2024 Anthony D. Dutoi
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
from collections import OrderedDict
import numpy
import tensorly
from qode.util.dynamic_array import dynamic_array, cached
from qode.math.tensornet     import evaluate, raw, tl_tensor
from diagrams.build_diagram  import resolve_density
from hummr_dens.streamed     import StreamedDensityType

# pre-contracted intermediates with at most this many orbital indices (besides the bra and ket state indices) are
# cached for the lifetime of the precontract object; larger ones (only a few XR[2] dimer terms) are kept in an LRU
# cache of at most LARGE_CACHE_BYTES (shared by all precontract objects), from which XR_term.dimer_matrix releases
# those of a fragment transition once all blocks using it are done (release_intermediates)
MAX_CACHED_ORB_INDICES = 3
LARGE_CACHE_BYTES = 1 << 30

_large = OrderedDict()    # (id(storage), indices) -> (storage, bytes, owner, fragment, transition)
_large_bytes = 0

def set_large_cache_limit(n_bytes):
    """byte limit of the cache of the large intermediates (0: not cached, computed for every diagram call)"""
    global LARGE_CACHE_BYTES
    LARGE_CACHE_BYTES = int(n_bytes)
    _trim_large()

def _remove_large(key):
    global _large_bytes
    storage, n_bytes, _, _, _ = _large.pop(key)
    storage.pop(key[1], None)
    _large_bytes -= n_bytes

def _trim_large():
    while _large and _large_bytes > LARGE_CACHE_BYTES:
        _remove_large(next(iter(_large)))

def release_large(owner, fragment, transition):
    """drops the large intermediates of the density of fragment for the transition ((chg_i, mult_i), (chg_j, mult_j))"""
    for key in [key for key, (_, _, o, f, t) in _large.items() if o is owner and f == fragment and t == transition]:
        _remove_large(key)

class cached_if_small(object):
    def __init__(self, rule, owner=None, fragment=None):
        self.rule = rule
        self.storage = {}
        self.owner, self.fragment = owner, fragment
    def __call__(self, *indices):
        global _large_bytes
        if indices in self.storage:
            key = (id(self.storage), indices)
            if key in _large:
                _large.move_to_end(key)
            return self.storage[indices]
        result = self.rule(*indices)
        if result is None or len(result.shape) - 2 <= MAX_CACHED_ORB_INDICES:
            self.storage[indices] = result
        elif self.owner is not None and LARGE_CACHE_BYTES > 0:
            # indices: (chg_i, chg_j, mult_bra, mult_ket, rank2, restr_ca, path, views)
            key = (id(self.storage), indices)
            n_bytes = 8 * int(numpy.prod(result.shape))
            self.storage[indices] = result
            _large[key] = (self.storage, n_bytes, self.owner, self.fragment,
                           ((indices[0], indices[2]), (indices[1], indices[3])))
            _large_bytes += n_bytes
            _trim_large()
        return result



#"cca#pqX_U#pq##"
#cca(p,q,0) @ V(p,q,1,2)

def precontract(densities, integrals, timings):
    n_frag = len(densities)
    owner = object()    # identifies the large intermediates of this object in the shared cache

    def mother_rule(label):
        n_indices = label.count("#")

        def contract_rho_int(*indices):
            rho_label, int_label = label.split("_")
            rho_type, rho_idxstr = rho_label.split("#")
            int_type, int_idxstr = int_label[0], int_label[1:]
            rho_indices = []
            int_indices = []
            int_blocks  = []
            block_idx = 0
            free_idx  = 2
            if int_type=="U":
                block_idx += 1
                int_blocks += [indices[block_idx]]
                int_idxstr = int_idxstr[1:]
            for idx in rho_idxstr:
                if idx=="X":
                    rho_indices += [free_idx]
                    free_idx += 1
                else:
                    rho_indices += [idx]
            for idx in int_idxstr:
                if idx=="#":
                    int_indices += [free_idx]
                    free_idx += 1
                    block_idx += 1
                    int_blocks += [indices[block_idx]]
                else:
                    int_indices += [idx]
                    int_blocks += [indices[0]]
            int_blocks = tuple(int_blocks)
            if int_type=="S":  ints = integrals[int_blocks]    # makes the calling signature inhomogeneous to force error otherwise
            if int_type=="T":  ints = integrals.T[int_blocks]
            if int_type=="U":  ints = integrals.U[int_blocks]
            if int_type=="V":  ints = integrals.V[int_blocks]
            densities_m = densities[indices[0]]
            Dchg = rho_type.count("a") - rho_type.count("c")    # get rid of this and allow the exception

            def contract_rho_int_m(chg_i,chg_j,*channel):
                # channel: (mult_bra, mult_ket, rank2, restr_ca, path, views) for spin-adapted densities, see
                # coupl_fac.diagram_couplings (the view maps the density onto the index order of this label)
                if chg_i-chg_j==Dchg:
                    if isinstance(densities_m[rho_type], StreamedDensityType):
                        # large type: contraction evaluated by the provider (hummr_dens/streamed.py)
                        if channel is None or len(channel) < 6:
                            raise NotImplementedError("streamed densities require spin-adapted channels with views")
                        positions, sign = dict(channel[5])[label]
                        canonical = [rho_indices[p] for p in positions]    # see build_diagram.density_view
                        timings.start()
                        try:
                            result = densities_m[rho_type].provider.contract(rho_type, (chg_i, chg_j), tuple(channel[:5]),
                                                                             canonical, [(raw(ints), int_indices)])
                        except KeyError as err:
                            raise RuntimeError(f"missing density {rho_type} for charges {(chg_i,chg_j)} and channel {channel[:5]}: {err}")
                        if sign != 1:
                            result *= sign    # in place, the provider returns new arrays
                        result = tl_tensor.init(result)
                        timings.record(label)
                        return result
                    try:
                        rho = resolve_density(densities_m[rho_type][chg_i,chg_j], channel, label)
                    except KeyError as err:
                        raise RuntimeError(f"missing density {rho_type} for charges {(chg_i,chg_j)} and channel {channel[:5]}: {err}")
                    timings.start()
                    result = evaluate(rho(*([0,1]+rho_indices)) @ ints(*int_indices))
                    timings.record(label)
                    return result
                else:
                    print("None was returned here")
                    return None
            #return dynamic_array(cached(contract_rho_int_m), [densities_m["n_states"].keys()]*2)  # no specification between bra and ket needed, because only keys are required
            # no specification between bra and ket needed, because only keys are required
            unique_mults = tuple({
                mult 
                for chg_dict in densities_m["n_states"].values() 
                for mult in chg_dict.keys()
            })
            possible_rank2 = (0,1,2,3,4,5,6,7,8)  # god help us, if these are not sufficient
            # since sigma is block diagonal for hermitian subsystem Hamiltonians and we are
            # looking at maximum at 2 particle operators, this should never be higher than 2
            possible_restr_ca = (0,1,2,3)
            # path and access views are arbitrary hashable objects (see coupl_fac.diagram_couplings)
            if None in unique_mults:    # not spin adapted
                possible_rank2, possible_restr_ca = (None,), (None,)
            ranges_with_mults = [densities_m["n_states"].keys()]*2 + [unique_mults]*2 + [possible_rank2] + [possible_restr_ca] + [None]*2
            if getattr(getattr(densities_m.get(rho_type), "provider", None), "uncached", False):
                return dynamic_array(contract_rho_int_m, ranges_with_mults)    # e.g. the probe of embedded.py
            return dynamic_array(cached_if_small(contract_rho_int_m, owner, indices[0]), ranges_with_mults)

        def contract_rho_rho_int(*indices):
            raise NotImplementedError

        if label.count("_")==1:    # good enough for now
            return dynamic_array(cached(contract_rho_int), [range(n_frag)]*n_indices)

    class _hack(object):    # wraps single string index as 1-tuple to not confuse dynamic_array
        def __init__(self, dyn_array):
            self.dyn_array = dyn_array
        def __getitem__(self, key):
            return self.dyn_array[(key,)]
        def release(self, fragment, transition):
            """drops the large intermediates of fragment for the transition ((chg_i, mult_i), (chg_j, mult_j))"""
            release_large(owner, fragment, transition)

    return _hack(dynamic_array(cached(mother_rule), [None]))
