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

"""
Threaded versions of the bandwidth-bound numpy operations around the (threaded) MKL products of the streamed
contractions (todo_bridge_performance.md, R3): contiguous (transposed) copies, in-place additions and a tensordot
built on them.  numpy releases the GIL in its copy and ufunc loops, so chunks of the output along one axis can be
processed by a thread pool; the results are identical to the plain numpy ones (the same elementwise operations).

Arrays with fewer than MIN_ELEMENTS elements take the plain numpy path.  The module also serves as an opt_einsum
backend (contract(..., backend="hummr_dens.fast_np"): tensordot, transpose and einsum by name).

Measured on a 16-core laptop (memory bandwidth bound): a 0.26 GB transposed copy 3.0 -> 7.0 GB/s with 8 threads,
in-place additions no gain (one thread already saturates the bandwidth); a many-core node has much more bandwidth
per socket than one thread can draw.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

MIN_ELEMENTS = 1 << 21          # 16 MB of doubles
ENABLED = os.environ.get("HUMMR_FAST_NP", "1") != "0"

_pool = None
_n_threads = None


def n_threads():
    global _n_threads
    if _n_threads is None:
        env = os.environ.get("HUMMR_FAST_NP_THREADS") or os.environ.get("OMP_NUM_THREADS")
        try:
            _n_threads = max(1, int(env)) if env else (os.cpu_count() or 1)
        except ValueError:
            _n_threads = os.cpu_count() or 1
    return _n_threads


def _executor():
    global _pool
    if _pool is None:
        _pool = ThreadPoolExecutor(max_workers=n_threads(), thread_name_prefix="fast_np")
    return _pool


def _chunk_slices(shape):
    """slices of the output into about 2 x n_threads chunks along one axis (the first one long enough, otherwise the
    longest one), or None if it is not worth it"""
    nt = n_threads()
    if nt < 2 or not ENABLED or int(np.prod(shape, dtype=np.int64)) < MIN_ELEMENTS or len(shape) == 0:
        return None
    axis = next((i for i, n in enumerate(shape) if n >= 2 * nt), int(np.argmax(shape)))
    n = shape[axis]
    if n < 2:
        return None
    bounds = np.linspace(0, n, min(n, 2 * nt) + 1).astype(int)
    slices = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b > a:
            sl = [slice(None)] * len(shape)
            sl[axis] = slice(a, b)
            slices.append(tuple(sl))
    return slices


def _run(job, slices):
    list(_executor().map(job, slices))


def copy_into(dst, src):
    """dst[...] = src (same shape)"""
    slices = _chunk_slices(dst.shape)
    if slices is None:
        dst[...] = src
        return dst
    def job(sl):
        dst[sl] = src[sl]
    _run(job, slices)
    return dst


def contiguous(a):
    """C-contiguous copy of a (a itself if it already is)"""
    a = np.asarray(a)
    if a.flags.c_contiguous:
        return a
    if _chunk_slices(a.shape) is None:
        return np.ascontiguousarray(a)
    return copy_into(np.empty(a.shape, dtype=a.dtype), a)


def transpose_copy(a, perm):
    """np.ascontiguousarray(np.transpose(a, perm)), threaded"""
    return contiguous(np.transpose(a, perm))


def iadd(view, term, coef=1.0):
    """view += coef * term in place (view may be a strided view of a larger array)"""
    slices = _chunk_slices(view.shape)
    if slices is None:
        if coef == 1.0:
            view += term
        else:
            view += coef * term
        return
    if coef == 1.0:
        def job(sl):
            v = view[sl]
            np.add(v, term[sl], out=v)
    else:
        def job(sl):
            v = view[sl]
            v += coef * term[sl]
    _run(job, slices)


def tensordot(a, b, axes=2):
    """np.tensordot with the transposed copies of the operands threaded (the product itself is MKL's dot)"""
    a, b = np.asarray(a), np.asarray(b)
    if isinstance(axes, (int, np.integer)):
        axes_a, axes_b = list(range(a.ndim - axes, a.ndim)), list(range(axes))
    else:
        axes_a, axes_b = axes
        axes_a = [axes_a] if np.ndim(axes_a) == 0 else list(axes_a)
        axes_b = [axes_b] if np.ndim(axes_b) == 0 else list(axes_b)
    axes_a = [int(x) % a.ndim for x in axes_a] if a.ndim else []
    axes_b = [int(x) % b.ndim for x in axes_b] if b.ndim else []
    if len(axes_a) != len(axes_b) or any(a.shape[i] != b.shape[j] for i, j in zip(axes_a, axes_b)):
        raise ValueError("shape-mismatch for sum")
    free_a = [k for k in range(a.ndim) if k not in axes_a]
    free_b = [k for k in range(b.ndim) if k not in axes_b]
    n_sum = int(np.prod([a.shape[i] for i in axes_a], dtype=np.int64))
    at = transpose_copy(a, free_a + axes_a).reshape(-1, n_sum)
    bt = transpose_copy(b, axes_b + free_b).reshape(n_sum, -1)
    return np.dot(at, bt).reshape([a.shape[i] for i in free_a] + [b.shape[i] for i in free_b])


transpose = np.transpose
einsum = np.einsum


def use_with_tensorly(optimize="auto-hq"):
    """
    tensorly's einsum (the Qode tensornet contractions, tensorly_backend.contract) through opt_einsum with this module
    as the backend: the same path cache as tensorly.plugins.use_opt_einsum (the same optimize), but its tensordot
    steps with the threaded transposed copies above instead of numpy.tensordot (single-threaded copies, ~2.7e3 s in
    the pair-basis diagrams of Be2 cc-pVTZ 3.0 A, todo_bridge_performance.md R3).  The plain plugin if not ENABLED.
    """
    import tensorly as tl
    if not ENABLED:
        tl.plugins.use_opt_einsum(optimize)
        return
    import opt_einsum as oe
    cache = {}
    def cached_einsum(equation, *args):
        key = equation + "," + ",".join(f"{np.shape(a)}" for a in args)
        expression = cache.get(key)
        if expression is None:
            expression = cache[key] = oe.contract_expression(equation, *[np.shape(a) for a in args], optimize=optimize)
        if not all(isinstance(a, np.ndarray) for a in args):
            return expression(*args)
        return expression(*args, backend=__name__)
    tl.plugins.use_opt_einsum(optimize)      # (records tensorly's own einsum for use_default_einsum)
    tl.backend.BackendManager.register_backend_method("einsum", cached_einsum)
