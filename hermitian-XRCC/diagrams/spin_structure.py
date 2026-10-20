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

# Spin structure of the (spin-orbital) diagrams for the evaluation with spin-adapted fragment densities.
#
# The diagrams were derived for spin orbitals.  Their comment blocks document the derivation, i.e. the
# fragment densities as normal-ordered operator strings (e.g. X.ccaa0(i0,j0,p,t,s,r) = <c_p c_t a_s a_r>)
# and the integral tensors, whose indices carry the spin summations:
#     s##(x,y), t##(x,y), u#_##(x,y):  sum_s c_xs a_ys                (x, y share one spin)
#     v####(p,q,r,s):                  sum_st c_ps c_qt a_st a_rs      (p, r and q, s share one spin each)
# The code accesses these densities directly (X.ca1(i1,j1,p,u)) or precontracted with an integral
# (X.ca0Xq_T1q(i0,j0,t,p)), possibly with its indices in a different order than in the comment (the
# spin-orbital densities are antisymmetric, the resulting sign is part of the code's prefactor).
#
# For spin-adapted densities, the operators of each fragment are brought into the canonical layout of
# density_providers/spin_tensors.py: operator-local pairs (c and a sharing a spin summation on the same
# fragment) become the rank-0 blocks, the remaining (free) c's and a's fill the other slots in order.
# Every access then gets a view (which canonical slot sits at which accessed position, and the sign of
# the permutation of the c's and of the a's), and the spin sum of the whole diagram is expanded as
#
#     sum_{K, P0, P1} w(K, P0, P1) [T^K(P0) x T^K(P1)]^0      (dimer)     or     sum_P w(P) T^0(P)  (monomer)
#
# in the canonical tensors of the fragments (exact, otherwise an error is raised).  The matrix elements
# of [T^K(P0) x T^K(P1)]^0 between spin-coupled product states are given by coupl_fac.recoupling_factor.

import re
import inspect
import itertools
from functools import lru_cache

import numpy as np

from spin_tensors import paths, components, layout_pairs, layout_kinds, cg2


_density_re = re.compile(r"X\.([ca]+)(\d)\(([^)]*)\)")
_tensor_re  = re.compile(r"X\.(s|t|v|u)(\d)?_?(\d+)\(([^)]*)\)")
_access_re  = re.compile(r"X\.([A-Za-z0-9_]+)\(([^)]*)\)")


def _find_diagram(label):
    from . import S_diagrams, ST_diagrams, SU_diagrams, SV_diagrams
    for module in (S_diagrams, ST_diagrams, SU_diagrams, SV_diagrams):
        for catalog in module.catalog.values():
            if label in catalog:
                return catalog[label]
    raise KeyError(f"diagram {label} not found")

def _source_lines(label):
    fn = _find_diagram(label)
    contraction = getattr(fn, "__wrapped_contraction__", None)
    if contraction is None:
        # build_diagram closures keep the contraction as a free variable
        closure = inspect.getclosurevars(fn).nonlocals
        contraction = closure.get("contraction", fn)
    return inspect.getsource(contraction).splitlines()

def _parse(label):
    """
    {frag: {"letters": [...], "kinds": [...]}} of the comment densities, the spin label of every index letter,
    and the accesses {attr: (frag, letters in access order)} of the code
    """
    if label == "identity":
        return {}, {}, {}
    if label == "u100":
        # special code (Kronecker delta on fragment 1, see SU_2mer_0.u100), derivation as in u000
        return ({0: {"letters": ["p", "q"], "kinds": ["c", "a"]}, 1: {"letters": [], "kinds": []}},
                {"p": "u", "q": "u"}, {"ca0pq_U1pq": (0, ["p", "q"])})
    densities, spin, accesses = {}, {}, {}
    n_tensor = 0
    for line in _source_lines(label):
        stripped = line.strip()
        if stripped.startswith("#"):
            body = stripped.lstrip("#@ ").strip()
            m = _density_re.fullmatch(body)
            if m:
                name, frag, args = m.group(1), int(m.group(2)), [a.strip() for a in m.group(3).split(",")]
                letters = args[2:]
                densities[frag] = {"letters": letters, "kinds": list(name)}
                continue
            m = _tensor_re.fullmatch(body)
            if m:
                kind, args = m.group(1), [a.strip() for a in m.group(4).split(",")]
                n_tensor += 1
                if kind == "v":
                    p, q, r, s = args
                    spin[p] = spin[r] = f"{n_tensor}s"
                    spin[q] = spin[s] = f"{n_tensor}t"
                else:
                    x, y = args
                    spin[x] = spin[y] = f"{n_tensor}s"
            continue
        for attr, args in _access_re.findall(line):
            args = [a.strip() for a in args.split(",")]
            if re.fullmatch(r"[ca]+\d", attr):
                accesses[attr] = (int(attr[-1]), args[2:])
            elif "_" in attr:
                rho_label = attr.split("_")[0]
                m = re.fullmatch(r"([ca]+)(\d)([A-Za-z]*)", rho_label)
                if m is None:
                    continue
                frag, letters = int(m.group(2)), list(m.group(3))
                free = iter(args[2:])
                letters = [next(free) if x == "X" else x for x in letters]
                accesses[attr] = (frag, letters)
    for frag in (0, 1):
        densities.setdefault(frag, {"letters": [], "kinds": []})
    return densities, spin, accesses

def _parity(seq):
    sign = 1
    seq = list(seq)
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            if seq[i] > seq[j]:
                sign = -sign
    return sign

class FragmentStructure(object):
    """canonical layout of the density of one fragment of a diagram"""
    def __init__(self, letters, kinds, spin):
        self.letters, self.kinds = letters, kinds
        self.n_c, self.n_a = kinds.count("c"), kinds.count("a")
        cs = [i for i, k in enumerate(kinds) if k == "c"]
        as_ = [i for i, k in enumerate(kinds) if k == "a"]
        if kinds != ["c"] * self.n_c + ["a"] * self.n_a:
            raise ValueError(f"density operators not normal ordered: {kinds}")
        # operator-local pairs: a c and an a sharing a spin summation
        local = []
        used_a = set()
        for c in cs:
            for a in as_:
                if a not in used_a and spin[letters[c]] == spin[letters[a]]:
                    local.append((c, a))
                    used_a.add(a)
                    break
        self.restr = len(local)
        used = set(x for pair in local for x in pair)
        free_c = [c for c in cs if c not in used]
        free_a = [a for a in as_ if a not in used]
        local_slots, free_c_slots, free_a_slots = layout_pairs(self.n_c, self.n_a, self.restr)
        # canonical slot -> position in the comment (derivation) order
        self.slot_pos = [None] * (self.n_c + self.n_a)
        for (cs_, as_s), (c, a) in zip(local_slots, local):
            self.slot_pos[cs_], self.slot_pos[as_s] = c, a
        for s, c in zip(free_c_slots, free_c):
            self.slot_pos[s] = c
        for s, a in zip(free_a_slots, free_a):
            self.slot_pos[s] = a
        self.slot_letters = [letters[p] for p in self.slot_pos]
        self.slot_spins = [spin[x] for x in self.slot_letters]
        self.slot_kinds = layout_kinds(self.n_c, self.n_a)

    def view(self, access_letters):
        """
        (positions, sign): canonical slot s is accessed at position positions[s], and
        rho_access = sign * rho_canonical (relabeled)
        """
        positions = tuple(access_letters.index(x) for x in self.slot_letters)
        c_order = [positions[s] for s in range(self.n_c)]
        a_order = [positions[s] for s in range(self.n_c, self.n_c + self.n_a)]
        # the sign of reordering the canonical string into the access order
        return positions, _parity(np.argsort(c_order)) * _parity(np.argsort(a_order))

    def type_name(self):
        return "c" * self.n_c + "a" * self.n_a


_LEG = {("c", 1): (1.0, 1), ("c", -1): (1.0, -1), ("a", 1): (-1.0, -1), ("a", -1): (1.0, 1)}
# spin orbital operator (kind, 2*sigma) -> (coefficient, m2) of the leg:  a_alpha = -at_{-1/2}, a_beta = at_{+1/2}

@lru_cache(maxsize=None)
def diagram_structure(label, n_frag):
    """
    n_frag: number of fragments of the diagram (1 for monomer, 2 for dimer diagrams)
    (fragments, accesses, weights):
        fragments  [FragmentStructure of fragment 0, of fragment 1] (diagram fragment order, monomer: one)
        accesses   {frag: {attr template (digits -> #): (positions, sign)}}
        weights    {(rank2, path_0, path_1): w}  (dimer)  or  {(0, path_0): w}  (monomer)
    """
    densities, spin, access_letters = _parse(label)
    if label == "identity":
        return [], {}, {(0,): 1.0}
    frags = [FragmentStructure(densities[f]["letters"], densities[f]["kinds"], spin) for f in range(n_frag)]
    accesses = {f: {} for f in range(n_frag)}
    for attr, (frag, letters) in access_letters.items():
        template = re.sub(r"\d", "#", attr).replace("U#_", "U#")    # as in build_diagram.frag_resolve
        accesses[frag][template] = frags[frag].view(letters)
    weights = _spin_weights(frags)
    return frags, accesses, weights

def _spin_weights(frags):
    spins = sorted(set(x for f in frags for x in f.slot_spins))
    target = {}
    for values in itertools.product((1, -1), repeat=len(spins)):
        assign = dict(zip(spins, values))
        coef, ms = 1.0, []
        for f in frags:
            m_f = []
            for kind, x in zip(f.slot_kinds, f.slot_spins):
                c, m2 = _LEG[(kind, assign[x])]
                coef *= c
                m_f.append(m2)
            ms.append(tuple(m_f))
        target[tuple(ms)] = target.get(tuple(ms), 0.0) + coef
    basis = []
    if len(frags) == 1:
        f = frags[0]
        for P in paths(f.n_c, f.n_a, 0, f.restr):
            basis.append(((0, P), {(m,): c for m, c in components(f.n_c, f.n_a, P, 0).items()}))
    else:
        f0, f1 = frags
        max_rank2 = min(f0.n_c + f0.n_a, f1.n_c + f1.n_a)
        for K2 in range((f0.n_c + f0.n_a) % 2, max_rank2 + 1, 2):
            for P0 in paths(f0.n_c, f0.n_a, K2, f0.restr):
                for P1 in paths(f1.n_c, f1.n_a, K2, f1.restr):
                    vec = {}
                    for q2 in range(-K2, K2 + 1, 2):
                        cg = cg2(K2, q2, K2, -q2, 0, 0)
                        for m0, c0 in components(f0.n_c, f0.n_a, P0, q2).items():
                            for m1, c1 in components(f1.n_c, f1.n_a, P1, -q2).items():
                                vec[(m0, m1)] = vec.get((m0, m1), 0.0) + cg * c0 * c1
                    basis.append(((K2, P0, P1), vec))
    keys = sorted(set(target) | set(k for _, v in basis for k in v))
    index = {k: i for i, k in enumerate(keys)}
    M = np.zeros((len(keys), len(basis)))
    for j, (_, v) in enumerate(basis):
        for k, c in v.items():
            M[index[k], j] = c
    o = np.zeros(len(keys))
    for k, c in target.items():
        o[index[k]] = c
    w, *_ = np.linalg.lstsq(M, o, rcond=None)
    if np.linalg.norm(M @ w - o) > 1e-10 * max(1, np.linalg.norm(o)):
        raise RuntimeError("spin sum of the diagram cannot be expressed in the fragment density tensors")
    return {key: float(x) for (key, _), x in zip(basis, w) if abs(x) > 1e-12}
