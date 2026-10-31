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

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import tensorly
from sympy import Rational, sqrt
from sympy.physics.quantum.cg import CG

from qode.util.PyC import Double
from qode.math.tensornet import tl_tensor
from dens_util import multiplicity_iterator, reversed_densities_spin_adapt
from densities import _tens_wrap


def _tens_wrap_view(data):
    """
    Like densities._tens_wrap, but without copying for the numpy backend (tensorly.tensor is np.array there,
    which always copies, also turning transposed views into contiguous copies).  tensornet never modifies the
    raw tensors in place, so the densities and their reversed partners can share the data.
    """
    if tensorly.get_backend() == "numpy":
        return tl_tensor.init(np.asarray(data, dtype=Double.numpy))
    return _tens_wrap(data)


StateRoots = Dict[int, Dict[int, int]]
StateVectors = Dict[int, Dict[int, list[np.ndarray]]]


def configure_openmp(n_threads: Optional[int] = None) -> None:
    """
    Configure the single OpenMP pool shared by Python and HUMMR.

    This should be called before importing hummr_python and, preferably,
    before importing NumPy/SciPy/other libraries which initialize their
    threaded runtimes.
    """

    if n_threads is None:
        n_threads = os.cpu_count() or 1

    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["OMP_PROC_BIND"] = "TRUE"
    os.environ["OMP_PLACES"] = "cores"
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"


def roots_to_text(roots: Mapping[int, Mapping[int, int]]) -> str:
    """
    Human-readable representation useful for diagnostics.
    """

    pieces = []

    for charge in sorted(roots):
        mults = []

        for mult in sorted(roots[charge]):
            mults.append(
                f"{mult}:{int(roots[charge][mult])}"
            )

        pieces.append(
            f"{charge}:{{{', '.join(mults)}}}"
        )

    return "{" + ", ".join(pieces) + "}"


def validate_roots(
    roots: Mapping[int, Mapping[int, int]]
) -> None:

    if not roots:
        raise ValueError("roots must not be empty")

    for charge, by_mult in roots.items():

        if not by_mult:
            raise ValueError(
                f"charge {charge} has no multiplicities"
            )

        for mult, nroots in by_mult.items():

            if int(mult) < 1:
                raise ValueError(
                    f"invalid multiplicity {mult}; "
                    "multiplicities must be positive"
                )

            if int(nroots) <= 0:
                raise ValueError(
                    f"charge={charge}, mult={mult}: "
                    "n_roots must be positive"
                )


def write_hummr_input(
    path: Path,
    *,
    calc_type: str,
    charge: int,
    multiplicities: Sequence[int],
    orb_guess: str,
    ints: str,
    basis: str,
    aux_basis: str,
    n_electrons: int,
    n_orbitals: int,
    nroots: Sequence[int],
    ci_solver: str,
    max_iter: int,
    orb_step: str,
    switch_orb_step: str,
    xr_level: int,
    geometry: Sequence[str],
    export_json: bool = False,
) -> Path:
    """
    Write an ordinary HUMMR input file.

    This is intentionally a Python function rather than a C++ configuration
    object. If HUMMR gains another keyword, this is the only layer that needs
    to know about it.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    if len(multiplicities) != len(nroots):
        raise ValueError(
            "multiplicities and nroots must have identical lengths"
        )

    lines = [
        "General",
        f"  CalcType {calc_type}",
        f"  Charge {charge}",
        "  Mult " + " ".join(
            str(x) for x in multiplicities
        ),
        f"  OrbGuessName {orb_guess}",
        f"  Ints {ints}",
        f"  Basis {basis}",
        f"  AuxBasis {aux_basis}",
        f"  ExportJSON {'True' if export_json else 'False'}",
        "End",
        "",
        "CASSCF",
        f"  NEl {n_electrons}",
        f"  NOrb {n_orbitals}",
        "  NRoots " + " ".join(
            str(x) for x in nroots
        ),
        f"  CISolver {ci_solver}",
        f"  MaxIter {max_iter}",
        f"  OrbStep {orb_step}",
        f"  SwitchOrbStep {switch_orb_step}",
        #f"  XRLevel {xr_level}",
        "End",
        "",
        "Geom",
    ]

    lines.extend(
        f"  {line}"
        for line in geometry
    )

    lines.extend([
        "End",
        "",
    ])

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return path

def cg_factor(channel):
    """the CG coefficient <S_J S_J K q|S_I S_I> of a channel (mult_bra, mult_ket, rank2, ...), see to_cg_convention"""
    return _cg_factor(*channel[:3])

@lru_cache(maxsize=None)    # (the sympy CG coefficients are slow)
def _cg_factor(mult_bra, mult_ket, rank2):
    S_I, S_J = Rational(mult_bra - 1, 2), Rational(mult_ket - 1, 2)
    k, q = Rational(rank2, 2), S_I - S_J
    return float(CG(S_J, S_J, k, q, S_I, S_I).doit())

def to_cg_convention(dens_name, channel, data):
    """
    The XR machinery (coupl_fac.recoupling_factor, dens_util.reversed_densities_spin_adapt)
    expects all spin-adapted densities in the CG convention

        <I S_I S_I| T^K_q |J S_J S_J> = <S_J S_J K q|S_I S_I> rho_IJ,      q = S_I - S_J,

    with the tensors T^K_q(path) of spin_tensors.py (the channel key is
    (mult_bra, mult_ket, rank2, restr_ca, path)).

    hummr returns stretched elements <I S_I S_I| T^K_q |J S_J S_J> for all types and paths (including the
    rank-1 ca channels, T^1 = -sqrt(2) [c x at]^1), i.e. the CG coefficient has to be divided out (it is 1 for
    S_I = S_J + K, e.g. for ccaa with restr_ca 2, rank 0 and equal multiplicities).  This was verified against
    reference densities from an exact Fock-space FCI (tests/spin_adapted).

    The conversion is done in place (data has to be a fresh array, e.g. as returned by hummr).
    """
    cg = cg_factor(channel)
    if cg != 1.0:
        np.divide(data, cg, out=data)
    return data

def embed_frozen_core(dens, n_core, n_mo, overlaps=None, consume=False):
    """
    Embed densities of an active space (orbitals n_core ... n_core+n_act-1 of the n_mo fragment MOs)
    into the full fragment MO space, with the n_core core orbitals doubly occupied in every state
    (all types and paths, see density_providers/frozen_core.py).  E.g.

        ca   (rank 0)   E_KL                           = 2 delta_KL delta_IJ
        caa  E_pr at_s  [K,K,s] = 2 a_s,               [K,r,K] = -a_r

    The delta_IJ are the overlaps <I|J> of the active parts: overlaps[(chg, mult)] (bra x ket) if given,
    otherwise bra and ket states have to be the same (orthonormal) set.
    consume: the active densities are removed from dens as soon as they are no longer needed (peak memory)
    """
    from frozen_core import embed
    return embed(dens, n_core, n_mo, overlaps, consume=consume)

def wrap_dens_and_build_transpose(dens_inp, n_core=0, n_mo=None, overlaps=None, parts="both", keep_active=False):
    """
    Convert the densities returned by hummr into the CG convention, embed them into the full MO space
    (frozen core), add their adjoint partners and wrap them with the internal tensor wrapper.

    Internal hierarchy
    ------------------
    densities[density_name][charge_pair][channel_key]

    where

        charge_pair =
            (charge_bra, charge_ket)

        channel_key =
            (mult_bra, mult_ket, rank2, restr_ca, path)

    with rank2 = 2*K and the spin coupling path of spin_tensors.py.

    parts: "both"     the computed densities and their reversed (adjoint) partners
           "direct"   only the computed densities
           "reversed" only the reversed partners (of channels, which are not computed directly)
    The latter two are used for different bra and ket state sets, where the partners have to be obtained
    from a calculation with bra and ket swapped (see HummrXRFragment.compute_densities).
    overlaps: <bra|ket> per (chg, mult) for the frozen-core embedding (see embed_frozen_core)

    dens_inp is consumed (emptied, and its arrays are modified in place) to keep the peak memory down: apart
    from the frozen-core embedding into the full MO space, no densities are copied (the reversed partners are
    lazily scaled transposed views, or lazy sums of them, of the computed densities).
    """
    if parts not in ("both", "direct", "reversed"):
        raise ValueError(f"unknown parts {parts!r}")

    # convert into the CG convention (in place) and embed the active space into the full MO space (frozen core)
    dens_cg = {}
    for dens_name, dens_group in dens_inp.items():
        dens_cg[dens_name] = {}
        for charge_pair, second_layer in dens_group.items():
            dens_cg[dens_name][charge_pair] = {channel: to_cg_convention(dens_name, channel, np.asarray(state_group))
                                               for channel, state_group in multiplicity_iterator(second_layer)}
    dens_inp.clear()    # dens_cg holds the only references, so that embed can release the active densities
    # the active densities (CG convention), e.g. for the core terms of the streamed large types
    active = {name: {cp: dict(by_channel) for cp, by_channel in by_chg.items()} for name, by_chg in dens_cg.items()} \
             if keep_active else None
    dens_np = embed_frozen_core(dens_cg, n_core, n_mo, overlaps, consume=True)
    del dens_cg

    dens_out = {}
    if parts != "reversed":
        for dens_name, by_chg in dens_np.items():
            for charge_pair, by_channel in by_chg.items():
                for channel, data in by_channel.items():
                    dens_out.setdefault(dens_name, {}).setdefault(charge_pair, {})[channel] = _tens_wrap_view(data)
    if parts != "direct":
        # adjoint partners (directly computed channels take precedence), already wrapped
        for dens_name, by_chg in reversed_densities_spin_adapt(dens_np, wrap=_tens_wrap_view).items():
            for charge_pair, by_channel in by_chg.items():
                for channel, data in by_channel.items():
                    if channel not in dens_np.get(dens_name, {}).get(charge_pair, {}):
                        dens_out.setdefault(dens_name, {}).setdefault(charge_pair, {})[channel] = data
    if keep_active:
        return dens_out, active
    return dens_out


def frozen_core_from_input(input_file, n_mo):
    """
    Number of doubly occupied (frozen) core orbitals of a hummr CASSCF/CASCI input, i.e.
    (sum of nuclear charges - Charge - NEl) / 2.  hummr orders the MOs as core, active, virtual.
    Returns (n_core, n_act, n_act_elec, charge): inputs without an active space (no NEl/NOrb) are treated as
    all-electron; n_act_elec is the number of active electrons of the input charge.
    """
    from qode.atoms.elements import atomic_number
    blocks, block = {}, None
    for line in Path(input_file).read_text().splitlines():
        words = line.split()
        if not words or words[0].startswith("#"):
            continue
        if block is None:
            block = words[0]
            blocks[block] = []
        elif words[0] == "End":
            block = None
        else:
            blocks[block].append(words)
    general = {w[0]: w[1:] for w in blocks.get("General", [])}
    cas = {w[0]: w[1:] for w in blocks.get("CASSCF", [])}
    charge = int(general.get("Charge", ["0"])[0])
    n_elec = sum(atomic_number(w[0]) for w in blocks["Geom"]) - charge
    if "NEl" not in cas or "NOrb" not in cas:
        return 0, n_mo, n_elec, charge
    n_core2 = n_elec - int(cas["NEl"][0])
    if n_core2 < 0 or n_core2 % 2:
        raise ValueError(f"{input_file}: inconsistent active space ({n_elec} electrons, NEl {cas['NEl'][0]})")
    n_core, n_act = n_core2 // 2, int(cas["NOrb"][0])
    if n_core + n_act > n_mo:
        raise ValueError(f"{input_file}: {n_core} core + {n_act} active orbitals exceed the {n_mo} MOs")
    return n_core, n_act, n_elec - 2 * n_core, charge


class HummrXRFragment:
    """
    One unique fragment.

    This object contains no MPI machinery. HUMMR is linked directly into the
    Python process and uses the same OpenMP pool as the Python program.
    """

    def __init__(
        self,
        *,
        xr_level: int = 0,
        omp_threads: Optional[int] = None,
    ):
        self.xr_level = xr_level
        # None: the number of threads at initialize() (see HummrXREngine)
        self.omp_threads = omp_threads

        self._engine = None
        self.input_file: Optional[Path] = None
        self.orbitals: Optional[np.ndarray] = None
        # frozen core: densities are embedded into the full MO space with doubly occupied core
        self.n_core = 0
        self.n_act = None

    def prepare_input(
        self,
        *,
        scratch_dir: str | Path,
        filename: str,
        overwrite: bool = False,
        **kwargs,
    ) -> Path:
        """
        Create a HUMMR input file in the scratch directory.

        If it already exists and overwrite=False, it is left untouched.
        """

        scratch = Path(scratch_dir)
        scratch.mkdir(
            parents=True,
            exist_ok=True,
        )

        path = scratch / filename

        if path.exists() and not overwrite:
            self.input_file = path
            return path

        write_hummr_input(
            path,
            xr_level=self.xr_level,
            **kwargs,
        )

        self.input_file = path

        return path

    def initialize(
        self,
        input_file: str | Path,
    ) -> np.ndarray:
        """
        Initialize HUMMR from the supplied ordinary HUMMR input file.

        Returns the MO coefficient matrix.
        """

        # Import here so configure_openmp() can be called first.
        import hummr_python

        if self._engine is not None:
            raise RuntimeError(
                "fragment is already initialized"
            )

        self.input_file = Path(input_file)

        self._engine = (
            hummr_python.HummrForXRMain()
        )
        if self.omp_threads is not None:
            self._engine.set_num_threads(self.omp_threads)

        # initialize of the hummr engine lets hummr setup its MPI environment
        # once and then use this MPI environment across all initialized
        # hummr instances. The OpenMP pool can be defined from the
        # configure_openmp() function here.
        self.orbitals = np.asarray(
            self._engine.initialize(
                str(self.input_file)
            )
        )
        self.n_core, self.n_act, self._n_act_elec, self._input_charge = frozen_core_from_input(self.input_file, self.orbitals.shape[1])

        return self.orbitals

    def csf_layout(self):
        """{charge: {mult: [(cfg, first CSF position, number of CSFs)]}} of all sectors, in the CSF order of the states"""
        return self._engine.csf_layout()

    def multiplicities(self, charge):
        """all multiplicities with CSFs of the active space for the given fragment charge"""
        n = self._n_act_elec - (charge - self._input_charge)
        if n < 0 or n > 2 * self.n_act:
            return []
        return [S2 + 1 for S2 in range(n % 2, min(n, 2 * self.n_act - n) + 1, 2)]

    def set_selected_cfgs(self, selected):
        """
        Use fixed (selected) CFG spaces {charge: {mult: CFGs}} instead of the complete CSF spaces (hummr's
        HCIManager, e.g. the union of the selections of hummr_dens.space_selection): every requested sector of
        get_states is solved in its space (HCI without further selection), and the densities resolve the identity
        exactly over the CFGs connected to the bra and ket spaces.  Before get_states.
        """
        self._engine.set_selected_cfgs({int(chg): {int(mult): sorted(cfgs) for mult, cfgs in by_mult.items()}
                                        for chg, by_mult in selected.items()})
        self.selected = True

    def with_intermediate_sectors(self, roots):
        """
        hummr resolves the identity in the composite densities (aa, caa, ccaa and all XR[1]/XR[2] types) only over
        the (charge, multiplicity) sectors requested in get_states, using their full CSF spaces.  Returns roots
        extended by one root for every multiplicity with CSFs of each requested charge, which is not requested.
        Selected spaces (set_selected_cfgs) do not need them (roots unchanged).
        """
        if getattr(self, "selected", False):
            return roots
        out = {chg: dict(by_mult) for chg, by_mult in roots.items()}
        for chg in roots:
            for mult in self.multiplicities(chg):
                out[chg].setdefault(mult, 1)
        return out

    def get_states(
        self,
        roots: Mapping[int, Mapping[int, int]],
    ) -> StateVectors:
        """
        Generate the requested FCI states.

        roots[charge][multiplicity] = n_roots
        """

        if self._engine is None:
            raise RuntimeError(
                "fragment has not been initialized"
            )

        validate_roots(roots)

        return self._engine.get_states(
            {
                int(charge): {
                    int(mult): int(nroots)
                    for mult, nroots in by_mult.items()
                }
                for charge, by_mult in roots.items()
            },
            self.xr_level
        )

    def contract_density_sector(self, n_c, n_a, restr_ca, bra_sector, ket_sector, bra_vecs, ket_vecs, channel,
                                slot_axis, tensor):
        """
        contraction of one channel of these active densities (stretched elements, as hummr returns them) with
        tensor: slot s (operator order) is contracted with axis slot_axis[s] of tensor or free (-1); result
        (bra, ket, free slots in slot order, free axes of tensor in axis order), without building the density
        """
        path = list(channel[4][0]) + list(channel[4][1])
        return self._engine.contract_density_sector(n_c, n_a, restr_ca, bra_sector[0], bra_sector[1], ket_sector[0],
                                                    ket_sector[1], list(bra_vecs), list(ket_vecs), channel[2], path,
                                                    list(slot_axis), np.ascontiguousarray(tensor, dtype=float))

    def density_sector(self, n_c, n_a, restr_ca, bra_sector, ket_sector, bra_vecs, ket_vecs):
        """
        active densities of the operator string (n_c, n_a, restr_ca) for one bra x ket sector pair
        ((chg, mult) each), all channels, as hummr returns them: {channel: ndarray (bra, ket, orbitals...)}
        """
        return self._engine.compute_density_sector(n_c, n_a, restr_ca, bra_sector[0], bra_sector[1], ket_sector[0],
                                                   ket_sector[1], list(bra_vecs), list(ket_vecs))

    def transformed_contracted_sector(self, n_c, n_a, restr_ca, bra_sector, ket_sector, bra_vecs, ket_vecs, factors,
                                      t_slot, tensor, chunk_bytes):
        """
        transformed_density_sector with the slots j of t_slot (flags) contracted with tensor (these slots in slot
        order, then the free axes) instead of a matrix: {channel: ndarray (bra, ket, K of the other slots..., free
        entries of tensor flattened)}
        """
        tensor = np.ascontiguousarray(tensor, dtype=float)
        n_t = sum(bool(t) for t in t_slot)
        g = int(np.prod(tensor.shape[n_t:]))
        return self._engine.compute_transformed_contracted_sector(n_c, n_a, restr_ca, bra_sector[0], bra_sector[1],
                                                                 ket_sector[0], ket_sector[1], list(bra_vecs),
                                                                 list(ket_vecs), [np.asarray(f, dtype=float) for f in factors],
                                                                 [bool(t) for t in t_slot], tensor.reshape(-1), g,
                                                                 int(chunk_bytes))

    def transformed_density_sector(self, n_c, n_a, restr_ca, bra_sector, ket_sector, bra_vecs, ket_vecs, factors,
                                   chunk_bytes):
        """
        density_sector with factors[j] (n_act x K_j) contracted into slot j (operator order) of every channel,
        without forming the densities over all orbitals: {channel: ndarray (bra, ket, K_0, ..., K_{N-1})}; the
        intermediates are formed for at most chunk_bytes at a time
        """
        return self._engine.compute_transformed_density_sector(n_c, n_a, restr_ca, bra_sector[0], bra_sector[1],
                                                              ket_sector[0], ket_sector[1], list(bra_vecs),
                                                              list(ket_vecs), [np.asarray(f, dtype=float) for f in factors],
                                                              int(chunk_bytes))

    def compute_densities(
        self,
        bra: StateVectors,
        ket: StateVectors,
        xr_level: int,
        streamed: bool = False,
        memory_budget: Optional[int] = None,
        max_ops: int = -1,
    ):
        """
        Build densities between arbitrary supplied bra and ket state maps.

        Each state map has the structure

            charge -> multiplicity -> [CI vectors]

        streamed:      the large types (4 and more operators) are not built here; their entries are
                       StreamedDensityType objects, whose contractions (precontract.py) are evaluated by a
                       StreamedDensities provider (hummr_dens/streamed.py) keeping at most memory_budget bytes of
                       active densities (None: no limit)
        max_ops:       only the types with at most this many operators (-1: all types of xr_level), e.g. 2 for
                       the overlap (S) diagrams
        """

        if self._engine is None:
            raise RuntimeError(
                "fragment has not been initialized"
            )

        from hummr_dens.streamed import STREAMED_MIN_OPS
        if streamed and not 0 <= max_ops < STREAMED_MIN_OPS:
            return self._compute_densities_streamed(bra, ket, xr_level, memory_budget)

        n_mo = self.orbitals.shape[1]
        if bra is ket:
            return wrap_dens_and_build_transpose(
                self._engine.compute_densities(
                    bra,
                    ket,
                    xr_level,
                    max_ops
                ),
                n_core=self.n_core,
                n_mo=n_mo,
            )

        # Different bra and ket sets (e.g. single CSFs as bra for the state gradients): the adjoint partners
        # <bra|c|ket> etc. are the reversed densities of <ket|a|bra>, which requires a second calculation.
        overlaps = {}
        for chg, mult_dict in bra.items():
            for mult, vecs in mult_dict.items():
                if len(vecs) > 0 and len(ket.get(chg, {}).get(mult, [])) > 0:
                    overlaps[(chg, mult)] = np.array(vecs, dtype=float) @ np.array(ket[chg][mult], dtype=float).T
        overlaps_T = {key: S.T for key, S in overlaps.items()}
        dens = wrap_dens_and_build_transpose(self._engine.compute_densities(bra, ket, xr_level, max_ops),
                                             n_core=self.n_core, n_mo=n_mo, overlaps=overlaps, parts="direct")
        partners = wrap_dens_and_build_transpose(self._engine.compute_densities(ket, bra, xr_level, max_ops),
                                                 n_core=self.n_core, n_mo=n_mo, overlaps=overlaps_T, parts="reversed")
        for dens_name, by_chg in partners.items():
            for charge_pair, by_channel in by_chg.items():
                for channel, data in by_channel.items():
                    dens.setdefault(dens_name, {}).setdefault(charge_pair, {}).setdefault(channel, data)
        return dens

    def _compute_densities_streamed(self, bra, ket, xr_level, memory_budget):
        from hummr_dens.streamed import StreamedDensities, StreamedDensityType, STREAMED_MIN_OPS
        n_mo = self.orbitals.shape[1]
        max_ops = STREAMED_MIN_OPS - 1
        if bra is ket:
            dens, active = wrap_dens_and_build_transpose(self._engine.compute_densities(bra, ket, xr_level, max_ops),
                                                         n_core=self.n_core, n_mo=n_mo, keep_active=True)
            active_small, overlaps = {False: active, True: active}, None
        else:
            overlaps = {}
            for chg, mult_dict in bra.items():
                for mult, vecs in mult_dict.items():
                    if len(vecs) > 0 and len(ket.get(chg, {}).get(mult, [])) > 0:
                        overlaps[(chg, mult)] = np.array(vecs, dtype=float) @ np.array(ket[chg][mult], dtype=float).T
            overlaps_T = {key: S.T for key, S in overlaps.items()}
            dens, active = wrap_dens_and_build_transpose(self._engine.compute_densities(bra, ket, xr_level, max_ops),
                                                         n_core=self.n_core, n_mo=n_mo, overlaps=overlaps,
                                                         parts="direct", keep_active=True)
            partners, active_swapped = wrap_dens_and_build_transpose(
                self._engine.compute_densities(ket, bra, xr_level, max_ops), n_core=self.n_core, n_mo=n_mo,
                overlaps=overlaps_T, parts="reversed", keep_active=True)
            for dens_name, by_chg in partners.items():
                for charge_pair, by_channel in by_chg.items():
                    for channel, data in by_channel.items():
                        dens.setdefault(dens_name, {}).setdefault(charge_pair, {}).setdefault(channel, data)
            active_small, overlaps = {False: active, True: active_swapped}, {False: overlaps, True: overlaps_T}
        provider = StreamedDensities(self, bra, ket, xr_level, active_small, overlaps, memory_budget)
        for name in provider.streamed_types():
            dens[name] = StreamedDensityType(provider, name)
        return dens

    def finalize(self) -> None:

        if self._engine is not None:
            self._engine.finalize()

        self._engine = None
        self.orbitals = None


class HummrXREngine:
    """
    Small supervisor for sequential unique-fragment calculations.

    It intentionally does NOT parallelize fragments. The desired workflow is:

        Python work
          -> fragment A
          -> Python work
          -> fragment B
          -> ...
          -> Python work

    For parallelization the first hummr engine initialization sets up an
    MPI environment, which is used across all hummr instances. The MPI
    environment can be tuned via the input file. Furthermore, the OpenMP
    environment can be configured using either hummr internals or the
    configure() function of this class

    The OpenMP (and MKL) runtime is shared with all other libraries in the
    process, which can change the number of threads (e.g. "import psi4",
    which happens in get_ints, sets it to 1). Every hummr instance therefore
    re-applies its number of threads at the beginning of each get_states()
    and compute_densities() call: omp_threads, if given, otherwise the
    numbers at its initialization (OMP_NUM_THREADS or NThreads of the input).
    """

    def __init__(
        self,
        omp_threads: Optional[int] = None,
    ):
        self.omp_threads = omp_threads

        self.fragments: Dict[str, HummrXRFragment] = {}

    def configure(self) -> None:
        configure_openmp(
            self.omp_threads
        )

    def add_fragment(
        self,
        name: str,
        *,
        xr_level: int = 0,
    ) -> HummrXRFragment:

        if name in self.fragments:
            raise ValueError(
                f"fragment '{name}' already exists"
            )

        fragment = HummrXRFragment(
            xr_level=xr_level,
            omp_threads=self.omp_threads,
        )

        self.fragments[name] = fragment

        return fragment

    def finalize(self) -> None:

        for fragment in self.fragments.values():
            fragment.finalize()

        self.fragments.clear()
