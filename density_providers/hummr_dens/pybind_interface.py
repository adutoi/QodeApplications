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
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from dens_util import multiplicity_iterator, build_reverse_density_spin_adapt
from densities import _tens_wrap


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

def wrap_dens_and_build_transpose(dens_inp):
    """
    Load transition densities exported by hummr, wrap them with
    the internal tensor wrapper and build the transpose densities,
    which haven't been built explicitly yet.

    Internal hierarchy
    ------------------
    densities[density_name][charge_pair][channel_key]

    where

        charge_pair =
            (charge_bra, charge_ket)

        channel_key =
            (mult_bra, mult_ket, rank2, restr_ca)

    and rank2 = 2*k.
    """

    dens_out = {}

    # Note that there is no -2 factor included here, because it is not included
    # in the XR specific density build of hummr anymore.

    #print("apply -2 definition on ccaa in json load, note that this definition" \
    #"is only in line with the internal hummr definition, but not with the" \
    #"ccaa build of the density builder from the XR module.")

    #densities["spin_adapted"] = spin_adapted

    for dens_name, dens_group in dens_inp.items():

        #
        # Do not treat our own metadata as a density.
        #
        #if dens_name == "spin_adapted":
        #    continue

        top_map = {}

        for charge_pair, second_layer in dens_group.items():

            #charge_pair = tuple(map(int, chg_key.split("_")))

            channel_map = {}

            for channel, state_group in multiplicity_iterator(
                second_layer
            ):
                #print(dens_name, charge_pair, channel)

                # pybind11 interface already included the state
                # indices into the numpy density tensor and released the
                # original tensors, which didn't include the state
                # indices in the tensor.
                channel_map[channel] = _tens_wrap(state_group)

                #
                # Generate the reversed density immediately.
                #
                build_reverse_density_spin_adapt(
                    densities=dens_out,
                    dens_name=dens_name,
                    data=state_group,
                    charge_pair=charge_pair,
                    channel=channel,
                )

            top_map[charge_pair] = channel_map

        dens_out[dens_name] = top_map

    return dens_out


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
    ):
        self.xr_level = xr_level

        self._engine = None
        self.input_file: Optional[Path] = None
        self.orbitals: Optional[np.ndarray] = None

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

        # initialize of the hummr engine lets hummr setup its MPI environment
        # once and then use this MPI environment across all initialized
        # hummr instances. The OpenMP pool can be defined from the
        # configure_openmp() function here.
        self.orbitals = np.asarray(
            self._engine.initialize(
                str(self.input_file)
            )
        )

        return self.orbitals

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

    def compute_densities(
        self,
        bra: StateVectors,
        ket: StateVectors,
        xr_level: int,
    ):
        """
        Build densities between arbitrary supplied bra and ket state maps.

        Each state map has the structure

            charge -> multiplicity -> [CI vectors]
        """

        if self._engine is None:
            raise RuntimeError(
                "fragment has not been initialized"
            )

        return wrap_dens_and_build_transpose(
            self._engine.compute_densities(
                bra,
                ket,
                xr_level
            )
        )

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
    """

    def __init__(
        self,
        omp_threads: Optional[int] = None,
    ):
        self.omp_threads = (
            omp_threads
            if omp_threads is not None
            else os.cpu_count()  # or 1
        )

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
        )

        self.fragments[name] = fragment

        return fragment

    def finalize(self) -> None:

        for fragment in self.fragments.values():
            fragment.finalize()

        self.fragments.clear()
