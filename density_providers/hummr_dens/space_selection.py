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
Selection of the CI spaces of the fragments for XR with hummr's HCI.

Per fragment and (charge, multiplicity) sector, several independent HCI selections (standalone hummr runs, none
starting from the space of another one) are combined into one union of configurations (CFGs):
    - the bare, isolated fragment (internal correlation, separated-fragment limit),
    - point charges for the charge-transfer partner: the opposite charge of the sector on every neighbouring
      fragment (one run per neighbour), and the atomic partial charges of the neutral neighbours for the neutral
      sectors (if given),
    - weak uniform electric fields +-F along x, y and z (induction and the leading dispersion part, which needs
      the perpendicular directions as well).
The embedding only biases the selected space: the XR Hamiltonian is built from the bare fragment integrals, and the
densities do not depend on the Hamiltonian.  All runs use the orbitals of the given input (MaxIter 0), so that the
CFGs of the different runs refer to the same orbitals.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np


def read_blocks(input_file):
    """hummr input as [(block name, [lines])] (block order and lines kept)"""
    blocks, current = [], None
    for line in Path(input_file).read_text().splitlines():
        words = line.split()
        if not words or words[0].startswith("#"):
            continue
        if current is None:
            current = (words[0], [])
            blocks.append(current)
        elif words[0] == "End":
            current = None
        else:
            current[1].append(line.strip())
    return blocks

def write_blocks(blocks, input_file):
    text = ""
    for name, lines in blocks:
        text += name + "\n" + "".join(f"  {line}\n" for line in lines) + "End\n\n"
    Path(input_file).write_text(text)

def _set_keyword(lines, keyword, value):
    """replace (or append) the keyword line, value None removes it"""
    lines[:] = [line for line in lines if line.split()[0].lower() != keyword.lower()]
    if value is not None:
        lines.append(f"{keyword} {value}")

def _get_keyword(lines, keyword):
    for line in lines:
        words = line.split()
        if words[0].lower() == keyword.lower():
            return words[1:]
    return None

def selection_input(base_blocks, charge, mult, n_roots, epsilon, cfgs_out, point_charge_file=None, field=None,
                    hci_options=None):
    """
    blocks of the hummr input of one selection run: the base input (frozen core and active space as for XR), for the
    sector (charge, mult) with HCI (threshold epsilon) at fixed orbitals, optionally with point charges or a field
    """
    blocks = [(name, list(lines)) for name, lines in base_blocks]
    names = [name.lower() for name, _ in blocks]
    general = blocks[names.index("general")][1]
    casscf = blocks[names.index("casscf")][1]
    if "ci" not in names:
        blocks.append(("CI", []))
        names.append("ci")
    ci = blocks[names.index("ci")][1]
    base_charge = int((_get_keyword(general, "Charge") or ["0"])[0])
    n_el = int(_get_keyword(casscf, "NEl")[0]) - (charge - base_charge)
    _set_keyword(general, "Charge", charge)
    _set_keyword(general, "Mult", mult)
    _set_keyword(general, "PointChargeFileName", point_charge_file)
    _set_keyword(general, "ElectricField", None if field is None else " ".join(f"{f:.10e}" for f in field))
    _set_keyword(casscf, "NEl", n_el)
    _set_keyword(casscf, "NRoots", n_roots)
    _set_keyword(casscf, "CISolver", "HCI")
    _set_keyword(casscf, "MaxIter", 0)
    _set_keyword(ci, "EpsilonHCI", f"{epsilon:.3e}")
    _set_keyword(ci, "HCICFGsFname", None)
    _set_keyword(ci, "HCIWriteCFGsFname", cfgs_out)
    for keyword, value in (hci_options or {}).items():
        _set_keyword(ci, keyword, value)
    return blocks

def default_hummr_exe():
    """the standalone hummr executable next to the hummr_python module"""
    import hummr_python
    return str(Path(hummr_python.__file__).resolve().parent / "hummr")

def run_selection(blocks, name, scratch_dir, hummr_exe):
    """runs hummr on the input blocks and returns the selected CFGs (the input has to write them to <name>.cfgs)"""
    scratch_dir = Path(scratch_dir)
    write_blocks(blocks, scratch_dir / f"{name}.inp")
    with open(scratch_dir / f"{name}.out", "w") as out:
        result = subprocess.run([hummr_exe, f"{name}.inp"], cwd=scratch_dir, stdout=out, stderr=subprocess.STDOUT)
    cfgs_file = scratch_dir / f"{name}.cfgs"
    if result.returncode != 0 or not cfgs_file.exists():
        raise RuntimeError(f"hummr selection run {name} failed, see {scratch_dir / (name + '.out')}")
    return set(cfgs_file.read_text().split())

def selection_runs(sector, environments, field_strength, include_fields=True):
    """
    the independent runs of one sector (charge, mult): [(name, point charges [(q, (x, y, z))] or None, field or None)]
    environments: one environment per position of the (unique) fragment, each a list of neighbouring fragments
                  {"atoms": [(element, (x, y, z)), ...], "partial_charges": [q per atom] or None} with the
                  coordinates in Angstrom in the frame of the hummr input of the fragment
    """
    charge, mult = sector
    runs = [("bare", None, None)]
    for ienv, environment in enumerate(environments):
        if charge != 0:
            # charge-transfer partner: the opposite charge, spread over the atoms of the neighbour
            for ineigh, neighbour in enumerate(environment):
                n_atoms = len(neighbour["atoms"])
                charges = [(-charge / n_atoms, xyz) for _, xyz in neighbour["atoms"]]
                runs.append((f"ct_e{ienv}_n{ineigh}", charges, None))
        else:
            charges = [(q, xyz) for neighbour in environment if neighbour.get("partial_charges") is not None
                       for (_, xyz), q in zip(neighbour["atoms"], neighbour["partial_charges"])]
            if charges:
                runs.append((f"pc_e{ienv}", charges, None))
    for axis, label in enumerate("xyz" if include_fields else ""):
        for sign, sign_label in ((+1, "p"), (-1, "m")):
            field = [0.0, 0.0, 0.0]
            field[axis] = sign * field_strength
            runs.append((f"field_{label}{sign_label}", None, field))
    return [(f"c{charge}_m{mult}_{name}", charges, field) for name, charges, field in runs]

def select_spaces(input_file, sectors, n_roots, epsilon, environments, field_strength=1e-3, scratch_dir="selection",
                  hummr_exe=None, hci_options=None, include_fields=True):
    """
    Union of the independent HCI selections (see module docstring) for every sector of one (unique) fragment.
    input_file:     hummr input of the fragment (orbital guess, basis, frozen core/active space as for XR)
    sectors:        {charge: [mult, ...]}
    n_roots:        {charge: {mult: number of roots}}, the states the selection is done for
    epsilon:        HCI selection threshold (EpsilonHCI), the same for all runs
    environments:   see selection_runs
    field_strength: F in a.u. (weak-response regime, large enough to change the selection at epsilon)
    include_fields: False skips the six field selections (e.g. environments=[] and False: bare selection only)
    Returns {charge: {mult: set of CFGs}} and writes the unions to <scratch_dir>/union_c<charge>_m<mult>.cfgs
    (to be read with HCICFGsFname).
    """
    scratch_dir = Path(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    hummr_exe = hummr_exe or default_hummr_exe()
    base_blocks = read_blocks(input_file)
    selected = {}
    for charge, mults in sectors.items():
        for mult in mults:
            union = set()
            for name, charges, field in selection_runs((charge, mult), environments, field_strength, include_fields):
                point_charge_file = None
                if charges is not None:
                    point_charge_file = f"{name}.pc"
                    (scratch_dir / point_charge_file).write_text(
                        "".join(f"{q:.10f} {x:.10f} {y:.10f} {z:.10f}\n" for q, (x, y, z) in charges))
                blocks = selection_input(base_blocks, charge, mult, n_roots[charge][mult], epsilon, f"{name}.cfgs",
                                         point_charge_file=point_charge_file, field=field, hci_options=hci_options)
                cfgs = run_selection(blocks, name, scratch_dir, hummr_exe)
                print(f"selection {name}: {len(cfgs)} CFGs ({len(cfgs - union)} new)", flush=True)
                union |= cfgs
            (scratch_dir / f"union_c{charge}_m{mult}.cfgs").write_text("".join(f"{cfg}\n" for cfg in sorted(union)))
            selected.setdefault(charge, {})[mult] = union
    return selected

def n_csfs(cfgs, mult):
    """number of CSFs of the multiplicity in the given CFGs (genealogical, f(N, S) per CFG with N open shells)"""
    from math import comb
    two_s = mult - 1
    total = 0
    for cfg in cfgs:
        n_open = cfg.count("1")
        if n_open >= two_s and (n_open - two_s) % 2 == 0:
            k = (n_open - two_s) // 2
            total += comb(n_open, k) - (comb(n_open, k - 1) if k > 0 else 0)
    return total

def selected_csf_indices(csf_layout, selected):
    """
    CSF indices of the selected CFGs in the complete CSF space of every sector, e.g. to run the existing (FCI)
    density engine in the selected space.  csf_layout: {charge: {mult: [(cfg, pos, dim)]}} (HummrXRFragment.csf_layout)
    """
    indices = {}
    for charge, mults in selected.items():
        for mult, cfgs in mults.items():
            layout = csf_layout[charge][mult]
            missing = cfgs - {cfg for cfg, _, _ in layout}
            if missing:
                raise ValueError(f"selected CFGs {sorted(missing)[:3]}... are not part of sector (chg {charge}, mult {mult})")
            indices.setdefault(charge, {})[mult] = np.array([pos + mu for cfg, pos, dim in layout if cfg in cfgs
                                                             for mu in range(dim)], dtype=int)
    return indices

def project_states(vecs, indices, n_keep, thresh=1e-8):
    """the vectors projected onto the CSFs indices and orthonormalized (in order), at most n_keep of them"""
    out = []
    for vec in vecs:
        v = np.zeros_like(np.asarray(vec, dtype=float))
        v[indices] = np.asarray(vec, dtype=float)[indices]
        for u in out:
            v -= (u @ v) * u
        norm = np.linalg.norm(v)
        if norm > thresh:
            out.append(v / norm)
        if len(out) == n_keep:
            break
    return out
