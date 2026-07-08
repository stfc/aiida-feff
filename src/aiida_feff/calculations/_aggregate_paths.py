#!/usr/bin/env python3
"""Aggregate FEFF scattering paths into a compact HDF5 file.

This script is written to the FEFF working directory by
:class:`~aiida_feff.calculations.feff.FeffCalculation` during
``prepare_for_submission``.  It is intended to be executed **on the remote
compute machine** by a wrapper script immediately after FEFF completes::

    #!/bin/bash
    set -e
    /path/to/feff     # or: python3 -m larch.apps.feff
    python3 _aggregate_paths.py

The script reads ``_feff_aggregate_config.json`` (also written by the CalcJob)
and ``files.dat`` (produced by FEFF), amplitude-filters ``feff????.dat`` files,
and writes ``contributions_raw.h5`` which AiiDA retrieves in lieu of the
individual path files.

If ``_feff_aggregate_config.json`` is absent the script exits immediately
(no-op), so it is safe to include unconditionally in the wrapper.

HDF5 schema (contributions_raw.h5)
-----------------------------------
/meta                   group
    attrs:
        format_version  int        = 1
        frame_idx       int        frame index within the MD trajectory
        site_idx        int        absorbing-site index within the structure
        absorber_element str       element symbol of the absorber
        threshold       float      cw_ratio amplitude threshold used
/paths                  group
    k_grid_params       float64[M] native (coarse) FEFF k grid
    feff_data           float64[P, M, 6]  columns: real_phc, mag_feff,
                                           pha_feff, red_fact, lam, rep
    r_eff               float64[P]
    nlegs               int32[P]
    degeneracy          float64[P]
    scatterer           str[P]     variable-length UTF-8
    cw_ratio            float64[P]

P = number of paths kept after amplitude filtering
M = number of k points in the native FEFF grid

Dependencies
------------
larch (xraylarch) — used for ``FeffDatFile`` parsing (primary).
numpy, h5py       — must also be available; larch pulls both in.
The pure-Python text parser is used as a fallback if larch fails on a
specific file but larch must be present in the environment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def _parse_files_dat(text: str) -> dict[str, dict]:
    """Parse FEFF ``files.dat`` → per-path amplitude ratios.

    After a variable-length preamble the data block starts on the line
    containing ``"amp ratio"``.
    """
    results: dict[str, dict] = {}
    in_data = False
    files_dat_columns = 6
    for line in text.splitlines():
        stripped = line.strip()
        if not in_data:
            if "amp ratio" in stripped:
                in_data = True
            continue
        if not stripped:
            continue
        tokens = stripped.split()
        if len(tokens) < files_dat_columns:
            continue
        try:
            results[tokens[0]] = {
                "cw_ratio": float(tokens[2]),
                "sig2": float(tokens[1]),
                "deg": float(tokens[3]),
                "nlegs": int(tokens[4]),
                "r_eff": float(tokens[5]),
            }
        except (ValueError, IndexError):
            continue
    return results


def _parse_with_larch(fpath: Path) -> dict:
    """Parse a ``feff????.dat`` file using larch's ``FeffDatFile``.

    Returns a dict with keys ``path_idx``, ``r_eff``, ``nlegs``,
    ``degeneracy``, ``scatterer``, ``k``, ``feff_data`` (ndarray N×6:
    real_phc, mag_feff, pha_feff, red_fact, lam, rep).
    Returns an empty dict on failure.
    """
    from larch.xafs.feffdat import FeffDatFile

    try:
        dat = FeffDatFile(filename=str(fpath))
    except Exception:
        return {}

    if dat.k is None or len(dat.k) == 0:
        return {}

    try:
        feff_data = np.column_stack(
            [
                dat.real_phc,
                dat.mag_feff,
                dat.pha_feff,
                dat.red_fact,
                dat.lam,
                dat.rep,
            ]
        )
    except (AttributeError, ValueError):
        return {}

    scatterer = "?"
    try:
        # geom entries: (atsym, iz, ipot, amass, x, y, z)
        scat = [g[0] for g in dat.geom if int(g[2]) != 0]
        if scat:
            scatterer = "-".join(scat)
    except Exception:
        pass

    try:
        path_idx = int(fpath.stem.replace("feff", ""))
    except ValueError:
        path_idx = 0

    return {
        "path_idx": path_idx,
        "r_eff": float(dat.reff),
        "nlegs": int(dat.nleg),
        "degeneracy": float(dat.degen),
        "scatterer": scatterer,
        "k": np.asarray(dat.k),
        "feff_data": feff_data,
    }


def _parse_with_text(fpath: Path) -> dict:
    """Fallback pure-Python ``feff????.dat`` parser."""
    import io

    try:
        lines = fpath.read_text().splitlines()
    except (OSError, FileNotFoundError):
        return {}
    path_idx = 0
    nlegs = 0
    degeneracy = 1.0
    r_eff = 0.0
    atom_lines: list[str] = []
    data_start: int | None = None
    nleg_line_seen = False
    atom_header_seen = False
    in_atoms = False

    atom_symbol_col_6 = 5
    atom_symbol_col_5 = 4
    atom_columns = 7
    for i, line in enumerate(lines):
        stripped = line.strip()
        if "Path" in line and "icalc" in line:
            parts = stripped.split()
            try:
                path_idx = int(parts[1])
            except (IndexError, ValueError):
                pass
            continue
        if "nleg" in line and "deg" in line and "reff" in line:
            parts = stripped.split()
            try:
                nlegs, degeneracy, r_eff = int(parts[0]), float(parts[1]), float(parts[2])
            except (IndexError, ValueError):
                pass
            nleg_line_seen = True
            continue
        if nleg_line_seen and not atom_header_seen and "pot" in stripped:
            atom_header_seen = True
            in_atoms = True
            continue
        if in_atoms:
            if len(atom_lines) >= nlegs:
                in_atoms = False
            else:
                parts = stripped.split()
                if len(parts) >= atom_symbol_col_6 + 1:
                    atom_lines.append(parts[atom_symbol_col_6])
                elif len(parts) >= atom_symbol_col_5 + 1:
                    atom_lines.append(parts[atom_symbol_col_5])
                else:
                    atom_lines.append("?")
                continue
        if line.rstrip().endswith("@#"):
            data_start = i + 1
            break

    if data_start is None:
        return {}

    scatterers = atom_lines[1:] if len(atom_lines) > 1 else list(atom_lines)
    scatterer = "-".join(s for s in scatterers if s) or "?"
    data_lines = [ln for ln in lines[data_start:] if ln.strip()]
    if not data_lines:
        return {}
    try:
        arr = np.loadtxt(io.StringIO("\n".join(data_lines)))
    except ValueError:
        return {}
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < atom_columns:
        return {}
    return {
        "path_idx": path_idx,
        "nlegs": nlegs,
        "degeneracy": degeneracy,
        "r_eff": r_eff,
        "scatterer": scatterer,
        "k": arr[:, 0],
        "feff_data": arr[:, 1:],
    }


def main() -> None:
    cwd = Path(".")

    # ----------------------------------------------------------------
    # Read config — no-op if absent
    # ----------------------------------------------------------------
    config_path = cwd / "_feff_aggregate_config.json"
    if not config_path.exists():
        sys.exit(0)

    with config_path.open() as fh:
        config = json.load(fh)

    threshold = float(config.get("threshold", 0.0))
    frame_idx = int(config.get("frame_idx", 0))
    site_idx = int(config.get("site_idx", 0))
    absorber_element = str(config.get("absorber_element", ""))

    # ----------------------------------------------------------------
    # Parse files.dat for amplitude ranking
    # ----------------------------------------------------------------
    files_dat = cwd / "files.dat"
    amplitude_map: dict[str, dict] = {}
    if files_dat.exists():
        amplitude_map = _parse_files_dat(files_dat.read_text())
    else:
        print("WARNING: files.dat not found; amplitude filtering disabled.", file=sys.stderr)

    # ----------------------------------------------------------------
    # Filter feff????.dat files by cw_ratio
    # ----------------------------------------------------------------
    all_dat = sorted(cwd.glob("feff????.dat"))
    if not all_dat:
        print("No feff????.dat files found; contributions_raw.h5 not written.", file=sys.stderr)
        sys.exit(0)

    if amplitude_map:
        qualifying = [
            f for f in all_dat if amplitude_map.get(f.name, {}).get("cw_ratio", 100.0) >= threshold
        ]
        print(
            f"Path amplitude filter (threshold={threshold}%): "
            f"keeping {len(qualifying)}/{len(all_dat)} paths.",
            file=sys.stderr,
        )
    else:
        qualifying = list(all_dat)

    if not qualifying:
        print(
            f"All {len(all_dat)} paths below threshold={threshold}%; "
            "contributions_raw.h5 not written.",
            file=sys.stderr,
        )
        sys.exit(0)

    # ----------------------------------------------------------------
    # Parse with larch (primary), text parser (fallback)
    # ----------------------------------------------------------------
    import h5py

    try:
        from larch.xafs.feffdat import FeffDatFile as _  # noqa: F401

        larch_ok = True
    except ImportError:
        larch_ok = False
        print(
            "WARNING: larch not available; falling back to text parser.",
            file=sys.stderr,
        )

    paths = []
    for fpath in qualifying:
        cw_ratio = amplitude_map.get(fpath.name, {}).get("cw_ratio", 0.0)
        parsed: dict = {}
        if larch_ok:
            parsed = _parse_with_larch(fpath)
            if not parsed:
                print(f"  larch failed on {fpath.name}; trying text parser.", file=sys.stderr)
        if not parsed:
            parsed = _parse_with_text(fpath)
        if not parsed:
            print(f"WARNING: could not parse {fpath.name}; skipping.", file=sys.stderr)
            continue
        parsed["cw_ratio"] = cw_ratio
        paths.append(parsed)

    if not paths:
        print("No paths parsed successfully; contributions_raw.h5 not written.", file=sys.stderr)
        sys.exit(0)

    # ----------------------------------------------------------------
    # Write contributions_raw.h5  (columnar schema v2)
    # ----------------------------------------------------------------
    _COMPRESS = {"compression": "gzip", "compression_opts": 6}

    k_grid_params = np.asarray(paths[0]["k"], dtype=np.float64)
    n_paths = len(paths)
    m_k = len(k_grid_params)

    feff_data_columns = 6
    feff_data_arr = np.zeros((n_paths, m_k, feff_data_columns), dtype=np.float64)
    r_eff_arr = np.zeros(n_paths, dtype=np.float64)
    nlegs_arr = np.zeros(n_paths, dtype=np.int32)
    degeneracy_arr = np.zeros(n_paths, dtype=np.float64)
    scatterer_list: list[str] = []
    cw_ratio_arr = np.zeros(n_paths, dtype=np.float64)

    for idx, p in enumerate(paths):
        fd = np.asarray(p["feff_data"], dtype=np.float64)
        # feff_data may have fewer than 6 columns (text parser gives col 1-6 of the file)
        if fd.shape[1] >= feff_data_columns:
            feff_data_arr[idx] = fd[:m_k, :feff_data_columns]
        else:
            feff_data_arr[idx, :, : fd.shape[1]] = fd[:m_k]
        r_eff_arr[idx] = float(p["r_eff"])
        nlegs_arr[idx] = int(p["nlegs"])
        degeneracy_arr[idx] = float(p["degeneracy"])
        scatterer_list.append(str(p["scatterer"]))
        cw_ratio_arr[idx] = float(p["cw_ratio"])

    out_path = cwd / "contributions_raw.h5"
    with h5py.File(out_path, "w") as hf:
        meta = hf.create_group("meta")
        meta.attrs["format_version"] = 1
        meta.attrs["frame_idx"] = frame_idx
        meta.attrs["site_idx"] = site_idx
        meta.attrs["absorber_element"] = absorber_element
        meta.attrs["threshold"] = threshold

        pg = hf.create_group("paths")
        pg.create_dataset("k_grid_params", data=k_grid_params, **_COMPRESS)
        pg.create_dataset("feff_data", data=feff_data_arr, **_COMPRESS)
        pg.create_dataset("r_eff", data=r_eff_arr, **_COMPRESS)
        pg.create_dataset("nlegs", data=nlegs_arr, **_COMPRESS)
        pg.create_dataset("degeneracy", data=degeneracy_arr, **_COMPRESS)
        dt_str = h5py.string_dtype(encoding="utf-8")
        pg.create_dataset(
            "scatterer",
            data=np.array(scatterer_list, dtype=dt_str),
            **_COMPRESS,
        )
        pg.create_dataset("cw_ratio", data=cw_ratio_arr, **_COMPRESS)

    print(f"Wrote {n_paths} paths to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
