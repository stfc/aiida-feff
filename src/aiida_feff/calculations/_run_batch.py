#!/usr/bin/env python3
r"""Batch FEFF driver — runs on the remote compute node, no AiiDA dependency.

Written to the job's working directory by
:class:`~aiida_feff.calculations.feff_batch.FeffBatchCalculation` during
``prepare_for_submission`` and executed as the job's main command::

    python3 _run_batch.py

Reads ``batch_config.json`` (also written by the CalcJob) and:

1. Copies pre-computed potential files (if any) into each snapshot run-dir.
2. Generates a per-run ``_run_feff.sh`` wrapper that applies the FEFF
   module-load environment and calls the FEFF executable.
3. Runs all FEFF instances in parallel via ``concurrent.futures``
   (one per allocated core, derived from ``SLURM_CPUS_ON_NODE``).
4. Optionally runs ``_aggregate_paths.py`` per run-dir (also parallel).

Partial failures (some FEFF runs crash) are logged to stderr and do NOT abort
the whole job — AiiDA's batch parser handles missing outputs gracefully.

``batch_config.json`` schema::

    {
        "pairs":          [[frame_idx, site_idx], ...],
        "feff_executable": "/full/path/to/feff8l",
        "feff_prepend":   "module load feff/8.5\n",
        "feff_append":    "",
        "n_workers":      null,
        "do_aggregate":   false,
        "threshold":      5.0
    }
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def snap_label(frame_idx: int, site_idx: int) -> str:
    """Canonical name of the run directory for a (frame, site) pair."""
    return f"snap_{frame_idx:04d}_site_{site_idx:04d}"


def copy_potentials(snap_dir: Path, site_idx: int) -> None:
    """Copy pre-computed potential files into *snap_dir* if available."""
    pot_dir = Path(f"potentials/site_{site_idx:04d}")
    if not pot_dir.exists():
        return
    for src in pot_dir.iterdir():
        shutil.copy(src, snap_dir / src.name)


def write_feff_wrapper(snap_dir: Path, feff_exe: str, prepend: str, append: str) -> Path:
    """Write a small bash wrapper that sets up the FEFF environment and runs FEFF."""
    wrapper = snap_dir / "_run_feff.sh"
    lines = ["#!/bin/bash", "set -e"]
    if prepend.strip():
        lines.append(prepend.rstrip())
    lines.append(feff_exe)
    if append.strip():
        lines.append(append.rstrip())
    wrapper.write_text("\n".join(lines) + "\n")
    return wrapper


def run_feff_one(label: str, feff_exe: str, prepend: str, append: str) -> str:
    """Copy potentials, write wrapper, run FEFF in *label* directory.

    Returns *label* on success.  Raises on failure (caller catches).
    """
    snap_dir = Path(label)
    snap_dir.mkdir(exist_ok=True)

    # Parse site index from label: snap_FFFF_site_SSSS
    site_idx = int(label.rsplit("_site_", maxsplit=1)[-1])
    copy_potentials(snap_dir, site_idx)

    wrapper = write_feff_wrapper(snap_dir, feff_exe, prepend, append)
    result = subprocess.run(  # noqa: PLW1510
        ["bash", str(wrapper.name)],
        cwd=str(snap_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    # Write FEFF stdout/stderr for retrieval
    (snap_dir / "log.dat").write_text(result.stdout)
    (snap_dir / "stderr.txt").write_text(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"FEFF exited {result.returncode}:\n{result.stderr[-500:]}")
    return label


def run_aggregate_one(label: str, agg_script: str) -> str:
    """Run ``_aggregate_paths.py`` inside the run directory.

    Returns *label* on success.  Raises on failure.
    """
    result = subprocess.run(
        [sys.executable, agg_script],
        cwd=label,
        capture_output=True,
        text=True,
        check=False,
    )
    # Append aggregation stderr to existing stderr file
    stderr_file = Path(label) / "stderr.txt"
    with stderr_file.open("a") as fh:
        fh.write("\n--- aggregation ---\n")
        fh.write(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"aggregate_paths exited {result.returncode}:\n{result.stderr[-300:]}")
    return label


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point."""
    config_path = Path("batch_config.json")
    if not config_path.exists():
        print("ERROR: batch_config.json not found", file=sys.stderr)
        sys.exit(1)

    with config_path.open() as fh:
        config = json.load(fh)

    pairs: list[list[int]] = config["pairs"]
    feff_exe: str = config["feff_executable"]
    feff_prepend: str = config.get("feff_prepend", "")
    feff_append: str = config.get("feff_append", "")
    do_aggregate: bool = config.get("do_aggregate", False)

    # ponytail: n_workers=null → read SLURM env; covers both ntasks and cpus-per-node configs
    n_workers: int | None = config.get("n_workers")
    if n_workers is None:
        n_workers = int(
            os.environ.get(
                "SLURM_CPUS_ON_NODE",
                os.environ.get("SLURM_NTASKS", "1"),
            )
        )

    labels = [snap_label(f, s) for f, s in pairs]
    agg_script = str(Path("_aggregate_paths.py").resolve())

    print(
        f"Batch FEFF: {len(labels)} runs, {n_workers} workers, aggregate={do_aggregate}",
        file=sys.stderr,
    )

    # ----------------------------------------------------------------
    # Step 1: run FEFF in parallel
    # ----------------------------------------------------------------
    successful: list[str] = []
    failed: list[str] = []

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(run_feff_one, lbl, feff_exe, feff_prepend, feff_append): lbl
            for lbl in labels
        }
        for fut in as_completed(futures):
            lbl = futures[fut]
            try:
                fut.result()
                successful.append(lbl)
                print(f"OK  {lbl}", file=sys.stderr)
            except Exception:  # noqa: BLE001
                failed.append(lbl)
                print(f"FAIL {lbl}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

    print(
        f"FEFF done: {len(successful)} OK, {len(failed)} failed",
        file=sys.stderr,
    )

    # ----------------------------------------------------------------
    # Step 2: aggregate paths (parallel, same pool size)
    # ----------------------------------------------------------------
    if do_aggregate and successful and Path(agg_script).exists():
        agg_failed: list[str] = []
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(run_aggregate_one, lbl, agg_script): lbl for lbl in successful}
            for fut in as_completed(futures):
                lbl = futures[fut]
                try:
                    fut.result()
                    print(f"AGG OK  {lbl}", file=sys.stderr)
                except Exception:  # noqa: BLE001
                    agg_failed.append(lbl)
                    print(f"AGG FAIL {lbl}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
        if agg_failed:
            print(
                f"Aggregation failed for {len(agg_failed)} run(s): {agg_failed}",
                file=sys.stderr,
            )

    # Always exit 0 — partial failures are handled by AiiDA's batch parser.
    sys.exit(0)


if __name__ == "__main__":
    main()
