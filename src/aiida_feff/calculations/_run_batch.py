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
   (one worker per allocated core, taken from ``batch_config.json``).
4. Optionally runs ``_aggregate_paths.py`` per run-dir (also parallel).

Partial failures (some FEFF runs crash) are logged to stderr and do NOT abort
the whole job — AiiDA's batch parser handles missing outputs gracefully.  A
run in which *every* FEFF failed exits non-zero, so a wholly broken batch is
visible from the scheduler rather than only from the parser.

``batch_config.json`` schema::

    {
        "pairs":          [[frame_idx, site_idx], ...],
        "feff_executable": "/full/path/to/feff8l",
        "feff_prepend":   "module load feff/8.5\n",
        "feff_append":    "",
        "n_workers":      64,
        "do_aggregate":   false,
        "threshold":      5.0,
        "run_timeout_seconds": 3240.0
    }
"""

from __future__ import annotations

import itertools
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
    wrapper.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return wrapper


def run_feff_one(
    label: str,
    feff_exe: str,
    prepend: str,
    append: str,
    timeout: float | None = None,
) -> str:
    """Copy potentials, write wrapper, run FEFF in *label* directory.

    Returns *label* on success.  Raises on failure (caller catches).
    A run exceeding *timeout* seconds is killed, so one hung FEFF cannot hold
    a worker until the scheduler kills the whole job.
    """
    snap_dir = Path(label)
    snap_dir.mkdir(exist_ok=True)

    # Parse site index from label: snap_FFFF_site_SSSS
    site_idx = int(label.rsplit("_site_", maxsplit=1)[-1])
    copy_potentials(snap_dir, site_idx)

    wrapper = write_feff_wrapper(snap_dir, feff_exe, prepend, append)
    try:
        result = subprocess.run(  # noqa: PLW1510
            [_bash(), str(wrapper.name)],
            cwd=str(snap_dir),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        _write_text(snap_dir / "log.dat", exc.stdout)
        _write_text(snap_dir / "stderr.txt", exc.stderr)
        raise RuntimeError(f"FEFF exceeded the {timeout:.0f} s run timeout") from exc

    # Write FEFF stdout/stderr for retrieval
    _write_text(snap_dir / "log.dat", result.stdout)
    _write_text(snap_dir / "stderr.txt", result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"FEFF exited {result.returncode}:\n{result.stderr[-500:]}")
    return label


def run_aggregate_one(label: str, agg_script: str, timeout: float | None = None) -> str:
    """Run ``_aggregate_paths.py`` inside the run directory.

    Returns *label* on success.  Raises on failure.
    """
    try:
        result = subprocess.run(  # noqa: PLW1510
            [sys.executable, agg_script],
            cwd=label,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"aggregate_paths exceeded the {timeout:.0f} s run timeout") from exc

    # Append aggregation stderr to existing stderr file
    stderr_file = Path(label) / "stderr.txt"
    with stderr_file.open("a", encoding="utf-8") as fh:
        fh.write("\n--- aggregation ---\n")
        fh.write(result.stderr or "")
    if result.returncode != 0:
        raise RuntimeError(f"aggregate_paths exited {result.returncode}:\n{result.stderr[-300:]}")
    return label


def _bash() -> str:
    """Absolute path to bash, falling back to the bare name."""
    return shutil.which("bash") or "bash"


def _write_text(path: Path, text: str | bytes | None) -> None:
    """Write *text* to *path*, tolerating ``None`` and undecodable bytes."""
    if text is None:
        text = ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    path.write_text(text, encoding="utf-8")


def resolve_n_workers(configured: int | None) -> int:
    """Return the worker count, preferring the value the CalcJob computed.

    The CalcJob derives it from ``metadata.options.resources``, which every
    scheduler fills in.  The environment is consulted only as a fallback, and
    values like SLURM's ``"32(x2)"`` are parsed rather than crashing ``int()``.
    """
    if configured:
        return max(1, int(configured))
    for var in ("SLURM_CPUS_ON_NODE", "SLURM_NTASKS", "NCPUS", "LSB_DJOB_NUMPROC"):
        raw = os.environ.get(var)
        if not raw:
            continue
        digits = "".join(itertools.takewhile(str.isdigit, raw.strip()))
        if digits:
            return max(1, int(digits))
    return 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point."""
    config_path = Path("batch_config.json")
    if not config_path.exists():
        print("ERROR: batch_config.json not found", file=sys.stderr)
        sys.exit(1)

    with config_path.open(encoding="utf-8") as fh:
        config = json.load(fh)

    pairs: list[list[int]] = config["pairs"]
    feff_exe: str = config["feff_executable"]
    feff_prepend: str = config.get("feff_prepend", "")
    feff_append: str = config.get("feff_append", "")
    do_aggregate: bool = config.get("do_aggregate", False)
    run_timeout: float | None = config.get("run_timeout_seconds")

    n_workers = resolve_n_workers(config.get("n_workers"))

    labels = [snap_label(f, s) for f, s in pairs]
    agg_path = Path("_aggregate_paths.py")
    agg_script = str(agg_path.resolve())

    print(
        f"Batch FEFF: {len(labels)} runs, {n_workers} workers, "
        f"aggregate={do_aggregate}, timeout={run_timeout}",
        file=sys.stderr,
    )

    # ----------------------------------------------------------------
    # Step 1: run FEFF in parallel
    # ----------------------------------------------------------------
    successful: list[str] = []
    failed: list[str] = []

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(run_feff_one, lbl, feff_exe, feff_prepend, feff_append, run_timeout): lbl
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
    if do_aggregate and not agg_path.exists():
        print(
            "ERROR: aggregation requested but _aggregate_paths.py is missing; "
            "path_contributions will be absent.",
            file=sys.stderr,
        )

    if do_aggregate and successful and agg_path.exists():
        agg_failed: list[str] = []
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(run_aggregate_one, lbl, agg_script, run_timeout): lbl
                for lbl in successful
            }
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

    # Partial failure is normal in an MD ensemble and the parser handles it.
    # A batch where nothing ran is a job failure, and saying so through the
    # exit status makes it visible to the scheduler as well as the parser.
    sys.exit(1 if successful == [] else 0)


if __name__ == "__main__":
    main()
