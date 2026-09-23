#!/usr/bin/env python3
r"""Batch FEFF driver — runs on the remote compute node, no AiiDA dependency.

Needs md-exafs on that interpreter, and checks for it before running any FEFF.
The shard it writes is what the ensemble average is built from, so a missing
md-exafs means the job cannot produce its deliverable; discovering that after
the batch has run costs the whole allocation.

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
        "run_timeout_seconds": 3240.0,
        "clean_scratch":  false,
        "stream_chunk_size": 256,
        "absorber_elements": ["Fe", "Fe"],
        "scratch_keep_files": ["chi.dat", "files.dat", "log.dat",
                               "paths.dat", "stderr.txt"]
    }

``clean_scratch`` strips each run directory down to ``scratch_keep_files`` once
its spectrum has been written to the shard.  That list is exactly what the
calcjob retrieves, so stripping cannot change any parsed quantity -- it only
removes the ``feffNNNN.dat`` path files and potential binaries that would
otherwise accumulate across the whole batch and exhaust scratch.
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
from typing import Any

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


def strip_run_dir(snap_dir: Path, keep: set[str]) -> int:
    """Delete everything in ``snap_dir`` except the basenames in ``keep``.

    ``keep`` is the calcjob's retrieve list, so what survives is exactly what
    AiiDA will pull back and the parser may read.  Stripping is therefore
    invisible to every output node: it reclaims the ``feffNNNN.dat`` path files
    and potential binaries, which are never retrieved, and nothing else.

    Returns the number of entries removed.
    """
    if not snap_dir.is_dir():
        return 0
    removed = 0
    for entry in snap_dir.iterdir():
        if entry.name in keep:
            continue
        try:
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink()
            removed += 1
        except OSError as exc:
            print(f"Warning: could not remove {entry}: {exc}", file=sys.stderr)
    return removed


def collect_chunk(
    writer: Any,
    chunk_pairs: list[list[int]],
    chunk_elements: list[str],
    chunk_successful: set[str],
    k_grid: Any,
    threshold: float,
    do_aggregate: bool,
) -> int:
    """Append each successful run in the chunk to the open shard.

    Returns the number of tasks actually written; a run whose ``chi.dat`` could
    not be parsed is skipped rather than zero-filled.
    """
    from md_exafs.execution import FeffTask, collect_task_into_shard

    n_written = 0
    for (f, s), element in zip(chunk_pairs, chunk_elements, strict=True):
        lbl = snap_label(f, s)
        if lbl not in chunk_successful:
            continue
        task = FeffTask(
            frame_idx=f,
            site_idx=s,
            input_dir=Path(lbl),
            absorber_element=element,
        )
        try:
            if collect_task_into_shard(
                writer,
                task,
                k_grid=k_grid,
                threshold=threshold,
                store_paths=do_aggregate,
            ):
                n_written += 1
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: collect_task_into_shard failed for {lbl}: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
    return n_written


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
    clean_scratch: bool = config.get("clean_scratch", False)
    stream_chunk_size: int | None = config.get("stream_chunk_size", 256)
    threshold: float = float(config.get("threshold", 0.0))
    elements: list[str] = config.get("absorber_elements") or [""] * len(pairs)
    keep_files: set[str] = set(config.get("scratch_keep_files", []))

    if clean_scratch and not keep_files:
        # Stripping with an empty keep-set would delete the outputs the parser
        # needs. Refuse rather than silently destroy the batch.
        print(
            "ERROR: clean_scratch is set but scratch_keep_files is empty; refusing to strip.",
            file=sys.stderr,
        )
        sys.exit(1)

    n_workers = resolve_n_workers(config.get("n_workers"))
    chunk_size = stream_chunk_size if stream_chunk_size and stream_chunk_size > 0 else len(pairs)

    agg_path = Path("_aggregate_paths.py")
    agg_script = str(agg_path.resolve())

    print(
        f"Batch FEFF: {len(pairs)} runs, {n_workers} workers, chunk_size={chunk_size}, "
        f"clean_scratch={clean_scratch}, aggregate={do_aggregate}, timeout={run_timeout}",
        file=sys.stderr,
    )

    if do_aggregate and not agg_path.exists():
        print(
            "ERROR: aggregation requested but _aggregate_paths.py is missing; "
            "path_contributions will be absent.",
            file=sys.stderr,
        )

    # md-exafs owns the shard format; we only ran FEFF. Reusing BatchShardWriter
    # and collect_task_into_shard keeps a shard written here byte-compatible with
    # one written by md-exafs' own BatchExecutor.
    #
    # Checked before the first FEFF starts. The shard is what the ensemble
    # average is built from, so an interpreter without md-exafs cannot produce
    # the deliverable, and finding that out after the batch has run costs the
    # whole allocation.
    try:
        from md_exafs.execution import DEFAULT_K_GRID, BatchShardWriter
    except ImportError as exc:
        print(
            f"ERROR: batch mode requires md-exafs on this interpreter ({sys.executable}), "
            f"which is the 'python_code' given to FeffBatchCalculation: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    k_grid = DEFAULT_K_GRID
    shard_path = Path("batch_shard.h5")
    writer = BatchShardWriter(shard_path, k_grid=k_grid, threshold=threshold)

    successful: list[str] = []
    failed: list[str] = []
    n_written = 0

    try:
        # One pool for the whole batch: chunking exists to bound peak disk, not
        # to partition the workers, and re-forking a pool per chunk costs
        # n_workers process spawns every time.
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for chunk_start in range(0, len(pairs), chunk_size):
                chunk_pairs = pairs[chunk_start : chunk_start + chunk_size]
                chunk_elements = elements[chunk_start : chunk_start + chunk_size]
                chunk_labels = [snap_label(f, s) for f, s in chunk_pairs]

                # --------------------------------------------------------
                # Step 1: run FEFF in parallel for this chunk
                # --------------------------------------------------------
                chunk_successful: list[str] = []
                futures = {
                    pool.submit(
                        run_feff_one, lbl, feff_exe, feff_prepend, feff_append, run_timeout
                    ): lbl
                    for lbl in chunk_labels
                }
                for fut in as_completed(futures):
                    lbl = futures[fut]
                    try:
                        fut.result()
                        chunk_successful.append(lbl)
                        successful.append(lbl)
                        print(f"OK  {lbl}", file=sys.stderr)
                    except Exception:  # noqa: BLE001
                        failed.append(lbl)
                        print(f"FAIL {lbl}", file=sys.stderr)
                        traceback.print_exc(file=sys.stderr)

                # --------------------------------------------------------
                # Step 2: aggregate paths for this chunk
                # --------------------------------------------------------
                if do_aggregate and chunk_successful and agg_path.exists():
                    agg_failed: list[str] = []
                    futures = {
                        pool.submit(run_aggregate_one, lbl, agg_script, run_timeout): lbl
                        for lbl in chunk_successful
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

                # --------------------------------------------------------
                # Step 3: collect into the shard, then reclaim scratch
                # --------------------------------------------------------
                n_written += collect_chunk(
                    writer,
                    chunk_pairs,
                    chunk_elements,
                    set(chunk_successful),
                    k_grid,
                    threshold,
                    do_aggregate,
                )

                # Strip every run in the chunk, successful or not: the files
                # removed are never retrieved, so a failed run keeps its
                # log.dat and stderr.txt for diagnosis either way.
                if clean_scratch:
                    n_stripped = sum(strip_run_dir(Path(lbl), keep_files) for lbl in chunk_labels)
                    print(
                        f"CLEAN  chunk at {chunk_start}: removed {n_stripped} scratch entries",
                        file=sys.stderr,
                    )
    finally:
        try:
            writer.close()
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: Failed to close batch_shard.h5: {exc}", file=sys.stderr)

    print(
        f"FEFF done: {len(successful)} OK, {len(failed)} failed",
        file=sys.stderr,
    )

    print(
        f"SHARD OK  batch_shard.h5 written ({n_written}/{len(successful)} tasks)",
        file=sys.stderr,
    )
    if n_written == 0:
        shard_path.unlink(missing_ok=True)
        print(
            "Warning: no task produced a parseable chi.dat; batch_shard.h5 not written",
            file=sys.stderr,
        )
    elif n_written < len(successful):
        print(
            f"Warning: {len(successful) - n_written} task(s) ran but produced no "
            "parseable chi.dat and were omitted from batch_shard.h5",
            file=sys.stderr,
        )

    # Partial failure is normal in an MD ensemble and the parser handles it.
    # A batch where nothing ran is a job failure, and saying so through the
    # exit status makes it visible to the scheduler as well as the parser.
    sys.exit(1 if successful == [] else 0)


if __name__ == "__main__":
    main()
