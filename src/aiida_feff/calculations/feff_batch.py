"""FeffBatchCalculation — single Slurm job that runs many FEFF calculations.

Instead of fanning out one :class:`FeffCalculation` per ``(frame, site)`` pair
(which creates N × M individual scheduler jobs), this CalcJob packs an
arbitrary number of pairs into **one** scheduler job.  A Python driver script
runs the FEFF instances in parallel using ``concurrent.futures`` with one
worker per allocated core, then optionally aggregates scattering paths.

Typical HPC resource specification (adapt to your cluster)::

    options = {
        "resources": {"num_machines": 1, "num_mpiprocs_per_machine": 64},
        "max_wallclock_seconds": 3600,
        "queue_name": "regular",
    }

The ``code`` input is the **Python interpreter** on the HPC (the same one
configured as ``python_code`` in the ensemble workflow), **not** FEFF.
FEFF itself is passed as ``feff_code``.  The Python driver calls FEFF via
``subprocess`` after embedding the FEFF executable path and its environment
setup (``prepend_text`` / ``append_text``) into a per-run shell wrapper.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from aiida import orm
from aiida.common import CalcInfo, CodeInfo, datastructures
from aiida.engine import CalcJob, CalcJobProcessSpec

from aiida_feff.calculations.feff import (
    CONTROL_NO_POT,
    FEFF_AGGREGATE_SCRIPT,
    FEFF_CHI_FILE,
    FEFF_FILES_DAT,
    FEFF_PATHS_FILE,
    FEFF_POTENTIAL_FILES,
    FeffCalculation,
    absorber_element,
    as_lf_bytes,
)
from aiida_feff.data.archive import ExafsArchiveData
from aiida_feff.data.parameters import FeffParameters

logger = logging.getLogger(__name__)

# Path to bundled driver (installed alongside this module)
_DRIVER_PATH = Path(__file__).parent / "_run_batch.py"
_AGGREGATE_PATH = Path(__file__).parent / FEFF_AGGREGATE_SCRIPT

# File names
BATCH_DRIVER = "_run_batch.py"
BATCH_CONFIG = "batch_config.json"
BATCH_LOG = "batch.log"
BATCH_ERR = "batch_err.log"

# Retrieve glob patterns for per-run outputs (depth=None preserves directory structure)
_SNAP_RETRIEVE = [
    ("batch_shard.h5", ".", 0),
    (f"snap_*/{FEFF_CHI_FILE}", ".", None),
    (f"snap_*/{FEFF_PATHS_FILE}", ".", None),
    (f"snap_*/{FEFF_FILES_DAT}", ".", None),
    ("snap_*/log.dat", ".", None),
    ("snap_*/stderr.txt", ".", None),
    (BATCH_LOG, ".", 0),
    (BATCH_ERR, ".", 0),
]
_CONTRIBUTIONS_GLOB = ("snap_*/contributions_raw.h5", ".", None)


def _scratch_keep_files(do_aggregate: bool) -> list[str]:
    """Basenames inside a ``snap_*`` directory that must survive ``clean_scratch``.

    Derived from the retrieve list rather than restated, so the set the driver
    preserves on the cluster cannot drift from the set AiiDA pulls back.  A file
    that is retrieved is by definition one the parser may read; deleting
    anything else is invisible to every downstream node, which is what makes
    ``clean_scratch`` lossless rather than a second, quieter physics setting.
    """
    patterns = [*_SNAP_RETRIEVE, _CONTRIBUTIONS_GLOB] if do_aggregate else list(_SNAP_RETRIEVE)
    prefix = "snap_*/"
    return sorted({p[len(prefix) :] for p, _, _ in patterns if p.startswith(prefix)})


# Fraction of the job's wallclock a single FEFF run may consume before the
# driver kills it.  Without a bound one hung run holds a worker until the
# scheduler kills the job, losing every result in the chunk.
_RUN_TIMEOUT_FRACTION = 0.9


class FeffBatchCalculation(CalcJob):
    r"""CalcJob that runs a batch of FEFF calculations in a single scheduler job.

    Inputs
    ------
    code : :class:`~aiida.orm.AbstractCode`
        **Python interpreter** on the HPC (same as ``python_code``
        in the ensemble workflow).  Used to run the batch driver script.
    feff_code : :class:`~aiida.orm.AbstractCode`
        The FEFF executable (``InstalledCode`` recommended).  Its
        ``prepend_text`` (module loads, etc.) is embedded in each run's
        shell wrapper at submission time.
    trajectory : :class:`~aiida.orm.TrajectoryData`
        MD trajectory; structures are extracted by the frame indices supplied
        in ``frame_indices``.
    frame_indices : :class:`~aiida.orm.List`
        Ordered list of integer frame indices into *trajectory* — one entry
        per ``(frame, site)`` pair to run.
    site_indices : :class:`~aiida.orm.List`
        Ordered list of integer absorber-site indices — parallel to
        ``frame_indices``.
    parameters : :class:`~aiida_feff.data.parameters.FeffParameters`
        Shared FEFF parameters.  ``absorbing_atom`` is overridden per site.
    remote_potentials : dynamic namespace of :class:`~aiida.orm.RemoteData`, optional
        Pre-computed potential files, keyed ``site_0000``, ``site_0001``, …
        (as produced by the potentials-only step in
        :class:`~aiida_feff.workflows.ensemble.EnsembleExafsWorkChain`).
        When present, potential files are copied into each run directory and
        ``CONTROL 0 0 0 1 1 1`` is applied so FEFF skips SCF.
    path_cw_threshold : :class:`~aiida.orm.Float`, optional, default -1.0
        Curved-wave amplitude threshold for path aggregation.  Values ≥ 0
        trigger ``_aggregate_paths.py`` after each FEFF run.
    n_workers : :class:`~aiida.orm.Int`, optional
        Number of parallel workers.  Defaults to the core count implied by
        ``metadata.options.resources``, which works on every scheduler; the
        driver only consults the environment if that is unavailable.

    Outputs
    -------
    xas_data : dynamic namespace of :class:`~aiida_feff.data.xasdata.XasData`
        One node per successful run, keyed ``snap_FFFF_site_SSSS``.
    path_contributions : dynamic namespace of PathContributionsData, optional
        One node per successful aggregated run, same key scheme.
    """

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:  # type: ignore[override]
        """Define inputs, outputs, exit codes."""
        super().define(spec)

        spec.input("feff_code", valid_type=orm.AbstractCode, help="FEFF executable on the HPC.")
        spec.input("trajectory", valid_type=orm.TrajectoryData)
        spec.input(
            "frame_indices",
            valid_type=orm.List,
            help="Trajectory step indices for each (frame, site) pair.",
        )
        spec.input(
            "site_indices",
            valid_type=orm.List,
            help="Absorber site indices, parallel to frame_indices.",
        )
        spec.input("parameters", valid_type=FeffParameters)
        spec.input_namespace(
            "remote_potentials",
            valid_type=orm.RemoteData,
            dynamic=True,
            required=False,
            help="Pre-computed potentials keyed site_0000, site_0001, …",
        )
        spec.input(
            "path_cw_threshold",
            valid_type=orm.Float,
            default=lambda: orm.Float(-1.0),
        )
        spec.input(
            "n_workers",
            valid_type=orm.Int,
            required=False,
            help="Parallel workers; defaults to the core count in metadata.options.resources.",
        )
        spec.input(
            "clean_scratch",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help=(
                "Strip each snapshot directory of files that are not retrieved, on "
                "the fly, once its spectrum is in batch_shard.h5. Removes the "
                "feffNNNN.dat path files and potential binaries that dominate "
                "scratch usage. Every retrieved file is kept, so parsed results are "
                "unchanged; only the remote working directory shrinks."
            ),
        )
        spec.input(
            "stream_chunk_size",
            valid_type=orm.Int,
            default=lambda: orm.Int(256),
            help=(
                "Number of runs executed before their output is collected and "
                "scratch is stripped. Bounds peak disk and inode usage at roughly "
                "this many fully-populated run directories."
            ),
        )

        spec.inputs["metadata"]["options"]["parser_name"].default = "feff.feff_batch"  # type: ignore[index]
        spec.inputs["metadata"]["options"]["withmpi"].default = False  # type: ignore[index]

        spec.output_namespace(
            "xas_data",
            valid_type=orm.ArrayData,
            dynamic=True,
            required=False,
            help="Parsed XasData nodes keyed snap_FFFF_site_SSSS.",
        )
        spec.output_namespace(
            "path_contributions",
            dynamic=True,
            required=False,
            help="PathContributionsData nodes keyed snap_FFFF_site_SSSS.",
        )
        spec.output(
            "archive",
            valid_type=ExafsArchiveData,
            required=False,
            help="Batch shard archive node (batch_shard.h5) (ADR 0004).",
        )

        spec.exit_code(400, "ERROR_PARSING_FAILED", message="Batch parser raised: {reason}.")
        spec.exit_code(
            301,
            "ERROR_ALL_RUNS_FAILED",
            message="Driver produced no chi.dat outputs.",
        )

        spec.inputs.validator = cls._validate_inputs  # type: ignore[assignment]

    @staticmethod
    def _validate_inputs(value, _port_namespace) -> str | None:
        """Reject inputs the driver could not act on.

        ``prepare_for_submission`` cannot return an exit code, so pairing and
        range checks happen here, before the node exists.
        """
        if value is None:
            return None

        frames = value.get("frame_indices")
        sites = value.get("site_indices")
        if frames is None or sites is None:
            return None

        frame_list = frames.get_list()
        site_list = sites.get_list()
        problems: list[str] = []

        if len(frame_list) != len(site_list):
            problems.append(
                f"frame_indices ({len(frame_list)}) and site_indices ({len(site_list)}) "
                "must have the same length: they are parallel lists of (frame, site) pairs."
            )
        if not frame_list:
            problems.append("frame_indices must not be empty.")

        trajectory = value.get("trajectory")
        if trajectory is not None and frame_list:
            n_steps = len(trajectory.get_array("positions"))
            out_of_range = sorted({f for f in frame_list if not 0 <= f < n_steps})
            if out_of_range:
                problems.append(
                    f"frame_indices {out_of_range} are outside the trajectory (0–{n_steps - 1})."
                )

        n_workers = value.get("n_workers")
        if n_workers is not None and n_workers.value < 1:
            problems.append(f"n_workers must be >= 1, got {n_workers.value}.")

        return " ".join(problems) or None

    # ------------------------------------------------------------------

    def prepare_for_submission(self, folder) -> CalcInfo:
        """Write all feff.inp files, the driver config, and the driver script."""
        from aiida_feff.utils import trajectory_to_structures

        frame_indices: list[int] = self.inputs.frame_indices.get_list()
        site_indices: list[int] = self.inputs.site_indices.get_list()

        use_precomputed = "remote_potentials" in self.inputs
        threshold = self.inputs.path_cw_threshold.value
        do_aggregate = threshold >= 0

        # ----------------------------------------------------------------
        # Load structures for each unique frame index
        # ----------------------------------------------------------------
        unique_frames = sorted(set(frame_indices))
        structures_for_frames = trajectory_to_structures(
            self.inputs.trajectory, step_ids=unique_frames
        )
        frame_to_structure = dict(zip(unique_frames, structures_for_frames, strict=True))

        # ----------------------------------------------------------------
        # Build per-(frame, site) input files
        # ----------------------------------------------------------------
        base_d = self.inputs.parameters.get_dict()
        base_d.pop("absorbing_atoms", None)
        # Mirror the normalisation the non-batch path applies, so both produce
        # the same feff.inp from the same parameters.
        if "scf" in base_d and base_d["scf"] is None:
            del base_d["scf"]
        if use_precomputed:
            base_d["control"] = CONTROL_NO_POT

        for frame_idx, site_idx in zip(frame_indices, site_indices, strict=True):
            run_label = _snap_label(frame_idx, site_idx)
            structure = frame_to_structure[frame_idx]

            d = dict(base_d)
            d["absorbing_atom"] = site_idx
            params = FeffParameters(dict=d)  # unstored; only used to build feff.inp

            # AiiDA's Folder.open does not create intermediate directories, so
            # create the per-snapshot subfolder before writing feff.inp /
            # _feff_aggregate_config.json into it.
            snap_folder = folder.get_subfolder(run_label, create=True)

            inp_text = FeffCalculation.build_feff_inp(structure, params)
            with snap_folder.open("feff.inp", "wb") as fh:
                fh.write(as_lf_bytes(inp_text))

            if do_aggregate:
                agg_cfg = {
                    "threshold": float(threshold),
                    "frame_idx": frame_idx,
                    "site_idx": site_idx,
                    "absorber_element": absorber_element(structure, site_idx),
                }
                with snap_folder.open("_feff_aggregate_config.json", "wb") as fh:
                    fh.write(as_lf_bytes(json.dumps(agg_cfg)))

        # ----------------------------------------------------------------
        # FEFF executable info — read from feff_code at submission time
        # so the environment setup is baked into the batch config,
        # not resolved at runtime on a potentially different environment.
        # ----------------------------------------------------------------
        feff_code = self.inputs.feff_code
        try:
            feff_exe = str(feff_code.filepath_executable)
        except AttributeError as exc:
            # Falling back to the label would put a non-executable name into
            # every wrapper script and fail hours later, one stderr.txt per
            # snapshot deep.  Refuse now instead.
            raise ValueError(
                f"feff_code {feff_code!r} has no filepath_executable. Batch mode drives "
                "FEFF through a shell wrapper and needs a concrete path, so the FEFF "
                "code must be an InstalledCode."
            ) from exc
        feff_prepend = getattr(feff_code, "prepend_text", "") or ""
        feff_append = getattr(feff_code, "append_text", "") or ""

        # ----------------------------------------------------------------
        # Write batch_config.json
        # ----------------------------------------------------------------
        if "n_workers" in self.inputs:
            n_workers_val = self.inputs.n_workers.value
        else:
            n_workers_val = workers_from_resources(self.options.resources)

        clean_scratch = self.inputs.clean_scratch.value
        stream_chunk_size = self.inputs.stream_chunk_size.value

        # One element per pair, in pair order. A single batch-wide symbol would
        # mislabel every task whenever ``absorbing_atoms`` is an explicit index
        # list spanning more than one species.
        absorber_elements = [
            absorber_element(frame_to_structure[f], s)
            for f, s in zip(frame_indices, site_indices, strict=True)
        ]
        batch_cfg = {
            "pairs": list(zip(frame_indices, site_indices, strict=True)),
            "feff_executable": feff_exe,
            "feff_prepend": feff_prepend,
            "feff_append": feff_append,
            "n_workers": n_workers_val,
            "do_aggregate": do_aggregate,
            "threshold": float(threshold),
            "run_timeout_seconds": self._run_timeout(),
            "clean_scratch": clean_scratch,
            "stream_chunk_size": stream_chunk_size,
            "absorber_elements": absorber_elements,
            "scratch_keep_files": _scratch_keep_files(do_aggregate),
        }
        with folder.open(BATCH_CONFIG, "wb") as fh:
            fh.write(as_lf_bytes(json.dumps(batch_cfg, indent=2)))

        # ----------------------------------------------------------------
        # Write driver and aggregation scripts
        # ----------------------------------------------------------------
        with folder.open(BATCH_DRIVER, "wb") as fh:
            fh.write(_DRIVER_PATH.read_bytes())
        if do_aggregate:
            with folder.open(FEFF_AGGREGATE_SCRIPT, "wb") as fh:
                fh.write(_AGGREGATE_PATH.read_bytes())

        # ----------------------------------------------------------------
        # CodeInfo: run "python _run_batch.py"
        # ----------------------------------------------------------------
        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = [BATCH_DRIVER]
        codeinfo.stdout_name = BATCH_LOG
        codeinfo.stderr_name = BATCH_ERR
        codeinfo.withmpi = False

        # ----------------------------------------------------------------
        # CalcInfo
        # ----------------------------------------------------------------
        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.codes_run_mode = datastructures.CodeRunMode.SERIAL

        # remote_copy_list: potentials per site → potentials/site_XXXX/
        calcinfo.remote_copy_list = []
        if use_precomputed:
            for key, remote in self.inputs.remote_potentials.items():
                # key is like 'site_0000'
                remote_path = remote.get_remote_path()
                computer_uuid = remote.computer.uuid
                for fname in FEFF_POTENTIAL_FILES:
                    calcinfo.remote_copy_list.append(
                        (computer_uuid, f"{remote_path}/{fname}", f"potentials/{key}/{fname}")
                    )

        # retrieve_list: glob each snap_* subdir for output files
        retrieve_list: list[tuple[str, str, int | None]] = list(_SNAP_RETRIEVE)
        if do_aggregate:
            retrieve_list.append(_CONTRIBUTIONS_GLOB)

        # CalcInfo.retrieve_list is annotated for the flat (name, dest, depth=str)
        # form; batch mode needs the depth=None variant that keeps snap_*/ nesting.
        calcinfo.retrieve_list = retrieve_list  # type: ignore[assignment]
        calcinfo.local_copy_list = []

        return calcinfo

    def _run_timeout(self) -> float | None:
        """Per-FEFF-run timeout in seconds, or ``None`` when unbounded.

        Derived from the job's own wallclock so a hung run is killed while
        there is still time to retrieve everything else in the chunk.
        """
        wallclock = self.options.get("max_wallclock_seconds")
        if not wallclock:
            return None
        return float(wallclock) * _RUN_TIMEOUT_FRACTION


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _snap_label(frame_idx: int, site_idx: int) -> str:
    """Canonical label for a (frame, site) pair, shared with the parser."""
    return f"snap_{frame_idx:04d}_site_{site_idx:04d}"


def workers_from_resources(resources: dict | None) -> int:
    """Derive a parallel-worker count from AiiDA's scheduler resources.

    ``metadata.options.resources`` is the one description of the allocation
    that every scheduler fills in, so it is a better default than reading
    ``SLURM_*`` from the environment — which silently yields one worker on
    PBS, LSF and ``core.direct``.

    Returns at least 1.
    """
    if not resources:
        return 1
    machines = int(resources.get("num_machines", 1) or 1)
    per_machine = resources.get("num_mpiprocs_per_machine") or resources.get(
        "num_cores_per_machine"
    )
    if per_machine:
        return max(1, machines * int(per_machine))
    total = resources.get("tot_num_mpiprocs") or resources.get("num_cores_per_mpiproc")
    if total:
        return max(1, int(total))
    return max(1, machines)
