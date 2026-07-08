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
    FEFF_AGGREGATE_SCRIPT,
    FEFF_CHI_FILE,
    FEFF_FILES_DAT,
    FEFF_PATHS_FILE,
    FEFF_POTENTIAL_FILES,
    FEFF_XMUDA_FILE,
    FeffCalculation,
)
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

# CONTROL strings (same as ensemble workflow)
_CONTROL_NO_POT = "0 0 0 1 1 1"

# Retrieve glob patterns for per-run outputs (depth=None preserves directory structure)
_SNAP_RETRIEVE = [
    (f"snap_*/{FEFF_XMUDA_FILE}", ".", None),
    (f"snap_*/{FEFF_CHI_FILE}", ".", None),
    (f"snap_*/{FEFF_PATHS_FILE}", ".", None),
    (f"snap_*/{FEFF_FILES_DAT}", ".", None),
    ("snap_*/log.dat", ".", None),
    ("snap_*/stderr.txt", ".", None),
    (BATCH_LOG, ".", 0),
    (BATCH_ERR, ".", 0),
]
_SNAP_RETRIEVE_PATHS = [("snap_*/_feff_aggregate_config.json", ".", None)]
_CONTRIBUTIONS_GLOB = ("snap_*/contributions_raw.h5", ".", None)


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
        Number of parallel workers.  Defaults to ``SLURM_CPUS_ON_NODE``
        (falling back to ``SLURM_NTASKS``, then 1).

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
            help="Parallel workers; defaults to SLURM_CPUS_ON_NODE at runtime.",
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

        spec.exit_code(300, "ERROR_INVALID_INPUT", message="Input validation failed: {reason}.")
        spec.exit_code(400, "ERROR_PARSING_FAILED", message="Batch parser raised: {reason}.")
        spec.exit_code(
            301,
            "ERROR_ALL_RUNS_FAILED",
            message="Driver produced no xmu.dat outputs.",
        )

    # ------------------------------------------------------------------

    def prepare_for_submission(self, folder) -> CalcInfo:
        """Write all feff.inp files, the driver config, and the driver script."""
        from aiida_feff.utils import trajectory_to_structures

        frame_indices: list[int] = self.inputs.frame_indices.get_list()
        site_indices: list[int] = self.inputs.site_indices.get_list()
        if len(frame_indices) != len(site_indices):
            return self.exit_codes.ERROR_INVALID_INPUT.format(  # type: ignore[no-any-return]
                reason="frame_indices and site_indices must have the same length"
            )

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
        if use_precomputed:
            base_d["control"] = _CONTROL_NO_POT

        for frame_idx, site_idx in zip(frame_indices, site_indices, strict=True):
            run_label = _snap_label(frame_idx, site_idx)
            structure = frame_to_structure[frame_idx]

            d = dict(base_d)
            d["absorbing_atom"] = site_idx
            params = FeffParameters(dict=d)  # unstored; only used to build feff.inp

            inp_text = FeffCalculation._build_feff_inp(structure, params)
            with folder.open(f"{run_label}/feff.inp", "w") as fh:
                fh.write(inp_text)

            if do_aggregate:
                # Compute absorber element for aggregate config
                try:
                    site = structure.sites[site_idx]
                    kind = structure.get_kind(site.kind_name)
                    absorber_element = str(kind.symbols[0])
                except Exception:  # noqa: BLE001
                    absorber_element = ""

                agg_cfg = {
                    "threshold": float(threshold),
                    "frame_idx": frame_idx,
                    "site_idx": site_idx,
                    "absorber_element": absorber_element,
                }
                with folder.open(f"{run_label}/_feff_aggregate_config.json", "w") as fh:
                    json.dump(agg_cfg, fh)

        # ----------------------------------------------------------------
        # FEFF executable info — read from feff_code at submission time
        # so the environment setup is baked into the batch config,
        # not resolved at runtime on a potentially different environment.
        # ----------------------------------------------------------------
        feff_code = self.inputs.feff_code
        try:
            feff_exe = str(feff_code.filepath_executable)
        except AttributeError:
            # ponytail: assumes InstalledCode; other types need extension
            feff_exe = feff_code.label
            logger.warning(
                "feff_code has no filepath_executable; using label %r as executable name",
                feff_exe,
            )
        feff_prepend = getattr(feff_code, "prepend_text", "") or ""
        feff_append = getattr(feff_code, "append_text", "") or ""

        # ----------------------------------------------------------------
        # Write batch_config.json
        # ----------------------------------------------------------------
        n_workers_val: int | None = None
        if "n_workers" in self.inputs:
            n_workers_val = self.inputs.n_workers.value

        batch_cfg = {
            "pairs": list(zip(frame_indices, site_indices, strict=True)),
            "feff_executable": feff_exe,
            "feff_prepend": feff_prepend,
            "feff_append": feff_append,
            "n_workers": n_workers_val,
            "do_aggregate": do_aggregate,
            "threshold": float(threshold),
        }
        with folder.open(BATCH_CONFIG, "w") as fh:
            json.dump(batch_cfg, fh, indent=2)

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
        retrieve_list = list(_SNAP_RETRIEVE)
        if do_aggregate:
            retrieve_list.append(_CONTRIBUTIONS_GLOB)

        calcinfo.retrieve_list = retrieve_list
        calcinfo.local_copy_list = []

        return calcinfo


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _snap_label(frame_idx: int, site_idx: int) -> str:
    """Canonical label for a (frame, site) pair, shared with the parser."""
    return f"snap_{frame_idx:04d}_site_{site_idx:04d}"
