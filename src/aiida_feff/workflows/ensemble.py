"""EnsembleExafsWorkChain — compute ensemble-averaged EXAFS from MD snapshots.

Workflow
--------
1. Accept a :class:`~aiida.orm.TrajectoryData` **or** a list of
   :class:`~aiida.orm.StructureData` nodes plus shared FEFF parameters.
2. When ``trajectory`` is supplied, split it into individual snapshots using
   the ``split_trajectory`` calcfunction (provenance-tracked) and apply
   ``sample_interval`` to sub-sample the trajectory.
3. Resolve absorber sites from ``parameters.absorbing_atoms`` (element string
   or list of indices).  All sites must be the same species.
4. Optionally, pre-compute FEFF potentials once **per absorber site** on a
   representative structure and reuse them across all MD frames (skipping
   the costly SCF step for each snapshot).
5. Launch one :class:`~aiida_feff.calculations.feff.FeffCalculation` per
   ``(frame, site)`` pair (fan-out: N_frames x N_sites jobs).
6. Wait for all children to finish.
7. Call :func:`~aiida_feff.calcfunctions.larch.average_xas_data` to
   produce per-site and overall ensemble-averaged
   :class:`~aiida_feff.data.xasdata.XasData` outputs.

Multi-site outputs
------------------
``averaged_xas.site_NNNN``  per-site average (one per absorber site index)
``averaged_xas.all``        grand average over all sites and all frames
``path_contributions``      merged HDF5; filter by ``site_idx`` column post-hoc

Exit codes
----------
300  All snapshot calculations failed.
301  Some snapshot calculations failed (partial average produced).
302  Potential pre-computation failed.
"""

from __future__ import annotations

import logging
import typing as t

from aiida import orm
from aiida.engine import ToContext, WorkChain, if_

from aiida_feff.calcfunctions.larch import average_xas_data
from aiida_feff.calcfunctions.path_contributions import merge_path_contributions
from aiida_feff.calculations.feff import FeffCalculation
from aiida_feff.calculations.feff_batch import FeffBatchCalculation, _snap_label
from aiida_feff.data.parameters import FeffParameters
from aiida_feff.data.pathcontributions import PathContributionsData
from aiida_feff.data.xasdata import XasData
from aiida_feff.utils import split_trajectory, structures_to_trajectory

logger = logging.getLogger(__name__)

# CONTROL string for the potentials-only FEFF run
_CONTROL_POT_ONLY = "1 1 1 0 0 0"
# CONTROL string for the paths+spectrum FEFF run using pre-computed potentials
_CONTROL_NO_POT = "0 0 0 1 1 1"

# Warn when this many absorber sites are selected (can produce huge fan-outs).
# ponytail: matches alc-dls-exafs LARGE_NUMBER_OF_SITES; upgrade to batching if needed
_LARGE_N_SITES = 20


def _resolve_absorber_sites(
    structure: orm.StructureData,
    spec: int | str | list,
) -> list[int]:
    """Resolve absorber specification to a validated list of 0-based atom indices.

    Accepted formats (all validated to be single-species):

    - ``int``            -- single absolute index, e.g. ``0``
    - ``list[int]``      -- explicit absolute indices, e.g. ``[0, 4, 8]``
    - ``"Cu"``           -- element symbol → all matching indices
    - ``"0,1,2"``        -- comma-separated absolute indices as a string
    - ``"Cu:0,1"``       -- element symbol + relative indices within that element
                           (e.g. ``"Cu:0,1"`` → 1st and 2nd Cu atoms)

    Parameters
    ----------
    structure:
        Reference structure (used to resolve element symbol to indices).
    spec:
        Absorber specification in any of the formats above.

    Returns:
    -------
    list[int]
        Validated absolute indices, all the same element.

    Raises:
    ------
    ValueError
        If indices are out of range, empty, the spec is ambiguous, or
        the selected atoms belong to more than one element.
    """
    atoms = structure.get_ase()
    symbols = atoms.get_chemical_symbols()
    n = len(symbols)

    if isinstance(spec, int):
        indices = [spec]

    elif isinstance(spec, list):
        if not spec:
            raise ValueError("absorbing_atoms list must not be empty.")
        indices = [int(x) for x in spec]

    elif isinstance(spec, str):
        spec = spec.strip()

        if ":" in spec:
            # "Cu:0,1" — relative indices within the element's sites
            element_part, idx_part = spec.split(":", 1)
            element = element_part.strip().capitalize()
            element_indices = [i for i, s in enumerate(symbols) if s == element]
            if not element_indices:
                raise ValueError(f"No atoms with element {element!r} found in structure.")
            try:
                rel = [int(x.strip()) for x in idx_part.split(",")]
            except ValueError as exc:
                raise ValueError(
                    f"Invalid absorber format {spec!r}. "
                    "Use 'Element:rel0,rel1' with integer relative indices."
                ) from exc
            bad_rel = [r for r in rel if not 0 <= r < len(element_indices)]
            if bad_rel:
                raise ValueError(
                    f"Relative indices {bad_rel} out of range for element {element!r} "
                    f"(0–{len(element_indices) - 1})."
                )
            indices = [element_indices[r] for r in rel]

        elif spec.replace(",", "").replace(" ", "").isdigit():
            # "0,1,2" — comma-separated absolute indices
            indices = [int(x.strip()) for x in spec.split(",")]

        else:
            # Element symbol → all matching sites
            element = spec.capitalize()
            indices = [i for i, s in enumerate(symbols) if s == element]
            if not indices:
                raise ValueError(f"No atoms with element {element!r} found in structure.")

    else:
        raise ValueError(f"absorbing_atoms must be int, str, or list[int]; got {type(spec)}")

    for idx in indices:
        if not 0 <= idx < n:
            raise ValueError(f"Absorber index {idx} out of range (0–{n - 1}).")

    # Single-species check
    element = symbols[indices[0]]
    bad = [idx for idx in indices if symbols[idx] != element]
    if bad:
        raise ValueError(
            f"All absorber indices must be the same element ({element!r}). "
            f"Indices {bad} are {[symbols[i] for i in bad]}."
        )

    if len(indices) > _LARGE_N_SITES:
        logger.warning(
            "Number of absorber sites (%d) is large — this will produce a very "
            "large fan-out. Consider passing an explicit subset of indices.",
            len(indices),
        )

    return indices


class EnsembleExafsWorkChain(WorkChain):
    """Ensemble-averaged EXAFS WorkChain.

    Inputs
    ------
    structures : :class:`~aiida.orm.List`, optional
        MD snapshots as a list of :class:`~aiida.orm.StructureData` nodes or
        integer PKs.  Either this **or** ``trajectory`` must be supplied.
    trajectory : :class:`~aiida.orm.TrajectoryData`, optional
        AiiDA trajectory node.  Will be split into
        :class:`~aiida.orm.StructureData` instances automatically.
        Use ``sample_interval`` to sub-sample long trajectories.
    sample_interval : :class:`~aiida.orm.Int`, optional, default 1
        Take every *N*-th frame from ``trajectory``.  Ignored when
        ``structures`` is used.
    parameters : :class:`~aiida_feff.data.parameters.FeffParameters`
        Shared FEFF parameters (edge, radius, …) applied to every snapshot.
    code : :class:`~aiida.orm.AbstractCode`
        The FEFF executable.
    python_code : :class:`~aiida.orm.AbstractCode`, optional
        Python 3 interpreter on the remote computer.  When provided together
        with ``path_cw_threshold >= 0``, each FeffCalculation runs
        ``_aggregate_paths.py`` as a second sequential step in the same job.
    options : :class:`~aiida.orm.Dict`, optional
        Scheduler resource options (``resources``, ``max_wallclock_seconds``, …).
    path_cw_threshold : :class:`~aiida.orm.Float`, optional, default -1.0
        Curved-wave amplitude threshold for storing scattering paths.
        Passed through to each FeffCalculation.
    path_r_bin : :class:`~aiida.orm.Float`, optional, default 0.15
        Bin width (Å) for grouping paths by effective path length when
        merging per-snapshot PathContributionsData nodes into the
        ensemble node.  Only used when path_cw_threshold >= 0.
    group_label : :class:`~aiida.orm.Str`, optional
        If provided, the finished workchain node is added to an AiiDA
        Group with this label (created if it does not exist).
    precompute_potentials : :class:`~aiida.orm.Bool`, optional, default False
        When True, run FEFF once per absorber site on a representative
        structure with ``CONTROL 1 1 1 0 0 0`` to compute potentials, then
        copy ``pot.pad`` and ``phase.pad`` into every MD-frame job (which
        runs with ``CONTROL 0 0 0 1 1 1``).  This skips the expensive SCF
        step for each snapshot and can dramatically reduce total wall time.
    potential_structure : :class:`~aiida.orm.StructureData`, optional
        Structure used for potential pre-computation.  Defaults to the last
        frame of the trajectory (or the last item in ``structures``).

    Outputs
    -------
    averaged_xas : namespace
        Per-site and overall averaged spectra:

        ``averaged_xas.site_NNNN`` — :class:`~aiida_feff.data.xasdata.XasData`
            Average over all frames for absorber site index NNNN.
        ``averaged_xas.all`` — :class:`~aiida_feff.data.xasdata.XasData`
            Grand average over all sites and all frames.
    n_failed : :class:`~aiida.orm.Int`
        Number of (frame, site) calculations that failed.
    path_contributions : :class:`~aiida_feff.data.pathcontributions.PathContributionsData`, optional
        Merged per-path FEFF data; filter by the ``site_idx`` column post-hoc
        to compare scattering paths per absorber site.

    Usage — single absorber site (backward-compatible)::

        params = FeffParameters(dict={"edge": "K", "absorbing_atom": 0, ...})
        node = submit(EnsembleExafsWorkChain, trajectory=traj, parameters=params, ...)

    Usage — all Cu sites (element string)::

        params = FeffParameters(dict={"edge": "K", "absorbing_atoms": "Cu", ...})
        node = submit(EnsembleExafsWorkChain, trajectory=traj, parameters=params, ...)

    Usage — explicit site indices::

        params = FeffParameters(dict={"edge": "K", "absorbing_atoms": [0, 1, 2], ...})
        node = submit(EnsembleExafsWorkChain, ...)

    Accessing outputs after completion::

        wc = load_node(pk)
        per_site = {k: v for k, v in wc.outputs.averaged_xas.items() if k != "all"}
        overall  = wc.outputs.averaged_xas.all
    """

    @classmethod
    def define(cls, spec) -> None:
        """Define inputs, outputs and outline of the workchain."""
        super().define(spec)

        spec.input(
            "structures",
            valid_type=orm.List,
            required=False,
            help="List of StructureData PKs (int) or StructureData nodes.",
        )
        spec.input(
            "trajectory",
            valid_type=orm.TrajectoryData,
            required=False,
            help="MD trajectory; split into StructureData snapshots automatically.",
        )
        spec.input(
            "sample_interval",
            valid_type=orm.Int,
            default=lambda: orm.Int(1),
            help="Take every N-th frame from a trajectory input.",
        )
        spec.input("parameters", valid_type=FeffParameters)
        spec.input("code", valid_type=orm.AbstractCode)
        spec.input(
            "python_code",
            valid_type=orm.AbstractCode,
            required=False,
            help="Python 3 interpreter for remote path aggregation (see FeffCalculation).",
        )
        spec.input(
            "options",
            valid_type=orm.Dict,
            required=False,
            help="Scheduler options forwarded to metadata.options.",
        )
        spec.input(
            "path_cw_threshold",
            valid_type=orm.Float,
            default=lambda: orm.Float(-1.0),
            help=(
                "Curved-wave amplitude threshold for storing scattering paths. "
                "Passed through to each FeffCalculation. "
                "Set to e.g. 5.0 to keep paths with ≥ 5%% of the peak amplitude; "
                "0.0 to store all paths; -1.0 (default) to skip path storage."
            ),
        )
        spec.input(
            "path_r_bin",
            valid_type=orm.Float,
            default=lambda: orm.Float(0.15),
            help=(
                "Bin width (Å) for grouping paths by effective path length when "
                "merging per-snapshot PathContributionsData nodes into the "
                "ensemble node.  Only used when path_cw_threshold >= 0."
            ),
        )
        spec.input(
            "precompute_potentials",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help=(
                "When True, run FEFF once per absorber site with CONTROL 1 1 1 0 0 0 "
                "to pre-compute potentials, then copy pot.pad and phase.pad into every "
                "MD-frame job (CONTROL 0 0 0 1 1 1) to skip the SCF step."
            ),
        )
        spec.input(
            "potential_structure",
            valid_type=orm.StructureData,
            required=False,
            help=(
                "Structure used for potential pre-computation.  Defaults to the last "
                "frame of the trajectory (or last item in structures)."
            ),
        )
        spec.input(
            "group_label",
            valid_type=orm.Str,
            required=False,
            help=(
                "If provided, the finished workchain node is added to an AiiDA "
                "Group with this label (created if it does not exist). "
                "Useful for organising large MD-EXAFS campaigns."
            ),
        )
        spec.input(
            "batch_size",
            valid_type=orm.Int,
            required=False,
            help=(
                "When set, groups (frame, site) pairs into chunks of this size and "
                "submits one FeffBatchCalculation per chunk instead of one "
                "FeffCalculation per pair.  Each chunk runs as a single Slurm job "
                "with one FEFF instance per allocated core.  "
                "Requires python_code to be set (it is used as the Python "
                "runner for the batch driver).  "
                "Set to the number of cores per node to use one Slurm job; "
                "set smaller for multi-job batching."
            ),
        )
        spec.input(
            "n_workers",
            valid_type=orm.Int,
            required=False,
            help=(
                "Number of parallel FEFF workers per batch job.  "
                "Passed to FeffBatchCalculation; defaults to SLURM_CPUS_ON_NODE "
                "at runtime if not set."
            ),
        )

        spec.output_namespace(
            "averaged_xas",
            valid_type=XasData,
            dynamic=True,
            help=(
                "Averaged XasData nodes. Keys: 'site_NNNN' (per absorber site) "
                "and 'all' (grand average over all sites and frames)."
            ),
        )
        spec.output("n_failed", valid_type=orm.Int)
        spec.output(
            "path_contributions",
            valid_type=PathContributionsData,
            required=False,
            help="Merged per-path FEFF data from all successful snapshots.",
        )

        spec.exit_code(300, "ERROR_ALL_FAILED", message="All snapshot FEFF calculations failed.")
        spec.exit_code(
            301,
            "ERROR_PARTIAL_FAILURE",
            message="{n_failed} of {n_total} snapshot calculations failed.",
        )
        spec.exit_code(
            302,
            "ERROR_POTENTIALS_FAILED",
            message="FEFF potential pre-computation failed for site {site_idx}.",
        )
        spec.exit_code(
            303,
            "ERROR_MISSING_AGGREGATION_CODE",
            message="batch_size requires python_code to be set (used as Python runner).",
        )

        spec.outline(
            cls.validate_inputs,
            if_(cls.should_precompute)(
                cls.precompute_potentials_step,
                cls.collect_potentials,
            ),
            if_(cls.use_batch)(
                cls.submit_batch_calculations,
                cls.inspect_batch_results,
            ).else_(
                cls.submit_feff_calculations,
                cls.inspect_results,
            ),
            cls.average_results,
        )

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def validate_inputs(self) -> None:
        """Load/split structure nodes into context."""
        if "structures" in self.inputs and "trajectory" in self.inputs:
            raise ValueError("Supply either 'structures' or 'trajectory', not both.")

        if "trajectory" in self.inputs:
            interval = self.inputs.sample_interval.value
            traj = self.inputs.trajectory
            n_steps = len(traj.get_array("positions"))
            step_ids = list(range(0, n_steps, interval))

            # split_trajectory is a @calcfunction: each StructureData gets a
            # CREATE link back to the TrajectoryData in the provenance graph.
            result = split_trajectory(traj, orm.Dict({"step_ids": step_ids}))
            structures: list[orm.StructureData] = [result[k] for k in sorted(result.keys())]
            self.report(
                f"Trajectory has {n_steps} frames; sampling every {interval} → "
                f"{len(structures)} snapshot(s)."
            )
        elif "structures" in self.inputs:
            raw = self.inputs.structures.get_list()
            structures = []
            for item in raw:
                if isinstance(item, int):
                    structures.append(t.cast(orm.StructureData, orm.load_node(item)))
                elif isinstance(item, orm.StructureData):
                    structures.append(item)
                else:
                    raise ValueError(
                        f"structures list must contain StructureData nodes or integer PKs, "
                        f"got {type(item)}"
                    )
            # Ensure all nodes are stored before placing in ctx (checkpoint safety).
            structures = [s if s.is_stored else s.store() for s in structures]
        else:
            raise ValueError("Either 'structures' or 'trajectory' must be supplied.")

        self.ctx.structures = structures
        self.report(f"Validated {len(structures)} snapshot structure(s).")

        # Resolve absorber sites (backwards-compat: fall back to absorbing_atom)
        params_dict = self.inputs.parameters.get_dict()
        absorbing_atoms = params_dict.get("absorbing_atoms", None)
        if absorbing_atoms is not None:
            site_indices = _resolve_absorber_sites(structures[-1], absorbing_atoms)
        else:
            site_indices = [int(params_dict.get("absorbing_atom", 0))]
        self.ctx.site_indices = site_indices
        self.report(f"Absorber site indices: {site_indices}")

    def should_precompute(self) -> bool:
        """Return True when potential pre-computation is requested."""
        return bool(self.inputs.precompute_potentials.value)

    def use_batch(self) -> bool:
        """Return True when batch_size is set (batch CalcJob mode)."""
        return "batch_size" in self.inputs

    def precompute_potentials_step(self):
        """Submit one FEFF potentials-only run per absorber site.

        Uses CONTROL 1 1 1 0 0 0 so FEFF computes potentials and writes
        pot.pad / phase.pad but does not run the paths or spectrum modules.
        The representative structure defaults to the last frame/structure.
        """
        if "potential_structure" in self.inputs:
            pot_structure = self.inputs.potential_structure
        else:
            pot_structure = self.ctx.structures[-1]

        d = self.inputs.parameters.get_dict()
        d["control"] = _CONTROL_POT_ONLY
        if "scf" in d and d["scf"] is None:
            del d["scf"]
        # Remove absorbing_atoms — each pot run uses a concrete absorbing_atom
        d.pop("absorbing_atoms", None)

        options = self.inputs.get("options", orm.Dict()).get_dict()
        futures: dict[str, t.Any] = {}

        for site_idx in self.ctx.site_indices:
            site_d = dict(d)
            site_d["absorbing_atom"] = site_idx
            pot_params = FeffParameters(dict=site_d)
            pot_params.label = (
                f"potentials-only site {site_idx} "
                f"(derived from parameters<{self.inputs.parameters.pk}>)"
            )
            label = f"pot_site_{site_idx:04d}"
            feff_inputs: dict[str, t.Any] = {
                "code": self.inputs.code,
                "structure": pot_structure,
                "parameters": pot_params,
                "path_cw_threshold": orm.Float(-1.0),
                "metadata": {
                    "label": label,
                    "call_link_label": label,
                    "options": options,
                },
            }
            future = self.submit(FeffCalculation, **feff_inputs)
            self.report(
                f"Submitted potentials-only FeffCalculation for site {site_idx} → {future.pk}"
            )
            futures[label] = future

        return ToContext(**futures)  # type: ignore[arg-type]

    def collect_potentials(self):
        """Check each potentials run and build ctx.pot_remote[site_idx → RemoteData].

        Exit codes 310 (no xmu.dat) and 311 (no chi.dat) are expected for a
        potentials-only run — anything else is a real failure.
        """
        acceptable = {0, 310, 311}
        pot_remote: dict[int, orm.RemoteData] = {}

        for site_idx in self.ctx.site_indices:
            label = f"pot_site_{site_idx:04d}"
            child = self.ctx[label]
            if child.exit_status not in acceptable:
                self.report(
                    f"Potentials run for site {site_idx} ({child.pk}) failed "
                    f"with exit status {child.exit_status}."
                )
                return self.exit_codes.ERROR_POTENTIALS_FAILED.format(  # type: ignore[no-any-return]
                    site_idx=site_idx
                )
            pot_remote[site_idx] = child.outputs.remote_folder
            self.report(
                f"Potentials for site {site_idx} ready "
                f"(remote pk={child.outputs.remote_folder.pk})."
            )

        self.ctx.pot_remote = pot_remote

    def submit_batch_calculations(self):
        """Group (frame, site) pairs into chunks; submit one FeffBatchCalculation each.

        Requires ``python_code`` (used as the Python runner for the batch
        driver).  The FEFF code is passed as ``feff_code``.
        """
        if "python_code" not in self.inputs:
            return self.exit_codes.ERROR_MISSING_AGGREGATION_CODE  # type: ignore[no-any-return]

        batch_size = self.inputs.batch_size.value
        options = self.inputs.get("options", orm.Dict()).get_dict()
        use_precomputed = self.should_precompute() and hasattr(self.ctx, "pot_remote")

        base_d = self.inputs.parameters.get_dict()
        base_d.pop("absorbing_atoms", None)

        # Build ordered list of all (frame, site) pairs (same order as non-batch path)
        job_pairs: list[tuple[int, int]] = []
        for site_idx in self.ctx.site_indices:
            for i in range(len(self.ctx.structures)):
                job_pairs.append((i, site_idx))

        self.ctx.job_pairs = job_pairs
        self.ctx.n_total = len(job_pairs)

        # Convert structures list → TrajectoryData if we don't already have one.
        # The batch CalcJob always receives a TrajectoryData (single node, clean provenance).
        if "trajectory" in self.inputs:
            traj = self.inputs.trajectory
        else:
            # structures input path: pack into a trajectory via provenance-tracked calcfunction
            traj = structures_to_trajectory(
                **{f"s{i:04d}": s for i, s in enumerate(self.ctx.structures)},
                metadata={"call_link_label": "structures_to_trajectory_for_batch"},
            )

        # Split pairs into chunks and submit one FeffBatchCalculation per chunk
        calcs: dict[str, t.Any] = {}
        self.ctx.batch_chunks = {}

        for chunk_start in range(0, len(job_pairs), batch_size):
            chunk = job_pairs[chunk_start : chunk_start + batch_size]
            chunk_idx = chunk_start // batch_size
            batch_label = f"batch_{chunk_idx:04d}"

            frame_indices = orm.List([p[0] for p in chunk])
            site_indices = orm.List([p[1] for p in chunk])

            batch_inputs: dict[str, t.Any] = {
                "code": self.inputs.python_code,  # Python runner
                "feff_code": self.inputs.code,  # FEFF executable
                "trajectory": traj,
                "frame_indices": frame_indices,
                "site_indices": site_indices,
                "parameters": self.inputs.parameters,
                "path_cw_threshold": self.inputs.path_cw_threshold,
                "metadata": {
                    "label": batch_label,
                    "call_link_label": batch_label,
                    "options": options,
                },
            }
            if "n_workers" in self.inputs:
                batch_inputs["n_workers"] = self.inputs.n_workers
            if use_precomputed:
                batch_inputs["remote_potentials"] = {
                    f"site_{k:04d}": v for k, v in self.ctx.pot_remote.items()
                }

            future = self.submit(FeffBatchCalculation, **batch_inputs)
            calcs[batch_label] = future
            self.ctx.batch_chunks[batch_label] = chunk
            self.report(
                f"Submitted FeffBatchCalculation {batch_label} ({len(chunk)} pairs) → {future.pk}"
            )

        return ToContext(**calcs)  # type: ignore[arg-type]

    def inspect_batch_results(self) -> None:
        """Collect XasData outputs from batch CalcJob children."""
        per_site: dict[int, dict[str, XasData]] = {s: {} for s in self.ctx.site_indices}
        all_xas: dict[str, XasData] = {}
        successful_paths: dict[str, PathContributionsData] = {}
        n_failed = 0

        for batch_label, chunk in self.ctx.batch_chunks.items():
            child = self.ctx[batch_label]

            if not child.is_finished_ok:
                self.report(
                    f"{batch_label} ({child.pk}) failed with exit status "
                    f"{child.exit_status}; counting all {len(chunk)} pairs as failed."
                )
                n_failed += len(chunk)
                continue

            # Harvest per-pair outputs from the dynamic namespace
            for frame_idx, site_idx in chunk:
                label = _snap_label(frame_idx, site_idx)
                xas = _get_dynamic_output(child, "xas_data", label)
                if xas is None:
                    self.report(
                        f"{batch_label}: {label} has no xas_data output; counting as failed."
                    )
                    n_failed += 1
                    continue

                per_site[site_idx][label] = xas
                all_xas[label] = xas

                pc = _get_dynamic_output(child, "path_contributions", label)
                if pc is not None:
                    successful_paths[label] = pc

        self.ctx.per_site_xas = per_site
        self.ctx.all_xas = all_xas
        self.ctx.successful_paths = successful_paths
        self.ctx.n_failed = n_failed

    def submit_feff_calculations(self):
        """Fan out: submit one FeffCalculation per (frame, site) pair."""
        calcs: dict[str, t.Any] = {}
        options = self.inputs.get("options", orm.Dict()).get_dict()
        use_precomputed = self.should_precompute() and hasattr(self.ctx, "pot_remote")
        job_pairs: list[tuple[int, int]] = []

        base_d = self.inputs.parameters.get_dict()
        base_d.pop("absorbing_atoms", None)  # replaced per-site below

        for site_idx in self.ctx.site_indices:
            for i, structure in enumerate(self.ctx.structures):
                d = dict(base_d)
                d["absorbing_atom"] = site_idx
                if use_precomputed:
                    d["control"] = _CONTROL_NO_POT
                params = FeffParameters(dict=d)
                if use_precomputed:
                    params.label = (
                        f"no-SCF snap_{i:04d} site_{site_idx:04d} "
                        f"(derived from parameters<{self.inputs.parameters.pk}>)"
                    )

                snap_label = f"snap_{i:04d}_site_{site_idx:04d}"
                feff_inputs: dict[str, t.Any] = {
                    "code": self.inputs.code,
                    "structure": structure,
                    "parameters": params,
                    "path_cw_threshold": self.inputs.path_cw_threshold,
                    "frame_idx": orm.Int(i),
                    "site_idx": orm.Int(site_idx),
                    "metadata": {
                        "label": f"feff_{snap_label}",
                        "call_link_label": snap_label,
                        "options": options,
                    },
                }
                if use_precomputed:
                    feff_inputs["remote_potentials"] = self.ctx.pot_remote[site_idx]
                if "python_code" in self.inputs:
                    feff_inputs["python_code"] = self.inputs.python_code

                future = self.submit(FeffCalculation, **feff_inputs)
                calcs[snap_label] = future
                job_pairs.append((i, site_idx))
                self.report(
                    f"Submitted FeffCalculation for snapshot {i} site {site_idx} → {future.pk}"
                )

        self.ctx.job_pairs = job_pairs
        self.ctx.n_total = len(job_pairs)
        return ToContext(**calcs)  # type: ignore[arg-type]

    def inspect_results(self) -> None:
        """Collect successful XasData outputs, bucketed by site_idx."""
        # per_site: site_idx → {label: XasData}
        per_site: dict[int, dict[str, XasData]] = {s: {} for s in self.ctx.site_indices}
        all_xas: dict[str, XasData] = {}
        successful_paths: dict[str, PathContributionsData] = {}
        n_failed = 0

        for frame_idx, site_idx in self.ctx.job_pairs:
            label = f"snap_{frame_idx:04d}_site_{site_idx:04d}"
            child = self.ctx[label]
            if not child.is_finished_ok:
                self.report(
                    f"Snapshot {frame_idx} site {site_idx} ({child.pk}) "
                    f"failed with exit status {child.exit_status}."
                )
                n_failed += 1
            elif "xas_data" in child.outputs:
                per_site[site_idx][label] = child.outputs.xas_data
                all_xas[label] = child.outputs.xas_data
                if "path_contributions" in child.outputs:
                    successful_paths[label] = child.outputs.path_contributions
            else:
                self.report(  # noqa: E501
                    f"Snapshot {frame_idx} site {site_idx} finished OK but xas_data missing."
                )
                n_failed += 1

        self.ctx.per_site_xas = per_site
        self.ctx.all_xas = all_xas
        self.ctx.successful_paths = successful_paths
        self.ctx.n_failed = n_failed

    def average_results(self) -> None:
        """Produce per-site and grand-average XasData outputs."""
        n_ok = len(self.ctx.all_xas)
        n_failed = self.ctx.n_failed
        n_total = self.ctx.n_total

        if n_ok == 0:
            self.out("n_failed", orm.Int(n_failed).store())
            self.report("All snapshot calculations failed.")
            return self.exit_codes.ERROR_ALL_FAILED  # type: ignore[no-any-return]

        # Per-site averages
        for site_idx, xas_dict in self.ctx.per_site_xas.items():
            if not xas_dict:
                continue
            averaged = average_xas_data(
                metadata={
                    "call_link_label": f"average_xas_site_{site_idx:04d}",
                    "label": f"ensemble_average_site_{site_idx:04d}",
                },
                **xas_dict,
            )
            self.out(f"averaged_xas.site_{site_idx:04d}", averaged)

        # Grand average over all sites and frames
        grand_avg = average_xas_data(
            metadata={"call_link_label": "average_xas_all", "label": "ensemble_average_all"},
            **self.ctx.all_xas,
        )
        self.out("averaged_xas.all", grand_avg)

        self.out("n_failed", orm.Int(n_failed).store())

        if self.inputs.path_cw_threshold.value >= 0 and self.ctx.successful_paths:
            merged = merge_path_contributions(
                self.inputs.path_r_bin,
                metadata={"call_link_label": "merge_paths", "label": "merged_path_contributions"},
                **self.ctx.successful_paths,
            )
            self.out("path_contributions", merged)

        if "group_label" in self.inputs:
            label = self.inputs.group_label.value
            group, created = orm.Group.collection.get_or_create(label)
            group.add_nodes(self.node)
            action = "created" if created else "updated"
            self.report(f"Group '{label}' {action}: added workchain pk={self.node.pk}")

        if n_failed > 0:
            self.report(
                f"Warning: {n_failed}/{n_total} snapshots failed. "
                f"Average computed from {n_ok} snapshots."
            )
            return self.exit_codes.ERROR_PARTIAL_FAILURE.format(  # type: ignore[no-any-return]
                n_failed=n_failed, n_total=n_total
            )

        self.report(f"Ensemble average complete: {n_ok} snapshots.")
        return None


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _get_dynamic_output(node, namespace: str, key: str):
    """Retrieve a named output from a dynamic output namespace.

    AiiDA exposes dynamic namespace outputs as output link labels of the form
    ``{namespace}.{key}``.  We walk the outgoing links to find it so we do not
    rely on attribute access (which raises on missing keys).

    Args:
        node: The finished process node.
        namespace: Name of the dynamic output namespace (e.g. ``'xas_data'``).
        key: The specific key within the namespace (e.g. ``'snap_0000_site_0001'``).

    Returns:
        The output node, or ``None`` if not found.
    """
    target_label = f"{namespace}.{key}"
    for link in node.get_outgoing().all():
        if link.link_label == target_label:
            return link.node
    return None
