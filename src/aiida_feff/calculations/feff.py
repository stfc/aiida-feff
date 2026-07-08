"""FeffCalculation — CalcJob wrapping a single FEFF execution."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from aiida import orm
from aiida.common import CalcInfo, CodeInfo, datastructures
from aiida.engine import CalcJob, CalcJobProcessSpec
from pymatgen.core import Structure

from aiida_feff.data.parameters import FeffParameters
from aiida_feff.data.pathcontributions import PathContributionsData

logger = logging.getLogger(__name__)

# Files written to / retrieved from the remote working directory
FEFF_INPUT_FILE = "feff.inp"
FEFF_LOG_FILE = "log.dat"
FEFF_XMUDA_FILE = "xmu.dat"
FEFF_CHI_FILE = "chi.dat"
FEFF_PATHS_FILE = "paths.dat"
FEFF_FILES_DAT = "files.dat"  # per-path amplitude ranking table
FEFF_AGGREGATE_SCRIPT = "_aggregate_paths.py"  # written to sandbox by prepare_for_submission
FEFF_AGGREGATE_CONFIG = "_feff_aggregate_config.json"  # threshold config for the above
FEFF_CONTRIBUTIONS_RAW = "contributions_raw.h5"  # produced by the aggregate script

# Files produced by the FEFF potentials step (CONTROL 1 1 1 0 0 0)
# that can be reused in subsequent runs (CONTROL 0 0 0 1 1 1).
# Derived from empirical testing — all are needed for module 6 (chi).
FEFF_POTENTIAL_FILES = (
    "pot.pad",
    "phase.pad",
    "xsect.json",
    "xsph.json",
    "genfmt.json",
    "ff2x.json",
    "geom.json",
    "atoms.json",
    "pot.json",
    "global.json",
    "path.json",
    "libpotph.json",
)

# Path to the bundled aggregation script (installed alongside this module)
_AGGREGATE_SCRIPT_PATH = Path(__file__).parent / FEFF_AGGREGATE_SCRIPT


class FeffCalculation(CalcJob):
    r"""CalcJob that runs a single FEFF calculation.

    Inputs
    ------
    code : :class:`~aiida.orm.AbstractCode`
        The FEFF executable.
    python_code : :class:`~aiida.orm.AbstractCode`, optional
        Python 3 interpreter on the remote computer.  Required when
        ``path_cw_threshold >= 0``.  AiiDA runs ``_aggregate_paths.py`` as a
        second sequential command in the same job.  Register once per
        computer::

            verdi code create core.code.installed \\
                --computer=<c> --filepath-executable=$(which python3)
    structure : :class:`~aiida.orm.StructureData`
        Crystal / cluster structure.  The absorbing atom is identified by
        ``parameters.absorbing_atom`` (0-based site index).
    parameters : :class:`~aiida_feff.data.parameters.FeffParameters`
        Calculation control parameters.
    feff_input_file : :class:`~aiida.orm.SinglefileData`, optional
        If supplied, this pre-written ``feff.inp`` is used *as-is*,
        bypassing automatic generation from ``structure`` + ``parameters``.
        Useful for importing existing calculations.
    settings : :class:`~aiida.orm.Dict`, optional
        Low-level scheduler / plugin settings (see below).

    Outputs
    -------
    xas_data : :class:`~aiida_feff.data.xasdata.XasData`
        Parsed μ(E) and χ(k) spectra.
    retrieved : :class:`~aiida.orm.FolderData`
        All retrieved files.

    Exit codes
    ----------
    300  Input validation error.
    310  FEFF did not produce ``xmu.dat``.
    311  FEFF did not produce ``chi.dat``.
    400  Unrecoverable parser error.
    """

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------
    _DEFAULT_RETRIEVE_LIST = [
        FEFF_LOG_FILE,
        FEFF_XMUDA_FILE,
        FEFF_CHI_FILE,
        FEFF_PATHS_FILE,
        FEFF_FILES_DAT,  # amplitude ranking; small file, kept for provenance
    ]
    _WITHMPI = False  # FEFF is serial by default; set via metadata

    # ------------------------------------------------------------------
    # define
    # ------------------------------------------------------------------

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:  # type: ignore[override]
        """Define inputs, outputs and exit codes."""
        super().define(spec)

        # --- inputs -----------------------------------------------------------
        spec.input(
            "structure",
            valid_type=orm.StructureData,
            required=False,
            help="Cluster structure.  Required unless feff_input_file is provided.",
        )
        spec.input(
            "parameters",
            valid_type=FeffParameters,
            required=False,
            help="FEFF control parameters (edge, rpath, …).",
        )
        spec.input(
            "feff_input_file",
            valid_type=orm.SinglefileData,
            required=False,
            help="Pre-written feff.inp to use verbatim, bypassing auto-generation.",
        )
        spec.input(
            "remote_potentials",
            valid_type=orm.RemoteData,
            required=False,
            help=(
                "Remote working directory of a completed FEFF potentials-only run "
                "(e.g. CONTROL 1 1 1 0 0 0).  When supplied, pot.pad and phase.pad "
                "are copied from that directory into this job's working directory, "
                "so the SCF step can be skipped (use CONTROL 0 0 0 1 1 1 in parameters)."
            ),
        )
        spec.input(
            "settings",
            valid_type=orm.Dict,
            required=False,
            help=(
                "Miscellaneous plugin settings:\n"
                "  additional_retrieve_list : list[str]  — extra glob patterns\n"
                "  cmdline_params          : list[str]  — extra CLI flags\n"
            ),
        )
        spec.input(
            "python_code",
            valid_type=orm.AbstractCode,
            required=False,
            help=(
                "Python 3 interpreter on the remote computer.  Required when "
                "path_cw_threshold >= 0.  AiiDA runs '_aggregate_paths.py' as "
                "a second sequential command in the same job script."
            ),
        )
        spec.input(
            "path_cw_threshold",
            valid_type=orm.Float,
            default=lambda: orm.Float(-1.0),
            help=(
                "Minimum curved-wave amplitude ratio (0–100) for storing a "
                "scattering path.  FEFF's files.dat ranks all paths relative "
                "to the strongest path (= 100).  Set to e.g. 5.0 to keep only "
                "paths contributing ≥ 5 %% of the peak amplitude.  "
                "Set to 0 to store all paths.  "
                "Set to -1 (or any negative value) to skip path storage entirely.  "
                "Requires python_code when >= 0."
            ),
        )
        spec.input(
            "frame_idx",
            valid_type=orm.Int,
            default=lambda: orm.Int(0),
            help=(
                "Frame index of this calculation within an MD trajectory.  "
                "Written into the aggregation config and stored as metadata in "
                "the PathContributionsData HDF5.  Set automatically by "
                "EnsembleExafsWorkChain; leave at 0 for single-site runs."
            ),
        )
        spec.input(
            "site_idx",
            valid_type=orm.Int,
            default=lambda: orm.Int(0),
            help=(
                "Absorbing-site index within the structure for this calculation.  "
                "Written into the aggregation config and stored as metadata in "
                "the PathContributionsData HDF5.  Leave at 0 unless running "
                "multi-site ensemble calculations."
            ),
        )
        spec.inputs["metadata"]["options"]["parser_name"].default = "feff.feff"  # type: ignore[index]
        spec.inputs["metadata"]["options"]["withmpi"].default = cls._WITHMPI  # type: ignore[index]

        # --- outputs ----------------------------------------------------------
        spec.output(
            "xas_data",
            valid_type=orm.ArrayData,
            required=False,
            help="Parsed XAS / EXAFS data arrays.",
        )
        spec.output(
            "path_contributions",
            valid_type=PathContributionsData,
            required=False,
            help=(
                "Amplitude-filtered per-path FEFF scattering factors packed "
                "into a single HDF5 file.  Emitted when python_code is "
                "provided and path_cw_threshold >= 0."
            ),
        )
        spec.output(
            "paths_file",
            valid_type=orm.SinglefileData,
            required=False,
            help="paths.dat produced by FEFF (scattering paths).",
        )

        # --- exit codes -------------------------------------------------------
        spec.exit_code(300, "ERROR_INVALID_INPUT", message="Input validation failed: {reason}.")
        spec.exit_code(310, "ERROR_MISSING_XMUDA", message="FEFF did not produce xmu.dat.")
        spec.exit_code(
            311, "ERROR_MISSING_CHI", message="FEFF did not produce chi.dat (EXAFS mode only)."
        )
        spec.exit_code(400, "ERROR_PARSING_FAILED", message="Parser raised an exception: {reason}.")

    # ------------------------------------------------------------------
    # prepare_for_submission
    # ------------------------------------------------------------------

    def prepare_for_submission(self, folder) -> CalcInfo:
        """Write input files to the *sandbox* folder and return CalcInfo.

        This is the only method plugins must implement.  AiiDA calls it
        before uploading the folder to the remote computer.
        """
        settings = self.inputs.get("settings", orm.Dict()).get_dict()

        # ----------------------------------------------------------------
        # 1. Build / copy feff.inp
        # ----------------------------------------------------------------
        if "feff_input_file" in self.inputs:
            # User-supplied verbatim input
            src = self.inputs.feff_input_file
            with folder.open(FEFF_INPUT_FILE, "wb") as fh:
                fh.write(src.get_content("rb"))
        else:
            if "structure" not in self.inputs or "parameters" not in self.inputs:
                err = self.exit_codes.ERROR_INVALID_INPUT.format(
                    reason="Either feff_input_file or both structure+parameters must be supplied"
                )
                return err  # type: ignore[no-any-return]
            inp_text = self._build_feff_inp(
                self.inputs.structure,
                self.inputs.parameters,
            )
            with folder.open(FEFF_INPUT_FILE, "w") as fh:
                fh.write(inp_text)

        # ----------------------------------------------------------------
        # 2. CodeInfo
        # ----------------------------------------------------------------
        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = settings.get("cmdline_params", [])
        codeinfo.stdout_name = FEFF_LOG_FILE
        codeinfo.stderr_name = "stderr.txt"

        # ----------------------------------------------------------------
        # 3. CalcInfo
        # ----------------------------------------------------------------
        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = []
        calcinfo.remote_copy_list = []

        # When pre-computed potentials are supplied, copy pot.pad and
        # phase.pad from the reference job's remote directory.  All file
        # transfers stay on the HPC — nothing is downloaded.
        if "remote_potentials" in self.inputs:
            remote = self.inputs.remote_potentials
            computer_uuid = remote.computer.uuid
            remote_path = remote.get_remote_path()
            for fname in FEFF_POTENTIAL_FILES:
                calcinfo.remote_copy_list.append((computer_uuid, f"{remote_path}/{fname}", fname))

        retrieve_list: list[str | tuple[str, str, str]] = list(self._DEFAULT_RETRIEVE_LIST)
        retrieve_list.extend(settings.get("additional_retrieve_list", []))
        calcinfo.retrieve_list = retrieve_list

        # When path contributions are requested, write the aggregation script
        # and config to the sandbox, then run them via a second CodeInfo.
        # feff????.dat files are never retrieved — only contributions_raw.h5.
        threshold = self.inputs.path_cw_threshold.value
        if threshold >= 0:
            if "python_code" not in self.inputs:
                err = self.exit_codes.ERROR_INVALID_INPUT.format(
                    reason="python_code is required when path_cw_threshold >= 0"
                )
                return err  # type: ignore[no-any-return]
            with folder.open(FEFF_AGGREGATE_SCRIPT, "wb") as fh:
                fh.write(_AGGREGATE_SCRIPT_PATH.read_bytes())
            with folder.open(FEFF_AGGREGATE_CONFIG, "w") as fh:
                json.dump(
                    {
                        "threshold": float(threshold),
                        "frame_idx": self.inputs.frame_idx.value,
                        "site_idx": self.inputs.site_idx.value,
                        "absorber_element": self._absorber_element(),
                    },
                    fh,
                )
            calcinfo.retrieve_list = list(retrieve_list) + [FEFF_CONTRIBUTIONS_RAW]

            agg_codeinfo = CodeInfo()
            agg_codeinfo.code_uuid = self.inputs.python_code.uuid
            agg_codeinfo.cmdline_params = [FEFF_AGGREGATE_SCRIPT]
            agg_codeinfo.withmpi = False
            calcinfo.codes_info = [codeinfo, agg_codeinfo]
            calcinfo.codes_run_mode = datastructures.CodeRunMode.SERIAL

        return calcinfo

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _absorber_element(self) -> str:
        """Return the element symbol of the absorbing atom, or empty string."""
        try:
            structure = self.inputs.structure
            parameters = self.inputs.parameters
            absorbing_idx = parameters.get("absorbing_atom", 0)
            site = structure.sites[absorbing_idx]
            kind = structure.get_kind(site.kind_name)
            return str(kind.symbols[0])
        except Exception:  # noqa: BLE001
            return ""

    @staticmethod
    def _build_feff_inp(
        structure: orm.StructureData,
        parameters: FeffParameters,
    ) -> str:
        """Construct the full text of ``feff.inp`` from AiiDA objects."""
        import tempfile

        from pymatgen.io.feff.sets import MPEXAFSSet

        absorbing_idx = parameters.get("absorbing_atom", 0)
        exclude_h = bool(parameters.get("exclude_hydrogen", False))

        pmg_structure: Structure = structure.get_pymatgen_structure()

        if exclude_h:
            symbols = [site.species_string for site in pmg_structure.sites]
            non_h = [i for i, sym in enumerate(symbols) if sym != "H"]
            if absorbing_idx not in non_h:
                raise ValueError(
                    f"absorbing_atom index {absorbing_idx} is a hydrogen atom "
                    "but exclude_hydrogen=True."
                )
            absorbing_idx = non_h.index(absorbing_idx)
            pmg_structure.remove_sites([i for i, sym in enumerate(symbols) if sym == "H"])

        user_settings = parameters.to_pymatgen_user_tags()
        user_settings["RPATH"] = str(parameters.radius)

        del_value = user_settings.pop("_del", None)
        del_list: list[str] = []
        if del_value is None:
            pass
        elif isinstance(del_value, str):
            del_list = [del_value]
        else:
            del_list = list(del_value)

        for kw in ("COREHOLE", "COREHOLE FSR"):
            if kw not in del_list:
                del_list.append(kw)
        if del_list:
            user_settings["_del"] = del_list

        # spglib 2.7 deprecates the old error-handling path that pymatgen's
        # SpacegroupAnalyzer still triggers; opt into the new path explicitly.
        import spglib.error

        spglib.error.OLD_ERROR_HANDLING = False

        feff_set = MPEXAFSSet(
            absorbing_atom=absorbing_idx,
            structure=pmg_structure,
            edge=parameters.edge,
            radius=parameters.radius,
            user_tag_settings=user_settings,
        )

        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                feff_set.write_input(tmpdir)
            finally:
                os.chdir(original_cwd)
            inp_path = Path(tmpdir) / "feff.inp"
            return inp_path.read_text()
