"""FeffBatchParser — parses output files from a FeffBatchCalculation."""

from __future__ import annotations

import traceback

from aiida.engine import ExitCode
from aiida.parsers import Parser
from typing_extensions import Any

from aiida_feff.calculations.feff import FEFF_CHI_FILE, FEFF_CONTRIBUTIONS_RAW, FEFF_XMUDA_FILE
from aiida_feff.calculations.feff_batch import _snap_label
from aiida_feff.parsers.feff import _parse_xas


class FeffBatchParser(Parser):
    """Parser for :class:`~aiida_feff.calculations.feff_batch.FeffBatchCalculation`.

    Iterates over every ``snap_FFFF_site_SSSS/`` subdirectory in the retrieved
    ``FolderData`` and emits one :class:`~aiida_feff.data.xasdata.XasData` output
    per successful run in the dynamic namespace ``xas_data``, and one
    :class:`~aiida_feff.data.pathcontributions.PathContributionsData` per run
    in the dynamic namespace ``path_contributions`` (when aggregation was
    requested and succeeded).

    Partial failure is accepted: missing ``xmu.dat`` for a run is logged and
    skipped.  The CalcJob is only marked failed if *no* outputs were produced
    at all.
    """

    def parse(self, **kwargs: Any) -> ExitCode | None:
        """Entry point called by the AiiDA daemon after retrieval."""
        try:
            return self._parse_impl()  # type: ignore[no-any-return]
        except Exception:  # noqa: BLE001
            tb = traceback.format_exc()
            self.logger.error("Batch parser raised an exception:\n%s", tb)
            return self.exit_codes.ERROR_PARSING_FAILED.format(reason=tb[:200])  # type: ignore[no-any-return]

    # ------------------------------------------------------------------

    def _parse_impl(self) -> ExitCode | None:
        retrieved = self.retrieved
        top_names = retrieved.base.repository.list_object_names()

        # Collect all snap_* subdirectories present in the retrieved folder
        snap_dirs = sorted(n for n in top_names if n.startswith("snap_"))

        if not snap_dirs:
            self.logger.error(
                "No snap_* directories found in retrieved folder; "
                "the driver may have failed entirely. Check batch_err.log."
            )
            return self.exit_codes.ERROR_ALL_RUNS_FAILED  # type: ignore[no-any-return]

        try:
            store_paths = self.node.inputs.path_cw_threshold.value >= 0
        except AttributeError:
            store_paths = False

        n_ok = 0
        for snap_dir in snap_dirs:
            frame_idx, site_idx = _parse_snap_dir_name(snap_dir)
            if frame_idx is None or site_idx is None:
                self.logger.warning(
                    "Could not parse frame/site from dir name %r; skipping", snap_dir
                )
                continue

            label = _snap_label(frame_idx, site_idx)
            files_in_dir = retrieved.base.repository.list_object_names(snap_dir)

            if FEFF_XMUDA_FILE not in files_in_dir:
                self.logger.warning("%s: xmu.dat missing; run likely failed", snap_dir)
                continue

            xmu_bytes = retrieved.base.repository.get_object_content(
                f"{snap_dir}/{FEFF_XMUDA_FILE}", mode="rb"
            )
            chi_bytes: bytes | None = None
            if FEFF_CHI_FILE in files_in_dir:
                chi_bytes = retrieved.base.repository.get_object_content(
                    f"{snap_dir}/{FEFF_CHI_FILE}", mode="rb"
                )

            xas = _parse_xas(xmu_bytes, chi_bytes, logger=self.logger)
            if xas is None:
                self.logger.warning("%s: XAS parsing returned None; skipping", snap_dir)
                continue

            self.out(f"xas_data.{label}", xas)
            n_ok += 1

            if store_paths and FEFF_CONTRIBUTIONS_RAW in files_in_dir:
                self._parse_path_contributions(snap_dir, label)

        if n_ok == 0:
            self.logger.error("All %d runs failed to produce xmu.dat", len(snap_dirs))
            return self.exit_codes.ERROR_ALL_RUNS_FAILED  # type: ignore[no-any-return]

        self.logger.info("Batch parse complete: %d/%d runs produced XasData", n_ok, len(snap_dirs))
        return ExitCode(0)

    def _parse_path_contributions(self, snap_dir: str, label: str) -> None:
        """Parse one contributions_raw.h5 and emit it under path_contributions.*."""
        from aiida_feff.data.pathcontributions import PathContributionsData

        try:
            raw_bytes = self.retrieved.base.repository.get_object_content(
                f"{snap_dir}/{FEFF_CONTRIBUTIONS_RAW}", mode="rb"
            )
        except OSError:
            self.logger.warning("%s: contributions_raw.h5 not found; skipping", snap_dir)
            return

        pc = PathContributionsData.from_hdf5_bytes(raw_bytes)
        self.out(f"path_contributions.{label}", pc)


# ---------------------------------------------------------------------------


def _parse_snap_dir_name(name: str) -> tuple[int | None, int | None]:
    """Extract (frame_idx, site_idx) from a directory name ``snap_FFFF_site_SSSS``.

    Returns (None, None) if the name does not match the expected pattern.
    """
    # Expected: snap_0000_site_0001
    try:
        after_snap = name.removeprefix("snap_")
        frame_str, site_str = after_snap.split("_site_")
        return int(frame_str), int(site_str)
    except (ValueError, AttributeError):
        return None, None
