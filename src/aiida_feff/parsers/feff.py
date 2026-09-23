"""FeffParser — parses output files from a FEFF calculation."""

from __future__ import annotations

import io
import traceback

import numpy as np
from aiida.engine import ExitCode
from aiida.orm import SinglefileData
from aiida.parsers import Parser
from typing_extensions import Any

from aiida_feff.calculations.feff import (
    FEFF_CHI_FILE,
    FEFF_CONTRIBUTIONS_RAW,
    FEFF_LOG_FILE,
    FEFF_PATHS_FILE,
    absorber_element,
    is_potentials_only,
)
from aiida_feff.data.xasdata import XasData
from aiida_feff.versions import (
    FEFF_VERSION_ATTR,
    VERSIONS_ATTR,
    dependency_versions,
    parse_feff_version,
)

#: Longest traceback excerpt carried into an exit message.
_TB_EXCERPT_CHARS = 400


class FeffParser(Parser):
    """Parser for :class:`~aiida_feff.calculations.feff.FeffCalculation`.

    Reads the ``retrieved`` FolderData and emits an
    :class:`~aiida_feff.data.xasdata.XasData` output node ``xas_data``
    and, when available, a ``SinglefileData`` node for ``paths.dat``.
    """

    def parse(self, **kwargs: Any) -> ExitCode | None:
        """Entry point called by the AiiDA daemon after retrieval."""
        try:
            result = self._parse_impl()
            return result  # type: ignore[no-any-return]
        except Exception:  # noqa: BLE001
            tb = traceback.format_exc()
            self.logger.error("Parser raised an exception:\n%s", tb)
            return self.exit_codes.ERROR_PARSING_FAILED.format(reason=excerpt_traceback(tb))  # type: ignore[no-any-return]

    # ------------------------------------------------------------------

    def _is_potentials_only(self) -> bool:
        """True when this job deliberately produced no spectrum."""
        try:
            control = self.node.inputs.parameters.get("control")
        except (AttributeError, KeyError):
            return False
        return is_potentials_only(control)

    def _parse_impl(self) -> ExitCode | None:
        retrieved = self.retrieved
        names = retrieved.base.repository.list_object_names()

        feff_version = None
        if FEFF_LOG_FILE in names:
            log_text = retrieved.base.repository.get_object_content(FEFF_LOG_FILE, mode="rb")
            feff_version = parse_feff_version(log_text.decode("utf-8", errors="replace"))

        # ----------------------------------------------------------------
        # Check mandatory output file (chi.dat)
        # ----------------------------------------------------------------
        if FEFF_CHI_FILE not in names:
            if self._is_potentials_only():
                # CONTROL switched the spectrum modules off; pot.pad / phase.pad
                # in the remote folder are the deliverable, and that is success.
                #
                # But a run that crashed before FEFF started also produces no
                # chi.dat, and would otherwise be indistinguishable from
                # success -- collect_potentials would then hand an empty
                # remote folder to every snapshot in the ensemble, and the
                # whole run would quietly use no potentials at all. The
                # potential files themselves stay on the remote and are not
                # retrieved, so they cannot be checked here; FEFF's banner on
                # stdout is the available evidence that it ran at all.
                if feff_version is None:
                    self.logger.error(
                        "Potentials-only run produced no FEFF banner in "
                        f"{FEFF_LOG_FILE}: FEFF did not start."
                    )
                    return self.exit_codes.ERROR_POTENTIALS_INCOMPLETE  # type: ignore[no-any-return]
                self.logger.info("Potentials-only run: no chi.dat expected.")
                return ExitCode(0)
            return self.exit_codes.ERROR_MISSING_CHIDAT  # type: ignore[no-any-return]

        chi_bytes = retrieved.base.repository.get_object_content(FEFF_CHI_FILE, mode="rb")
        xas = _parse_xas(chi_bytes, logger=self.logger, feff_version=feff_version)
        if xas is None:
            return self.exit_codes.ERROR_MISSING_CHIDAT  # type: ignore[no-any-return]

        # Where in the trajectory this spectrum came from, and what absorbed.
        # create_serial_shard reads these back when it assembles the archive,
        # so a missing or wrong value here becomes a mislabelled snapshot
        # there rather than a visible failure.  The element comes from
        # ``parameters.absorbing_atom``, which is what FEFF actually ran on;
        # ``site_idx`` is the ensemble's bookkeeping index and the two only
        # coincide because EnsembleExafsWorkChain sets both.
        xas.base.attributes.set(
            "absorber_element",
            absorber_element(
                self.node.inputs.structure,
                self.node.inputs.parameters.get("absorbing_atom", 0),
            ),
        )
        xas.base.attributes.set("site_index", int(self.node.inputs.site_idx.value))
        xas.base.attributes.set("frame_index", int(self.node.inputs.frame_idx.value))

        self.out("xas_data", xas)

        # ----------------------------------------------------------------
        # Expose paths.dat as SinglefileData
        # ----------------------------------------------------------------
        if FEFF_PATHS_FILE in retrieved.base.repository.list_object_names():
            paths_content = retrieved.base.repository.get_object_content(FEFF_PATHS_FILE, mode="rb")
            sfd = SinglefileData(io.BytesIO(paths_content), filename=FEFF_PATHS_FILE)
            self.out("paths_file", sfd)

        # ----------------------------------------------------------------
        # Per-path contributions
        # ----------------------------------------------------------------
        try:
            store_paths = self.node.inputs.path_cw_threshold.value >= 0
        except AttributeError:
            store_paths = False

        k_chi = xas.get_array("k") if "k" in xas.get_arraynames() else None
        chi_k = xas.get_array("chi_k") if "chi_k" in xas.get_arraynames() else None

        if (
            store_paths
            and k_chi is not None
            and FEFF_CONTRIBUTIONS_RAW in retrieved.base.repository.list_object_names()
        ):
            self._parse_path_contributions(k_chi, chi_k)

        return ExitCode(0)

    def _parse_path_contributions(self, k_chi: Any, chi_k: Any) -> None:
        """Build PathContributionsData from the remotely aggregated HDF5."""
        from aiida_feff.data.pathcontributions import PathContributionsData

        try:
            raw_bytes = self.retrieved.base.repository.get_object_content(
                FEFF_CONTRIBUTIONS_RAW, mode="rb"
            )
        except OSError:
            self.logger.warning(
                f"{FEFF_CONTRIBUTIONS_RAW} not found in retrieved; skipping path_contributions."
            )
            return

        pc = PathContributionsData.from_hdf5_bytes(raw_bytes)
        self.out("path_contributions", pc)


# ---------------------------------------------------------------------------
# Module-level helpers — shared with FeffBatchParser
# ---------------------------------------------------------------------------


def excerpt_traceback(tb: str) -> str:
    """Return the informative tail of a traceback, not its header.

    Truncating from the start keeps ``Traceback (most recent call last):`` and
    discards the exception type and message, which is exactly backwards for an
    exit-code reason.
    """
    return tb.strip()[-_TB_EXCERPT_CHARS:]


def _parse_xas(
    chi_bytes: bytes,
    *,
    logger: Any = None,
    feff_version: str | None = None,
) -> XasData | None:
    """Turn raw ``chi.dat`` bytes into an XasData node.

    Shared by :class:`FeffParser` and
    :class:`~aiida_feff.parsers.feff_batch.FeffBatchParser` so a serial run and
    a batched one produce byte-identical χ(k) for the same FEFF output.

    Args:
        chi_bytes: Raw bytes of the FEFF ``chi.dat`` file.
        logger: Optional logger for warnings.
        feff_version: FEFF banner parsed from ``log.dat``, recorded on the node.

    Returns:
        Populated :class:`~aiida_feff.data.xasdata.XasData`, or ``None`` if the
        file could not be parsed.
    """
    try:
        text = chi_bytes.decode("utf-8", errors="replace")
        k, chi_k = _parse_chi(text)
    except Exception as exc:  # noqa: BLE001
        if logger:
            logger.warning("Failed to parse chi.dat: %s", exc)
        return None

    xas = XasData()
    xas.set_chi(k, chi_k)
    xas.base.attributes.set("chi_source", "feff.chi.dat")
    xas.base.attributes.set(VERSIONS_ATTR, dependency_versions())
    if feff_version:
        xas.base.attributes.set(FEFF_VERSION_ATTR, feff_version)

    return xas


def _parse_chi(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse ``chi.dat``.

    Columns: k  chi(k)  |chi(k)|  phase(k)
    """
    data_lines = [
        line for line in text.splitlines() if line.strip() and not line.strip().startswith("#")
    ]
    if not data_lines:
        raise ValueError("chi.dat contains no data lines")
    arr = np.loadtxt(io.StringIO("\n".join(data_lines)))
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    k = arr[:, 0]
    chi_k = arr[:, 1]
    return k, chi_k
