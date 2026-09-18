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
    FEFF_XMUDA_FILE,
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
        # Check mandatory output file
        # ----------------------------------------------------------------
        if FEFF_XMUDA_FILE not in names:
            if self._is_potentials_only():
                # CONTROL switched the spectrum modules off; pot.pad / phase.pad
                # in the remote folder are the deliverable, and that is success.
                #
                # But a run that crashed before FEFF started also produces no
                # xmu.dat, and would otherwise be indistinguishable from
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
                self.logger.info("Potentials-only run: no xmu.dat expected.")
                return ExitCode(0)
            return self.exit_codes.ERROR_MISSING_XMUDA  # type: ignore[no-any-return]

        xmu_bytes = retrieved.base.repository.get_object_content(FEFF_XMUDA_FILE, mode="rb")
        chi_bytes: bytes | None = None
        if FEFF_CHI_FILE in names:
            chi_bytes = retrieved.base.repository.get_object_content(FEFF_CHI_FILE, mode="rb")

        xas = _parse_xas(xmu_bytes, chi_bytes, logger=self.logger, feff_version=feff_version)
        if xas is None:
            return self.exit_codes.ERROR_MISSING_XMUDA  # type: ignore[no-any-return]

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
    xmu_bytes: bytes,
    chi_bytes: bytes | None,
    logger: Any = None,
    feff_version: str | None = None,
) -> XasData | None:
    """Parse raw ``xmu.dat`` bytes (and optionally ``chi.dat``) into an XasData node.

    This is the core larch parsing logic factored out so both :class:`FeffParser`
    and :class:`~aiida_feff.parsers.feff_batch.FeffBatchParser` can call it
    without duplicating the larch import / pre_edge / autobk dance.

    Args:
        xmu_bytes: Raw bytes of the FEFF ``xmu.dat`` file.
        chi_bytes: Raw bytes of FEFF ``chi.dat``, used as fallback if autobk fails.
            ``None`` skips the fallback.
        logger: Optional logger for warnings (``logging.Logger`` or AiiDA parser logger).
        feff_version: FEFF banner parsed from ``log.dat``, recorded on the node.

    Returns:
        Populated :class:`~aiida_feff.data.xasdata.XasData` node, or ``None``
        if ``xmu_bytes`` could not be parsed.
    """
    import os
    import tempfile

    from larch.io import read_ascii
    from larch.xafs import autobk, pre_edge

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            xmu_path = os.path.join(tmpdir, FEFF_XMUDA_FILE)
            with open(xmu_path, "wb") as fh:
                fh.write(xmu_bytes)
            grp = read_ascii(xmu_path)
    except Exception as exc:  # noqa: BLE001
        if logger:
            logger.warning("larch read_ascii failed: %s", exc)
        return None

    grp.energy = grp.omega

    try:
        pre_edge(grp.energy, grp.mu, group=grp, e0=0)
    except Exception as exc:  # noqa: BLE001
        if logger:
            logger.warning("larch pre_edge failed: %s", exc)

    autobk_ok = False
    autobk_min_points = 5
    autobk_min_kmax = 2.0
    try:
        autobk(grp.energy, grp.mu, group=grp)
        k_arr = getattr(grp, "k", None)
        autobk_ok = (
            k_arr is not None
            and len(k_arr) > autobk_min_points
            and float(k_arr.max()) > autobk_min_kmax
        )
    except Exception as exc:  # noqa: BLE001
        if logger:
            logger.warning("larch autobk failed; will fall back to FEFF chi.dat: %s", exc)

    e0_absolute = 0.0
    for line in getattr(grp, "header", []):
        if "e0" in line.lower() and "=" in line:
            try:
                e0_absolute = float(line.split("=")[-1])
                break
            except ValueError:
                pass

    xas = XasData()
    mu0 = getattr(grp, "mu0", None)
    # grp.energy is FEFF's omega column: energy relative to E0.  e0 carries the
    # absolute threshold so the two can be recombined.
    xas.set_spectrum(
        np.asarray(grp.energy),
        np.asarray(grp.mu),
        np.asarray(mu0) if mu0 is not None else None,
        e0=e0_absolute,
    )

    # Which background algorithm produced chi(k) changes its meaning, so the
    # choice is recorded rather than left implicit in the numbers.
    chi_source = "none"
    if autobk_ok:
        xas.set_chi(np.asarray(grp.k), np.asarray(grp.chi))
        chi_source = "larch.autobk"
    elif chi_bytes is not None:
        try:
            k_chi_fb, chi_k_fb = _parse_chi(chi_bytes.decode("utf-8", errors="replace"))
            xas.set_chi(k_chi_fb, chi_k_fb)
            chi_source = "feff.chi.dat"
        except Exception as exc:  # noqa: BLE001
            if logger:
                logger.warning("chi.dat fallback failed: %s", exc)
    elif logger:
        logger.info("autobk insufficient and no chi.dat; chi(k) will not be available.")

    xas.base.attributes.set("chi_source", chi_source)
    xas.base.attributes.set(VERSIONS_ATTR, dependency_versions())
    if feff_version:
        xas.base.attributes.set(FEFF_VERSION_ATTR, feff_version)

    return xas


def _parse_chi(text: str):
    """Parse ``chi.dat``.

    Columns: k  chi(k)  |chi(k)|  phase(k)
    """
    data_lines = [
        line for line in text.splitlines() if line.strip() and not line.strip().startswith("#")
    ]
    arr = np.loadtxt(io.StringIO("\n".join(data_lines)))
    k = arr[:, 0]
    chi_k = arr[:, 1]
    return k, chi_k
