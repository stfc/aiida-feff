"""Larch-based calcfunctions for EXAFS post-processing.

These are pure Python functions decorated with ``@calcfunction`` so that
every invocation is tracked in the AiiDA provenance graph.

Available functions
-------------------
archive_to_averaged_xas
    Project the grand and per-site averages held in an
    :class:`~aiida_feff.data.archive.ExafsArchiveData` into XasData nodes.
chi_k_to_r
    Fourier-transform χ(k) → χ(R) using larch.
xftf_arrays
    The plain-array Fourier transform both of the above and
    :mod:`aiida_feff.visualise` share, so there is one set of FT defaults.
"""

from __future__ import annotations

from aiida.engine import calcfunction
from aiida.orm import ArrayData, Dict
from md_exafs.spectra import FT_DEFAULTS, resolve_ft_params, xftf_arrays

from aiida_feff.data.archive import ExafsArchiveData
from aiida_feff.data.xasdata import XasData
from aiida_feff.versions import VERSIONS_ATTR, dependency_versions

__all__ = [
    "FT_DEFAULTS",
    "archive_to_averaged_xas",
    "chi_k_to_r",
    "resolve_ft_params",
    "xftf_arrays",
]

# ---------------------------------------------------------------------------
# Ensemble archive projection to XasData
# ---------------------------------------------------------------------------


@calcfunction
def archive_to_averaged_xas(archive: ExafsArchiveData) -> dict[str, XasData]:
    """Unpack an ensemble archive into the ``averaged_xas`` output namespace.

    Returns ``all`` (the grand average) plus one ``site_SSSS`` entry per
    absorber site that the archive holds an average for.  Going through a
    calcfunction rather than reading the archive in the workchain keeps the
    link ``archive → averaged_xas`` in the provenance graph, so the arrays
    downstream tooling plots can be traced back to the file they came from.

    The archive is the single source: these nodes are a projection of it, not
    a second average taken over the per-snapshot XasData nodes.
    """
    out: dict[str, XasData] = {"all": archive.to_xas_data()}
    with archive.reader() as reader:
        site_indices = reader.site_indices
    for site_idx in site_indices:
        out[f"site_{site_idx:04d}"] = archive.to_site_xas_data(site_idx)
    return out


# ---------------------------------------------------------------------------
# χ(k) → χ(R) Fourier transform
# ---------------------------------------------------------------------------


@calcfunction
def chi_k_to_r(xas_data: XasData, ft_params: Dict) -> ArrayData:
    """Fourier-transform χ(k) to χ(R) using larch.

    Parameters
    ----------
    xas_data:
        Node containing ``k`` and ``chi_k`` arrays.
    ft_params:
        Any subset of :data:`FT_DEFAULTS`:

          - kmin, kmax (float) — k-range of the window
          - kweight (int, default 2) — k-weighting exponent
          - dk (float, default 1.0) — window taper width
          - rmax (float, default 8.0) — max R in Å
          - window (str, default ``"kaiser"``) — larch window function

    Returns:
    -------
    :class:`~aiida.orm.ArrayData`
        Arrays ``r``, ``chir_mag``, ``chir_re``, ``chir_im``.  χ(R) carries
        units of Å^-(kweight+1), so the resolved FT parameters — including
        ``kweight`` — are stored in the ``fourier_params`` attribute.
    """
    result = xftf_arrays(
        xas_data.get_array("k"),
        xas_data.get_array("chi_k"),
        ft_params.get_dict(),
    )

    out = ArrayData()
    for name in ("r", "chir_mag", "chir_re", "chir_im"):
        out.set_array(name, result[name])
    out.base.attributes.set("fourier_params", result["ft_params"])
    out.base.attributes.set(VERSIONS_ATTR, dependency_versions())
    return out
