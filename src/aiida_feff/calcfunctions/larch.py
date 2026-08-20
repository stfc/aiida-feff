"""Larch-based calcfunctions for EXAFS post-processing.

These are pure Python functions decorated with ``@calcfunction`` so that
every invocation is tracked in the AiiDA provenance graph.

Available functions
-------------------
average_xas_data
    Ensemble-average a collection of :class:`~aiida_feff.data.xasdata.XasData`
    nodes onto a common energy/k grid.
chi_k_to_r
    Fourier-transform χ(k) → χ(R) using larch.
xftf_arrays
    The plain-array Fourier transform both of the above and
    :mod:`aiida_feff.visualise` share, so there is one set of FT defaults.
"""

from __future__ import annotations

import numpy as np
from aiida.engine import calcfunction
from aiida.orm import ArrayData, Dict

from aiida_feff.data.xasdata import XasData
from aiida_feff.versions import VERSIONS_ATTR, dependency_versions

# ---------------------------------------------------------------------------
# Fourier transform — single source of truth for FT defaults
# ---------------------------------------------------------------------------

#: Defaults applied by :func:`xftf_arrays` when a key is absent.  ``window``
#: is listed explicitly because larch's own default is Kaiser–Bessel, not the
#: Hanning window EXAFS documentation often assumes.
FT_DEFAULTS: dict = {
    "kmin": 3.0,
    "kmax": 15.0,
    "kweight": 2,
    "dk": 1.0,
    "rmax": 8.0,
    "window": "kaiser",
}


def resolve_ft_params(ft_params: dict | None) -> dict:
    """Fill *ft_params* with :data:`FT_DEFAULTS` and reject unknown keys."""
    p = dict(ft_params or {})
    unknown = sorted(set(p) - set(FT_DEFAULTS))
    if unknown:
        raise ValueError(
            f"Unknown Fourier-transform parameter(s): {unknown}. "
            f"Recognised keys: {sorted(FT_DEFAULTS)}"
        )
    return {**FT_DEFAULTS, **p}


def xftf_arrays(k: np.ndarray, chi: np.ndarray, ft_params: dict | None = None) -> dict:
    """Fourier-transform χ(k) → χ(R) and return plain numpy arrays.

    Args:
        k: Photoelectron wavenumber grid (Å⁻¹).
        chi: χ(k) on that grid.
        ft_params: Any subset of :data:`FT_DEFAULTS`.

    Returns:
        Dict with keys ``r``, ``chir_mag``, ``chir_re``, ``chir_im`` and
        ``ft_params`` (the fully-resolved parameters actually used).
    """
    try:
        from larch import Group
        from larch.xafs import xftf
    except ImportError as exc:
        raise ImportError(
            "larch is required for the Fourier transform.  Install with: pip install xraylarch"
        ) from exc

    p = resolve_ft_params(ft_params)
    grp = Group(k=np.asarray(k, dtype=float), chi=np.asarray(chi, dtype=float))
    xftf(
        grp,
        kmin=p["kmin"],
        kmax=p["kmax"],
        kweight=p["kweight"],
        dk=p["dk"],
        window=p["window"],
        rmax_out=p["rmax"],
    )
    return {
        "r": grp.r,
        "chir_mag": np.abs(grp.chir),
        "chir_re": grp.chir.real,
        "chir_im": grp.chir.imag,
        "ft_params": p,
    }


# ---------------------------------------------------------------------------
# Ensemble averaging
# ---------------------------------------------------------------------------


def _average_xas_data_impl(**xas_nodes) -> XasData:
    """Pure-Python ensemble average — no AiiDA tracking.

    Call this directly in unit tests; the :func:`average_xas_data` calcfunction
    delegates to it so the two code paths stay in sync.

    The reference grid comes from the first node in **sorted key order**, so
    the result does not depend on dict iteration order.  Members are
    interpolated onto that grid; χ(k) is masked outside each member's own
    k-range rather than zero-filled, because zero-filling drags the mean
    towards zero at high k — exactly where the Debye–Waller information lives.
    """
    if not xas_nodes:
        raise ValueError("No XasData nodes supplied.")

    labels = sorted(xas_nodes)
    nodes: list[XasData] = [xas_nodes[label] for label in labels]

    with_chi = [n for n in nodes if {"k", "chi_k"} <= set(n.get_arraynames())]
    if not with_chi:
        raise ValueError("None of the supplied XasData nodes contains a chi(k) spectrum.")

    ref = nodes[0]
    energy_ref = ref.get_array("energy")
    k_ref = with_chi[0].get_array("k")

    mu_stack = [
        np.interp(energy_ref, n.get_array("energy"), n.get_array("mu"))
        for n in nodes
        if "mu" in n.get_arraynames()
    ]

    # np.nan outside a member's k-range, then nan-aware statistics: a member
    # that stops at lower kmax simply stops contributing there.
    chi_stack = []
    for node in with_chi:
        k_node = node.get_array("k")
        chi_interp = np.interp(k_ref, k_node, node.get_array("chi_k"), left=np.nan, right=np.nan)
        chi_stack.append(chi_interp)

    out = XasData()

    if mu_stack:
        out.set_spectrum(energy_ref, np.mean(mu_stack, axis=0), e0=float(ref.e0))
        out.set_array("mu_std", _sample_std(np.asarray(mu_stack)))

    chi_arr = np.asarray(chi_stack)
    out.set_chi(k_ref, np.nanmean(chi_arr, axis=0))
    out.set_array("chi_k_std", _sample_std(chi_arr))
    # How many members actually contributed at each k point.
    out.set_array("chi_k_count", np.sum(~np.isnan(chi_arr), axis=0).astype(float))

    out.base.attributes.set("n_snapshots", len(nodes))
    out.base.attributes.set("n_snapshots_chi", len(with_chi))
    out.base.attributes.set("member_labels", labels)
    out.base.attributes.set(VERSIONS_ATTR, dependency_versions())
    return out


def _sample_std(stack: np.ndarray) -> np.ndarray:
    """Sample standard deviation (``ddof=1``) over axis 0, nan-aware.

    ``ddof=1`` throughout the package: ensemble members and MD frames are
    samples from a distribution, not the whole population.

    Points backed by a single member get ``nan``, not zero — one sample gives
    no spread estimate, and reporting zero would understate the uncertainty
    exactly where the ensemble has thinned out.  ``chi_k_count`` says where
    that happened.
    """
    single_member = 2
    if stack.shape[0] < single_member:
        return np.full(stack.shape[1:], np.nan, dtype=float)
    with np.errstate(invalid="ignore"):
        # nanstd warns rather than raising when a slice has too few points;
        # the nan it returns is the intended answer here.
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanstd(stack, axis=0, ddof=1)


@calcfunction
def average_xas_data(**xas_nodes) -> XasData:
    """Average an ensemble of :class:`~aiida_feff.data.xasdata.XasData` nodes.

    Inputs are passed as keyword arguments so that AiiDA builds the correct
    provenance links::

        averaged = average_xas_data(snap_0=xas_0, snap_1=xas_1, …)

    The function interpolates each spectrum onto the grid of the first node in
    sorted key order before averaging, so minor grid differences are tolerated.

    Returns:
    -------
    :class:`~aiida_feff.data.xasdata.XasData`
        Ensemble-averaged spectra, plus ``mu_std`` / ``chi_k_std`` (sample
        standard deviation) and ``chi_k_count`` (members contributing per k).
    """
    return _average_xas_data_impl(**xas_nodes)


@calcfunction
def tag_averaged_xas(averaged: XasData, ft_params: Dict) -> XasData:
    """Copy an averaged XasData and attach Fourier-transform parameters.

    Wrapping the attribute mutation in a calcfunction keeps the provenance
    graph intact; workflows cannot mutate stored Data nodes directly.
    """
    out = XasData()
    for name in averaged.get_arraynames():
        out.set_array(name, averaged.get_array(name))
    for key, val in averaged.base.attributes.all.items():
        if not key.startswith("array|"):
            out.base.attributes.set(key, val)
    out.base.attributes.set("fourier_params", resolve_ft_params(ft_params.get_dict()))
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
