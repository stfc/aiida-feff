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
"""

from __future__ import annotations

import numpy as np
from aiida.engine import calcfunction
from aiida.orm import ArrayData, Dict

from aiida_feff.data.xasdata import XasData

# ---------------------------------------------------------------------------
# Ensemble averaging
# ---------------------------------------------------------------------------


def _average_xas_data_impl(**xas_nodes) -> XasData:
    """Pure-Python ensemble average — no AiiDA tracking.

    Call this directly in unit tests; the :func:`average_xas_data` calcfunction
    delegates to it so the two code paths stay in sync.
    """
    nodes: list[XasData] = list(xas_nodes.values())
    if not nodes:
        raise ValueError("No XasData nodes supplied.")

    # Reference grid from first node
    ref = nodes[0]
    k_ref = ref.get_array("k")
    energy_ref = ref.get_array("energy")

    mu_stack = []
    chi_stack = []

    for node in nodes:
        mu_interp = np.interp(energy_ref, node.get_array("energy"), node.get_array("mu"))
        mu_stack.append(mu_interp)
        if "k" in node.get_arraynames():
            chi_interp = np.interp(
                k_ref, node.get_array("k"), node.get_array("chi_k"), left=0.0, right=0.0
            )
            chi_stack.append(chi_interp)

    out = XasData()
    mu_avg = np.mean(mu_stack, axis=0)
    mu_std = np.std(mu_stack, axis=0)
    out.set_spectrum(energy_ref, mu_avg)
    out.set_array("mu_std", mu_std)

    if chi_stack:
        chi_avg = np.mean(chi_stack, axis=0)
        chi_std = np.std(chi_stack, axis=0)
        out.set_chi(k_ref, chi_avg)
        out.set_array("chi_k_std", chi_std)

    out.base.extras.set("n_snapshots", len(nodes))
    return out


@calcfunction
def average_xas_data(**xas_nodes) -> XasData:
    """Average an ensemble of :class:`~aiida_feff.data.xasdata.XasData` nodes.

    Inputs are passed as keyword arguments so that AiiDA builds the correct
    provenance links::

        averaged = average_xas_data(snap_0=xas_0, snap_1=xas_1, …)

    The function interpolates each spectrum onto the k-grid of the first
    node before averaging, so minor grid differences are tolerated.

    Returns:
    -------
    :class:`~aiida_feff.data.xasdata.XasData`
        Ensemble-averaged spectra.
    """
    return _average_xas_data_impl(**xas_nodes)


@calcfunction
def tag_averaged_xas(averaged: XasData, ft_params: Dict) -> XasData:
    """Copy an averaged XasData and attach Fourier-transform parameters as extras.

    Wrapping the extras mutation in a calcfunction keeps the provenance graph
    intact; workflows cannot mutate unstored Data nodes directly.
    """
    out = XasData()
    for name in averaged.get_arraynames():
        out.set_array(name, averaged.get_array(name))
    for key, val in averaged.base.extras.all.items():
        out.base.extras.set(key, val)
    out.base.extras.set("fourier_params", ft_params.get_dict())
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
        Dict with keys:
          - kmin, kmax (float) — k-range for window
          - kweight (int, default 2) — k-weighting exponent
          - dk (float, default 1.0) — Hanning window width
          - rmax (float, default 8.0) — max R in Å

    Returns:
    -------
    :class:`~aiida.orm.ArrayData`
        Arrays ``r``, ``chir_mag``, ``chir_re``, ``chir_im``.
    """
    try:
        from larch import Group
        from larch.xafs import xftf
    except ImportError as exc:
        raise ImportError(
            "larch is required for chi_k_to_r.  Install with: pip install xraylarch"
        ) from exc

    p = ft_params.get_dict()
    k = xas_data.get_array("k")
    chi = xas_data.get_array("chi_k")

    grp = Group(k=k, chi=chi)
    xftf(
        grp,
        kmin=p.get("kmin", 3.0),
        kmax=p.get("kmax", 15.0),
        kweight=p.get("kweight", 2),
        dk=p.get("dk", 1.0),
        rmax_out=p.get("rmax", 8.0),
    )

    out = ArrayData()
    out.set_array("r", grp.r)
    out.set_array("chir_mag", np.abs(grp.chir))
    out.set_array("chir_re", grp.chir.real)
    out.set_array("chir_im", grp.chir.imag)
    return out
