"""Plotting helpers for aiida-feff.

These are plain Python functions — *not* calcfunctions — that accept AiiDA
nodes and return ``matplotlib`` figures.  They are intentionally kept
outside ``calcfunctions/`` because plotting produces side-effects (figures)
and is an interactive concern, not a provenance-tracked computation step.

Typical usage::

    from aiida.orm import load_node
    from aiida_feff.visualise import plot_mu_e, plot_chi_k, plot_chi_r

    xas = load_node(<pk>)                    # XasData
    fig = plot_mu_e(xas)

    # R-space: run the FT calcfunction first, then plot
    from aiida.orm import Dict
    from aiida_feff.calcfunctions.larch import chi_k_to_r
    chir = chi_k_to_r(xas_data=xas, ft_params=Dict({"kmin": 3, "kmax": 14}))
    fig = plot_chi_r(chir)

    # Or let plot_chi_r run the FT on the fly (not tracked)
    fig = plot_chi_r(xas, ft_params={"kmin": 3, "kmax": 14})

Requires the ``plots`` optional dependency::

    pip install aiida-feff[plots]
"""

from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

import numpy as np

from aiida_feff.data.xasdata import XasData

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = ["plot_mu_e", "plot_chi_k", "plot_chi_r"]


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting.  Install with: pip install aiida-feff[plots]"
        ) from exc


# ---------------------------------------------------------------------------
# μ(E)
# ---------------------------------------------------------------------------


def plot_mu_e(
    xas_data: XasData,
    *,
    ax=None,
    label: str | None = None,
    show_mu0: bool = False,
    energy_offset: float = 0.0,
) -> Figure:
    """Plot the absorption spectrum μ(E)."""
    plt = _require_matplotlib()

    energy = xas_data.get_array("energy") + energy_offset
    mu = xas_data.get_array("mu")

    fig = None
    if ax is None:
        fig, ax = plt.subplots()

    _label = label or (xas_data.label or f"pk={xas_data.pk}")
    ax.plot(energy, mu, label=_label)

    if show_mu0 and "mu0" in xas_data.get_arraynames():
        ax.plot(energy, xas_data.get_array("mu0"), linestyle="--", label=f"{_label} μ₀")

    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("μ(E)")
    ax.set_title("μ(E)")
    ax.legend()

    return t.cast("Figure", fig or ax.get_figure())


# ---------------------------------------------------------------------------
# χ(k) — k-space EXAFS
# ---------------------------------------------------------------------------


def plot_chi_k(
    xas_data: XasData,
    *,
    kweight: int = 2,
    ax=None,
    label: str | None = None,
    plot_envelope: bool = False,
) -> Figure:
    """Plot k-weighted χ(k).

    Parameters
    ----------
    xas_data:
        Node containing ``k`` and ``chi_k`` arrays.
    kweight:
        Exponent for k-weighting (0, 1, 2, or 3).
    ax:
        Existing axes; a new figure is created when *None*.
    label:
        Legend label.
    plot_envelope:
        If the node contains ``chi_k_std`` (ensemble average), shade ±1σ.

    Returns:
    -------
    matplotlib.figure.Figure
    """
    plt = _require_matplotlib()

    k = xas_data.get_array("k")
    chi = xas_data.get_array("chi_k")
    kw_chi = k**kweight * chi

    fig = None
    if ax is None:
        fig, ax = plt.subplots()

    _label = label or (xas_data.label or f"pk={xas_data.pk}")
    ax.plot(k, kw_chi, label=_label)

    if plot_envelope and "chi_k_std" in xas_data.get_arraynames():
        std = xas_data.get_array("chi_k_std")
        kw_std = k**kweight * std
        ax.fill_between(k, kw_chi - kw_std, kw_chi + kw_std, alpha=0.25)

    ax.set_xlabel("k (Å⁻¹)")
    ax.set_ylabel(f"k$^{{{kweight}}}$χ(k) (Å$^{{-{kweight}}}$)")
    ax.set_title("EXAFS χ(k)")
    ax.legend()

    return t.cast("Figure", fig or ax.get_figure())


# ---------------------------------------------------------------------------
# χ(R) — R-space EXAFS
# ---------------------------------------------------------------------------


def plot_chi_r(
    source,
    *,
    ft_params: dict | None = None,
    component: str = "mag",
    ax=None,
    label: str | None = None,
    rmax: float | None = None,
) -> Figure:
    """Plot χ(R) from either an FT result node or an XasData node.

    Parameters
    ----------
    source:
        Either:

        * An :class:`~aiida.orm.ArrayData` node already containing ``r``,
          ``chir_mag``, ``chir_re``, ``chir_im`` — i.e. the output of
          :func:`~aiida_feff.calcfunctions.larch.chi_k_to_r`.
        * An :class:`~aiida_feff.data.xasdata.XasData` node.  In this case
          the FT is computed on-the-fly using larch (not provenance-tracked).
          Pass *ft_params* to control the transform.

    ft_params:
        FT parameters forwarded to larch ``xftf`` when *source* is an
        ``XasData``.  Keys: ``kmin``, ``kmax``, ``kweight``, ``dk``, ``rmax``.
        Ignored when *source* is an ``ArrayData`` FT-result node.
    component:
        Which component to plot: ``"mag"`` (default), ``"re"``, or ``"im"``.
    ax:
        Existing axes; a new figure is created when *None*.
    label:
        Legend label.
    rmax:
        Clip the x-axis at this R value.

    Returns:
    -------
    matplotlib.figure.Figure
    """
    plt = _require_matplotlib()
    from aiida.orm import ArrayData

    # -- resolve r / chir arrays --------------------------------------------
    if isinstance(source, XasData):
        r, chir_mag, chir_re, chir_im = _xftf_inline(source, ft_params or {})
    elif isinstance(source, ArrayData):
        r = source.get_array("r")
        chir_mag = source.get_array("chir_mag")
        chir_re = source.get_array("chir_re")
        chir_im = source.get_array("chir_im")
    else:
        raise TypeError(f"Expected XasData or ArrayData, got {type(source)}")

    component_map = {"mag": chir_mag, "re": chir_re, "im": chir_im}
    if component not in component_map:
        raise ValueError(f"component must be one of {list(component_map)}; got {component!r}")
    y = component_map[component]

    # -- plot ----------------------------------------------------------------
    fig = None
    if ax is None:
        fig, ax = plt.subplots()

    _label = label or (getattr(source, "label", None) or f"pk={source.pk}")
    mask = (r <= rmax) if rmax is not None else slice(None)
    ax.plot(r[mask], y[mask], label=_label)

    component_label = {"mag": "|χ(R)|", "re": "Re[χ(R)]", "im": "Im[χ(R)]"}[component]
    ax.set_xlabel("R (Å)")
    ax.set_ylabel(f"{component_label} (Å⁻³)")
    ax.set_title("EXAFS χ(R)")
    ax.legend()

    return t.cast("Figure", fig or ax.get_figure())


# ---------------------------------------------------------------------------
# Internal: on-the-fly FT without provenance tracking
# ---------------------------------------------------------------------------


def _xftf_inline(xas_data: XasData, ft_params: dict):
    """Run larch xftf without the @calcfunction wrapper (no AiiDA tracking)."""
    try:
        import larch
        from larch.xafs import xftf
    except ImportError as exc:
        raise ImportError(
            "larch is required for on-the-fly FT.  Install with: pip install xraylarch"
        ) from exc

    k = xas_data.get_array("k")
    chi = xas_data.get_array("chi_k")

    session = larch.Interpreter()
    grp = larch.Group(k=k, chi=chi)
    xftf(
        grp,
        kmin=ft_params.get("kmin", 3.0),
        kmax=ft_params.get("kmax", 15.0),
        kweight=ft_params.get("kweight", 2),
        dk=ft_params.get("dk", 1.0),
        rmax_out=ft_params.get("rmax", 8.0),
        _larch=session,
    )
    return grp.r, np.abs(grp.chir), grp.chir.real, grp.chir.imag
