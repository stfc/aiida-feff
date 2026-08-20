r"""Reconstruct χ(k) from FEFF per-path scattering factors.

:class:`~aiida_feff.data.pathcontributions.PathContributionsData` stores the
six raw columns of every ``feff????.dat`` file that survived amplitude
filtering.  Turning those columns back into χ(k) — applying S₀², a
Debye–Waller factor and an E₀ shift — is the whole reason the store exists,
and it is what this module does.

The implementation follows the EXAFS equation exactly as larch's
``FeffPathGroup._calc_chi`` evaluates it, so results can be checked against
larch path-by-path (see ``tests/test_exafs.py``):

.. math::

    \chi(k) = \Im \left[
        \frac{N\,S_0^2\,A(q)}{q\,(R_{\mathrm{eff}} + \Delta R)^2}
        \; e^{-2 R_{\mathrm{eff}} \Im p \; - \; 2 p^2 \sigma^2}
        \; e^{\,i\,(2 q R_{\mathrm{eff}} + \phi(q)
              + 2 p (\Delta R - 2\sigma^2 / R_{\mathrm{eff}}))}
    \right]

with the complex momentum :math:`p = \mathrm{rep} + i/\lambda` taken from the
``rep`` and ``lam`` columns.  Using ``rep`` rather than the nominal ``k`` grid
matters: ``rep`` carries the inner-potential correction, and substituting
``k`` for it introduces a systematic phase error indistinguishable from an
uncorrected E₀ shift.

``A = mag_feff · red_fact`` and ``φ = real_phc + pha_feff`` reproduce larch's
``amp`` and ``pha``.  ``red_fact`` does **not** contain S₀²; that is applied
separately here.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from aiida_feff.constants import ETOK
from aiida_feff.data.pathcontributions import (
    FEFF_DATA_COLS,
    PathContributionsData,
    PathResult,
)

_COL = {name: i for i, name in enumerate(FEFF_DATA_COLS)}


def _resample(k_native: np.ndarray, values: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Interpolate a ``feff????.dat`` column onto the shifted wavenumber grid.

    Cubic, matching larch's ``UnivariateSpline(..., s=0)``: FEFF's native grid
    has 0.1 Å⁻¹ spacing, coarse enough that linear interpolation visibly
    distorts χ(k) near its zero crossings.  Falls back to linear when there
    are too few points for a cubic spline.
    """
    min_points_for_cubic = 4
    if len(k_native) < min_points_for_cubic:
        return np.asarray(np.interp(q, k_native, values), dtype=float)
    from scipy.interpolate import UnivariateSpline

    return np.asarray(UnivariateSpline(k_native, values, s=0)(q), dtype=float)


def path_chi(
    k_native: np.ndarray,
    feff_data: np.ndarray,
    r_eff: float,
    degeneracy: float,
    k_out: np.ndarray,
    *,
    sigma2: float = 0.0,
    s02: float = 1.0,
    e0_shift: float = 0.0,
    deltar: float = 0.0,
) -> np.ndarray:
    """Evaluate one scattering path's χ(k) on *k_out*.

    Args:
        k_native: The path's own k grid (``PathResult.k``), Å⁻¹.
        feff_data: ``(M, 6)`` array with columns :data:`FEFF_DATA_COLS`.
        r_eff: Effective path length in Å — half the total path length for
            multiple-scattering paths, per FEFF's convention.
        degeneracy: Path degeneracy N as reported by FEFF.
        k_out: Output wavenumber grid, Å⁻¹.
        sigma2: Debye–Waller factor σ² in Å² (a variance, not σ).
        s02: Amplitude reduction factor S₀².
        e0_shift: ΔE₀ in eV.  Positive shifts theory to lower k.
        deltar: ΔR in Å, added to ``r_eff``.

    Returns:
        χ(k) on ``k_out``.  Points below the shifted threshold are zero.
    """
    k_out = np.asarray(k_out, dtype=float)
    feff_data = np.asarray(feff_data, dtype=float)
    if feff_data.shape[1] != len(FEFF_DATA_COLS):
        raise ValueError(
            f"feff_data must have {len(FEFF_DATA_COLS)} columns "
            f"{FEFF_DATA_COLS}, got shape {feff_data.shape}."
        )
    if r_eff <= 0:
        raise ValueError(f"r_eff must be > 0, got {r_eff}")

    # E0-shifted wavenumber q.  Below the shifted threshold the energy is
    # negative; larch keeps the sign, which makes the transform continuous.
    energy = k_out**2 - float(e0_shift) * ETOK
    q = np.sign(energy) * np.sqrt(np.abs(energy))

    k_native = np.asarray(k_native, dtype=float)
    amp = _resample(k_native, feff_data[:, _COL["mag_feff"]] * feff_data[:, _COL["red_fact"]], q)
    pha = _resample(k_native, feff_data[:, _COL["real_phc"]] + feff_data[:, _COL["pha_feff"]], q)
    rep = _resample(k_native, feff_data[:, _COL["rep"]], q)
    lam = _resample(k_native, feff_data[:, _COL["lam"]], q)

    reff = float(r_eff)
    p_sq = (rep + 1j / lam) ** 2
    p = np.sqrt(p_sq)

    cchi = np.exp(
        -2 * reff * p.imag
        - 2 * p_sq * sigma2
        + 1j * (2 * q * reff + pha + 2 * p * (deltar - 2 * sigma2 / reff))
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        cchi = degeneracy * float(s02) * amp * cchi / (q * (reff + deltar) ** 2)

    chi = np.asarray(cchi.imag, dtype=float)
    chi[~np.isfinite(chi)] = 0.0
    # At k = 0 the 1/q prefactor is singular, and once E0 is shifted the
    # lookup there also falls outside the tabulated k range.  larch replaces
    # the point by linear extrapolation from its neighbours; matching that
    # keeps the two implementations comparable point for point.
    min_points_for_extrapolation = 3
    if len(chi) >= min_points_for_extrapolation and k_out[0] == 0.0:
        chi[0] = 2 * chi[1] - chi[2]
    return chi


def path_result_chi(path: PathResult, k_out: np.ndarray, **kwargs) -> np.ndarray:
    """Evaluate :func:`path_chi` for a :class:`PathResult` row."""
    return path_chi(
        path.k,
        path.feff_data,
        path.r_eff,
        path.degeneracy,
        k_out,
        **kwargs,
    )


def total_chi(
    node: PathContributionsData,
    k_out: np.ndarray,
    *,
    sigma2: float | dict[str, float] = 0.0,
    s02: float = 1.0,
    e0_shift: float = 0.0,
    frame_idx: int | None = None,
    site_idx: int | None = None,
) -> np.ndarray:
    """Sum χ(k) over every path stored in *node*.

    Args:
        node: Per-calculation or merged path store.
        k_out: Output wavenumber grid, Å⁻¹.
        sigma2: Either one σ² applied to every path, or a mapping from
            ``scatterer`` to σ² (Å²) — e.g. the output of
            :func:`~aiida_feff.calcfunctions.debye_waller.compute_msrd`
            re-keyed by path type.
        s02: Amplitude reduction factor.
        e0_shift: ΔE₀ in eV.
        frame_idx: Restrict to one MD frame (merged nodes only).
        site_idx: Restrict to one absorber site (merged nodes only).

    Returns:
        Summed χ(k).  When a merged node spanning several frames is passed
        without a ``frame_idx`` filter the result is the **sum**, not the
        ensemble average; divide by the number of frames yourself, or filter.
    """
    k_out = np.asarray(k_out, dtype=float)
    total = np.zeros_like(k_out)
    for path in node.iter_paths():
        if frame_idx is not None and path.frame_idx != frame_idx:
            continue
        if site_idx is not None and path.site_idx != site_idx:
            continue
        s2 = sigma2.get(path.scatterer, 0.0) if isinstance(sigma2, dict) else float(sigma2)
        total += path_result_chi(path, k_out, sigma2=s2, s02=s02, e0_shift=e0_shift)
    return total


def group_paths_by_key(node: PathContributionsData, r_bin: float = 0.15) -> dict[str, list]:
    """Group a node's paths by :func:`make_path_key`.

    Returns a mapping ``path_key -> list[PathResult]``, using the same key
    function as :func:`~aiida_feff.calcfunctions.path_contributions.merge_path_contributions`
    so that grouping is consistent wherever it is done.
    """
    from aiida_feff.calcfunctions.path_contributions import make_path_key

    groups: dict[str, list] = defaultdict(list)
    for path in node.iter_paths():
        groups[make_path_key(path.scatterer, path.nlegs, path.r_eff, r_bin)].append(path)
    return dict(groups)
