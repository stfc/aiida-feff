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
from md_exafs.paths import path_chi

from aiida_feff.data.pathcontributions import (
    PathContributionsData,
    PathResult,
)


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
