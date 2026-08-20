"""Debye-Waller / MSRD functions for MD-EXAFS analysis.

Computes per-path mean-square relative displacements (MSRD, σ²) and
anisotropic displacement parameters (ADP / B-factors) directly from an
AiiDA :class:`~aiida.orm.TrajectoryData` node.

The pure-Python computation functions are adapted from the ``alc-dls-exafs``
package (https://github.com/stfc/alc-dls-exafs), stripped of all file-I/O.

Available functions
-------------------
compute_msrd(trajectory_node, params_dict)
    Plain Python function — returns a plain ``dict``.  Not recorded in the
    AiiDA database.  Use this for interactive exploration of parameters.
store_msrd(trajectory, params)
    ``@calcfunction`` wrapper around ``compute_msrd`` — inputs and output are
    AiiDA nodes, call is recorded in the provenance graph.
compute_adp(trajectory_node, params_dict)
    Plain Python function — returns a plain ``dict``.
store_adp(trajectory, params)
    ``@calcfunction`` wrapper around ``compute_adp``.

Dependencies
------------
All required packages (``numpy``, ``ase``) are part of the core
``aiida-feff`` install.  No extra optional dependency is needed.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np
from aiida.engine import calcfunction
from aiida.orm import ArrayData, Dict
from aiida.orm import TrajectoryData as _TrajectoryData
from ase.geometry import find_mic

logger = logging.getLogger(__name__)

#: Keys :func:`compute_msrd` reads.  ``align`` is deliberately absent: MSRD is
#: computed from raw coordinates because Kabsch alignment rotates the frame
#: away from the cell used for the minimum-image convention.
MSRD_PARAM_KEYS = frozenset(
    {
        "absorber_site",
        "cutoff",
        "tol_dist",
        "tol_angle",
        "cutoff_3body",
        "skip_frames",
        "exclude_hydrogen",
        "allow_unsafe_cutoff",
    }
)

#: Keys :func:`compute_adp` reads.
ADP_PARAM_KEYS = frozenset({"skip_frames", "align"})


def _reject_unknown_params(params: dict, allowed: frozenset[str], func_name: str) -> None:
    """Raise on parameters the function does not read.

    A key that is accepted and ignored still changes the hash of the ``Dict``
    input node, so AiiDA records two provenance-distinct calls that produced
    identical output — while the user believes their setting took effect.
    """
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ValueError(
            f"{func_name} does not use parameter(s) {unknown}. Recognised keys: {sorted(allowed)}"
        )


# ===========================================================================
# Pure computation helpers (no AiiDA, no file I/O)
# Adapted from https://github.com/stfc/alc-dls-exafs
# ===========================================================================


def _unwrap_positions_pbc(
    positions: np.ndarray,
    cells: np.ndarray,
) -> np.ndarray:
    """Unwrap atomic positions across PBC for a continuous trajectory.

    Parameters
    ----------
    positions:
        Shape ``(n_frames, n_atoms, 3)`` — wrapped Cartesian positions.
    cells:
        Shape ``(n_frames, 3, 3)`` — cell matrices per frame.

    Returns:
    -------
    np.ndarray  shape ``(n_frames, n_atoms, 3)``  unwrapped positions.
    """
    n_frames, n_atoms, _ = positions.shape
    unwrapped = np.zeros_like(positions)
    unwrapped[0] = positions[0]

    for i in range(1, n_frames):
        cell = cells[i]
        if np.allclose(cell, 0):
            unwrapped[i] = positions[i]
            continue
        inv_cell = np.linalg.inv(cell)
        frac_current = positions[i] @ inv_cell.T
        frac_prev = unwrapped[i - 1] @ inv_cell.T
        frac_disp = frac_current - frac_prev
        frac_disp -= np.round(frac_disp)
        unwrapped[i] = unwrapped[i - 1] + (frac_disp @ cell)

    return unwrapped


def _kabsch_align(
    positions: np.ndarray,
    reference: np.ndarray | None = None,
) -> np.ndarray:
    """Kabsch-align all frames in *positions* to *reference*.

    Parameters
    ----------
    positions:
        Shape ``(n_frames, n_atoms, 3)``.
    reference:
        Shape ``(n_atoms, 3)``.  Defaults to frame 0.

    Returns:
    -------
    np.ndarray  same shape as *positions*, aligned.
    """
    ref = positions[0] if reference is None else reference
    ref_com = ref.mean(axis=0)
    ref_c = ref - ref_com

    aligned = np.zeros_like(positions)
    for i, pos in enumerate(positions):
        com = pos.mean(axis=0)
        pos_c = pos - com
        H = ref_c.T @ pos_c
        U, _S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        aligned[i] = pos_c @ R + ref_com

    return aligned


def _process_positions(
    positions: np.ndarray,
    cells: np.ndarray,
    *,
    align: bool = True,
) -> np.ndarray:
    """Unwrap PBC and optionally two-pass Kabsch-align trajectory positions."""
    unwrapped = _unwrap_positions_pbc(positions, cells)
    if align:
        rough = _kabsch_align(unwrapped)
        avg = rough.mean(axis=0)
        unwrapped = _kabsch_align(unwrapped, reference=avg)
    return unwrapped


def _compute_adp_impl(
    positions_proc: np.ndarray,
    symbols: list[str],
) -> dict[str, Any]:
    """Compute ADP tensors and B-factors from processed positions.

    ``U_ij`` is the **sample** covariance of the displacements (``ddof=1``),
    matching the σ² convention used for MSRD in this module so the two second
    moments are the same estimator.  For uncorrelated Gaussian displacements
    of width σ this gives ``B → 8π²σ²``.
    """
    n_frames = len(positions_proc)
    min_frames_for_variance = 2
    if n_frames < min_frames_for_variance:
        raise ValueError(f"ADPs need at least 2 frames, got {n_frames}.")
    avg_pos = positions_proc.mean(axis=0)
    disp = positions_proc - avg_pos[np.newaxis]
    u_tensor = np.einsum("fni,fnj->nij", disp, disp) / (n_frames - 1)
    b_factors = 8 * np.pi**2 * np.trace(u_tensor, axis1=1, axis2=2) / 3
    return {
        "avg_positions": avg_pos,
        "u_tensor": u_tensor,
        "b_factors": b_factors,
        "symbols": symbols,
    }


def _max_safe_mic_cutoff(cell: np.ndarray) -> float | None:
    """Largest sphere radius that fits inside a parallelepiped cell.

    For the MIC to be unambiguous the cutoff must be less than half the
    smallest perpendicular distance between opposite faces of the cell.

    Returns ``None`` when the cell has zero volume.
    """
    cell = np.asarray(cell)
    if cell.shape != (3, 3):
        return None
    volume = float(np.abs(np.linalg.det(cell)))
    if volume <= 0.0:
        return None
    area_ab = float(np.linalg.norm(np.cross(cell[0], cell[1])))
    area_bc = float(np.linalg.norm(np.cross(cell[1], cell[2])))
    area_ca = float(np.linalg.norm(np.cross(cell[2], cell[0])))
    if 0.0 in (area_ab, area_bc, area_ca):
        return None
    return min(volume / area_ab, volume / area_bc, volume / area_ca) / 2.0


def _check_mic_cutoffs(
    cell: np.ndarray,
    cutoff: float,
    cutoff_3body: float | None,
    allow_unsafe: bool,
) -> None:
    """Raise (or warn) when a neighbour cutoff outruns the minimum-image convention.

    ``find_mic`` returns the shortest image displacement.  Once the cutoff
    exceeds the largest sphere that fits inside the cell, that shortest image
    can be a different atom than the one intended, so distances are biased
    downward and σ² collapses.  Nothing downstream can detect this, which is
    why it is an error rather than a warning.
    """
    max_safe = _max_safe_mic_cutoff(cell)
    if max_safe is None:
        return
    for name, value in (("cutoff", cutoff), ("cutoff_3body", cutoff_3body)):
        if value is None or value <= 0 or value <= max_safe:
            continue
        message = (
            f"{name}={value:.3f} Å exceeds this cell's maximum safe "
            f"minimum-image cutoff ({max_safe:.3f} Å): distances would be biased "
            f"towards periodic images. Build a supercell, or reduce {name} to "
            f"<= {max_safe:.3f} Å."
        )
        if allow_unsafe:
            logger.warning("%s (allow_unsafe_cutoff=True, continuing)", message)
        else:
            raise ValueError(message + " Pass allow_unsafe_cutoff=True to override.")


def _cluster_by_tolerance(items: list[dict], key: str, tol: float) -> list[list[dict]]:
    """Sort *items* by *key* and greedily cluster them, bounding cluster width by *tol*.

    Complete linkage: an item joins the current cluster only if the resulting
    cluster still spans no more than *tol*.  Comparing against the cluster's
    running *mean* instead lets a dense ladder of paths chain into one cluster
    far wider than the tolerance the caller asked for, which then pools
    genuinely different shells into a single σ².

    An empty *items* yields no clusters rather than raising.
    """
    if not items:
        return []
    items = sorted(items, key=lambda x: x[key])
    clusters: list[list[dict]] = [[items[0]]]
    for item in items[1:]:
        if item[key] - clusters[-1][0][key] <= tol:
            clusters[-1].append(item)
        else:
            clusters.append([item])
    return clusters


def _calculate_grouped_msrd_impl(
    positions: np.ndarray,
    symbols: list[str],
    cell: np.ndarray,
    pbc: list[bool],
    central_indices: list[int],
    *,
    cutoff: float = 3.5,
    tol_dist: float = 0.1,
    tol_angle: float = 5.0,
    cutoff_3body: float | None = None,
    exclude_hydrogen: bool = False,
    allow_unsafe_cutoff: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute grouped MSRD σ² for 2-body and 3-body scattering paths.

    Parameters
    ----------
    positions:
        Shape ``(n_frames, n_atoms, 3)`` — raw (non-aligned) positions.
        Must NOT be Kabsch-aligned: alignment rotates the frame, making
        coordinates inconsistent with the static cell matrix used by find_mic.
    symbols:
        Chemical symbol per atom.
    cell / pbc:
        Reference cell matrix and PBC flags.
    central_indices:
        Zero-based absorber atom indices.
    cutoff:
        Neighbour cutoff for 2-body paths (Å).
    tol_dist:
        Distance tolerance for grouping paths into shells (Å).
    tol_angle:
        Angle tolerance for 3-body path grouping (degrees).  The angle is the
        **included angle at the first scatterer**, i.e. 180° − FEFF's
        scattering angle.
    cutoff_3body:
        Leg cutoff for 3-body paths.  ``None`` / ``0`` disables 3-body.
    exclude_hydrogen:
        When True, H atoms are excluded from the neighbour search.
    allow_unsafe_cutoff:
        Proceed even when a cutoff exceeds the cell's inscribed-sphere radius.
        Off by default: past that radius the minimum-image convention picks
        the nearest periodic image, which biases distances downward and σ²
        by tens of percent — silently, and in the direction that looks like
        better-ordered material.
    """
    _check_mic_cutoffs(cell, cutoff, cutoff_3body, allow_unsafe_cutoff)

    central_element = symbols[central_indices[0]]

    # Reference distances from frame 0
    ref_pos = positions[0]

    eligible = {i for i, sym in enumerate(symbols) if (not exclude_hydrogen or sym != "H")}

    pair_list: list[dict[str, Any]] = []
    triplet_list: list[dict[str, Any]] = []

    for c_idx in central_indices:
        # Find neighbours within cutoff using reference frame
        all_cand = [i for i in range(len(symbols)) if i != c_idx and i in eligible]
        raw_disp = ref_pos[all_cand] - ref_pos[c_idx]
        _, ref_dists = find_mic(raw_disp, cell, pbc)  # type: ignore[arg-type]
        neighbors = [all_cand[i] for i, d in enumerate(ref_dists) if d < cutoff]

        # Pre-compute MIC vectors over all frames for each neighbour
        mic_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for n_idx in neighbors:
            raw = positions[:, n_idx, :] - positions[:, c_idx, :]
            mic_v, dists = find_mic(raw, cell, pbc)  # type: ignore[arg-type]
            mic_cache[n_idx] = (mic_v, dists)

        # --- 2-body ---
        for n_idx in neighbors:
            _, dists = mic_cache[n_idx]
            pair_list.append(
                {
                    "element": symbols[n_idx],
                    "dists": dists,
                    "mean_d": float(dists.mean()),
                    "label": f"{central_element}-{symbols[n_idx]}",
                    "c_idx": c_idx,
                    "n_idx": n_idx,
                }
            )

        # --- 3-body ---
        if not cutoff_3body:
            continue
        nb3 = (
            [n for n in neighbors if mic_cache[n][1].mean() <= cutoff_3body]
            if cutoff_3body < cutoff
            else neighbors
        )

        for i in range(len(nb3)):
            for j in range(i + 1, len(nb3)):
                n1, n2 = nb3[i], nb3[j]
                v01, d01 = mic_cache[n1]
                v02, d02 = mic_cache[n2]
                # The scattering path is absorber -> n1 -> n2 -> absorber for
                # the two neighbour *images* already chosen relative to the
                # absorber, so the n1->n2 leg is fixed by those choices.
                # Minimum-imaging it independently can pick a third image that
                # belongs to no triangle, leaving L longer or shorter than any
                # real path and the vertex angle inconsistent with it.
                v12 = v02 - v01
                d12 = np.linalg.norm(v12, axis=1)
                L = d01 + d12 + d02

                # Included angle at the first scatterer: 180° - FEFF's
                # scattering angle.
                v1 = -v01
                v2 = v12
                v1u = v1 / np.maximum(np.linalg.norm(v1, axis=1, keepdims=True), 1e-10)
                v2u = v2 / np.maximum(np.linalg.norm(v2, axis=1, keepdims=True), 1e-10)
                cos_t = np.clip((v1u * v2u).sum(axis=1), -1, 1)
                angles_deg = np.degrees(np.arccos(cos_t))

                elem_pair = tuple(sorted([symbols[n1], symbols[n2]]))
                triplet_list.append(
                    {
                        "elements": elem_pair,
                        "reff_series": L / 2.0,
                        "mean_L": float((L / 2.0).mean()),
                        "angle": float(angles_deg.mean()),
                        "c_idx": c_idx,
                        "n1_idx": n1,
                        "n2_idx": n2,
                    }
                )

    # --- cluster 2-body paths by element + distance shell ---
    res_2b: list[dict[str, Any]] = []
    by_elem: defaultdict[str, list] = defaultdict(list)
    for p in pair_list:
        by_elem[p["element"]].append(p)

    for _elem, paths in by_elem.items():
        for cl in _cluster_by_tolerance(paths, "mean_d", tol_dist):
            all_d = np.concatenate([p["dists"] for p in cl])
            res_2b.append(
                {
                    "type": cl[0]["label"],
                    "reff": float(all_d.mean()),
                    "sigma2": float(np.var(all_d, ddof=1)),
                    "count": len(cl),
                    "atom_indices": [(p["c_idx"], p["n_idx"]) for p in cl],
                }
            )

    # --- cluster 3-body paths ---
    res_3b: list[dict[str, Any]] = []
    by_pair: defaultdict[tuple, list] = defaultdict(list)
    for p in triplet_list:
        by_pair[p["elements"]].append(p)

    for elem_pair, paths in by_pair.items():
        for ang_cl in _cluster_by_tolerance(paths, "angle", tol_angle):
            for cl in _cluster_by_tolerance(ang_cl, "mean_L", tol_dist):
                all_r = np.concatenate([p["reff_series"] for p in cl])
                res_3b.append(
                    {
                        "type": f"{central_element}-{elem_pair[0]}-{elem_pair[1]}",
                        "reff": float(all_r.mean()),
                        "sigma2": float(np.var(all_r, ddof=1)),
                        "angle": float(np.mean([p["angle"] for p in cl])),
                        "count": len(cl),
                        "atom_indices": [(p["c_idx"], p["n1_idx"], p["n2_idx"]) for p in cl],
                    }
                )

    return (sorted(res_2b, key=lambda x: x["reff"]), sorted(res_3b, key=lambda x: x["reff"]))


def _parse_site_spec(spec: str, symbols: list[str]) -> list[int]:
    """Parse a site specification into zero-based atom indices.

    Formats::

        "Fe"      → all Fe atoms
        "Fe.1"    → first Fe (1-based within element)
        "Fe.1-3"  → first three Fe atoms
        "11"      → atom 11 (1-based, absolute)
        "11-20"   → atoms 11–20 inclusive

    Raises:
    ------
    ValueError  if the specification matches nothing.
    """
    spec = spec.strip()
    if spec.replace("-", "").isdigit():
        if "-" in spec:
            lo, hi = spec.split("-")
            return list(range(int(lo) - 1, int(hi)))
        return [int(spec) - 1]

    if "." in spec:
        elem, idx_part = spec.split(".", 1)
        elem_idx = [i for i, s in enumerate(symbols) if s == elem]
        if not elem_idx:
            raise ValueError(f"No atoms of element '{elem}' found")
        if "-" in idx_part:
            lo, hi = idx_part.split("-")
            return elem_idx[int(lo) - 1 : int(hi)]
        return [elem_idx[int(idx_part) - 1]]

    matching = [i for i, s in enumerate(symbols) if s == spec]
    if not matching:
        raise ValueError(f"No atoms of element '{spec}' found")
    return matching


# ===========================================================================
# Public API
# ===========================================================================


def compute_msrd(
    trajectory: _TrajectoryData,
    params: dict,
) -> dict:
    """Compute per-path MSRD (σ²) from an MD trajectory.

    This is a plain Python function — calls are **not** recorded in the AiiDA
    database.  Use it for interactive exploration of cutoff / tolerance
    parameters.  Call :func:`store_msrd` when you want provenance.

    Parameters
    ----------
    trajectory:
        AiiDA :class:`~aiida.orm.TrajectoryData` node.  Must contain
        ``positions`` and ``cells`` arrays and a ``symbols`` attribute.
    params:
        Plain Python dict with keys:

        ``absorber_site`` : str, required
            Site specification for the absorbing atom(s).
            Examples: ``"Fe"``, ``"Fe.1"``, ``"Fe.1-3"``, ``"11"``.
        ``cutoff`` : float, default 3.5
            Neighbour cutoff radius in Å.
        ``tol_dist`` : float, default 0.1
            Distance tolerance for grouping paths into shells (Å).
        ``tol_angle`` : float, default 5.0
            Angle tolerance for 3-body path grouping (degrees).
        ``cutoff_3body`` : float | None, default None
            Leg cutoff for 3-body paths.  ``None`` / ``0`` → 2-body only.
        ``skip_frames`` : int, default 0
            Discard the first N frames (equilibration).
        ``exclude_hydrogen`` : bool, default False
            Exclude H atoms from the neighbour search.
        ``allow_unsafe_cutoff`` : bool, default False
            Permit a cutoff larger than the cell's inscribed-sphere radius.
            Leave this off unless you know the bias is acceptable.

    Conventions
    -----------
    * σ² is a **variance** in Å², pooled over frames *and* over the paths
      grouped into a shell.  It therefore contains both the thermal and the
      static (configurational) spread; with a loose ``tol_dist`` the static
      term can dominate.
    * σ² uses the sample estimator (``ddof=1``), matching ``compute_adp``.
    * For 3-body paths ``reff`` is half the total path length, per FEFF, and
      ``angle`` is the included angle at the first scatterer — 180° minus
      FEFF's scattering angle.  The scatterer–scatterer leg is taken as the
      difference of the two absorber-relative minimum-image vectors, so the
      triangle always closes.
    * The minimum-image convention uses the **frame-0 cell**.  For an NPT
      trajectory with a drifting cell this is an approximation.

    Returns:
    -------
    dict
        Flat dictionary.  Each key is a path label such as
        ``"Fe-Fe_2p48_2body"``; each value is a sub-dict with::

            {
                "reff":    float,   # mean effective path length (Å)
                "sigma2":  float,   # MSRD σ² (Å², variance, ddof=1)
                "count":   int,     # number of paths averaged into this shell
                "n_body":  int,     # 2 or 3
                "angle":   float | None,  # mean included angle, 3-body (°)
            }
    """
    _reject_unknown_params(params, MSRD_PARAM_KEYS, "compute_msrd")

    absorber_site = params.get("absorber_site")
    if not absorber_site:
        raise ValueError("params must include 'absorber_site'")

    cutoff = float(params.get("cutoff", 3.5))
    tol_dist = float(params.get("tol_dist", 0.1))
    tol_angle = float(params.get("tol_angle", 5.0))
    cutoff_3body = params.get("cutoff_3body")
    skip = int(params.get("skip_frames", 0))
    excl_h = bool(params.get("exclude_hydrogen", False))
    allow_unsafe = bool(params.get("allow_unsafe_cutoff", False))

    positions = trajectory.get_array("positions")[skip:]
    symbols: list[str] = trajectory.base.attributes.get("symbols")
    try:
        cells = trajectory.get_array("cells")[skip:]
    except KeyError:
        # A zero cell makes ASE's find_mic treat the system as non-periodic,
        # which is the right answer for a cluster — but say so, because the
        # safe-cutoff check cannot run without a cell.
        logger.warning(
            "TrajectoryData has no 'cells' array; treating the system as "
            "non-periodic and skipping the minimum-image cutoff check."
        )
        cells = np.zeros((len(positions), 3, 3))

    # TrajectoryData carries no pbc flags, so full periodicity is assumed.
    # For a slab or a molecule in a box this creates spurious images along the
    # vacuum direction; pad such systems or pass a non-periodic (zero) cell.
    pbc = [True, True, True]
    ref_cell = cells[0]
    if len(cells) > 1 and not np.allclose(cells, ref_cell):
        logger.warning(
            "Cell varies across frames (NPT?); the minimum-image convention "
            "uses frame 0 for every frame."
        )

    central_indices = _parse_site_spec(absorber_site, symbols)
    logger.info(
        "compute_msrd: absorber='%s' → %d site(s), %d frames",
        absorber_site,
        len(central_indices),
        len(positions),
    )

    # Raw positions passed directly — Kabsch alignment would rotate the frame,
    # making coords inconsistent with the static cell matrix used by find_mic
    # (wrong MIC for non-orthogonal cells). _process_positions is for compute_adp only.
    res_2b, res_3b = _calculate_grouped_msrd_impl(
        positions,
        symbols,
        ref_cell,
        pbc,
        central_indices,
        cutoff=cutoff,
        tol_dist=tol_dist,
        tol_angle=tol_angle,
        cutoff_3body=cutoff_3body,
        exclude_hydrogen=excl_h,
        allow_unsafe_cutoff=allow_unsafe,
    )

    output: dict[str, Any] = {}
    for path in res_2b:
        reff_str = f"{path['reff']:.2f}".replace(".", "p")
        key = f"{path['type']}_{reff_str}_2body"
        output[key] = {
            "reff": path["reff"],
            "sigma2": path["sigma2"],
            "count": path["count"],
            "n_body": 2,
            "angle": None,
        }
    for path in res_3b:
        reff_str = f"{path['reff']:.2f}".replace(".", "p")
        key = f"{path['type']}_{reff_str}_3body"
        output[key] = {
            "reff": path["reff"],
            "sigma2": path["sigma2"],
            "count": path["count"],
            "n_body": 3,
            "angle": path["angle"],
        }

    return output


@calcfunction
def store_msrd(trajectory: _TrajectoryData, params: Dict) -> Dict:
    """Provenance-tracked wrapper around :func:`compute_msrd`.

    Accepts AiiDA nodes; records inputs, output and the call itself in the
    database.  Use :func:`compute_msrd` directly when exploring parameters
    interactively.
    """
    return Dict(dict=compute_msrd(trajectory, params.get_dict()))


def compute_adp(
    trajectory: _TrajectoryData,
    params: dict,
) -> dict:
    """Compute per-atom ADP tensors and B-factors from an MD trajectory.

    Plain Python function — not recorded in the AiiDA database.
    Call :func:`store_adp` when you want provenance.

    Parameters
    ----------
    trajectory:
        AiiDA :class:`~aiida.orm.TrajectoryData` node.
    params:
        Plain Python dict with optional keys:

        ``skip_frames`` : int, default 0
        ``align`` : bool, default True

    Returns:
    -------
    dict
        Keys: ``b_factors`` (ndarray, shape ``(n_atoms,)``),
        ``u_tensor`` (ndarray, shape ``(n_atoms, 3, 3)``),
        ``avg_positions`` (ndarray, shape ``(n_atoms, 3)``),
        ``symbols`` (list[str]).
    """
    _reject_unknown_params(params, ADP_PARAM_KEYS, "compute_adp")

    skip = int(params.get("skip_frames", 0))
    align = bool(params.get("align", True))

    positions = trajectory.get_array("positions")[skip:]
    symbols: list[str] = trajectory.base.attributes.get("symbols")
    try:
        cells = trajectory.get_array("cells")[skip:]
    except KeyError:
        cells = np.zeros((len(positions), 3, 3))

    positions_proc = _process_positions(positions, cells, align=align)
    adp = _compute_adp_impl(positions_proc, symbols)
    return adp


@calcfunction
def store_adp(trajectory: _TrajectoryData, params: Dict) -> ArrayData:
    """Provenance-tracked wrapper around :func:`compute_adp`.

    Accepts AiiDA nodes; records inputs, output and the call itself in the
    database.  Use :func:`compute_adp` directly when exploring parameters
    interactively.
    """
    adp = compute_adp(trajectory, params.get_dict())
    out = ArrayData()
    out.set_array("b_factors", adp["b_factors"])
    out.set_array("u_tensor", adp["u_tensor"])
    out.set_array("avg_positions", adp["avg_positions"])
    out.base.attributes.set("symbols", adp["symbols"])
    return out
