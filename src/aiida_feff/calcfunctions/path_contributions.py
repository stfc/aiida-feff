"""Calcfunctions for per-path FEFF contribution data.

AiiDA calcfunction
------------------
merge_path_contributions(r_bin, **kwargs)
    Concatenate N per-snapshot
    :class:`~aiida_feff.data.pathcontributions.PathContributionsData` nodes
    into one ensemble-wide node, tagging every row with a ``path_key`` so
    equivalent paths can be grouped across frames afterwards.  Rows are *not*
    averaged; :func:`aiida_feff.calcfunctions.exafs.group_paths_by_key` does
    the grouping on read.
"""

from __future__ import annotations

import io
import math

import numpy as np
from aiida.engine import calcfunction

from aiida_feff.data.pathcontributions import (
    FEFF_DATA_COLS,
    H5_VERSION_MERGED,
    PathContributionsData,
    _require_h5py,
)

# ---------------------------------------------------------------------------
# Path grouping key
# ---------------------------------------------------------------------------

#: Relative tolerance applied when locating a bin edge.  ``r_eff`` that lands
#: exactly on a multiple of ``r_bin`` is otherwise at the mercy of floating
#: point: ``3.0 // 0.15`` is 20 while ``2.9999999999 // 0.15`` is 19, which
#: puts two indistinguishable paths in bins 0.15 Å apart.
_BIN_EDGE_TOL = 1e-9


def make_path_key(scatterer: str, nlegs: int, r_eff: float, r_bin: float) -> str:
    """Create a stable string key grouping FEFF paths across MD frames.

    Paths with the same scatterer chain, number of legs, and ``r_eff`` in the
    same ``r_bin``-wide window are considered equivalent.  Bin *k* covers
    ``[k·r_bin, (k+1)·r_bin)`` and is labelled by its centre.

    Examples (``r_bin=0.15``)::

        make_path_key("Fe", 2, 2.48, 0.15)     -> "SS_Fe_2.475"
        make_path_key("Fe-Fe", 3, 6.12, 0.15)  -> "MS3_Fe-Fe_6.075"
    """
    if r_bin <= 0:
        raise ValueError(f"r_bin must be > 0, got {r_bin}")
    two_legs = 2
    bin_index = math.floor(r_eff / r_bin + _BIN_EDGE_TOL)
    bin_centre = round((bin_index + 0.5) * r_bin, 4)
    prefix = "SS" if nlegs == two_legs else f"MS{nlegs}"
    return f"{prefix}_{scatterer}_{bin_centre}"


#: Backwards-compatible alias for the pre-rename private name.
_make_path_key = make_path_key


# ---------------------------------------------------------------------------
# AiiDA calcfunction: merge N per-snapshot nodes
# ---------------------------------------------------------------------------


@calcfunction
def merge_path_contributions(r_bin, **kwargs) -> PathContributionsData:
    """Merge per-snapshot PathContributionsData nodes into one ensemble node.

    Keyword arguments (beyond ``r_bin``) must be
    :class:`~aiida_feff.data.pathcontributions.PathContributionsData` nodes.
    Keys are used only for ordering (e.g. ``snap_0000``, ``snap_0001``).

    Every row is tagged with :func:`make_path_key` for the given ``r_bin``
    (in Å), so paths in the same bin across different frames share a key.
    Rows are concatenated, not averaged — averaging would have to choose a
    weighting over degeneracies that differ frame to frame, which is the
    caller's decision.

    The merged node uses schema ``format_version = 2``: the per-calc columnar
    layout plus per-row ``frame_idx``, ``site_idx`` and ``path_key``.

    Parameters
    ----------
    r_bin : orm.Float
        Bin width (Å) for grouping paths by effective path length.
        Default in the wrapper is 0.15 Å.
    **kwargs :
        Mapping of label → PathContributionsData (one per snapshot).

    Returns:
    -------
    PathContributionsData
        Merged ensemble node.
    """
    h5py = _require_h5py()
    r_bin_val = float(r_bin)

    # Collect all path rows across all nodes.
    # Each row: (frame_idx, site_idx, path_key, r_eff, nlegs, degeneracy,
    #            scatterer, cw_ratio, k[M], feff_data[M,6])
    rows: list[dict] = []
    k_grid: np.ndarray | None = None
    absorber_elements: set[str] = set()

    for key in sorted(kwargs.keys()):
        node = kwargs[key]
        absorber_elements.add(node.absorber_element)
        for pr in node.iter_paths():
            if k_grid is None:
                k_grid = pr.k
            elif len(pr.k) != len(k_grid) or not np.allclose(pr.k, k_grid):
                # Interpolating here would silently resample every column;
                # a k-grid mismatch means the inputs came from different FEFF
                # settings and should not be merged without the caller saying so.
                raise ValueError(
                    f"Path in node {key!r} has a k grid of length {len(pr.k)} that "
                    f"differs from the reference grid (length {len(k_grid)}). "
                    "All merged nodes must share one native FEFF k grid."
                )
            rows.append(
                {
                    "frame_idx": pr.frame_idx,
                    "site_idx": pr.site_idx,
                    "path_key": make_path_key(pr.scatterer, pr.nlegs, pr.r_eff, r_bin_val),
                    "r_eff": pr.r_eff,
                    "nlegs": pr.nlegs,
                    "degeneracy": pr.degeneracy,
                    "scatterer": pr.scatterer,
                    "cw_ratio": pr.cw_ratio,
                    "sig2": pr.sig2,
                    "feff_data": pr.feff_data,
                }
            )

    if not rows or k_grid is None:
        raise ValueError("No path data found in any of the input nodes.")

    n = len(rows)
    m_k = len(k_grid)
    n_cols = len(FEFF_DATA_COLS)

    feff_data_arr = np.zeros((n, m_k, n_cols), dtype=np.float64)
    scalar_cols = ("r_eff", "degeneracy", "cw_ratio", "sig2")
    scalars = {name: np.zeros(n, dtype=np.float64) for name in scalar_cols}
    nlegs_arr = np.zeros(n, dtype=np.int32)
    frame_idx_arr = np.zeros(n, dtype=np.int32)
    site_idx_arr = np.zeros(n, dtype=np.int32)
    scatterer_list: list[str] = []
    path_key_list: list[str] = []

    for i, row in enumerate(rows):
        feff_data_arr[i] = row["feff_data"]
        for name in scalar_cols:
            scalars[name][i] = row[name]
        nlegs_arr[i] = row["nlegs"]
        frame_idx_arr[i] = row["frame_idx"]
        site_idx_arr[i] = row["site_idx"]
        scatterer_list.append(row["scatterer"])
        path_key_list.append(row["path_key"])

    _COMPRESS = {"compression": "gzip", "compression_opts": 6}
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["format_version"] = H5_VERSION_MERGED
        # Distinct MD frames, not the number of input nodes: with several
        # absorber sites per frame there are n_frames x n_sites inputs.
        meta.attrs["n_frames"] = int(len(np.unique(frame_idx_arr)))
        meta.attrs["n_sites"] = int(len(np.unique(site_idx_arr)))
        meta.attrs["n_inputs"] = int(len(kwargs))
        meta.attrs["r_bin"] = r_bin_val
        meta.attrs["absorber_element"] = ",".join(sorted(e for e in absorber_elements if e))

        pg = f.create_group("paths")
        pg.create_dataset("k_grid_params", data=k_grid, **_COMPRESS)
        pg.create_dataset("feff_data", data=feff_data_arr, **_COMPRESS)
        for name in scalar_cols:
            pg.create_dataset(name, data=scalars[name], **_COMPRESS)
        pg.create_dataset("nlegs", data=nlegs_arr, **_COMPRESS)
        pg.create_dataset("frame_idx", data=frame_idx_arr, **_COMPRESS)
        pg.create_dataset("site_idx", data=site_idx_arr, **_COMPRESS)
        dt_str = h5py.string_dtype(encoding="utf-8")
        pg.create_dataset("scatterer", data=np.array(scatterer_list, dtype=dt_str), **_COMPRESS)
        pg.create_dataset("path_key", data=np.array(path_key_list, dtype=dt_str), **_COMPRESS)

    buf.seek(0)
    return PathContributionsData.from_hdf5_bytes(buf.getvalue())
