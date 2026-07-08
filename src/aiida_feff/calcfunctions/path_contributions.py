"""Calcfunctions for per-path FEFF contribution data.

AiiDA calcfunction
------------------
merge_path_contributions(r_bin, **kwargs)
    Merge N per-snapshot :class:`~aiida_feff.data.pathcontributions.PathContributionsData`
    nodes (keyed by ``snap_NNNN``) into one ensemble-wide node, grouping
    paths by scatterer type and effective path length bin.
"""

from __future__ import annotations

import io

import numpy as np
from aiida.engine import calcfunction

from aiida_feff.data.pathcontributions import (
    _H5_KEY,
    _H5_VERSION,
    PathContributionsData,
    _require_h5py,
)

# ---------------------------------------------------------------------------
# Path grouping key (mirrors larch_cli_wrapper.exafs_data.make_path_key)
# ---------------------------------------------------------------------------


def _make_path_key(scatterer: str, nlegs: int, r_eff: float, r_bin: float) -> str:
    """Create a stable string key grouping FEFF paths across MD frames.

    Paths with the same scatterer chain, number of legs, and r_eff within
    the same r_bin-width window are considered equivalent.

    Examples (r_bin=0.15)::

        _make_path_key("Fe", 2, 2.48)  -> "SS_Fe_2.55"
        _make_path_key("Fe-Fe", 3, 6.12) -> "MS3_Fe-Fe_6.15"
    """
    two_legs = 2
    bin_centre = round((r_eff // r_bin + 0.5) * r_bin, 4)
    prefix = "SS" if nlegs == two_legs else f"MS{nlegs}"
    return f"{prefix}_{scatterer}_{bin_centre}"


# ---------------------------------------------------------------------------
# AiiDA calcfunction: merge N per-snapshot nodes
# ---------------------------------------------------------------------------


@calcfunction
def merge_path_contributions(r_bin, **kwargs) -> PathContributionsData:
    """Merge per-snapshot PathContributionsData nodes into one ensemble node.

    Keyword arguments (beyond ``r_bin``) must be
    :class:`~aiida_feff.data.pathcontributions.PathContributionsData` nodes.
    Keys are used only for ordering (e.g. ``snap_0000``, ``snap_0001``).

    Paths are grouped across frames using ``_make_path_key`` with the given
    ``r_bin`` (in Å).  Paths in the same bin across different frames share
    the same path-key column in the output HDF5.

    The merged node uses the same columnar schema as the per-calc nodes but
    with an additional ``frame_idx`` and ``site_idx`` per path row.

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

    for key in sorted(kwargs.keys()):
        node = kwargs[key]
        for pr in node.iter_paths():
            if k_grid is None:
                k_grid = pr.k
            path_key = _make_path_key(pr.scatterer, pr.nlegs, pr.r_eff, r_bin_val)
            rows.append(
                {
                    "frame_idx": pr.frame_idx,
                    "site_idx": pr.site_idx,
                    "path_key": path_key,
                    "r_eff": pr.r_eff,
                    "nlegs": pr.nlegs,
                    "degeneracy": pr.degeneracy,
                    "scatterer": pr.scatterer,
                    "cw_ratio": pr.cw_ratio,
                    "feff_data": pr.feff_data,
                }
            )

    if not rows or k_grid is None:
        raise ValueError("No path data found in any of the input nodes.")

    n = len(rows)
    m_k = len(k_grid)

    feff_data_arr = np.zeros((n, m_k, 6), dtype=np.float64)
    r_eff_arr = np.zeros(n, dtype=np.float64)
    nlegs_arr = np.zeros(n, dtype=np.int32)
    degeneracy_arr = np.zeros(n, dtype=np.float64)
    cw_ratio_arr = np.zeros(n, dtype=np.float64)
    frame_idx_arr = np.zeros(n, dtype=np.int32)
    site_idx_arr = np.zeros(n, dtype=np.int32)
    scatterer_list: list[str] = []
    path_key_list: list[str] = []

    for i, row in enumerate(rows):
        feff_data_arr[i] = row["feff_data"]
        r_eff_arr[i] = row["r_eff"]
        nlegs_arr[i] = row["nlegs"]
        degeneracy_arr[i] = row["degeneracy"]
        cw_ratio_arr[i] = row["cw_ratio"]
        frame_idx_arr[i] = row["frame_idx"]
        site_idx_arr[i] = row["site_idx"]
        scatterer_list.append(row["scatterer"])
        path_key_list.append(row["path_key"])

    _COMPRESS = {"compression": "gzip", "compression_opts": 6}
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["format_version"] = _H5_VERSION
        meta.attrs["n_frames"] = int(len(kwargs))
        meta.attrs["r_bin"] = r_bin_val

        pg = f.create_group("paths")
        pg.create_dataset("k_grid_params", data=k_grid, **_COMPRESS)
        pg.create_dataset("feff_data", data=feff_data_arr, **_COMPRESS)
        pg.create_dataset("r_eff", data=r_eff_arr, **_COMPRESS)
        pg.create_dataset("nlegs", data=nlegs_arr, **_COMPRESS)
        pg.create_dataset("degeneracy", data=degeneracy_arr, **_COMPRESS)
        pg.create_dataset("cw_ratio", data=cw_ratio_arr, **_COMPRESS)
        pg.create_dataset("frame_idx", data=frame_idx_arr, **_COMPRESS)
        pg.create_dataset("site_idx", data=site_idx_arr, **_COMPRESS)
        dt_str = h5py.string_dtype(encoding="utf-8")
        pg.create_dataset("scatterer", data=np.array(scatterer_list, dtype=dt_str), **_COMPRESS)
        pg.create_dataset("path_key", data=np.array(path_key_list, dtype=dt_str), **_COMPRESS)

    buf.seek(0)
    merged = PathContributionsData()
    merged.base.repository.put_object_from_filelike(buf, _H5_KEY)  # type: ignore[arg-type]
    return merged
