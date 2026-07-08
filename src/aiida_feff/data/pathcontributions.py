"""PathContributionsData — HDF5-backed per-path FEFF contribution store."""

from __future__ import annotations

import io
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from aiida import orm

_H5_KEY = "contributions.h5"
_H5_VERSION = 1

# Column names for the 6 columns in feff_data (matching larch FeffDatFile attrs).
FEFF_DATA_COLS = ("real_phc", "mag_feff", "pha_feff", "red_fact", "lam", "rep")


def _require_h5py():
    """Import h5py or raise a helpful ImportError."""
    try:
        import h5py  # noqa: PLC0415

        return h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required for PathContributionsData. Install it with: pip install h5py"
        ) from exc


@dataclass
class PathResult:
    """Data yielded by :meth:`PathContributionsData.iter_paths`."""

    frame_idx: int
    site_idx: int
    r_eff: float
    nlegs: int
    degeneracy: float
    scatterer: str
    cw_ratio: float  # curved-wave amplitude ratio relative to strongest path (0–100)
    k: np.ndarray  # native FEFF k grid
    feff_data: np.ndarray  # shape (M, 6): columns = FEFF_DATA_COLS


class PathContributionsData(orm.Data):
    """AiiDA Data node wrapping a single gzip-compressed HDF5 file.

    Stores amplitude-filtered per-path FEFF scattering factors for one
    FeffCalculation (one MD frame, one absorbing site).

    HDF5 schema (format_version=1, columnar)
    -----------------------------------------
    /meta                   group
        attrs:
            format_version  int   = 2
            frame_idx       int   frame index within the MD trajectory
            site_idx        int   absorbing-site index within the structure
            absorber_element str  element symbol of the absorber
            threshold       float cw_ratio amplitude threshold used
    /paths                  group
        k_grid_params       float64[M]       native (coarse) FEFF k grid
        feff_data           float64[P, M, 6] columns: FEFF_DATA_COLS
        r_eff               float64[P]
        nlegs               int32[P]
        degeneracy          float64[P]
        scatterer           str[P]           variable-length UTF-8
        cw_ratio            float64[P]

    P = number of paths kept after amplitude filtering
    M = number of k points in the native FEFF grid
    factors emitted by :class:`~aiida_feff.calculations.feff.FeffCalculation`.

    HDF5 schema::

        contributions.h5
        ├── meta/
        │   ├── k_grid_sites   float64[N]   (k-grid from chi.dat)
        │   └── k_grid_paths   float64[M]   (k-grid from feff*.dat)
        └── frames/
            └── frame_NNNN/
                └── sites/
                    └── site_NNNN/
                        ├── chi    float64[N]   gzip=6
                        │          attrs: frame_idx, site_idx,
                        │                 absorber_element, success
                        └── paths/   (only when path data is available)
                            └── path_NNNN/
                                └── feff_data  float64[M, 6]   gzip=6
                                    attrs: r_eff, nlegs, degeneracy,
                                           scatterer, cw_ratio, columns

    Columns of ``feff_data`` (in order, matching larch FeffDatFile attrs):
    ``real_phc``, ``mag_feff``, ``pha_feff``, ``red_fact``, ``lam``, ``rep``.
    """

    @classmethod
    def from_hdf5_bytes(cls, raw: bytes) -> PathContributionsData:
        """Construct a node from raw HDF5 bytes (used by the parser)."""
        node = cls()
        node.base.repository.put_object_from_filelike(io.BytesIO(raw), _H5_KEY)  # type: ignore[arg-type]
        return node

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def frame_idx(self) -> int:
        """Frame index stored in HDF5 meta attrs."""
        h5py = _require_h5py()
        raw = self.base.repository.get_object_content(_H5_KEY, mode="rb")
        with h5py.File(io.BytesIO(raw), "r") as f:
            return int(f["meta"].attrs.get("frame_idx", 0))

    @property
    def site_idx(self) -> int:
        """Absorbing-site index stored in HDF5 meta attrs."""
        h5py = _require_h5py()
        raw = self.base.repository.get_object_content(_H5_KEY, mode="rb")
        with h5py.File(io.BytesIO(raw), "r") as f:
            return int(f["meta"].attrs.get("site_idx", 0))

    @property
    def absorber_element(self) -> str:
        """Absorber element symbol stored in HDF5 meta attrs."""
        h5py = _require_h5py()
        raw = self.base.repository.get_object_content(_H5_KEY, mode="rb")
        with h5py.File(io.BytesIO(raw), "r") as f:
            val = f["meta"].attrs.get("absorber_element", "")
            return val.decode("utf-8") if isinstance(val, bytes) else str(val)

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def iter_paths(self) -> Iterator[PathResult]:
        """Yield one :class:`PathResult` per path stored in this node."""
        h5py = _require_h5py()
        raw = self.base.repository.get_object_content(_H5_KEY, mode="rb")
        with h5py.File(io.BytesIO(raw), "r") as f:
            meta = f["meta"]
            # Per-node default frame/site (used when no per-row arrays exist).
            default_frame_idx = int(meta.attrs.get("frame_idx", 0))
            default_site_idx = int(meta.attrs.get("site_idx", 0))
            pg = f["paths"]
            k = np.array(pg["k_grid_params"])
            feff_data = np.array(pg["feff_data"])
            r_eff = np.array(pg["r_eff"])
            nlegs = np.array(pg["nlegs"])
            degeneracy = np.array(pg["degeneracy"])
            scatterers = [
                s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in pg["scatterer"]
            ]
            cw_ratio = np.array(pg["cw_ratio"])
            # Merged ensemble nodes store per-row frame/site arrays.
            frame_idx_arr = (
                np.array(pg["frame_idx"])
                if "frame_idx" in pg
                else np.full(len(r_eff), default_frame_idx, dtype=np.int32)
            )
            site_idx_arr = (
                np.array(pg["site_idx"])
                if "site_idx" in pg
                else np.full(len(r_eff), default_site_idx, dtype=np.int32)
            )

        for i in range(len(r_eff)):
            yield PathResult(
                frame_idx=int(frame_idx_arr[i]),
                site_idx=int(site_idx_arr[i]),
                r_eff=float(r_eff[i]),
                nlegs=int(nlegs[i]),
                degeneracy=float(degeneracy[i]),
                scatterer=scatterers[i],
                cw_ratio=float(cw_ratio[i]),
                k=k.copy(),
                feff_data=feff_data[i].copy(),
            )

    def info(self) -> dict:
        """Return a summary dict (frame_idx, site_idx, n_paths, file_size_mb)."""
        h5py = _require_h5py()
        raw = self.base.repository.get_object_content(_H5_KEY, mode="rb")
        with h5py.File(io.BytesIO(raw), "r") as f:
            meta = f["meta"]
            pg = f["paths"]
            n_paths = len(pg["r_eff"])

            # Detect merged ensemble node or single calc node
            is_merged = "n_frames" in meta.attrs

            if is_merged:
                n_frames = int(meta.attrs.get("n_frames", 1))
                r_bin = float(meta.attrs.get("r_bin", 0.15))
                if "site_idx" in pg:
                    n_sites = int(len(np.unique(pg["site_idx"])))
                else:
                    n_sites = 1

                return {
                    "is_merged": True,
                    "n_frames": n_frames,
                    "n_sites": n_sites,
                    "r_bin": r_bin,
                    "n_paths": n_paths,
                    "frame_idx": 0,
                    "site_idx": 0,
                    "absorber_element": "",
                    "file_size_mb": round(len(raw) / 1024 / 1024, 3),
                }
            else:
                frame_idx = int(meta.attrs.get("frame_idx", 0))
                site_idx = int(meta.attrs.get("site_idx", 0))
                absorber_element = meta.attrs.get("absorber_element", "")
                if isinstance(absorber_element, bytes):
                    absorber_element = absorber_element.decode("utf-8")

                return {
                    "is_merged": False,
                    "frame_idx": frame_idx,
                    "site_idx": site_idx,
                    "absorber_element": str(absorber_element),
                    "n_frames": 1,
                    "n_sites": 1,
                    "n_paths": n_paths,
                    "file_size_mb": round(len(raw) / 1024 / 1024, 3),
                }
