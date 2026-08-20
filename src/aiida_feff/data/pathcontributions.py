"""PathContributionsData — HDF5-backed per-path FEFF contribution store."""

from __future__ import annotations

import io
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
from aiida import orm

_H5_KEY = "contributions.h5"

#: Schema written by ``_aggregate_paths.py`` for a single FEFF run.
H5_VERSION_SINGLE = 1
#: Schema written by ``merge_path_contributions`` — adds per-row ``frame_idx``,
#: ``site_idx`` and ``path_key`` datasets, so it is a distinct version rather
#: than the same number with extra columns.
H5_VERSION_MERGED = 2
SUPPORTED_H5_VERSIONS = frozenset({H5_VERSION_SINGLE, H5_VERSION_MERGED})

# Retained for backwards compatibility with older imports.
_H5_VERSION = H5_VERSION_SINGLE

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


def _as_str(value) -> str:
    """Decode an HDF5 string attribute or dataset element to ``str``."""
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _check_h5_version(raw: bytes) -> int:
    """Validate the schema version of an HDF5 blob and return it."""
    h5py = _require_h5py()
    with h5py.File(io.BytesIO(raw), "r") as f:
        if "meta" not in f or "paths" not in f:
            raise ValueError("Not a PathContributions HDF5 file: expected /meta and /paths groups.")
        version = int(f["meta"].attrs.get("format_version", -1))
    if version not in SUPPORTED_H5_VERSIONS:
        raise ValueError(
            f"Unsupported PathContributions schema version {version}; "
            f"this build reads {sorted(SUPPORTED_H5_VERSIONS)}."
        )
    return version


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
    sig2: float = 0.0  # σ² (Å²) FEFF itself applied, from files.dat; 0 when unset


class PathContributionsData(orm.Data):
    """AiiDA Data node wrapping a single gzip-compressed HDF5 file.

    Stores amplitude-filtered per-path FEFF scattering factors, either for one
    FeffCalculation (one MD frame, one absorbing site) or, after
    :func:`~aiida_feff.calcfunctions.path_contributions.merge_path_contributions`,
    for a whole ensemble.

    Turn the stored columns back into χ(k) with
    :func:`aiida_feff.calcfunctions.exafs.total_chi`.

    HDF5 schema, ``format_version = 1`` (single calculation)
    --------------------------------------------------------
    /meta                   group
        attrs:
            format_version  int   = 1
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
        sig2                float64[P]       σ² FEFF itself applied (Å²)

    P = number of paths kept after amplitude filtering
    M = number of k points in the native FEFF grid

    HDF5 schema, ``format_version = 2`` (merged ensemble)
    ------------------------------------------------------
    As above, except ``/meta`` carries ``n_frames``, ``n_sites``, ``r_bin``
    and ``absorber_element`` instead of a single ``frame_idx`` / ``site_idx``,
    and ``/paths`` gains per-row ``frame_idx``, ``site_idx`` and ``path_key``
    datasets.

    Columns of ``feff_data`` (in order, matching larch FeffDatFile attrs):
    ``real_phc``, ``mag_feff``, ``pha_feff``, ``red_fact``, ``lam``, ``rep``.
    """

    @classmethod
    def from_hdf5_bytes(cls, raw: bytes) -> PathContributionsData:
        """Construct a node from raw HDF5 bytes (used by the parser).

        The schema version is checked here rather than on first access, so an
        unreadable file fails at parse time with a clear message instead of
        raising ``KeyError`` deep inside :meth:`iter_paths`.
        """
        _check_h5_version(raw)
        node = cls()
        node.base.repository.put_object_from_filelike(io.BytesIO(raw), _H5_KEY)  # type: ignore[arg-type]
        return node

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _raw(self) -> bytes:
        """Return the stored HDF5 blob, cached per instance.

        Each metadata property used to re-read and re-parse the whole file;
        for a merged ensemble node that is hundreds of megabytes per attribute
        access.
        """
        cached = getattr(self, "_raw_cache", None)
        if cached is None:
            cached = self.base.repository.get_object_content(_H5_KEY, mode="rb")
            self._raw_cache = cached
        return cached  # type: ignore[no-any-return]

    @contextmanager
    def _open(self):
        """Yield an open :class:`h5py.File` over the stored blob."""
        h5py = _require_h5py()
        with h5py.File(io.BytesIO(self._raw()), "r") as f:
            yield f

    @property
    def format_version(self) -> int:
        """Schema version of the stored HDF5 file."""
        with self._open() as f:
            return int(f["meta"].attrs.get("format_version", H5_VERSION_SINGLE))

    @property
    def is_merged(self) -> bool:
        """True when this node holds a merged ensemble rather than one run."""
        return self.format_version == H5_VERSION_MERGED

    @property
    def frame_idx(self) -> int:
        """Frame index stored in HDF5 meta attrs.

        Raises for a merged node, where frames live per row in
        :meth:`iter_paths` and no single value is meaningful.
        """
        with self._open() as f:
            if self._merged_attrs(f):
                raise ValueError(
                    "frame_idx is not defined for a merged node; "
                    "read PathResult.frame_idx from iter_paths() instead."
                )
            return int(f["meta"].attrs.get("frame_idx", 0))

    @property
    def site_idx(self) -> int:
        """Absorbing-site index stored in HDF5 meta attrs.

        Raises for a merged node, for the same reason as :attr:`frame_idx`.
        """
        with self._open() as f:
            if self._merged_attrs(f):
                raise ValueError(
                    "site_idx is not defined for a merged node; "
                    "read PathResult.site_idx from iter_paths() instead."
                )
            return int(f["meta"].attrs.get("site_idx", 0))

    @property
    def absorber_element(self) -> str:
        """Absorber element symbol stored in HDF5 meta attrs."""
        with self._open() as f:
            return _as_str(f["meta"].attrs.get("absorber_element", ""))

    @staticmethod
    def _merged_attrs(f) -> bool:
        return int(f["meta"].attrs.get("format_version", H5_VERSION_SINGLE)) == H5_VERSION_MERGED

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def iter_paths(self) -> Iterator[PathResult]:
        """Yield one :class:`PathResult` per path stored in this node."""
        with self._open() as f:
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
            scatterers = [_as_str(s) for s in pg["scatterer"]]
            cw_ratio = np.array(pg["cw_ratio"])
            # sig2 was added after the first release; older files lack it.
            sig2_arr = (
                np.array(pg["sig2"]) if "sig2" in pg else np.zeros(len(r_eff), dtype=np.float64)
            )
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
                sig2=float(sig2_arr[i]),
                k=k.copy(),
                feff_data=feff_data[i].copy(),
            )

    def info(self) -> dict:
        """Return a summary dict describing the stored paths.

        A merged node has no single frame or site, so those keys are simply
        absent rather than reported as zero.
        """
        raw = self._raw()
        with self._open() as f:
            meta = f["meta"]
            pg = f["paths"]
            summary = {
                "format_version": int(meta.attrs.get("format_version", H5_VERSION_SINGLE)),
                "is_merged": self._merged_attrs(f),
                "n_paths": int(len(pg["r_eff"])),
                "absorber_element": _as_str(meta.attrs.get("absorber_element", "")),
                "file_size_mb": round(len(raw) / 1024 / 1024, 3),
            }

            if summary["is_merged"]:
                summary["n_frames"] = int(meta.attrs.get("n_frames", 1))
                summary["n_sites"] = int(meta.attrs.get("n_sites", 1))
                summary["r_bin"] = float(meta.attrs.get("r_bin", 0.15))
                summary["n_path_keys"] = (
                    int(len(np.unique(np.asarray(pg["path_key"])))) if "path_key" in pg else 0
                )
            else:
                summary["frame_idx"] = int(meta.attrs.get("frame_idx", 0))
                summary["site_idx"] = int(meta.attrs.get("site_idx", 0))
                summary["threshold"] = float(meta.attrs.get("threshold", 0.0))

            return summary
