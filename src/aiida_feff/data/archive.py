"""ExafsArchiveData: Consolidated AiiDA data node for MD-EXAFS results (ADR 0004).

Wraps either a batch shard (batch_shard.h5) or an ensemble archive (ensemble_results.h5),
delegating spectral and path queries directly to md_exafs.hdf5.ArchiveReader.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from aiida.orm import SinglefileData
from md_exafs.hdf5 import ArchiveReader
from md_exafs.paths import PathResult

if TYPE_CHECKING:
    from aiida_feff.data.xasdata import XasData


class ExafsArchiveData(SinglefileData):
    """Consolidated AiiDA data node for batch shards and ensemble archives (ADR 0004).

    Replaces decoupled XasData and PathContributionsData pairs with a single
    provenance node wrapping the underlying HDF5 archive.

    The file lives in the AiiDA repository, so every read has to materialise it on
    the local filesystem first.  That extraction is done once per node instance and
    cached for the instance's lifetime — reading ``k``, ``chi``, ``r`` and
    ``chir_mag`` off a freshly loaded node therefore costs one copy, not four.
    Use :meth:`reader` when making several queries in a row.
    """

    def _cached_path(self) -> Path:
        """Return a local path to the archive, extracting it at most once per instance.

        ``SinglefileData.as_path()`` is a context manager that deletes the temporary
        copy on exit, so an :class:`~md_exafs.hdf5.ArchiveReader` built inside it is
        dangling by the time the caller uses it.  We therefore own the temporary
        directory ourselves and tie its lifetime to this node instance.
        """
        cached: Path | None = getattr(self, "_archive_local_path", None)
        if cached is not None and cached.exists():
            return cached

        tmp_dir = tempfile.TemporaryDirectory(prefix="aiida-feff-archive-")
        local = Path(tmp_dir.name) / self.filename
        with self.as_path() as src:
            local.write_bytes(Path(src).read_bytes())

        # TemporaryDirectory cleans itself up when garbage collected, so holding it
        # on the node ties the extracted copy's lifetime to the node object's.
        self._archive_tmp_dir = tmp_dir
        self._archive_local_path = local
        return local

    @contextmanager
    def reader(self) -> Iterator[ArchiveReader]:
        """Yield an :class:`~md_exafs.hdf5.ArchiveReader` for this node's file.

        Preferred entry point for anything beyond a single attribute read::

            with node.reader() as archive:
                k, chi = archive.k, archive.chi
        """
        yield ArchiveReader(self._cached_path())

    @property
    def is_ensemble(self) -> bool:
        """Whether this archive is a consolidated ensemble archive."""
        return bool(ArchiveReader(self._cached_path()).is_ensemble)

    @property
    def is_shard(self) -> bool:
        """Whether this archive is a batch shard."""
        return bool(ArchiveReader(self._cached_path()).is_shard)

    @property
    def k(self) -> np.ndarray:
        """Photoelectron wavenumber grid (Å⁻¹)."""
        return ArchiveReader(self._cached_path()).k

    @property
    def chi(self) -> np.ndarray:
        """Grand ensemble average or shard mean χ(k)."""
        return ArchiveReader(self._cached_path()).chi

    @property
    def chi_std(self) -> np.ndarray | None:
        """Sample standard deviation of χ(k), if stored."""
        return ArchiveReader(self._cached_path()).chi_std

    @property
    def r(self) -> np.ndarray | None:
        """Fourier transform distance grid R (Å)."""
        return ArchiveReader(self._cached_path()).r

    @property
    def chir_mag(self) -> np.ndarray | None:
        """Fourier transform magnitude |χ(R)|."""
        return ArchiveReader(self._cached_path()).chir_mag

    @property
    def chir_re(self) -> np.ndarray | None:
        """Fourier transform real part."""
        return ArchiveReader(self._cached_path()).chir_re

    @property
    def chir_im(self) -> np.ndarray | None:
        """Fourier transform imaginary part."""
        return ArchiveReader(self._cached_path()).chir_im

    def iter_paths(self) -> list[PathResult]:
        """Return all scattering paths stored in the archive."""
        return list(ArchiveReader(self._cached_path()).iter_paths())

    def get_site_average(self, site_idx: int) -> dict[str, np.ndarray]:
        """Return average spectra for a given site index."""
        return ArchiveReader(self._cached_path()).get_site_average(site_idx)

    def get_frame_average(self, frame_idx: int) -> dict[str, np.ndarray]:
        """Return average spectra for a given frame index."""
        return ArchiveReader(self._cached_path()).get_frame_average(frame_idx)

    def to_xas_data(self) -> XasData:
        """Convert the overall average to a legacy XasData node for backwards compatibility."""
        from aiida_feff.data.xasdata import XasData

        with self.reader() as archive:
            k, chi = archive.k, archive.chi

        out = XasData()
        out.set_chi(k, chi)
        return out


__all__ = [
    "ExafsArchiveData",
]
