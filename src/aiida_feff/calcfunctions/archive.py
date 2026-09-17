"""Calcfunctions for merging ExafsArchiveData nodes (ADR 0002, ADR 0004)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from aiida.engine import calcfunction
from md_exafs.execution import merge_shards

from aiida_feff.data.archive import ExafsArchiveData


@calcfunction
def merge_exafs_shards(**shards) -> ExafsArchiveData:
    """Merge K batch shards into a single consolidated ensemble archive.

    Keyword arguments must be
    :class:`~aiida_feff.data.archive.ExafsArchiveData` nodes, passed by keyword
    (``shard_0=s0, shard_1=s1, …``) so that AiiDA builds the provenance links.
    """
    if not shards:
        raise ValueError("No ExafsArchiveData shards supplied for merging.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        shard_paths: list[Path] = []
        for name, shard_node in shards.items():
            s_path = tmp_p / f"{name}.h5"
            with shard_node.as_path() as src:
                s_path.write_bytes(Path(src).read_bytes())
            shard_paths.append(s_path)

        out_ensemble = tmp_p / "ensemble_results.h5"
        merge_shards(shard_paths, ensemble_path=out_ensemble)

        return ExafsArchiveData(file=str(out_ensemble))


__all__ = [
    "merge_exafs_shards",
]
