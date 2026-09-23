"""Calcfunctions for merging ExafsArchiveData nodes (ADR 0002, ADR 0004)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

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


@calcfunction
def create_serial_shard(**kwargs) -> ExafsArchiveData:
    """Write the outputs of a serial ensemble into one batch shard.

    The batch path gets its shard from the remote driver.  Building the
    equivalent here means :func:`merge_exafs_shards` runs on both routes, so
    the ``archive`` output exists either way and one code path produces the
    ensemble average.

    Note that this puts the serial route's χ(k) onto md-exafs'
    :data:`~md_exafs.execution.DEFAULT_K_GRID`, as the batch route already
    does.  Spectra have to share a grid before they can be averaged; FEFF's
    own grid varies with the ``EXAFS`` k_max card.

    Keyword arguments carry the provenance links and are named
    ``xas__snap_FFFF_site_SSSS`` (:class:`~aiida_feff.data.xasdata.XasData`,
    required) and ``paths__snap_FFFF_site_SSSS``
    (:class:`~aiida_feff.data.pathcontributions.PathContributionsData`,
    optional).
    """
    from md_exafs.execution import DEFAULT_K_GRID
    from md_exafs.hdf5 import BatchShardWriter
    from md_exafs.spectra import resample_chi

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_shard = Path(tmp_dir) / "serial_shard.h5"
        with BatchShardWriter(out_shard, k_grid=DEFAULT_K_GRID) as writer:
            for key in sorted(kwargs):
                if not key.startswith("xas__"):
                    continue
                node = kwargs[key]
                paths_node = kwargs.get(f"paths__{key.removeprefix('xas__')}")

                # Frame and site come off the node, where FeffParser copied
                # them from the calcjob's own inputs, rather than being parsed
                # back out of the link label.
                writer.add_task_result(
                    frame_idx=node.base.attributes.get("frame_index"),
                    site_idx=node.base.attributes.get("site_index"),
                    absorber_element=node.base.attributes.get("absorber_element"),
                    chi=resample_chi(node.get_array("k"), node.get_array("chi_k"), DEFAULT_K_GRID),
                    paths=_as_shard_paths(paths_node) if paths_node else None,
                )

        return ExafsArchiveData(file=str(out_shard))


def _as_shard_paths(node: Any) -> list[Any]:
    """Convert this package's ``PathResult`` into the one md-exafs writes.

    Two different dataclasses carry that name.  md-exafs' has ``angle`` and
    ``path_index``; the one :meth:`PathContributionsData.iter_paths` yields
    has neither, and :meth:`BatchShardWriter.add_task_result` reads ``angle``.
    Handing ours straight over raises ``AttributeError`` at write time, so the
    import is aliased here to keep the two visibly distinct.

    aiida-feff records no scattering angle anywhere, so it stays ``None`` and
    the writer stores its -1.0 "unset" sentinel.
    """
    from md_exafs.paths import PathResult as ShardPathResult

    return [
        ShardPathResult(
            frame_idx=p.frame_idx,
            site_idx=p.site_idx,
            r_eff=p.r_eff,
            nlegs=p.nlegs,
            degeneracy=p.degeneracy,
            scatterer=p.scatterer,
            cw_ratio=p.cw_ratio,
            k=p.k,
            feff_data=p.feff_data,
            sig2=p.sig2,
        )
        for p in node.iter_paths()
    ]


__all__ = [
    "create_serial_shard",
    "merge_exafs_shards",
]
