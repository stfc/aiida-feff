"""Tests for PathContributionsData and merge_path_contributions."""

from __future__ import annotations

import io

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers: build PathContributionsData nodes using the new columnar HDF5 schema
# ---------------------------------------------------------------------------


def _make_raw_hdf5(
    frame_idx: int = 0,
    site_idx: int = 0,
    absorber_element: str = "Fe",
    n_paths: int = 1,
    r_eff_base: float = 2.48,
) -> bytes:
    """Build a minimal contributions_raw.h5 bytes blob (schema v2)."""
    pytest.importorskip("h5py")
    import h5py as _h5py

    k = np.array([0.5, 1.0, 2.0, 3.0], dtype=np.float64)
    m_k = len(k)

    feff_data = np.tile(
        np.column_stack(
            [
                [3.8, 3.5, 3.0, 2.5],
                [0.5, 0.7, 0.9, 1.0],
                [-6.0, -7.0, -8.0, -9.0],
                [2.0, 2.1, 2.2, 2.3],
                [1.0, 1.1, 1.2, 1.3],
                [2.5, 2.6, 2.7, 2.8],
            ]
        ),
        (n_paths, 1, 1),
    ).reshape(n_paths, m_k, 6)

    buf = io.BytesIO()
    with _h5py.File(buf, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["format_version"] = 1
        meta.attrs["frame_idx"] = frame_idx
        meta.attrs["site_idx"] = site_idx
        meta.attrs["absorber_element"] = absorber_element
        meta.attrs["threshold"] = 0.0

        pg = f.create_group("paths")
        pg.create_dataset("k_grid_params", data=k)
        pg.create_dataset("feff_data", data=feff_data)
        pg.create_dataset("r_eff", data=np.array([r_eff_base + i * 0.5 for i in range(n_paths)]))
        pg.create_dataset("nlegs", data=np.full(n_paths, 2, dtype=np.int32))
        pg.create_dataset("degeneracy", data=np.ones(n_paths))
        dt_str = _h5py.string_dtype(encoding="utf-8")
        pg.create_dataset("scatterer", data=np.array(["Fe"] * n_paths, dtype=dt_str))
        pg.create_dataset("cw_ratio", data=np.array([100.0 - i * 5 for i in range(n_paths)]))

    buf.seek(0)
    return buf.read()


def _make_pc_node(
    frame_idx: int = 0, site_idx: int = 0, n_paths: int = 1, r_eff_base: float = 2.48
):
    """Build (but do not store) a PathContributionsData node from raw HDF5 bytes."""
    from aiida_feff.data.pathcontributions import PathContributionsData

    raw = _make_raw_hdf5(
        frame_idx=frame_idx, site_idx=site_idx, n_paths=n_paths, r_eff_base=r_eff_base
    )
    return PathContributionsData.from_hdf5_bytes(raw)


# ---------------------------------------------------------------------------
# PathContributionsData — round-trip (no AiiDA profile needed)
# ---------------------------------------------------------------------------


class TestPathContributionsData:
    def test_frame_and_site_idx(self):
        pc = _make_pc_node(frame_idx=3, site_idx=1)
        assert pc.frame_idx == 3
        assert pc.site_idx == 1

    def test_absorber_element(self):
        pc = _make_pc_node()
        assert pc.absorber_element == "Fe"

    def test_iter_paths_count(self):
        pc = _make_pc_node(n_paths=3)
        paths = list(pc.iter_paths())
        assert len(paths) == 3

    def test_iter_paths_r_eff(self):
        pc = _make_pc_node(n_paths=1, r_eff_base=2.48)
        p = list(pc.iter_paths())[0]
        assert p.r_eff == pytest.approx(2.48)

    def test_iter_paths_nlegs(self):
        pc = _make_pc_node()
        p = list(pc.iter_paths())[0]
        assert p.nlegs == 2

    def test_iter_paths_scatterer(self):
        pc = _make_pc_node()
        p = list(pc.iter_paths())[0]
        assert p.scatterer == "Fe"

    def test_iter_paths_frame_site_propagated(self):
        pc = _make_pc_node(frame_idx=7, site_idx=2)
        p = list(pc.iter_paths())[0]
        assert p.frame_idx == 7
        assert p.site_idx == 2

    def test_iter_paths_feff_data_shape(self):
        pc = _make_pc_node(n_paths=1)
        p = list(pc.iter_paths())[0]
        assert p.feff_data.shape == (4, 6)

    def test_feff_data_roundtrip(self):
        """feff_data values survive the HDF5 round-trip."""
        pc = _make_pc_node(n_paths=1)
        p = list(pc.iter_paths())[0]
        np.testing.assert_allclose(p.feff_data[:, 1], [0.5, 0.7, 0.9, 1.0], rtol=1e-6)

    def test_info_n_paths(self):
        pc = _make_pc_node(n_paths=4)
        info = pc.info()
        assert info["n_paths"] == 4

    def test_info_frame_site(self):
        pc = _make_pc_node(frame_idx=2, site_idx=0)
        info = pc.info()
        assert info["frame_idx"] == 2
        assert info["site_idx"] == 0

    def test_info_file_size_positive(self):
        pc = _make_pc_node()
        assert pc.info()["file_size_mb"] > 0


# ---------------------------------------------------------------------------
# _make_path_key
# ---------------------------------------------------------------------------


class TestMakePathKey:
    def test_ss_key(self):
        from aiida_feff.calcfunctions.path_contributions import _make_path_key

        key = _make_path_key("Fe", 2, 2.48, 0.15)
        assert key.startswith("SS_Fe_")

    def test_ms_key(self):
        from aiida_feff.calcfunctions.path_contributions import _make_path_key

        key = _make_path_key("Fe-Fe", 3, 6.12, 0.15)
        assert key.startswith("MS3_Fe-Fe_")

    def test_same_bin_same_key(self):
        from aiida_feff.calcfunctions.path_contributions import _make_path_key

        k1 = _make_path_key("Fe", 2, 2.47, 0.15)
        k2 = _make_path_key("Fe", 2, 2.51, 0.15)
        assert k1 == k2

    def test_different_bins_different_key(self):
        from aiida_feff.calcfunctions.path_contributions import _make_path_key

        k1 = _make_path_key("Fe", 2, 2.48, 0.15)
        k2 = _make_path_key("Fe", 2, 2.80, 0.15)
        assert k1 != k2


# ---------------------------------------------------------------------------
# merge_path_contributions  (requires AiiDA profile)
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("aiida_profile_clean")
class TestMergePathContributions:
    def test_merge_two_nodes_path_count(self):
        from aiida import orm

        from aiida_feff.calcfunctions.path_contributions import merge_path_contributions

        node0 = _make_pc_node(frame_idx=0, n_paths=1)
        node1 = _make_pc_node(frame_idx=1, n_paths=1)
        merged = merge_path_contributions(orm.Float(0.15), snap_0000=node0, snap_0001=node1)
        paths = list(merged.iter_paths())
        assert len(paths) == 2

    def test_merge_frame_indices_preserved(self):
        from aiida import orm

        from aiida_feff.calcfunctions.path_contributions import merge_path_contributions

        node0 = _make_pc_node(frame_idx=0)
        node5 = _make_pc_node(frame_idx=5)
        merged = merge_path_contributions(orm.Float(0.15), snap_0000=node0, snap_0005=node5)
        frame_indices = sorted(p.frame_idx for p in merged.iter_paths())
        assert frame_indices == [0, 5]

    def test_merge_path_keys_present(self):
        from aiida import orm

        from aiida_feff.calcfunctions.path_contributions import merge_path_contributions
        from aiida_feff.data.pathcontributions import _H5_KEY, _require_h5py

        node0 = _make_pc_node(frame_idx=0)
        node1 = _make_pc_node(frame_idx=1)
        merged = merge_path_contributions(orm.Float(0.15), snap_0000=node0, snap_0001=node1)

        h5py = _require_h5py()
        raw = merged.base.repository.get_object_content(_H5_KEY, mode="rb")
        with h5py.File(io.BytesIO(raw), "r") as f:
            assert "path_key" in f["paths"]
            assert len(f["paths"]["path_key"]) == 2

    def test_merge_different_r_eff_separate_keys(self):
        """Paths with r_eff in different bins get different path_keys."""
        from aiida import orm

        from aiida_feff.calcfunctions.path_contributions import merge_path_contributions
        from aiida_feff.data.pathcontributions import _H5_KEY, _require_h5py

        node0 = _make_pc_node(frame_idx=0, r_eff_base=2.48)
        node1 = _make_pc_node(frame_idx=1, r_eff_base=3.50)
        merged = merge_path_contributions(orm.Float(0.15), snap_0000=node0, snap_0001=node1)

        h5py = _require_h5py()
        raw = merged.base.repository.get_object_content(_H5_KEY, mode="rb")
        with h5py.File(io.BytesIO(raw), "r") as f:
            keys = [
                k.decode("utf-8") if isinstance(k, bytes) else k for k in f["paths"]["path_key"][:]
            ]
        assert len(set(keys)) == 2

    def test_merge_same_r_eff_same_key(self):
        """Paths with r_eff in the same bin share the path_key."""
        from aiida import orm

        from aiida_feff.calcfunctions.path_contributions import merge_path_contributions
        from aiida_feff.data.pathcontributions import _H5_KEY, _require_h5py

        node0 = _make_pc_node(frame_idx=0, r_eff_base=2.48)
        node1 = _make_pc_node(frame_idx=1, r_eff_base=2.50)  # same 0.15 Å bin
        merged = merge_path_contributions(orm.Float(0.15), snap_0000=node0, snap_0001=node1)

        h5py = _require_h5py()
        raw = merged.base.repository.get_object_content(_H5_KEY, mode="rb")
        with h5py.File(io.BytesIO(raw), "r") as f:
            keys = [
                k.decode("utf-8") if isinstance(k, bytes) else k for k in f["paths"]["path_key"][:]
            ]
        assert len(set(keys)) == 1

    def test_merge_feff_data_shape(self):
        from aiida import orm

        from aiida_feff.calcfunctions.path_contributions import merge_path_contributions

        node0 = _make_pc_node(frame_idx=0, n_paths=2)
        node1 = _make_pc_node(frame_idx=1, n_paths=3)
        merged = merge_path_contributions(orm.Float(0.15), snap_0000=node0, snap_0001=node1)
        paths = list(merged.iter_paths())
        assert len(paths) == 5
        assert all(p.feff_data.shape == (4, 6) for p in paths)
