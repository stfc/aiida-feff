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
    n_k: int = 4,
) -> bytes:
    """Build a minimal contributions_raw.h5 bytes blob (single-calc schema).

    ``TestSyntheticSchemaMatchesTheWriter`` pins this against the real writer
    so the two cannot drift apart unnoticed.
    """
    pytest.importorskip("h5py")
    import h5py as _h5py

    k = np.linspace(0.5, 3.0, n_k)
    m_k = len(k)

    columns = np.column_stack(
        [
            np.linspace(3.8, 2.5, m_k),
            np.linspace(0.5, 1.0, m_k),
            np.linspace(-6.0, -9.0, m_k),
            np.linspace(2.0, 2.3, m_k),
            np.linspace(1.0, 1.3, m_k),
            np.linspace(2.5, 2.8, m_k),
        ]
    )
    feff_data = np.tile(columns, (n_paths, 1, 1)).reshape(n_paths, m_k, 6)

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
        pg.create_dataset("sig2", data=np.zeros(n_paths))

    buf.seek(0)
    return buf.read()


def _make_pc_node(
    frame_idx: int = 0,
    site_idx: int = 0,
    n_paths: int = 1,
    r_eff_base: float = 2.48,
    n_k: int = 4,
):
    """Build (but do not store) a PathContributionsData node from raw HDF5 bytes."""
    from aiida_feff.data.pathcontributions import PathContributionsData

    raw = _make_raw_hdf5(
        frame_idx=frame_idx,
        site_idx=site_idx,
        n_paths=n_paths,
        r_eff_base=r_eff_base,
        n_k=n_k,
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
        np.testing.assert_allclose(p.feff_data[:, 1], np.linspace(0.5, 1.0, 4), rtol=1e-6)

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


class TestPathKeyStability:
    """Bin assignment must not depend on floating-point noise."""

    def test_documented_examples(self):
        from aiida_feff.calcfunctions.path_contributions import make_path_key

        assert make_path_key("Fe", 2, 2.48, 0.15) == "SS_Fe_2.475"
        assert make_path_key("Fe-Fe", 3, 6.12, 0.15) == "MS3_Fe-Fe_6.075"

    def test_exact_bin_edge_is_not_at_the_mercy_of_representation(self):
        from aiida_feff.calcfunctions.path_contributions import make_path_key

        # 3.0 // 0.15 is 20 while 2.9999999999 // 0.15 is 19, so plain floor
        # division puts two indistinguishable paths in bins 0.15 A apart.
        assert make_path_key("X", 2, 3.0, 0.15) == make_path_key("X", 2, 2.9999999999, 0.15)

    def test_bin_index_increases_monotonically(self):
        from aiida_feff.calcfunctions.path_contributions import make_path_key

        centres = [
            float(make_path_key("X", 2, r, 0.15).rsplit("_", 1)[1])
            for r in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        ]
        assert centres == sorted(centres)

    def test_multiples_of_the_bin_width_are_exact(self):
        from aiida_feff.calcfunctions.path_contributions import make_path_key

        # r_eff exactly on an edge belongs to the bin it opens.
        assert make_path_key("X", 2, 0.30, 0.15) == "SS_X_0.375"

    def test_non_positive_bin_width_rejected(self):
        from aiida_feff.calcfunctions.path_contributions import make_path_key

        with pytest.raises(ValueError, match="r_bin"):
            make_path_key("X", 2, 2.0, 0.0)


@pytest.mark.usefixtures("aiida_profile_clean")
class TestMergedNodeMetadata:
    """The merged node must describe itself honestly."""

    def test_n_frames_counts_frames_not_input_nodes(self):
        from aiida import orm

        from aiida_feff.calcfunctions.path_contributions import merge_path_contributions

        # Two frames x two sites = four inputs, but only two frames.
        nodes = {
            f"snap_{f:04d}_site_{s:04d}": _make_pc_node(frame_idx=f, site_idx=s, n_paths=1)
            for f in (0, 1)
            for s in (0, 1)
        }
        merged = merge_path_contributions(orm.Float(0.15), **nodes)
        info = merged.info()
        assert info["n_frames"] == 2
        assert info["n_sites"] == 2
        assert info["n_paths"] == 4

    def test_merged_node_declares_its_own_schema_version(self):
        from aiida import orm

        from aiida_feff.calcfunctions.path_contributions import merge_path_contributions
        from aiida_feff.data.pathcontributions import H5_VERSION_MERGED

        merged = merge_path_contributions(
            orm.Float(0.15), snap_0000=_make_pc_node(frame_idx=0, n_paths=1)
        )
        assert merged.format_version == H5_VERSION_MERGED
        assert merged.is_merged

    def test_frame_idx_property_refuses_to_invent_a_value(self):
        from aiida import orm

        from aiida_feff.calcfunctions.path_contributions import merge_path_contributions

        merged = merge_path_contributions(
            orm.Float(0.15),
            snap_0000=_make_pc_node(frame_idx=0, n_paths=1),
            snap_0001=_make_pc_node(frame_idx=7, n_paths=1),
        )
        # Reporting 0 here, as it used to, is a lie about which frame this is.
        with pytest.raises(ValueError, match="not defined for a merged node"):
            _ = merged.frame_idx

    def test_mismatched_k_grids_are_refused(self):
        from aiida import orm

        from aiida_feff.calcfunctions.path_contributions import merge_path_contributions

        with pytest.raises(ValueError, match="k grid"):
            merge_path_contributions(
                orm.Float(0.15),
                snap_0000=_make_pc_node(frame_idx=0, n_paths=1, n_k=20),
                snap_0001=_make_pc_node(frame_idx=1, n_paths=1, n_k=25),
            )


@pytest.mark.usefixtures("aiida_profile_clean")
class TestSchemaValidation:
    def test_unknown_schema_version_is_refused_on_load(self):
        import io

        import h5py

        from aiida_feff.data.pathcontributions import PathContributionsData

        buf = io.BytesIO()
        with h5py.File(buf, "w") as f:
            f.create_group("meta").attrs["format_version"] = 99
            f.create_group("paths")
        with pytest.raises(ValueError, match="Unsupported PathContributions schema"):
            PathContributionsData.from_hdf5_bytes(buf.getvalue())

    def test_non_path_hdf5_is_refused(self):
        import io

        import h5py

        from aiida_feff.data.pathcontributions import PathContributionsData

        buf = io.BytesIO()
        with h5py.File(buf, "w") as f:
            f.create_dataset("something_else", data=[1, 2, 3])
        with pytest.raises(ValueError, match="Not a PathContributions HDF5 file"):
            PathContributionsData.from_hdf5_bytes(buf.getvalue())


class TestSyntheticSchemaMatchesTheWriter:
    """Guard against the hand-built fixture drifting from _aggregate_paths.py."""

    def test_same_datasets_and_attributes(self, aggregated_hdf5):
        import h5py

        def describe(raw):
            with h5py.File(io.BytesIO(raw), "r") as f:
                return set(f["paths"]), set(f["meta"].attrs)

        assert describe(_make_raw_hdf5()) == describe(aggregated_hdf5)
