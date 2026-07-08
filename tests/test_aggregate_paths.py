"""Tests for the remote path-aggregation script."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Paths to real FEFF fixtures (checked into repo)
# ---------------------------------------------------------------------------
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "aggregate_paths"

FILES_DAT = FIXTURE_DIR / "files.dat"
FEFF_DATS = sorted(FIXTURE_DIR.glob("feff????.dat"))

# ---------------------------------------------------------------------------
# Module under test (imported late so syntax errors are caught by pytest)
# ---------------------------------------------------------------------------


def _import_module():
    """Import the aggregation script as a module."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_aggregate_paths",
        Path(__file__).parent.parent
        / "src"
        / "aiida_feff"
        / "calculations"
        / "_aggregate_paths.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def aggregate_paths():
    """Return the imported aggregation module."""
    return _import_module()


@pytest.fixture()
def feff_workdir(tmp_path: Path):
    """Return a temporary directory pre-populated with real FEFF outputs."""
    for src in FEFF_DATS:
        shutil.copy(src, tmp_path / src.name)
    shutil.copy(FILES_DAT, tmp_path / "files.dat")
    return tmp_path


# ---------------------------------------------------------------------------
# _parse_files_dat
# ---------------------------------------------------------------------------


class TestParseFilesDat:
    def test_parses_all_entries(self, aggregate_paths):
        text = FILES_DAT.read_text()
        result = aggregate_paths._parse_files_dat(text)
        # The fixture files.dat has >700 entries; we only copied 9 .dat files
        # but the parser should read every line in the text.
        assert len(result) >= 9
        assert "feff0001.dat" in result
        assert "feff0006.dat" in result

    def test_values_correct(self, aggregate_paths):
        text = FILES_DAT.read_text()
        result = aggregate_paths._parse_files_dat(text)
        entry = result["feff0001.dat"]
        assert entry["cw_ratio"] == pytest.approx(100.0)
        assert entry["deg"] == pytest.approx(1.0)
        assert entry["nlegs"] == 2
        assert entry["r_eff"] == pytest.approx(1.8590, abs=1e-4)

    def test_empty_text(self, aggregate_paths):
        assert aggregate_paths._parse_files_dat("") == {}

    def test_no_amp_ratio_line(self, aggregate_paths):
        assert aggregate_paths._parse_files_dat("some preamble\n1 2 3\n") == {}


# ---------------------------------------------------------------------------
# _parse_with_text
# ---------------------------------------------------------------------------


class TestParseWithText:
    def test_success(self, aggregate_paths):
        fpath = FIXTURE_DIR / "feff0001.dat"
        parsed = aggregate_paths._parse_with_text(fpath)
        assert parsed
        assert parsed["path_idx"] == 1
        assert parsed["nlegs"] == 2
        assert parsed["degeneracy"] == pytest.approx(1.0)
        assert parsed["r_eff"] == pytest.approx(1.8590, abs=1e-4)
        assert "k" in parsed
        assert "feff_data" in parsed
        assert parsed["k"].ndim == 1
        assert parsed["feff_data"].ndim == 2
        assert parsed["feff_data"].shape[1] == 6

    def test_scatterer_extracted(self, aggregate_paths):
        # feff0007 is a 3-leg path (should have a scatterer chain)
        fpath = FIXTURE_DIR / "feff0007.dat"
        parsed = aggregate_paths._parse_with_text(fpath)
        assert parsed
        assert parsed["nlegs"] == 3
        assert parsed["scatterer"] != "?"

    def test_missing_file(self, aggregate_paths, tmp_path: Path):
        assert aggregate_paths._parse_with_text(tmp_path / "nonexistent.dat") == {}

    def test_malformed_no_data(self, aggregate_paths, tmp_path: Path):
        bad = tmp_path / "feff0999.dat"
        bad.write_text("nleg 2 deg 1.0 reff 2.0\nno data block here\n")
        assert aggregate_paths._parse_with_text(bad) == {}


# ---------------------------------------------------------------------------
# _parse_with_larch (skip if larch not installed)
# ---------------------------------------------------------------------------


class TestParseWithLarch:
    def test_success(self, aggregate_paths):
        pytest.importorskip("larch.xafs.feffdat", reason="larch not installed")
        fpath = FIXTURE_DIR / "feff0001.dat"
        parsed = aggregate_paths._parse_with_larch(fpath)
        assert parsed
        assert parsed["path_idx"] == 1
        assert parsed["nlegs"] == 2
        assert "k" in parsed
        assert "feff_data" in parsed
        assert parsed["feff_data"].shape[1] == 6

    def test_consistent_with_text_parser(self, aggregate_paths):
        """Both parsers should return the same scalar metadata for real files."""
        pytest.importorskip("larch.xafs.feffdat", reason="larch not installed")
        fpath = FIXTURE_DIR / "feff0001.dat"
        larch_parsed = aggregate_paths._parse_with_larch(fpath)
        text_parsed = aggregate_paths._parse_with_text(fpath)
        assert larch_parsed["path_idx"] == text_parsed["path_idx"]
        assert larch_parsed["nlegs"] == text_parsed["nlegs"]
        assert larch_parsed["degeneracy"] == pytest.approx(text_parsed["degeneracy"], abs=1e-3)
        assert larch_parsed["r_eff"] == pytest.approx(text_parsed["r_eff"], abs=1e-3)
        np.testing.assert_allclose(larch_parsed["k"], text_parsed["k"], atol=1e-4)


# ---------------------------------------------------------------------------
# main() integration tests
# ---------------------------------------------------------------------------


class TestMainIntegration:
    def test_no_config_exits_immediately(self, aggregate_paths, feff_workdir, monkeypatch):
        monkeypatch.chdir(feff_workdir)
        with pytest.raises(SystemExit) as exc_info:
            aggregate_paths.main()
        assert exc_info.value.code == 0
        assert not (feff_workdir / "contributions_raw.h5").exists()

    def test_all_paths_threshold_zero(self, aggregate_paths, feff_workdir, monkeypatch):
        monkeypatch.chdir(feff_workdir)
        config = {"threshold": 0.0, "frame_idx": 0, "site_idx": 0, "absorber_element": "Fe"}
        (feff_workdir / "_feff_aggregate_config.json").write_text(json.dumps(config))

        aggregate_paths.main()

        out = feff_workdir / "contributions_raw.h5"
        assert out.exists()

        import h5py

        with h5py.File(out, "r") as hf:
            assert hf["meta"].attrs["format_version"] == 1
            assert "k_grid_params" in hf["paths"]
            assert "feff_data" in hf["paths"]
            assert "r_eff" in hf["paths"]
            # All 9 copied .dat files should be present
            assert len(hf["paths"]["r_eff"]) == 9

    def test_threshold_filters_paths(self, aggregate_paths, feff_workdir, monkeypatch, capsys):
        monkeypatch.chdir(feff_workdir)
        # From the fixture files.dat:
        #   feff0001 100.0, feff0002 95.08, feff0003 90.68, feff0004 88.88,
        #   feff0005 85.67, feff0006 81.91, feff0007 13.01, feff0008 11.81,
        #   feff0011 30.87
        config = {"threshold": 90.0, "frame_idx": 0, "site_idx": 0, "absorber_element": "Fe"}
        (feff_workdir / "_feff_aggregate_config.json").write_text(json.dumps(config))

        aggregate_paths.main()

        out = feff_workdir / "contributions_raw.h5"
        assert out.exists()

        import h5py

        with h5py.File(out, "r") as hf:
            # Paths >= 90%: 0001, 0002, 0003
            assert len(hf["paths"]["r_eff"]) == 3

        captured = capsys.readouterr()
        assert "keeping 3/9 paths" in captured.err

    def test_missing_files_dat_warns_but_runs(
        self, aggregate_paths, feff_workdir, monkeypatch, capsys
    ):
        monkeypatch.chdir(feff_workdir)
        (feff_workdir / "files.dat").unlink()
        config = {"threshold": 0.0, "frame_idx": 0, "site_idx": 0, "absorber_element": "Fe"}
        (feff_workdir / "_feff_aggregate_config.json").write_text(json.dumps(config))

        aggregate_paths.main()

        out = feff_workdir / "contributions_raw.h5"
        assert out.exists()
        captured = capsys.readouterr()
        assert "files.dat not found" in captured.err
        assert "amplitude filtering disabled" in captured.err

    def test_no_feff_dat_files(self, aggregate_paths, feff_workdir, monkeypatch, capsys):
        monkeypatch.chdir(feff_workdir)
        for f in feff_workdir.glob("feff????.dat"):
            f.unlink()
        config = {"threshold": 0.0, "frame_idx": 0, "site_idx": 0, "absorber_element": "Fe"}
        (feff_workdir / "_feff_aggregate_config.json").write_text(json.dumps(config))

        with pytest.raises(SystemExit) as exc_info:
            aggregate_paths.main()
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "No feff????.dat files found" in captured.err

    def test_threshold_excludes_all_paths(self, aggregate_paths, feff_workdir, monkeypatch, capsys):
        monkeypatch.chdir(feff_workdir)
        config = {"threshold": 101.0, "frame_idx": 0, "site_idx": 0, "absorber_element": "Fe"}
        (feff_workdir / "_feff_aggregate_config.json").write_text(json.dumps(config))

        with pytest.raises(SystemExit) as exc_info:
            aggregate_paths.main()
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "All 9 paths below threshold" in captured.err

    def test_hdf5_columnar_schema(self, aggregate_paths, feff_workdir, monkeypatch):
        monkeypatch.chdir(feff_workdir)
        config = {"threshold": 0.0, "frame_idx": 2, "site_idx": 1, "absorber_element": "Cu"}
        (feff_workdir / "_feff_aggregate_config.json").write_text(json.dumps(config))

        aggregate_paths.main()

        import h5py

        with h5py.File(feff_workdir / "contributions_raw.h5", "r") as hf:
            meta = hf["meta"]
            assert meta.attrs["format_version"] == 1
            assert meta.attrs["frame_idx"] == 2
            assert meta.attrs["site_idx"] == 1
            assert meta.attrs["absorber_element"] in ("Cu", b"Cu")
            pg = hf["paths"]
            n = len(pg["r_eff"])
            assert n == 9
            assert pg["feff_data"].shape == (n, len(pg["k_grid_params"]), 6)
            assert pg["nlegs"][0] >= 2
            assert pg["cw_ratio"][0] == pytest.approx(100.0)

    def test_k_grid_consistency(self, aggregate_paths, feff_workdir, monkeypatch):
        monkeypatch.chdir(feff_workdir)
        config = {"threshold": 0.0, "frame_idx": 0, "site_idx": 0, "absorber_element": "Fe"}
        (feff_workdir / "_feff_aggregate_config.json").write_text(json.dumps(config))

        aggregate_paths.main()

        import h5py

        with h5py.File(feff_workdir / "contributions_raw.h5", "r") as hf:
            pg = hf["paths"]
            m_k = len(pg["k_grid_params"])
            n_paths = len(pg["r_eff"])
            # feff_data must be (n_paths, m_k, 6)
            assert pg["feff_data"].shape == (n_paths, m_k, 6)
