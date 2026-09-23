"""Tests for the batch CalcJob, its parser, and the remote driver.

The batch path is what runs on the cluster, so the pieces that only execute
there — worker sizing, the shell wrapper, the run timeout — are exercised here
against the real scripts rather than being taken on trust.

Note the code inversion this CalcJob inherits: ``code`` is the **Python
interpreter** and ``feff_code`` is FEFF.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from aiida import orm

from aiida_feff.calculations.feff_batch import (
    _RUN_TIMEOUT_FRACTION,
    BATCH_CONFIG,
    BATCH_DRIVER,
    _snap_label,
    workers_from_resources,
)
from tests.helpers import origin_atoms


class TestWorkersFromResources:
    """Worker count comes from the allocation, not from SLURM-only env vars."""

    @pytest.mark.parametrize(
        ("resources", "expected"),
        [
            ({"num_machines": 1, "num_mpiprocs_per_machine": 64}, 64),
            ({"num_machines": 2, "num_mpiprocs_per_machine": 8}, 16),
            ({"num_machines": 1, "num_cores_per_machine": 12}, 12),
            ({"tot_num_mpiprocs": 40}, 40),
            ({"num_machines": 3}, 3),
            ({}, 1),
            (None, 1),
        ],
    )
    def test_derives_count(self, resources, expected):
        assert workers_from_resources(resources) == expected

    def test_never_returns_zero(self):
        assert workers_from_resources({"num_machines": 0}) == 1


class TestResolveNWorkers:
    """The driver prefers the configured value and parses odd env formats."""

    @pytest.fixture()
    def driver(self):
        from tests.conftest import import_remote_script

        return import_remote_script("_run_batch.py")

    def test_configured_value_wins(self, driver, monkeypatch):
        monkeypatch.setenv("SLURM_CPUS_ON_NODE", "8")
        assert driver.resolve_n_workers(64) == 64

    def test_falls_back_to_environment(self, driver, monkeypatch):
        monkeypatch.delenv("SLURM_NTASKS", raising=False)
        monkeypatch.setenv("SLURM_CPUS_ON_NODE", "12")
        assert driver.resolve_n_workers(None) == 12

    def test_parses_slurm_multiplier_syntax(self, driver, monkeypatch):
        # SLURM reports heterogeneous allocations as e.g. "32(x2)"; int() alone
        # raises on that and would abort the whole batch.
        monkeypatch.setenv("SLURM_CPUS_ON_NODE", "32(x2)")
        assert driver.resolve_n_workers(None) == 32

    def test_defaults_to_one_without_any_hint(self, driver, monkeypatch):
        for var in ("SLURM_CPUS_ON_NODE", "SLURM_NTASKS", "NCPUS", "LSB_DJOB_NUMPROC"):
            monkeypatch.delenv(var, raising=False)
        assert driver.resolve_n_workers(None) == 1


class TestRunFeffOne:
    """The per-run wrapper, exercised with a stand-in for the FEFF binary."""

    @pytest.fixture()
    def driver(self):
        from tests.conftest import import_remote_script

        return import_remote_script("_run_batch.py")

    def test_captures_stdout_and_stderr(self, driver, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        label = _snap_label(0, 0)
        driver.run_feff_one(label, "echo hello; echo oops >&2", "", "")
        assert (tmp_path / label / "log.dat").read_text().strip() == "hello"
        assert "oops" in (tmp_path / label / "stderr.txt").read_text()

    def test_non_zero_exit_raises(self, driver, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(RuntimeError, match="FEFF exited"):
            driver.run_feff_one(_snap_label(0, 0), "exit 3", "", "")

    def test_timeout_raises_and_keeps_partial_output(self, driver, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        label = _snap_label(0, 0)
        with pytest.raises(RuntimeError, match="run timeout"):
            driver.run_feff_one(label, "sleep 30", "", "", timeout=0.5)
        assert (tmp_path / label / "log.dat").exists()

    def test_prepend_text_runs_before_feff(self, driver, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        label = _snap_label(0, 0)
        driver.run_feff_one(label, "echo second", "echo first", "")
        assert (tmp_path / label / "log.dat").read_text().split() == ["first", "second"]

    def test_potentials_are_copied_into_the_run_dir(self, driver, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        pot_dir = tmp_path / "potentials" / "site_0002"
        pot_dir.mkdir(parents=True)
        (pot_dir / "pot.pad").write_text("potential")
        label = _snap_label(7, 2)
        driver.run_feff_one(label, "true", "", "")
        assert (tmp_path / label / "pot.pad").read_text() == "potential"


class TestDriverExitStatus:
    """A batch where nothing ran must fail visibly, not silently succeed."""

    def _run(self, tmp_path, command, pairs=((0, 0), (1, 0))):
        from aiida_feff.calculations import _run_batch

        (tmp_path / BATCH_CONFIG).write_text(
            json.dumps(
                {
                    "pairs": [list(p) for p in pairs],
                    "feff_executable": command,
                    "feff_prepend": "",
                    "feff_append": "",
                    "n_workers": 1,
                    "do_aggregate": False,
                    "threshold": 0.0,
                    "run_timeout_seconds": 30.0,
                }
            )
        )
        return subprocess.run(
            [sys.executable, str(Path(_run_batch.__file__))],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_all_runs_failing_exits_non_zero(self, tmp_path):
        assert self._run(tmp_path, "exit 1").returncode != 0

    def test_partial_failure_still_exits_zero(self, tmp_path):
        # AiiDA's parser handles missing snapshots; only a total wipe-out is a
        # job failure.
        result = self._run(
            tmp_path, '[ "$(basename "$PWD")" = "snap_0000_site_0000" ] && exit 1; true'
        )
        assert result.returncode == 0

    def test_all_runs_succeeding_exits_zero(self, tmp_path):
        result = self._run(tmp_path, "true")
        assert result.returncode == 0
        assert (tmp_path / _snap_label(0, 0) / "log.dat").exists()

    def test_missing_config_exits_non_zero(self, tmp_path):
        from aiida_feff.calculations import _run_batch

        result = subprocess.run(
            [sys.executable, str(Path(_run_batch.__file__))],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode != 0
        assert "batch_config.json" in result.stderr

    def _run_driver(self, tmp_path, **config_overrides):
        """Drive the batch script over a stand-in that writes chi.dat plus bulk."""
        from aiida_feff.calculations import _run_batch

        command = (
            'python3 -c "import pathlib, numpy as np; p=pathlib.Path.cwd(); '
            "(p/'chi.dat').write_text('# k chi\\n' + '\\n'.join(f'{k:.2f} {np.sin(k):.6f}' "
            "for k in np.linspace(0.05, 20.0, 100))); (p/'log.dat').write_text('ok'); "
            "(p/'xmu.dat').write_text('# xmu'); "
            "[(p/f'feff{i:04d}.dat').write_text('bulk') for i in range(1, 6)]; "
            "(p/'phase.pad').write_text('bulk')\""
        )
        config = {
            "pairs": [[0, 0], [1, 0]],
            "feff_executable": command,
            "feff_prepend": "",
            "feff_append": "",
            "n_workers": 1,
            "do_aggregate": False,
            "threshold": 0.0,
            "run_timeout_seconds": 30.0,
            **config_overrides,
        }
        (tmp_path / BATCH_CONFIG).write_text(json.dumps(config))
        return subprocess.run(
            [sys.executable, str(Path(_run_batch.__file__))],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_clean_scratch_reclaims_bulk_but_keeps_retrieved_files(self, tmp_path):
        """clean_scratch must remove only what the calcjob never retrieves.

        The feffNNNN.dat path files and potential binaries are what exhaust
        scratch; xmu.dat / chi.dat / log.dat are what the parser reads. Deleting
        the latter would change parsed physics, so assert both halves.
        """
        keep = ["chi.dat", "files.dat", "log.dat", "paths.dat", "stderr.txt"]
        res = self._run_driver(
            tmp_path, clean_scratch=True, stream_chunk_size=1, scratch_keep_files=keep
        )
        assert res.returncode == 0, res.stderr
        assert (tmp_path / "batch_shard.h5").exists()

        for label in (_snap_label(0, 0), _snap_label(1, 0)):
            snap = tmp_path / label
            assert snap.is_dir(), "stripping must not remove the run directory itself"
            assert (snap / "chi.dat").exists()
            assert (snap / "log.dat").exists()
            assert not list(snap.glob("feff[0-9]*.dat")), "path files should be reclaimed"
            assert not (snap / "phase.pad").exists()
            assert not (snap / "feff.inp").exists()

    def test_clean_scratch_off_keeps_everything(self, tmp_path):
        """The default must not delete anything -- an option that cleans when off is a trap."""
        res = self._run_driver(tmp_path, clean_scratch=False, stream_chunk_size=1)
        assert res.returncode == 0, res.stderr
        snap = tmp_path / _snap_label(0, 0)
        assert len(list(snap.glob("feff[0-9]*.dat"))) == 5
        assert (snap / "phase.pad").exists()

    def test_clean_scratch_without_keep_list_refuses_to_strip(self, tmp_path):
        """An empty keep-set would delete the outputs; the driver must refuse."""
        res = self._run_driver(
            tmp_path, clean_scratch=True, stream_chunk_size=1, scratch_keep_files=[]
        )
        assert res.returncode != 0
        assert "scratch_keep_files" in res.stderr
        # Refused up front, before any FEFF ran, so nothing was produced to destroy.
        assert not (tmp_path / _snap_label(0, 0)).exists()
        assert not (tmp_path / "batch_shard.h5").exists()

    def test_shard_is_identical_with_and_without_clean_scratch(self, tmp_path_factory):
        """Stripping is lossless, so the shard contents must not depend on it."""
        import h5py

        keep = ["chi.dat", "files.dat", "log.dat", "paths.dat", "stderr.txt"]

        def chis(clean):
            d = tmp_path_factory.mktemp(f"clean_{clean}")
            res = self._run_driver(
                d, clean_scratch=clean, stream_chunk_size=1, scratch_keep_files=keep
            )
            assert res.returncode == 0, res.stderr
            with h5py.File(d / "batch_shard.h5", "r") as f:
                return {name: np.array(g["chi"]) for name, g in f["tasks"].items()}

        on, off = chis(True), chis(False)
        assert set(on) == set(off) and on
        for name in on:
            np.testing.assert_array_equal(on[name], off[name])


@pytest.fixture()
def batch_inputs(generate_trajectory, generate_feff_parameters, aiida_localhost):
    """Inputs for a two-pair FeffBatchCalculation."""
    traj = generate_trajectory(n_frames=3, reps=1).store()
    python_code = orm.InstalledCode(
        label="python-test", computer=aiida_localhost, filepath_executable="/usr/bin/python3"
    ).store()
    feff_code = orm.InstalledCode(
        label="feff-test", computer=aiida_localhost, filepath_executable="/opt/feff/feff8l"
    )
    feff_code.prepend_text = "module load feff/8.5"
    feff_code.store()
    return {
        "code": python_code,  # the Python interpreter, not FEFF
        "feff_code": feff_code,
        "trajectory": traj,
        "frame_indices": orm.List([0, 1]),
        "site_indices": orm.List([0, 1]),
        "parameters": generate_feff_parameters(),
        "metadata": {
            "options": {
                "resources": {"num_machines": 1, "num_mpiprocs_per_machine": 16},
                "max_wallclock_seconds": 3600,
            }
        },
    }


class TestBatchPrepareForSubmission:
    """What the CalcJob actually writes into the sandbox."""

    def _prepare(self, generate_calc_job, fixture_sandbox, inputs):
        return generate_calc_job(
            folder=fixture_sandbox, entry_point_name="feff.feff_batch", inputs=inputs
        )

    def test_one_subfolder_per_pair(self, generate_calc_job, fixture_sandbox, batch_inputs):
        self._prepare(generate_calc_job, fixture_sandbox, batch_inputs)
        for frame, site in ((0, 0), (1, 1)):
            label = _snap_label(frame, site)
            assert fixture_sandbox.isfile(f"{label}/feff.inp")

    def test_absorbing_atom_differs_per_site(
        self, generate_calc_job, fixture_sandbox, batch_inputs
    ):
        self._prepare(generate_calc_job, fixture_sandbox, batch_inputs)
        texts = {
            label: Path(fixture_sandbox.get_abs_path(f"{label}/feff.inp")).read_text()
            for label in (_snap_label(0, 0), _snap_label(1, 1))
        }
        assert texts[_snap_label(0, 0)] != texts[_snap_label(1, 1)]

    def test_same_frame_different_site_gives_a_different_absorber(
        self, generate_calc_job, fixture_sandbox, batch_inputs
    ):
        """Two sites in the *same* frame must produce different inputs.

        The test above compares (frame 0, site 0) against (frame 1, site 1),
        which differ in the frame as well, so it passes even if site_indices
        were ignored entirely. Holding the frame fixed isolates the site.
        """
        inputs = copy.deepcopy(batch_inputs)
        inputs["frame_indices"] = orm.List([0, 0])
        inputs["site_indices"] = orm.List([0, 1])
        self._prepare(generate_calc_job, fixture_sandbox, inputs)

        first = Path(fixture_sandbox.get_abs_path(f"{_snap_label(0, 0)}/feff.inp")).read_text()
        second = Path(fixture_sandbox.get_abs_path(f"{_snap_label(0, 1)}/feff.inp")).read_text()
        assert first != second

        # Both must put *their own* absorber at the origin, which is the
        # actual contract -- not merely that the two files differ somehow.
        for text in (first, second):
            assert len(origin_atoms(text)) == 1, "expected exactly one atom at the origin"

    def test_config_records_the_feff_environment(
        self, generate_calc_job, fixture_sandbox, batch_inputs
    ):
        self._prepare(generate_calc_job, fixture_sandbox, batch_inputs)
        config = json.loads(Path(fixture_sandbox.get_abs_path(BATCH_CONFIG)).read_text())
        assert config["feff_executable"] == "/opt/feff/feff8l"
        assert config["feff_prepend"] == "module load feff/8.5"
        assert config["pairs"] == [[0, 0], [1, 1]]

    def test_workers_default_to_the_allocation(
        self, generate_calc_job, fixture_sandbox, batch_inputs
    ):
        self._prepare(generate_calc_job, fixture_sandbox, batch_inputs)
        config = json.loads(Path(fixture_sandbox.get_abs_path(BATCH_CONFIG)).read_text())
        assert config["n_workers"] == 16

    def test_run_timeout_is_bounded_by_the_wallclock(
        self, generate_calc_job, fixture_sandbox, batch_inputs
    ):
        self._prepare(generate_calc_job, fixture_sandbox, batch_inputs)
        config = json.loads(Path(fixture_sandbox.get_abs_path(BATCH_CONFIG)).read_text())
        # Pin the actual value. `0 < x < 3600` held for any fraction in (0, 1)
        # and so pinned nothing -- including a fraction of 0.999 that would
        # leave no time to retrieve the chunk's results after a hung run.
        wallclock = batch_inputs["metadata"]["options"]["max_wallclock_seconds"]
        assert config["run_timeout_seconds"] == pytest.approx(wallclock * _RUN_TIMEOUT_FRACTION)
        # The margin has to be big enough to actually retrieve results in.
        assert wallclock - config["run_timeout_seconds"] >= 60

    def test_run_timeout_is_absent_when_the_wallclock_is(
        self, generate_calc_job, fixture_sandbox, batch_inputs
    ):
        """No wallclock means no per-run timeout -- a hung FEFF blocks the chunk.

        This branch silently disables the timeout on a cluster, so it needs
        to be a deliberate, visible behaviour rather than an untested one.
        """
        batch_inputs = copy.deepcopy(batch_inputs)
        del batch_inputs["metadata"]["options"]["max_wallclock_seconds"]
        self._prepare(generate_calc_job, fixture_sandbox, batch_inputs)
        config = json.loads(Path(fixture_sandbox.get_abs_path(BATCH_CONFIG)).read_text())
        assert config["run_timeout_seconds"] is None

    def test_driver_script_is_shipped(self, generate_calc_job, fixture_sandbox, batch_inputs):
        self._prepare(generate_calc_job, fixture_sandbox, batch_inputs)
        assert fixture_sandbox.isfile(BATCH_DRIVER)

    def test_feff_inp_uses_lf_endings(self, generate_calc_job, fixture_sandbox, batch_inputs):
        self._prepare(generate_calc_job, fixture_sandbox, batch_inputs)
        raw = Path(fixture_sandbox.get_abs_path(f"{_snap_label(0, 0)}/feff.inp")).read_bytes()
        assert b"\r" not in raw


class TestBatchInputValidation:
    """Rejected before a node exists, because prepare_for_submission cannot."""

    def _submit(self, inputs):
        from aiida.engine.utils import instantiate_process
        from aiida.manage import get_manager
        from aiida.plugins import CalculationFactory

        runner = get_manager().get_runner()
        return instantiate_process(runner, CalculationFactory("feff.feff_batch"), **inputs)

    def test_mismatched_index_lists_rejected(self, batch_inputs):
        batch_inputs["site_indices"] = orm.List([0])
        with pytest.raises(ValueError, match="same length"):
            self._submit(batch_inputs)

    def test_frame_index_out_of_range_rejected(self, batch_inputs):
        batch_inputs["frame_indices"] = orm.List([0, 99])
        with pytest.raises(ValueError, match="outside the trajectory"):
            self._submit(batch_inputs)

    def test_empty_pair_list_rejected(self, batch_inputs):
        batch_inputs["frame_indices"] = orm.List([])
        batch_inputs["site_indices"] = orm.List([])
        with pytest.raises(ValueError, match="must not be empty"):
            self._submit(batch_inputs)

    def test_zero_workers_rejected(self, batch_inputs):
        batch_inputs["n_workers"] = orm.Int(0)
        with pytest.raises(ValueError, match="n_workers"):
            self._submit(batch_inputs)


class TestBatchParser:
    """Partial failure is tolerated; total failure is not."""

    @staticmethod
    def _parse(retrieved: dict[str, str], aiida_profile_clean):
        import io

        from aiida.common.links import LinkType
        from aiida.orm import CalcJobNode, FolderData
        from aiida.plugins import ParserFactory

        folder = FolderData()
        for name, content in retrieved.items():
            folder.base.repository.put_object_from_filelike(io.BytesIO(content.encode()), name)
        folder.store()

        node = CalcJobNode()
        node.set_process_type("aiida.calculations:feff.feff_batch")
        node.store()
        folder.base.links.add_incoming(node, link_type=LinkType.CREATE, link_label="retrieved")

        parser = ParserFactory("feff.feff_batch")(node)
        exit_code = parser.parse()
        # Parser.outputs keys dynamic namespaces as flat "namespace.key" labels.
        namespaced: dict[str, dict[str, object]] = {}
        for label, out_node in parser.outputs.items():
            namespace, _, key = label.partition(".")
            namespaced.setdefault(namespace, {})[key] = out_node
        return namespaced, (exit_code.status if exit_code else 0)

    CHI = (
        "# FEFF chi.dat\n"
        "#  k         chi(k)    |chi|     phase\n"
        + "\n".join(
            f"{k:8.3f} {np.sin(k) * 0.1:12.6f} 0.000000 0.000000"
            for k in np.linspace(0.05, 15.0, 60)
        )
        + "\n"
    )

    def test_one_output_per_successful_run(self, aiida_profile_clean):
        retrieved = {
            f"{_snap_label(0, 0)}/chi.dat": self.CHI,
            f"{_snap_label(1, 0)}/chi.dat": self.CHI,
        }
        outputs, status = self._parse(retrieved, aiida_profile_clean)
        assert status == 0
        assert set(outputs["xas_data"]) == {_snap_label(0, 0), _snap_label(1, 0)}

    def test_missing_run_is_skipped_not_fatal(self, aiida_profile_clean):
        retrieved = {
            f"{_snap_label(0, 0)}/chi.dat": self.CHI,
            f"{_snap_label(1, 0)}/stderr.txt": "FEFF crashed",
        }
        outputs, status = self._parse(retrieved, aiida_profile_clean)
        assert status == 0
        assert set(outputs["xas_data"]) == {_snap_label(0, 0)}

    def test_no_snapshots_at_all_is_fatal(self, aiida_profile_clean):
        _outputs, status = self._parse({"batch_err.log": "driver died"}, aiida_profile_clean)
        assert status != 0

    def test_feff_version_is_recorded(self, aiida_profile_clean):
        retrieved = {
            f"{_snap_label(0, 0)}/chi.dat": self.CHI,
            f"{_snap_label(0, 0)}/log.dat": "  Feff 8.50L\n  more banner\n",
        }
        outputs, status = self._parse(retrieved, aiida_profile_clean)
        assert status == 0
        node = outputs["xas_data"][_snap_label(0, 0)]
        assert node.base.attributes.get("feff_version") == "Feff8.50L"


class TestSnapDirNameParsing:
    """Snapshot directory names carry the (frame, site) identity of a run.

    A name that fails to parse must yield (None, None) so the caller can skip
    it, rather than silently mapping the run onto frame 0 / site 0 and
    corrupting the ensemble.
    """

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("snap_0000_site_0000", (0, 0)),
            ("snap_0003_site_0012", (3, 12)),
            ("snap_1234_site_5678", (1234, 5678)),
        ],
    )
    def test_well_formed_names(self, name, expected):
        from aiida_feff.parsers.feff_batch import _parse_snap_dir_name

        assert _parse_snap_dir_name(name) == expected

    @pytest.mark.parametrize(
        "name",
        [
            "snap_0000",  # no site part
            "snap_abcd_site_0000",  # non-numeric frame
            "snap_0000_site_xyz",  # non-numeric site
            "not_a_snapshot",
            "",
            "snap__site_",
        ],
    )
    def test_malformed_names_return_none_rather_than_guessing(self, name):
        from aiida_feff.parsers.feff_batch import _parse_snap_dir_name

        assert _parse_snap_dir_name(name) == (None, None)

    def test_round_trips_with_the_label_writer(self):
        """The parser must invert the same function the calcjob writes with."""
        from aiida_feff.parsers.feff_batch import _parse_snap_dir_name

        for frame, site in ((0, 0), (7, 3), (42, 11)):
            assert _parse_snap_dir_name(_snap_label(frame, site)) == (frame, site)
