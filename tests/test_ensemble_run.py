"""End-to-end runs of EnsembleExafsWorkChain against a stand-in for FEFF.

The workchain outline — trajectory splitting, absorber fan-out, potential
reuse, averaging, path merging — is the part of this plugin that a unit test
cannot reach, and it is also the part that only fails after a queue wait.  Here
a shell script plays the FEFF binary, writing plausible ``xmu.dat`` output, so
the whole outline runs in-process in seconds.
"""

from __future__ import annotations

import stat
import sys
import textwrap

import numpy as np
import pytest
from aiida import orm
from aiida.engine import run_get_node

from aiida_feff.workflows.ensemble import EnsembleExafsWorkChain

# A FEFF stand-in: writes the columns the parser reads, with an r-dependent
# oscillation so different sites give different spectra.
FAKE_FEFF = textwrap.dedent("""\
    #!/bin/bash
    python3 - <<'PY'
    import math
    rows = []
    for i in range(120):
        omega = -20.0 + i * 2.0
        k = math.sqrt(max(omega, 0.0) * 0.2624684)
        mu = 1.0 + 0.1 * math.sin(2 * k * 2.5) * math.exp(-0.05 * k * k)
        rows.append(f"{omega:10.4f} {omega:10.4f} {k:10.4f} {mu:10.5f} {1.0:10.5f} {0.0:10.5f}")
    header = [
        "# Feff8L (EXAFS)  0.1",
        "# e0 = 7112.00",
        "#   omega      e        k        mu       mu0      chi",
    ]
    open("xmu.dat", "w").write("\\n".join(header + rows) + "\\n")
    open("files.dat", "w").write("Feff8L (EXAFS)  0.1\\n")
    PY
    """)


@pytest.fixture()
def fake_feff_code(tmp_path_factory, aiida_localhost):
    """An InstalledCode pointing at the stand-in script."""
    script = tmp_path_factory.mktemp("fakefeff") / "feff.sh"
    script.write_text(FAKE_FEFF)
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return orm.InstalledCode(
        label="fake-feff", computer=aiida_localhost, filepath_executable=str(script)
    ).store()


@pytest.fixture()
def two_site_trajectory(aiida_profile):
    """Three frames of a two-atom Fe cell, both atoms valid absorbers."""
    rng = np.random.default_rng(0)
    a = 2.87
    eq = np.array([[0.0, 0.0, 0.0], [a / 2, a / 2, a / 2]])
    positions = eq[np.newaxis] + rng.normal(scale=0.02, size=(3, 2, 3))
    traj = orm.TrajectoryData()
    traj.set_array("positions", positions)
    traj.set_array("cells", np.tile(np.eye(3) * a, (3, 1, 1)))
    traj.set_array("steps", np.arange(3))
    traj.base.attributes.set("symbols", ["Fe", "Fe"])
    return traj.store()


OPTIONS = {
    "resources": {"num_machines": 1},
    "max_wallclock_seconds": 120,
    "withmpi": False,
}


def run_workchain(**overrides):
    """Run the workchain in-process with localhost scheduler options."""
    return run_get_node(EnsembleExafsWorkChain, options=orm.Dict(OPTIONS), **overrides)


@pytest.mark.usefixtures("aiida_profile_clean")
class TestSingleSiteEnsemble:
    def test_averages_every_frame(self, fake_feff_code, two_site_trajectory):
        from aiida_feff.data.parameters import FeffParameters

        results, node = run_workchain(
            code=fake_feff_code,
            trajectory=two_site_trajectory,
            parameters=FeffParameters(dict={"edge": "K", "radius": 4.0, "absorbing_atom": 0}),
        )
        assert node.is_finished_ok, node.exit_message
        assert results["n_failed"].value == 0
        assert "all" in results["averaged_xas"]
        assert results["averaged_xas"]["all"].base.attributes.get("n_snapshots") == 3

    def test_sample_interval_reduces_the_fan_out(self, fake_feff_code, two_site_trajectory):
        from aiida_feff.data.parameters import FeffParameters

        results, node = run_workchain(
            code=fake_feff_code,
            trajectory=two_site_trajectory,
            sample_interval=orm.Int(2),
            parameters=FeffParameters(dict={"edge": "K", "radius": 4.0, "absorbing_atom": 0}),
        )
        assert node.is_finished_ok, node.exit_message
        assert results["averaged_xas"]["all"].base.attributes.get("n_snapshots") == 2


@pytest.mark.usefixtures("aiida_profile_clean")
class TestMultiSiteEnsemble:
    def test_one_average_per_site_plus_a_grand_average(self, fake_feff_code, two_site_trajectory):
        from aiida_feff.data.parameters import FeffParameters

        results, node = run_workchain(
            code=fake_feff_code,
            trajectory=two_site_trajectory,
            parameters=FeffParameters(dict={"edge": "K", "radius": 4.0, "absorbing_atoms": "Fe"}),
        )
        assert node.is_finished_ok, node.exit_message
        averaged = results["averaged_xas"]
        assert set(averaged) == {"site_0000", "site_0001", "all"}
        # Grand average spans both sites and all three frames.
        assert averaged["all"].base.attributes.get("n_snapshots") == 6
        assert averaged["site_0000"].base.attributes.get("n_snapshots") == 3


@pytest.mark.usefixtures("aiida_profile_clean")
class TestPotentialReuse:
    def test_potentials_only_child_is_accepted_without_a_spectrum(
        self, fake_feff_code, two_site_trajectory
    ):
        """The potentials run writes no xmu.dat, and that must count as success."""
        from aiida_feff.data.parameters import FeffParameters

        results, node = run_workchain(
            code=fake_feff_code,
            trajectory=two_site_trajectory,
            precompute_potentials=orm.Bool(True),
            parameters=FeffParameters(dict={"edge": "K", "radius": 4.0, "absorbing_atom": 0}),
        )
        assert node.is_finished_ok, node.exit_message
        assert results["n_failed"].value == 0

        labels = [c.label for c in node.called]
        assert any(label.startswith("pot_site_") for label in labels)


@pytest.mark.usefixtures("aiida_profile_clean")
class TestFailureHandling:
    @pytest.fixture()
    def broken_feff(self, tmp_path_factory, aiida_localhost):
        script = tmp_path_factory.mktemp("brokenfeff") / "feff.sh"
        script.write_text("#!/bin/bash\nexit 1\n")
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
        return orm.InstalledCode(
            label="broken-feff", computer=aiida_localhost, filepath_executable=str(script)
        ).store()

    def test_all_children_failing_is_reported(self, broken_feff, two_site_trajectory):
        from aiida_feff.data.parameters import FeffParameters

        _results, node = run_workchain(
            code=broken_feff,
            trajectory=two_site_trajectory,
            parameters=FeffParameters(dict={"edge": "K", "radius": 4.0, "absorbing_atom": 0}),
        )
        assert not node.is_finished_ok
        assert node.exit_status == EnsembleExafsWorkChain.exit_codes.ERROR_ALL_FAILED.status


@pytest.mark.usefixtures("aiida_profile_clean")
class TestInputValidation:
    def test_structures_and_trajectory_are_mutually_exclusive(
        self, fake_feff_code, two_site_trajectory, generate_structure
    ):
        from aiida_feff.data.parameters import FeffParameters

        with pytest.raises(ValueError, match="not both"):
            run_workchain(
                code=fake_feff_code,
                trajectory=two_site_trajectory,
                structures={"frame_000000": generate_structure().store()},
                parameters=FeffParameters(dict={"edge": "K", "radius": 4.0}),
            )

    def test_neither_input_is_rejected(self, fake_feff_code):
        from aiida_feff.data.parameters import FeffParameters

        with pytest.raises(ValueError, match="must be supplied"):
            run_workchain(
                code=fake_feff_code,
                parameters=FeffParameters(dict={"edge": "K", "radius": 4.0}),
            )


@pytest.mark.usefixtures("aiida_profile_clean")
class TestBatchMode:
    """One scheduler job per chunk, with the code/feff_code inversion applied."""

    @pytest.fixture()
    def python_code(self, aiida_localhost):
        return orm.InstalledCode(
            label="python-runner",
            computer=aiida_localhost,
            filepath_executable=sys.executable,
        ).store()

    def test_batch_run_produces_one_spectrum_per_pair(
        self, fake_feff_code, python_code, two_site_trajectory
    ):
        from aiida_feff.data.parameters import FeffParameters

        results, node = run_workchain(
            code=fake_feff_code,  # the workchain swaps these round for the batch CalcJob
            python_code=python_code,
            trajectory=two_site_trajectory,
            batch_size=orm.Int(2),
            parameters=FeffParameters(dict={"edge": "K", "radius": 4.0, "absorbing_atom": 0}),
        )
        assert node.is_finished_ok, node.exit_message
        assert results["n_failed"].value == 0
        # Three frames in chunks of two: two batch jobs, three spectra.
        assert results["averaged_xas"]["all"].base.attributes.get("n_snapshots") == 3
        batch_children = [c for c in node.called if c.label.startswith("batch_")]
        assert len(batch_children) == 2

    def test_batch_and_serial_paths_agree(self, fake_feff_code, python_code, two_site_trajectory):
        """Batching is a scheduling choice; it must not change the physics."""
        from aiida_feff.data.parameters import FeffParameters

        def average(**extra):
            params = FeffParameters(dict={"edge": "K", "radius": 4.0, "absorbing_atom": 0})
            results, node = run_workchain(
                code=fake_feff_code,
                trajectory=two_site_trajectory,
                parameters=params,
                **extra,
            )
            assert node.is_finished_ok, node.exit_message
            return results["averaged_xas"]["all"]

        serial = average()
        batched = average(python_code=python_code, batch_size=orm.Int(3))
        np.testing.assert_allclose(
            batched.get_array("chi_k"), serial.get_array("chi_k"), rtol=1e-10
        )

    def test_batch_size_without_python_code_is_refused(self, fake_feff_code, two_site_trajectory):
        from aiida_feff.data.parameters import FeffParameters

        _results, node = run_workchain(
            code=fake_feff_code,
            trajectory=two_site_trajectory,
            batch_size=orm.Int(2),
            parameters=FeffParameters(dict={"edge": "K", "radius": 4.0, "absorbing_atom": 0}),
        )
        expected = EnsembleExafsWorkChain.exit_codes.ERROR_MISSING_AGGREGATION_CODE
        assert node.exit_status == expected.status
