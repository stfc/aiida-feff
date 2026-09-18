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

# A FEFF stand-in that reads its own input.
#
# The previous version hardcoded R = 2.5 and never opened feff.inp, so every
# frame, every site and every absorber produced a byte-identical xmu.dat.
# That silently defeated the tests built on top of it: batch-vs-serial
# equivalence, per-site averaging and snapshot de-duplication all compared
# copies of one array and could not fail. It also could not catch a wrong
# frame->spectrum or site->spectrum assignment.
#
# This version derives R from the geometry FEFF was actually handed -- the
# nearest-neighbour distance in the ATOMS block, with the absorber at the
# origin by construction -- so each snapshot has a *known correct* spectrum
# and tests can assert identity rather than mere difference.
#
# Note on resolution: the trajectory fixture jitters positions by 0.02 A, so
# frame-to-frame Delta-R is ~0.03 A, just under the 0.0307 A chi(R) bin. Frames
# are therefore NOT separable in R space. In k space the phase difference
# 2*k*Delta-R reaches ~0.9 rad by k=15, which is comfortably resolvable, so
# every assertion below works in k space.
FAKE_FEFF = textwrap.dedent("""\
    #!/bin/bash
    python3 - <<'PY'
    import math

    def nearest_neighbour_distance(path="feff.inp"):
        \"\"\"Smallest absorber-scatterer distance in the ATOMS block.

        The absorber is ipot 0 at the origin, so this is just the smallest
        non-zero radius. Falls back to a sentinel only if ATOMS is missing,
        which would itself be a bug worth seeing in the output.
        \"\"\"
        try:
            lines = open(path).read().splitlines()
        except OSError:
            return -1.0
        try:
            start = next(i for i, l in enumerate(lines) if l.strip().startswith("ATOMS"))
        except StopIteration:
            return -1.0
        radii = []
        for line in lines[start + 1:]:
            if line.strip().startswith("END"):
                break
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            except ValueError:
                continue
            r = math.sqrt(x * x + y * y + z * z)
            if r > 1e-9:
                radii.append(r)
        return min(radii) if radii else -1.0

    R = nearest_neighbour_distance()
    rows = []
    for i in range(120):
        omega = -20.0 + i * 2.0
        k = math.sqrt(max(omega, 0.0) * 0.2624684)
        chi = 0.1 * math.sin(2 * k * R) * math.exp(-0.05 * k * k)
        mu = 1.0 + chi
        rows.append(f"{omega:10.4f} {omega:10.4f} {k:10.4f} {mu:10.5f} {1.0:10.5f} {0.0:10.5f}")

    # chi.dat gets its own uniform k grid rather than reusing the omega grid
    # above. The archive resamples chi.dat onto k = 0.05, 0.10, ... so a
    # sparse non-uniform source would leave interpolation error in every
    # comparison and force tests to use tolerances loose enough to hide a
    # real merge bug. Sampling the same grid the archive uses makes the
    # round trip exact.
    chi_rows = []
    for i in range(1, 401):
        k = 0.05 * i
        chi = 0.1 * math.sin(2 * k * R) * math.exp(-0.05 * k * k)
        chi_rows.append(f"{k:10.4f} {chi:12.6e} {abs(chi):12.6e} {0.0:10.4f}")

    header = [
        "# Feff8L (EXAFS)  0.1",
        "# e0 = 7112.00",
        "#   omega      e        k        mu       mu0      chi",
    ]
    open("xmu.dat", "w").write("\\n".join(header + rows) + "\\n")
    # chi.dat is what the batch shard is built from, so the stand-in has to
    # write it too or the batch tests silently skip the archive path.
    chi_header = ["# Feff8L (EXAFS)  0.1", "#    k          chi          mag        phase"]
    open("chi.dat", "w").write("\\n".join(chi_header + chi_rows) + "\\n")
    open("files.dat", "w").write("Feff8L (EXAFS)  0.1\\n")
    # Record the geometry this run actually saw, so tests can verify that the
    # right structure reached the right snapshot directory.
    open("stand_in_R.txt", "w").write(f"{R!r}\\n")
    PY
    """)


def expected_chi(k, r):
    """The stand-in's spectrum for a given nearest-neighbour distance.

    Kept in the test module so assertions state the expected physics rather
    than re-deriving it from whatever the stand-in happened to write.
    """
    return 0.1 * np.sin(2 * k * r) * np.exp(-0.05 * k * k)


def nn_distance(positions, cell, site):
    """Minimum-image nearest-neighbour distance from ``site`` in a cell."""
    deltas = positions - positions[site]
    frac = deltas @ np.linalg.inv(cell)
    frac -= np.round(frac)
    radii = np.linalg.norm(frac @ cell, axis=1)
    return float(radii[radii > 1e-9].min())


def _frame_nn_distances(trajectory, site):
    """Nearest-neighbour distance at ``site`` for every frame of a trajectory."""
    positions = trajectory.get_array("positions")
    cells = trajectory.get_array("cells")
    return [nn_distance(positions[f], cells[f], site) for f in range(len(positions))]


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


def feff_rejecting_short_bonds(directory, computer, min_r=2.0, label="picky-feff"):
    """A stand-in that fails on snapshots whose nearest bond is shorter than ``min_r``.

    Partial failure is the normal outcome of a real ensemble run -- one
    snapshot lands on a pathological geometry -- but every failure test here
    used to make *all* children fail, so the partial-failure and
    potentials-failure exit codes were never reached.

    Failure is keyed on the *input geometry* rather than on invocation
    order. An earlier version of this helper counted invocations in a shared
    file, which silently did not work: the workchain runs its children
    concurrently, so all three read the counter after all three had
    incremented it, every run saw N=3, and no run ever failed. Keying on the
    input makes which snapshot fails deterministic and independent of
    scheduling.
    """
    guard = textwrap.dedent(f"""\
        #!/bin/bash
        MIN_R={min_r}
        python3 - <<'CHECK' || exit 1
        import math, sys
        lines = open("feff.inp").read().splitlines()
        start = next(i for i, l in enumerate(lines) if l.strip().startswith("ATOMS"))
        radii = []
        for line in lines[start + 1:]:
            if line.strip().startswith("END"):
                break
            p = line.split()
            if len(p) < 4:
                continue
            try:
                r = math.sqrt(float(p[0])**2 + float(p[1])**2 + float(p[2])**2)
            except ValueError:
                continue
            if r > 1e-9:
                radii.append(r)
        sys.exit(1 if radii and min(radii) < {min_r} else 0)
        CHECK
        """)
    script = directory / f"{label}.sh"
    script.write_text(guard + FAKE_FEFF.split("\n", 1)[1])
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return orm.InstalledCode(
        label=label, computer=computer, filepath_executable=str(script)
    ).store()


def trajectory_with_bad_frames(bad_frames, n_frames=3, a=2.87):
    """Trajectory where ``bad_frames`` have a collapsed bond (~1.04 A).

    Good frames sit at the normal bcc nearest-neighbour distance (~2.48 A),
    so a stand-in thresholding at 2.0 A separates them cleanly.
    """
    rng = np.random.default_rng(0)
    positions = np.empty((n_frames, 2, 3))
    for frame in range(n_frames):
        second = [0.6, 0.6, 0.6] if frame in bad_frames else [a / 2, a / 2, a / 2]
        positions[frame] = np.array([[0.0, 0.0, 0.0], second])
        if frame not in bad_frames:
            positions[frame] += rng.normal(scale=0.02, size=(2, 3))

    traj = orm.TrajectoryData()
    traj.set_array("positions", positions)
    traj.set_array("cells", np.tile(np.eye(3) * a, (n_frames, 1, 1)))
    traj.set_array("steps", np.arange(n_frames))
    traj.base.attributes.set("symbols", ["Fe", "Fe"])
    return traj.store()


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

    def test_one_failure_among_several_is_reported_as_partial(
        self, tmp_path_factory, aiida_localhost, aiida_profile
    ):
        """Some snapshots failing must not be reported as total success or total failure.

        This is the normal outcome of a real ensemble run, and it was the
        single largest untested branch in the workchain.
        """
        from aiida_feff.data.parameters import FeffParameters

        tmp = tmp_path_factory.mktemp("partialfeff")
        code = feff_rejecting_short_bonds(tmp, aiida_localhost, label="partial-feff")

        results, node = run_workchain(
            code=code,
            trajectory=trajectory_with_bad_frames([1]),
            parameters=FeffParameters(dict={"edge": "K", "radius": 4.0, "absorbing_atom": 0}),
        )

        assert not node.is_finished_ok
        assert node.exit_status == EnsembleExafsWorkChain.exit_codes.ERROR_PARTIAL_FAILURE.status
        # The surviving snapshots must still be averaged and reported, rather
        # than the whole run being discarded.
        assert results["n_failed"].value == 1
        assert results["averaged_xas"]["all"].base.attributes.get("n_snapshots") == 2

    def test_a_failed_potentials_run_stops_the_workchain(
        self, tmp_path_factory, aiida_localhost, aiida_profile
    ):
        """Potentials are shared by every snapshot, so a failure there is fatal.

        Continuing would silently run the whole ensemble against absent or
        stale potentials.
        """
        from aiida_feff.data.parameters import FeffParameters

        tmp = tmp_path_factory.mktemp("potfeff")
        code = feff_rejecting_short_bonds(tmp, aiida_localhost, label="pot-broken-feff")

        # precompute_potentials_step uses ctx.structures[-1], so the bad
        # geometry has to be in the last frame for the potentials run to hit it.
        _results, node = run_workchain(
            code=code,
            trajectory=trajectory_with_bad_frames([2]),
            precompute_potentials=orm.Bool(True),
            parameters=FeffParameters(dict={"edge": "K", "radius": 4.0, "absorbing_atom": 0}),
        )

        assert not node.is_finished_ok
        assert node.exit_status == EnsembleExafsWorkChain.exit_codes.ERROR_POTENTIALS_FAILED.status
        # It must stop at the potentials stage, not fan out anyway.
        assert not [c for c in node.called if c.label.startswith("snap_")]


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

    def test_batch_run_produces_a_consolidated_archive(
        self, fake_feff_code, python_code, two_site_trajectory
    ):
        """Every batch shard must end up merged into the single 'archive' output (ADR 0004)."""
        from aiida_feff.data.archive import ExafsArchiveData
        from aiida_feff.data.parameters import FeffParameters

        results, node = run_workchain(
            code=fake_feff_code,
            python_code=python_code,
            trajectory=two_site_trajectory,
            batch_size=orm.Int(2),
            parameters=FeffParameters(dict={"edge": "K", "radius": 4.0, "absorbing_atom": 0}),
        )
        assert node.is_finished_ok, node.exit_message

        # Each batch CalcJob emits its own shard...
        shards = [c.outputs.archive for c in node.called if c.label.startswith("batch_")]
        assert len(shards) == 2
        assert all(s.is_shard for s in shards)

        # ...and the workchain consolidates them into one ensemble archive.
        archive = results["archive"]
        assert isinstance(archive, ExafsArchiveData)
        assert archive.is_ensemble
        assert archive.k.size > 0
        assert np.abs(archive.chi).max() > 0, "ensemble chi(k) is identically zero"

        # The shards must genuinely differ. If they did not, every assertion
        # below would hold for a merge that dropped or duplicated snapshots,
        # which is precisely the hole the old constant stand-in left.
        assert not np.allclose(shards[0].chi, shards[1].chi), (
            "shards are identical -- the stand-in is not reading feff.inp, "
            "so this test cannot detect a mis-merged ensemble"
        )

        # The ensemble archive must be the mean over snapshots of the spectrum
        # each snapshot's own geometry implies. A merge that dropped a frame,
        # double-counted one, or zero-filled a gap fails here.
        expected = np.mean(
            [expected_chi(archive.k, r) for r in _frame_nn_distances(two_site_trajectory, site=0)],
            axis=0,
        )
        # Amplitude-relative tolerance, not a bare atol and not rtol.
        #
        # rtol is wrong for an oscillating signal: it explodes at every zero
        # crossing of chi(k) and would force a tolerance loose enough to hide
        # a real merge bug. A bare atol would silently track the stand-in's
        # arbitrary 0.1 prefactor. Scaling by the measured amplitude is
        # scale-free and stays meaningful if the stand-in changes.
        #
        # The floor is 8.8e-6 of amplitude, set by R round-tripping through
        # feff.inp's finite coordinate precision (~2e-6 A), so 1e-4 leaves
        # about a factor of ten of headroom.
        tol = 1e-4 * np.abs(expected).max()
        np.testing.assert_allclose(archive.chi, expected, rtol=0.0, atol=tol)

        # Note: this is deliberately *not* compared against averaged_xas.  That
        # output comes from larch's autobk on xmu.dat, whereas the archive comes
        # from FEFF's own chi.dat; the two agree on real data but not on a
        # synthetic mu(E).

    def test_serial_path_reports_no_archive(self, fake_feff_code, two_site_trajectory):
        """FeffCalculation writes no shard, so the serial path must not claim one."""
        from aiida_feff.data.parameters import FeffParameters

        results, node = run_workchain(
            code=fake_feff_code,
            trajectory=two_site_trajectory,
            parameters=FeffParameters(dict={"edge": "K", "radius": 4.0, "absorbing_atom": 0}),
        )
        assert node.is_finished_ok, node.exit_message
        assert "archive" not in results

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

        serial_chi = serial.get_array("chi_k")
        batched_chi = batched.get_array("chi_k")

        # Guard: the comparison is only meaningful if the snapshots being
        # averaged actually differ from one another. With the old constant
        # stand-in both sides averaged copies of a single array, so this
        # assertion held for any implementation -- including one that
        # dropped every snapshot but the first.
        assert serial.base.attributes.get("n_snapshots") == 3
        assert np.abs(serial_chi).max() > 0
        assert serial.get_array("chi_k_std").max() > 0, (
            "snapshots are identical -- this test cannot distinguish a correct "
            "average from one that discarded all but one snapshot"
        )

        np.testing.assert_allclose(batched_chi, serial_chi, rtol=1e-10)

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
