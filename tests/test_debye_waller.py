"""Tests for the Debye-Waller / MSRD calcfunctions.

Every test in this repository loads a temporary AiiDA profile: aiida-core's
pytest plugin installs it session-scoped and autouse.  The backend is
``core.sqlite_dos``, so no PostgreSQL and no daemon are involved.

Distance-based tests use a 2x2x2 BCC supercell whose inscribed-sphere radius
(2.77 Angstrom) exceeds the cutoffs they pass.  Raise ``reps`` rather than the
cutoff if a test needs a longer range.
"""

import numpy as np
import pytest

from tests.conftest import BCC_FE_NN, bcc_supercell_positions

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestParSiteSpec:
    """Tests for _parse_site_spec."""

    def _symbols(self):
        return ["Fe", "Fe", "Al", "Al", "H"]

    def test_element_all(self):
        from aiida_feff.calcfunctions.debye_waller import _parse_site_spec

        assert _parse_site_spec("Fe", self._symbols()) == [0, 1]

    def test_element_indexed(self):
        from aiida_feff.calcfunctions.debye_waller import _parse_site_spec

        assert _parse_site_spec("Fe.1", self._symbols()) == [0]
        assert _parse_site_spec("Fe.2", self._symbols()) == [1]
        assert _parse_site_spec("Al.1", self._symbols()) == [2]

    def test_element_range(self):
        from aiida_feff.calcfunctions.debye_waller import _parse_site_spec

        assert _parse_site_spec("Fe.1-2", self._symbols()) == [0, 1]
        assert _parse_site_spec("Al.1-2", self._symbols()) == [2, 3]

    def test_absolute_index(self):
        from aiida_feff.calcfunctions.debye_waller import _parse_site_spec

        # 1-based: "1" → atom 0, "3" → atom 2
        assert _parse_site_spec("1", self._symbols()) == [0]
        assert _parse_site_spec("3", self._symbols()) == [2]

    def test_absolute_range(self):
        from aiida_feff.calcfunctions.debye_waller import _parse_site_spec

        assert _parse_site_spec("1-3", self._symbols()) == [0, 1, 2]

    def test_unknown_element_raises(self):
        from aiida_feff.calcfunctions.debye_waller import _parse_site_spec

        with pytest.raises(ValueError, match="No atoms of element"):
            _parse_site_spec("Xe", self._symbols())


class TestUnwrapPositionsPbc:
    """Tests for _unwrap_positions_pbc."""

    def test_no_pbc_crossing_is_identity(self):
        from aiida_feff.calcfunctions.debye_waller import _unwrap_positions_pbc

        a = 10.0
        pos = np.array(
            [
                [[1.0, 1.0, 1.0]],
                [[1.1, 1.1, 1.1]],
                [[1.2, 1.2, 1.2]],
            ]
        )  # shape (3, 1, 3) — one atom, no PBC crossing
        cells = np.tile(np.eye(3) * a, (3, 1, 1))
        result = _unwrap_positions_pbc(pos, cells)
        np.testing.assert_allclose(result, pos, atol=1e-10)

    def test_unwrap_single_crossing(self):
        from aiida_feff.calcfunctions.debye_waller import _unwrap_positions_pbc

        a = 5.0
        cell = np.eye(3) * a
        cells = np.tile(cell, (3, 1, 1))
        # Atom starts near one edge, crosses to wrapped position
        pos = np.array(
            [
                [[4.9, 0.0, 0.0]],
                [[0.1, 0.0, 0.0]],  # wrapped (would be 5.1 without PBC)
                [[0.3, 0.0, 0.0]],
            ]
        )
        result = _unwrap_positions_pbc(pos, cells)
        # After unwrapping frame 1 should be ~5.1, not 0.1
        assert result[1, 0, 0] > 4.0, "Atom should be unwrapped past the cell boundary"
        # Displacement from frame 0 → 1 should be small (~+0.2), not ~−4.8
        disp01 = result[1, 0, 0] - result[0, 0, 0]
        assert abs(disp01) < 1.0

    def test_output_shape_preserved(self):
        from aiida_feff.calcfunctions.debye_waller import _unwrap_positions_pbc

        rng = np.random.default_rng(42)
        pos = rng.random((20, 5, 3)) * 5.0
        cells = np.tile(np.eye(3) * 5.0, (20, 1, 1))
        result = _unwrap_positions_pbc(pos, cells)
        assert result.shape == pos.shape


class TestKabschAlign:
    """Tests for _kabsch_align."""

    def test_already_aligned_returns_same(self):
        from aiida_feff.calcfunctions.debye_waller import _kabsch_align

        rng = np.random.default_rng(0)
        pos = rng.random((10, 3, 3))
        result = _kabsch_align(pos, reference=pos[0])
        # Frame 0 should be unchanged (modulo float precision)
        np.testing.assert_allclose(result[0], pos[0], atol=1e-10)

    def test_rigid_rotation_reduces_rmsd(self):
        """Kabsch alignment should give a lower (or equal) RMSD to reference."""
        from aiida_feff.calcfunctions.debye_waller import _kabsch_align

        rng = np.random.default_rng(7)
        base = rng.random((8, 3))  # 8 atoms in 3D
        base -= base.mean(axis=0)

        # Build a random rotation matrix via QR decomposition
        Q, _ = np.linalg.qr(rng.random((3, 3)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1

        rotated = (base @ Q.T) + rng.normal(scale=0.01, size=base.shape)
        positions = np.stack([base, rotated])  # shape (2, 8, 3)

        aligned = _kabsch_align(positions, reference=base)
        rmsd_before = float(np.sqrt(np.mean((rotated - base) ** 2)))
        rmsd_after = float(np.sqrt(np.mean((aligned[1] - base) ** 2)))
        assert (
            rmsd_after <= rmsd_before + 1e-10
        ), f"Alignment should not increase RMSD: before={rmsd_before:.4f} after={rmsd_after:.4f}"

    def test_output_shape_preserved(self):
        from aiida_feff.calcfunctions.debye_waller import _kabsch_align

        pos = np.random.default_rng(1).random((15, 6, 3))
        result = _kabsch_align(pos)
        assert result.shape == pos.shape


class TestClusterByTolerance:
    """Cluster width must stay bounded by the tolerance the caller asked for."""

    def test_ladder_does_not_chain_beyond_tolerance(self):
        from aiida_feff.calcfunctions.debye_waller import _cluster_by_tolerance

        # Consecutive gaps of 0.08 < tol, but the full span is 0.32 > tol.
        items = [{"d": d} for d in (2.00, 2.08, 2.16, 2.24, 2.32)]
        clusters = _cluster_by_tolerance(items, "d", tol=0.1)
        for cluster in clusters:
            span = cluster[-1]["d"] - cluster[0]["d"]
            assert span <= 0.1 + 1e-12, f"cluster spans {span:.3f} Å, wider than tol=0.1"
        assert len(clusters) > 1

    def test_empty_input_yields_no_clusters(self):
        from aiida_feff.calcfunctions.debye_waller import _cluster_by_tolerance

        assert _cluster_by_tolerance([], "d", tol=0.1) == []


class TestMicCutoffGuard:
    """A cutoff beyond the inscribed sphere biases every distance downward."""

    def test_unsafe_cutoff_raises(self, generate_trajectory, aiida_profile):
        from aiida.orm import Dict

        from aiida_feff.calcfunctions.debye_waller import store_msrd

        # reps=1 gives a 2.77 Å cell, safe only to 1.435 Å.
        traj = generate_trajectory(n_frames=10, reps=1, sigma=0.02, seed=0)
        params = Dict({"absorber_site": "Fe", "cutoff": 3.5})
        with pytest.raises(ValueError, match="minimum-image cutoff"):
            store_msrd(trajectory=traj, params=params)

    def test_override_permits_unsafe_cutoff(self, generate_trajectory, aiida_profile):
        from aiida.orm import Dict

        from aiida_feff.calcfunctions.debye_waller import store_msrd

        traj = generate_trajectory(n_frames=10, reps=1, sigma=0.02, seed=0)
        params = Dict({"absorber_site": "Fe", "cutoff": 3.5, "allow_unsafe_cutoff": True})
        assert len(store_msrd(trajectory=traj, params=params).get_dict()) > 0

    def test_unsafe_cutoff_would_have_biased_sigma2_low(self):
        """The guard is worth having: show the size of the error it prevents."""
        from aiida_feff.calcfunctions.debye_waller import _calculate_grouped_msrd_impl

        sigma = 0.05
        rng = np.random.default_rng(0)

        def first_shell_sigma2(reps, cutoff, allow_unsafe):
            eq, cell = bcc_supercell_positions(reps)
            pos = eq[None] + rng.normal(scale=sigma, size=(2000, len(eq), 3))
            res2, _ = _calculate_grouped_msrd_impl(
                pos,
                ["Fe"] * len(eq),
                cell,
                [True] * 3,
                [0],
                cutoff=cutoff,
                allow_unsafe_cutoff=allow_unsafe,
            )
            return res2[0]["sigma2"]

        safe = first_shell_sigma2(reps=2, cutoff=2.7, allow_unsafe=False)
        unsafe = first_shell_sigma2(reps=1, cutoff=2.7, allow_unsafe=True)
        assert safe == pytest.approx(2 * sigma**2, rel=0.05)
        assert unsafe < 0.5 * safe, "expected the aliased cutoff to collapse sigma2"


class TestMsrdAnalyticLimits:
    """Uncorrelated Gaussian displacements have closed-form MSRD and <r>."""

    SIGMA = 0.05
    N_FRAMES = 4000

    @pytest.fixture()
    def first_shell(self, generate_trajectory, aiida_profile):
        from aiida.orm import Dict

        from aiida_feff.calcfunctions.debye_waller import store_msrd

        traj = generate_trajectory(n_frames=self.N_FRAMES, reps=2, sigma=self.SIGMA, seed=0)
        # 2.7 Å < the 2.77 Å inscribed-sphere radius of the 2x2x2 cell.
        params = Dict({"absorber_site": "Fe.1", "cutoff": 2.7})
        shells = store_msrd(trajectory=traj, params=params).get_dict()
        return min(shells.values(), key=lambda v: v["reff"])

    def test_sigma2_approaches_twice_the_displacement_variance(self, first_shell):
        # sigma2 of a bond = Var(u_A - u_B) = 2 sigma^2 for independent atoms.
        assert first_shell["sigma2"] == pytest.approx(2 * self.SIGMA**2, rel=0.05)

    def test_mean_distance_shows_perpendicular_displacement_bias(self, first_shell):
        # <r> exceeds the equilibrium bond length by <u_perp^2> / 2 r_eq, with
        # <u_perp^2> = 2 x 2 sigma^2 over the two perpendicular directions.
        expected = BCC_FE_NN + 2 * self.SIGMA**2 / BCC_FE_NN
        assert first_shell["reff"] > BCC_FE_NN
        assert first_shell["reff"] == pytest.approx(expected, abs=1e-3)

    def test_first_shell_has_eight_neighbours(self, first_shell):
        assert first_shell["count"] == 8  # BCC coordination number


class TestComputeMsrd:
    """Behaviour of the store_msrd calcfunction."""

    def test_keys_contain_path_labels(self, generate_trajectory, aiida_profile):
        from aiida.orm import Dict

        from aiida_feff.calcfunctions.debye_waller import store_msrd

        traj = generate_trajectory(n_frames=50, reps=2, sigma=0.05, seed=1)
        result = store_msrd(trajectory=traj, params=Dict({"absorber_site": "Fe.1", "cutoff": 2.7}))
        d = result.get_dict()
        assert len(d) > 0, "Expected at least one path in output"
        for key in d:
            assert "2body" in key or "3body" in key

    def test_skip_frames_keeps_the_same_shells(self, generate_trajectory, aiida_profile):
        from aiida.orm import Dict

        from aiida_feff.calcfunctions.debye_waller import store_msrd

        traj = generate_trajectory(n_frames=60, reps=2, sigma=0.05, seed=4)
        base = {"absorber_site": "Fe.1", "cutoff": 2.7}
        full = store_msrd(trajectory=traj, params=Dict(base)).get_dict()
        skipped = store_msrd(trajectory=traj, params=Dict({**base, "skip_frames": 10})).get_dict()
        # Same topology, so the same shells with the same occupancies; only the
        # statistics over frames differ.
        assert len(full) == len(skipped)
        assert sorted(v["count"] for v in full.values()) == sorted(
            v["count"] for v in skipped.values()
        )
        for a, b in zip(
            sorted(full.values(), key=lambda v: v["reff"]),
            sorted(skipped.values(), key=lambda v: v["reff"]),
            strict=True,
        ):
            assert a["reff"] == pytest.approx(b["reff"], abs=0.02)

    def test_missing_absorber_site_raises(self, generate_trajectory, aiida_profile):
        from aiida.orm import Dict

        from aiida_feff.calcfunctions.debye_waller import store_msrd

        traj = generate_trajectory(n_frames=10, reps=2, sigma=0.05, seed=5)
        with pytest.raises(ValueError, match="absorber_site"):
            store_msrd(trajectory=traj, params=Dict({"cutoff": 2.7}))

    def test_unknown_parameter_raises(self, generate_trajectory, aiida_profile):
        """A silently-ignored key is a wrong answer the user cannot see."""
        from aiida.orm import Dict

        from aiida_feff.calcfunctions.debye_waller import store_msrd

        traj = generate_trajectory(n_frames=10, reps=2, sigma=0.05, seed=5)
        params = Dict({"absorber_site": "Fe.1", "cutoff": 2.7, "align": True})
        with pytest.raises(ValueError, match="does not use parameter"):
            store_msrd(trajectory=traj, params=params)

    def test_two_body_only_without_cutoff_3body(self, generate_trajectory, aiida_profile):
        from aiida.orm import Dict

        from aiida_feff.calcfunctions.debye_waller import store_msrd

        traj = generate_trajectory(n_frames=50, reps=2, sigma=0.05, seed=6)
        result = store_msrd(trajectory=traj, params=Dict({"absorber_site": "Fe.1", "cutoff": 2.7}))
        assert all(v["n_body"] == 2 for v in result.get_dict().values())

    def test_three_body_paths_are_produced_and_labelled(self, generate_trajectory, aiida_profile):
        from aiida.orm import Dict

        from aiida_feff.calcfunctions.debye_waller import store_msrd

        traj = generate_trajectory(n_frames=40, reps=2, sigma=0.03, seed=7)
        params = Dict({"absorber_site": "Fe.1", "cutoff": 2.7, "cutoff_3body": 2.7})
        d = store_msrd(trajectory=traj, params=params).get_dict()
        three_body = [v for v in d.values() if v["n_body"] == 3]
        assert three_body, "expected 3-body paths when cutoff_3body is set"
        for path in three_body:
            # Included angle at the first scatterer, so within (0, 180].
            assert 0.0 < path["angle"] <= 180.0
            # reff is half the total path length, per FEFF's convention, so it
            # is longer than the 2-body bond but shorter than the perimeter.
            assert path["reff"] > BCC_FE_NN


class TestComputeAdp:
    """Tests for the store_adp calcfunction."""

    def test_b_factor_approaches_8pi2_sigma_squared(self, generate_trajectory, aiida_profile):
        from aiida.orm import Dict

        from aiida_feff.calcfunctions.debye_waller import store_adp

        sigma = 0.05
        traj = generate_trajectory(n_frames=4000, reps=2, sigma=sigma, seed=13)
        result = store_adp(trajectory=traj, params=Dict({"align": False}))
        b = result.get_array("b_factors")
        assert b.mean() == pytest.approx(8 * np.pi**2 * sigma**2, rel=0.02)

    def test_u_tensor_is_isotropic_for_isotropic_noise(self, generate_trajectory, aiida_profile):
        from aiida.orm import Dict

        from aiida_feff.calcfunctions.debye_waller import store_adp

        sigma = 0.05
        traj = generate_trajectory(n_frames=4000, reps=2, sigma=sigma, seed=14)
        u = store_adp(trajectory=traj, params=Dict({"align": False})).get_array("u_tensor")
        diagonal = np.einsum("nii->ni", u)
        off_diagonal = u[:, 0, 1]
        assert diagonal.mean() == pytest.approx(sigma**2, rel=0.05)
        assert abs(off_diagonal.mean()) < 0.05 * sigma**2

    def test_shapes_and_symbols(self, generate_trajectory, aiida_profile):
        from aiida.orm import Dict

        from aiida_feff.calcfunctions.debye_waller import store_adp

        traj = generate_trajectory(n_frames=30, reps=2, sigma=0.05, seed=11)
        n_atoms = len(traj.get_array("positions")[0])
        result = store_adp(trajectory=traj, params=Dict({}))
        assert result.get_array("b_factors").shape == (n_atoms,)
        assert result.get_array("u_tensor").shape == (n_atoms, 3, 3)
        assert result.get_array("avg_positions").shape == (n_atoms, 3)
        assert result.base.attributes.get("symbols") == ["Fe"] * n_atoms

    def test_unknown_parameter_raises(self, generate_trajectory, aiida_profile):
        from aiida.orm import Dict

        from aiida_feff.calcfunctions.debye_waller import store_adp

        traj = generate_trajectory(n_frames=10, reps=2, sigma=0.05, seed=15)
        with pytest.raises(ValueError, match="does not use parameter"):
            store_adp(trajectory=traj, params=Dict({"cutoff": 3.5}))
