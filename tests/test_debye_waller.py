"""Tests for the Debye-Waller / MSRD calcfunctions.

Pure helper functions are tested directly (no AiiDA profile needed).
Calcfunction tests use the ``aiida_profile`` fixture from aiida-core.
"""

import numpy as np
import pytest

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


# ---------------------------------------------------------------------------
# Calcfunctions (require AiiDA profile)
# ---------------------------------------------------------------------------


class TestComputeMsrd:
    """Tests for the store_msrd calcfunction."""

    def test_returns_dict_node(self, generate_trajectory, aiida_profile):
        from aiida.orm import Dict

        from aiida_feff.calcfunctions.debye_waller import store_msrd

        traj = generate_trajectory(n_frames=30, n_atoms=2, sigma=0.05, seed=0)
        params = Dict({"absorber_site": "Fe", "cutoff": 3.5})
        result = store_msrd(trajectory=traj, params=params)
        assert isinstance(result, Dict)

    def test_keys_contain_path_labels(self, generate_trajectory, aiida_profile):
        from aiida.orm import Dict

        from aiida_feff.calcfunctions.debye_waller import store_msrd

        traj = generate_trajectory(n_frames=50, n_atoms=2, sigma=0.05, seed=1)
        params = Dict({"absorber_site": "Fe", "cutoff": 3.5})
        result = store_msrd(trajectory=traj, params=params)
        d = result.get_dict()
        assert len(d) > 0, "Expected at least one path in output"
        for key in d:
            assert "2body" in key or "3body" in key

    def test_sigma2_positive(self, generate_trajectory, aiida_profile):
        from aiida.orm import Dict

        from aiida_feff.calcfunctions.debye_waller import store_msrd

        traj = generate_trajectory(n_frames=100, n_atoms=2, sigma=0.05, seed=2)
        params = Dict({"absorber_site": "Fe", "cutoff": 3.5})
        result = store_msrd(trajectory=traj, params=params)
        for key, val in result.get_dict().items():
            assert val["sigma2"] >= 0.0, f"sigma2 must be non-negative for path '{key}'"

    def test_reff_near_bcc_fe_nn(self, generate_trajectory, aiida_profile):
        """First-shell reff should be close to BCC-Fe nearest-neighbour distance."""
        from aiida.orm import Dict

        from aiida_feff.calcfunctions.debye_waller import store_msrd

        traj = generate_trajectory(n_frames=200, n_atoms=2, sigma=0.02, seed=3)
        params = Dict({"absorber_site": "Fe", "cutoff": 3.5})
        result = store_msrd(trajectory=traj, params=params)
        d = result.get_dict()
        # BCC Fe nearest-neighbour: a*sqrt(3)/2 ≈ 2.48 Å
        # Key format uses 'p' instead of '.' in reff (e.g. "Fe-Fe_2p48_2body")
        reffs = [v["reff"] for v in d.values() if v["n_body"] == 2]
        assert len(reffs) > 0, "Expected at least one 2-body path"
        assert any(abs(r - 2.48) < 0.15 for r in reffs), f"Expected a reff near 2.48 Å; got {reffs}"

    def test_skip_frames_reduces_frames(self, generate_trajectory, aiida_profile):
        """skip_frames should not crash and should still produce valid paths."""
        from aiida.orm import Dict

        from aiida_feff.calcfunctions.debye_waller import store_msrd

        traj = generate_trajectory(n_frames=60, n_atoms=2, sigma=0.05, seed=4)
        params_full = Dict({"absorber_site": "Fe", "cutoff": 3.5})
        params_skip = Dict({"absorber_site": "Fe", "cutoff": 3.5, "skip_frames": 10})
        result_full = store_msrd(trajectory=traj, params=params_full)
        result_skip = store_msrd(trajectory=traj, params=params_skip)
        # Both should produce at least one path
        assert len(result_full.get_dict()) > 0
        assert len(result_skip.get_dict()) > 0
        # Both should produce the same number of shells (identical topology)
        assert len(result_full.get_dict()) == len(result_skip.get_dict())

    def test_missing_absorber_site_raises(self, generate_trajectory, aiida_profile):
        from aiida.orm import Dict

        from aiida_feff.calcfunctions.debye_waller import store_msrd

        traj = generate_trajectory(n_frames=10, n_atoms=2, sigma=0.05, seed=5)
        params = Dict({"cutoff": 3.5})
        with pytest.raises(ValueError, match="absorber_site"):
            store_msrd(trajectory=traj, params=params)

    def test_n_body_field_is_2(self, generate_trajectory, aiida_profile):
        from aiida.orm import Dict

        from aiida_feff.calcfunctions.debye_waller import store_msrd

        traj = generate_trajectory(n_frames=50, n_atoms=2, sigma=0.05, seed=6)
        params = Dict({"absorber_site": "Fe", "cutoff": 3.5})
        result = store_msrd(trajectory=traj, params=params)
        for val in result.get_dict().values():
            assert val["n_body"] == 2  # no 3-body paths requested


class TestComputeAdp:
    """Tests for the store_adp calcfunction."""

    def test_returns_array_data(self, generate_trajectory, aiida_profile):
        from aiida.orm import ArrayData, Dict

        from aiida_feff.calcfunctions.debye_waller import store_adp

        traj = generate_trajectory(n_frames=30, n_atoms=2, sigma=0.05, seed=10)
        params = Dict({})
        result = store_adp(trajectory=traj, params=params)
        assert isinstance(result, ArrayData)

    def test_b_factors_shape(self, generate_trajectory, aiida_profile):
        from aiida.orm import Dict

        from aiida_feff.calcfunctions.debye_waller import store_adp

        n_atoms = 4
        traj = generate_trajectory(n_frames=30, n_atoms=n_atoms, sigma=0.05, seed=11)
        params = Dict({})
        result = store_adp(trajectory=traj, params=params)
        assert result.get_array("b_factors").shape == (n_atoms,)

    def test_u_tensor_shape(self, generate_trajectory, aiida_profile):
        from aiida.orm import Dict

        from aiida_feff.calcfunctions.debye_waller import store_adp

        n_atoms = 4
        traj = generate_trajectory(n_frames=30, n_atoms=n_atoms, sigma=0.05, seed=12)
        params = Dict({})
        result = store_adp(trajectory=traj, params=params)
        assert result.get_array("u_tensor").shape == (n_atoms, 3, 3)

    def test_b_factors_positive(self, generate_trajectory, aiida_profile):
        from aiida.orm import Dict

        from aiida_feff.calcfunctions.debye_waller import store_adp

        traj = generate_trajectory(n_frames=50, n_atoms=2, sigma=0.05, seed=13)
        params = Dict({})
        result = store_adp(trajectory=traj, params=params)
        b = result.get_array("b_factors")
        assert np.all(b > 0), f"All B-factors should be positive; got {b}"

    def test_symbols_attribute_preserved(self, generate_trajectory, aiida_profile):
        from aiida.orm import Dict

        from aiida_feff.calcfunctions.debye_waller import store_adp

        traj = generate_trajectory(n_frames=20, n_atoms=2, sigma=0.05, seed=14)
        params = Dict({})
        result = store_adp(trajectory=traj, params=params)
        symbols = result.base.attributes.get("symbols")
        assert symbols == ["Fe", "Fe"]

    def test_avg_positions_shape(self, generate_trajectory, aiida_profile):
        from aiida.orm import Dict

        from aiida_feff.calcfunctions.debye_waller import store_adp

        n_atoms = 2
        traj = generate_trajectory(n_frames=30, n_atoms=n_atoms, sigma=0.05, seed=15)
        params = Dict({})
        result = store_adp(trajectory=traj, params=params)
        assert result.get_array("avg_positions").shape == (n_atoms, 3)
