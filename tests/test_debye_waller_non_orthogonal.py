"""Tests for debye_waller MSRD with non-orthogonal unit cells.

Ported from larch_cli_wrapper/tests/test_debye_waller_non_orthogonal.py.
The upstream uses ASE Atoms + calculate_grouped_msrd; here we use the
aiida_feff internal functions directly, which is consistent with the
existing test_debye_waller.py style.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

# ---------------------------------------------------------------------------
# _max_safe_mic_cutoff
# ---------------------------------------------------------------------------


class TestMaxSafeMicCutoff:
    def test_orthorhombic(self):
        """For an orthorhombic cell the safe cutoff is half the smallest axis."""
        from aiida_feff.calcfunctions.debye_waller import _max_safe_mic_cutoff

        cell = np.diag([4.0, 5.0, 6.0])
        assert _max_safe_mic_cutoff(cell) == pytest.approx(2.0, rel=1e-10)

    def test_non_orthogonal(self):
        """Safe cutoff for a rhombohedral cell equals volume/(2*face_area)."""
        from aiida_feff.calcfunctions.debye_waller import _max_safe_mic_cutoff

        a = 3.0
        alpha = np.radians(60.0)
        cell = np.array(
            [
                [a, 0.0, 0.0],
                [a * np.cos(alpha), a * np.sin(alpha), 0.0],
                [
                    a * np.cos(alpha),
                    a * (np.cos(alpha) - np.cos(alpha) ** 2) / np.sin(alpha),
                    a
                    * np.sqrt(1 - 3 * np.cos(alpha) ** 2 + 2 * np.cos(alpha) ** 3)
                    / np.sin(alpha),
                ],
            ]
        )
        max_cutoff = _max_safe_mic_cutoff(cell)
        assert max_cutoff is not None
        assert max_cutoff > 0.0

        volume = abs(np.linalg.det(cell))
        area = np.linalg.norm(np.cross(cell[0], cell[1]))
        assert max_cutoff == pytest.approx(volume / area / 2.0, rel=1e-10)

    def test_zero_volume(self):
        from aiida_feff.calcfunctions.debye_waller import _max_safe_mic_cutoff

        assert _max_safe_mic_cutoff(np.zeros((3, 3))) is None


# ---------------------------------------------------------------------------
# _calculate_grouped_msrd_impl — non-orthogonal cell correctness
# ---------------------------------------------------------------------------


class TestMsrdNonOrthogonalCell:
    def _make_trajectory(self, n_frames=10, seed=42):
        """Return raw positions array and reference ASE distances."""
        cell = np.array([[3.0, 1.0, 0.5], [0.5, 3.0, 1.0], [1.0, 0.5, 3.0]])
        symbols = ["Mn", "O", "O"]
        base = np.array([[0.0, 0.0, 0.0], [1.5, 1.0, 0.8], [-1.0, 1.5, 1.2]])
        rng = np.random.default_rng(seed)
        positions = base[np.newaxis] + rng.normal(0, 0.05, (n_frames, 3, 3))

        ref_d_01, ref_d_02, ref_angles = [], [], []
        for pos in positions:
            atoms = Atoms(symbols, positions=pos, cell=cell, pbc=True)
            ref_d_01.append(atoms.get_distance(0, 1, mic=True))
            ref_d_02.append(atoms.get_distance(0, 2, mic=True))
            v01 = atoms.get_distances(0, [1], mic=True, vector=True)[0]
            v12 = atoms.get_distances(1, [2], mic=True, vector=True)[0]
            v1, v2 = -v01, v12
            cos_t = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            ref_angles.append(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0))))

        return cell, symbols, positions, ref_d_01, ref_d_02, ref_angles

    def test_2body_reff_matches_ase_mic(self):
        from aiida_feff.calcfunctions.debye_waller import _calculate_grouped_msrd_impl

        cell, symbols, positions, ref_d_01, ref_d_02, _ = self._make_trajectory()

        res_2b, _ = _calculate_grouped_msrd_impl(
            positions,
            symbols,
            cell,
            [True, True, True],
            [0],
            cutoff=4.0,
            cutoff_3body=None,
            exclude_hydrogen=False,
        )

        calc = sorted(r["reff"] for r in res_2b)
        expected = sorted([float(np.mean(ref_d_01)), float(np.mean(ref_d_02))])
        np.testing.assert_allclose(calc, expected, rtol=1e-5)

    def test_3body_angle_matches_ase_mic(self):
        from aiida_feff.calcfunctions.debye_waller import _calculate_grouped_msrd_impl

        cell, symbols, positions, _, _, ref_angles = self._make_trajectory()

        _, res_3b = _calculate_grouped_msrd_impl(
            positions,
            symbols,
            cell,
            [True, True, True],
            [0],
            cutoff=4.0,
            cutoff_3body=4.0,
            exclude_hydrogen=False,
        )

        assert len(res_3b) == 1
        np.testing.assert_allclose(res_3b[0]["angle"], float(np.mean(ref_angles)), rtol=1e-5)

    def test_ase_consistent_mic_nonorthogonal(self):
        """Raw positions + find_mic must agree with ASE get_distance(mic=True)."""
        from aiida_feff.calcfunctions.debye_waller import _calculate_grouped_msrd_impl

        cell = np.array([[4.0, 0.0, 0.0], [1.6, 3.7, 0.0], [0.0, 0.0, 4.2]])
        frames = [
            Atoms(
                "MnO", scaled_positions=[[0.03, 0.5, 0.5], [0.97, 0.5, 0.5]], cell=cell, pbc=True
            ),
            Atoms(
                "MnO", scaled_positions=[[0.04, 0.5, 0.5], [0.96, 0.5, 0.5]], cell=cell, pbc=True
            ),
        ]
        symbols = frames[0].get_chemical_symbols()
        positions = np.array([f.get_positions() for f in frames])
        expected_dists = np.array([f.get_distance(0, 1, mic=True) for f in frames])

        res_2b, res_3b = _calculate_grouped_msrd_impl(
            positions,
            symbols,
            cell,
            [True, True, True],
            [0],
            cutoff=1.0,
            cutoff_3body=None,
            exclude_hydrogen=False,
        )

        assert len(res_2b) == 1
        assert res_2b[0]["type"] == "Mn-O"
        assert np.isclose(res_2b[0]["reff"], expected_dists.mean())
        assert np.isclose(res_2b[0]["sigma2"], np.var(expected_dists, ddof=1))
        assert res_2b[0]["atom_indices"] == [(0, 1)]
        assert res_3b == []


# ---------------------------------------------------------------------------
# Cutoff safety warnings
# ---------------------------------------------------------------------------


class TestCutoffSafetyWarnings:
    def test_warns_when_cutoff_exceeds_safe_radius(self, caplog):
        from aiida_feff.calcfunctions.debye_waller import _calculate_grouped_msrd_impl

        cell = np.diag([2.0, 2.0, 2.0])  # safe radius = 1.0 Å
        positions = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            ]
        )
        symbols = ["Mn", "Mn"]

        with caplog.at_level("WARNING", logger="aiida_feff.calcfunctions.debye_waller"):
            _calculate_grouped_msrd_impl(
                positions,
                symbols,
                cell,
                [True, True, True],
                [0],
                cutoff=1.5,
                cutoff_3body=None,
            )

        assert any("exceeds the maximum safe MIC cutoff" in r.message for r in caplog.records)

    def test_no_warning_for_safe_cutoff(self, caplog):
        from aiida_feff.calcfunctions.debye_waller import _calculate_grouped_msrd_impl

        cell = np.diag([4.0, 4.0, 4.0])  # safe radius = 2.0 Å
        positions = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            ]
        )
        symbols = ["Mn", "Mn"]

        with caplog.at_level("WARNING", logger="aiida_feff.calcfunctions.debye_waller"):
            _calculate_grouped_msrd_impl(
                positions,
                symbols,
                cell,
                [True, True, True],
                [0],
                cutoff=1.5,
                cutoff_3body=None,
            )

        cutoff_warns = [r for r in caplog.records if "maximum safe MIC cutoff" in r.message]
        assert len(cutoff_warns) == 0
