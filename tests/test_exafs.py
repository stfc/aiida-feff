"""Tests for χ(k) reconstruction from stored FEFF path data.

The strongest check here is an oracle test: :func:`path_chi` is compared against
larch's own ``FeffPathGroup`` evaluated on the same ``feff????.dat`` file.  The
two are independent implementations of the same equation, so agreement pins
down the phase convention (``rep`` versus ``k``), the amplitude convention
(``mag_feff · red_fact``, with S₀² applied separately) and the σ² sign.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aiida_feff.calcfunctions.exafs import path_chi, total_chi
from aiida_feff.data.pathcontributions import FEFF_DATA_COLS

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "aggregate_paths"
FEFF_DATS = sorted(FIXTURE_DIR.glob("feff????.dat"))

# larch's own ETOK; if our derived constant drifted from it every phase would.
LARCH_ETOK = 0.26246842396479836


@pytest.fixture(params=[p.name for p in FEFF_DATS[:3]])
def feff_dat(request):
    """Path to one real Feff8L ``feff????.dat`` fixture."""
    return FIXTURE_DIR / request.param


@pytest.fixture()
def larch_path(feff_dat):
    """larch's FeffPathGroup for the same fixture, used as an oracle."""
    feffdat = pytest.importorskip("larch.xafs.feffdat")
    return feffdat.FeffPathGroup(filename=str(feff_dat))


def columns_from_larch(dat) -> np.ndarray:
    """Stack a larch FeffDatFile back into our 6-column layout."""
    return np.column_stack([getattr(dat, name) for name in FEFF_DATA_COLS])


class TestEtokConstant:
    """The energy-to-wavenumber constant must match the literature value."""

    def test_matches_larch(self):
        from aiida_feff.constants import ETOK

        # Asserted against an external number, not against the module's own
        # constant: a test that imports the constant it checks proves nothing.
        assert pytest.approx(LARCH_ETOK, rel=1e-6) == ETOK

    def test_inverse_is_hbar_squared_over_2m(self):
        from aiida_feff.constants import ETOK, HBAR2_OVER_2M_EV_ANGSTROM2

        assert pytest.approx(3.8099821, rel=1e-6) == HBAR2_OVER_2M_EV_ANGSTROM2
        assert pytest.approx(1.0, rel=1e-12) == ETOK * HBAR2_OVER_2M_EV_ANGSTROM2


class TestPathChiAgainstLarch:
    """path_chi must reproduce larch's FeffPathGroup on real FEFF output."""

    @staticmethod
    def _compare(larch_path, *, sigma2=0.0, s02=1.0, e0_shift=0.0):
        larch_path._calc_chi(sigma2=sigma2, s02=s02, e0=e0_shift, kstep=0.05, kmax=12.0)
        k = np.asarray(larch_path.k)
        dat = larch_path._feffdat

        ours = path_chi(
            np.asarray(dat.k),
            columns_from_larch(dat),
            float(dat.reff),
            float(dat.degen),
            k,
            sigma2=sigma2,
            s02=s02,
            e0_shift=e0_shift,
        )
        return np.asarray(larch_path.chi), ours

    def test_bare_path(self, larch_path):
        expected, ours = self._compare(larch_path)
        scale = np.abs(expected).max()
        np.testing.assert_allclose(ours, expected, atol=1e-9 * scale)

    def test_with_debye_waller(self, larch_path):
        expected, ours = self._compare(larch_path, sigma2=0.008)
        scale = np.abs(expected).max()
        np.testing.assert_allclose(ours, expected, atol=1e-9 * scale)

    def test_with_s02_and_e0_shift(self, larch_path):
        expected, ours = self._compare(larch_path, s02=0.85, e0_shift=3.0)
        scale = np.abs(expected).max()
        np.testing.assert_allclose(ours, expected, atol=1e-6 * scale)


class TestPathChiPhysics:
    """Limits that hold whatever the path."""

    @pytest.fixture()
    def path_args(self, larch_path):
        dat = larch_path._feffdat
        return np.asarray(dat.k), columns_from_larch(dat), float(dat.reff), float(dat.degen)

    def test_s02_scales_amplitude_linearly(self, path_args):
        k_native, columns, reff, degen = path_args
        k = np.linspace(2.0, 12.0, 200)
        one = path_chi(k_native, columns, reff, degen, k, s02=1.0)
        half = path_chi(k_native, columns, reff, degen, k, s02=0.5)
        np.testing.assert_allclose(half, 0.5 * one, rtol=1e-12)

    def test_debye_waller_damps_high_k_hardest(self, path_args):
        k_native, columns, reff, degen = path_args
        k = np.linspace(3.0, 12.0, 300)
        bare = path_chi(k_native, columns, reff, degen, k)
        damped = path_chi(k_native, columns, reff, degen, k, sigma2=0.01)

        # exp(-2 sigma^2 k^2) falls monotonically, so the surviving fraction of
        # the envelope must shrink with k.
        def envelope_ratio(lo, hi):
            window = (k >= lo) & (k < hi)
            return np.abs(damped[window]).max() / np.abs(bare[window]).max()

        assert envelope_ratio(3.0, 5.0) > envelope_ratio(9.0, 12.0)
        assert envelope_ratio(9.0, 12.0) < 1.0

    def test_degeneracy_scales_amplitude_linearly(self, path_args):
        k_native, columns, reff, degen = path_args
        k = np.linspace(2.0, 12.0, 100)
        single = path_chi(k_native, columns, reff, 1.0, k)
        quadruple = path_chi(k_native, columns, reff, 4.0, k)
        np.testing.assert_allclose(quadruple, 4.0 * single, rtol=1e-12)

    def test_rejects_wrong_column_count(self, path_args):
        k_native, columns, reff, degen = path_args
        with pytest.raises(ValueError, match="columns"):
            path_chi(k_native, columns[:, :4], reff, degen, np.linspace(2, 10, 10))

    def test_rejects_non_positive_reff(self, path_args):
        k_native, columns, _reff, degen = path_args
        with pytest.raises(ValueError, match="r_eff"):
            path_chi(k_native, columns, 0.0, degen, np.linspace(2, 10, 10))


class TestTotalChi:
    """Summation over a stored node."""

    def test_sums_every_path(self, aggregated_node):
        k = np.linspace(3.0, 12.0, 200)
        per_path = [
            path_chi(p.k, p.feff_data, p.r_eff, p.degeneracy, k)
            for p in aggregated_node.iter_paths()
        ]
        np.testing.assert_allclose(total_chi(aggregated_node, k), np.sum(per_path, axis=0))

    def test_per_scatterer_sigma2_is_applied(self, aggregated_node):
        k = np.linspace(3.0, 12.0, 200)
        scatterers = {p.scatterer for p in aggregated_node.iter_paths()}
        uniform = total_chi(aggregated_node, k, sigma2=0.01)
        mapped = total_chi(aggregated_node, k, sigma2=dict.fromkeys(scatterers, 0.01))
        np.testing.assert_allclose(mapped, uniform, rtol=1e-12)

    def test_frame_filter_selects_nothing_for_absent_frame(self, aggregated_node):
        k = np.linspace(3.0, 12.0, 50)
        np.testing.assert_allclose(total_chi(aggregated_node, k, frame_idx=999), np.zeros_like(k))
