"""Tests for the Fourier transform and ensemble averaging in calcfunctions.larch."""

from __future__ import annotations

import numpy as np
import pytest
from aiida.orm import Dict

from aiida_feff.calcfunctions.larch import (
    FT_DEFAULTS,
    archive_to_averaged_xas,
    chi_k_to_r,
    resolve_ft_params,
    xftf_arrays,
)
from aiida_feff.data.archive import ExafsArchiveData

pytest.importorskip("larch.xafs")


def single_shell_chi(k: np.ndarray, r0: float, sigma2: float = 0.0) -> np.ndarray:
    """A textbook single-frequency EXAFS signal peaking at ``r0`` in R space."""
    return np.sin(2 * k * r0) * np.exp(-2 * sigma2 * k**2)


class TestFourierTransformAnalyticLimit:
    """chi(k) = sin(2 k R0) must transform to a peak at R0."""

    K = np.linspace(0.0, 20.0, 401)

    @pytest.mark.parametrize("r0", [1.8, 2.5, 3.7])
    def test_peak_sits_at_r0(self, r0):
        result = xftf_arrays(
            self.K,
            single_shell_chi(self.K, r0),
            {"kmin": 2.0, "kmax": 18.0, "kweight": 0},
        )
        peak_r = result["r"][int(np.argmax(result["chir_mag"]))]
        # The FT r grid is 0.0307 A wide, so half a bin (0.0153 A) is the
        # resolution floor: a peak cannot be located more precisely than the
        # bin it falls in. Measured error here is 0.33-0.49 bins for these r0,
        # so 0.016 is the tightest honest tolerance. A looser one (the 0.05
        # used previously, ~1.6 bins) would pass through a systematic
        # one-bin offset in the R grid.
        assert peak_r == pytest.approx(r0, abs=0.016)

    def test_two_shells_give_two_peaks(self):
        chi = single_shell_chi(self.K, 2.0) + single_shell_chi(self.K, 3.5)
        result = xftf_arrays(self.K, chi, {"kmin": 2.0, "kmax": 18.0, "kweight": 0})
        r, mag = result["r"], result["chir_mag"]

        def peak_near(target):
            window = np.abs(r - target) < 0.4
            return r[window][int(np.argmax(mag[window]))]

        assert peak_near(2.0) == pytest.approx(2.0, abs=0.016)
        assert peak_near(3.5) == pytest.approx(3.5, abs=0.016)

    def test_debye_waller_broadens_and_damps_the_peak(self):
        sharp = xftf_arrays(self.K, single_shell_chi(self.K, 2.5), {"kweight": 0})
        damped = xftf_arrays(self.K, single_shell_chi(self.K, 2.5, sigma2=0.01), {"kweight": 0})
        assert damped["chir_mag"].max() < sharp["chir_mag"].max()


class TestFtParameterHandling:
    """FT defaults live in one place and unknown keys are refused."""

    def test_defaults_are_filled_in(self):
        # Spelled out rather than compared against FT_DEFAULTS: a test that
        # imports the constant it checks passes for any value of that
        # constant. These literals are the transform every untagged
        # calculation in the package silently runs, so a change to them
        # should surface as a diff here.
        assert resolve_ft_params({"kmin": 4.0}) == {
            "kmin": 4.0,
            "kmax": 15.0,
            "kweight": 2,
            "dk": 1.0,
            "rmax": 8.0,
            "window": "kaiser",
        }

    def test_defaults_constant_matches_the_resolver(self):
        """The exported constant and the resolver must not drift apart."""
        assert resolve_ft_params({}) == FT_DEFAULTS

    def test_window_default_is_kaiser(self):
        # larch's own default, and not the Hanning that EXAFS docs assume.
        assert FT_DEFAULTS["window"] == "kaiser"

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown Fourier-transform parameter"):
            resolve_ft_params({"kwieght": 2})

    def test_window_choice_changes_the_result(self):
        """Windows differ in sidelobes but must agree on where the shell is.

        Asserting only that the two differ would pass if one window were
        applied wrongly; pinning both peaks to r0 as well means the
        difference has to be a windowing difference rather than damage.
        """
        k = np.linspace(0.0, 20.0, 401)
        chi = single_shell_chi(k, 2.5)
        params = {"kmin": 2.0, "kmax": 18.0, "kweight": 0}
        kaiser = xftf_arrays(k, chi, {**params, "window": "kaiser"})
        hanning = xftf_arrays(k, chi, {**params, "window": "hanning"})

        assert not np.allclose(kaiser["chir_mag"], hanning["chir_mag"])
        for result in (kaiser, hanning):
            peak = result["r"][int(np.argmax(result["chir_mag"]))]
            assert peak == pytest.approx(2.5, abs=0.016)

    def test_chi_k_to_r_records_the_transform_it_ran(self, generate_xas_data, aiida_profile):
        result = chi_k_to_r(generate_xas_data(), Dict({"kweight": 3, "kmax": 12.0}))
        recorded = result.base.attributes.get("fourier_params")
        # kweight sets the units of chi(R), so it has to be recoverable.
        assert recorded["kweight"] == 3
        assert recorded["kmax"] == 12.0
        assert recorded["window"] == "kaiser"

    def test_chi_k_to_r_arrays_are_mutually_consistent(self, generate_xas_data, aiida_profile):
        """chir_mag must be the modulus of the complex transform it ships with.

        Checking only that the four arrays exist would pass if chir_mag were
        computed from a different transform than chir_re/chir_im -- which is
        exactly what a mis-wired kweight or window argument would produce.
        """
        result = chi_k_to_r(generate_xas_data(), Dict({"kmin": 3.0, "kmax": 12.0, "kweight": 2}))
        names = result.get_arraynames()
        assert {"r", "chir_mag", "chir_re", "chir_im"} <= set(names)

        mag = result.get_array("chir_mag")
        re, im = result.get_array("chir_re"), result.get_array("chir_im")
        np.testing.assert_allclose(mag, np.hypot(re, im), rtol=1e-12)
        assert result.get_array("r").shape == mag.shape
        # A transform that returned zeros would satisfy the identity above.
        assert mag.max() > 0.0


class TestArchiveProjection:
    """archive_to_averaged_xas must project ExafsArchiveData into bit-identical XasData nodes."""

    def test_archive_projection_is_bit_identical(self, tmp_path, aiida_profile):
        from md_exafs.hdf5 import EnsembleWriter

        h5_path = tmp_path / "ensemble.h5"
        k = np.linspace(0.05, 15.0, 100)
        chi_overall = np.sin(2 * k * 2.5)
        chi_std = np.full_like(k, 0.05)
        n_contrib = np.full(len(k), 4, dtype=np.int32)

        chi_site1 = np.cos(1.8 * k)
        n_contrib_site1 = np.full(len(k), 2, dtype=np.int32)

        with EnsembleWriter(h5_path, k_grid=k) as writer:
            writer.set_overall_average(k, chi_overall, chi_std=chi_std, n_contributors=n_contrib)
            writer.add_site_average(site_idx=1, k=k, chi=chi_site1, n_contributors=n_contrib_site1)

        archive = ExafsArchiveData(file=str(h5_path))
        projected = archive_to_averaged_xas(archive)

        assert "all" in projected
        assert "site_0001" in projected

        xas_all = projected["all"]
        # Bit-identical checks (decision 5)
        np.testing.assert_array_equal(xas_all.get_array("chi_k"), chi_overall)
        np.testing.assert_array_equal(xas_all.get_array("k"), k)
        np.testing.assert_array_equal(xas_all.get_array("chi_k_std"), chi_std)
        np.testing.assert_array_equal(xas_all.get_array("chi_k_count"), n_contrib.astype(float))
        assert xas_all.base.attributes.get("chi_source") == "feff.chi.dat"
        assert xas_all.base.attributes.get("n_snapshots") == 4

        xas_site1 = projected["site_0001"]
        np.testing.assert_array_equal(xas_site1.get_array("chi_k"), chi_site1)
        np.testing.assert_array_equal(xas_site1.get_array("k"), k)
        np.testing.assert_array_equal(
            xas_site1.get_array("chi_k_count"), n_contrib_site1.astype(float)
        )
        assert xas_site1.base.attributes.get("site_index") == 1
