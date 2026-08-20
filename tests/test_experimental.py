"""Tests for Larch-backed experimental spectrum imports."""

import numpy as np
import pytest


def test_import_k_chi_ascii(tmp_path):
    """Larch imports explicitly-labelled χ(k) text data without custom parsing."""
    from aiida_feff.calcfunctions.experimental import _import_experimental_spectrum_impl

    path = tmp_path / "experimental_chi.dat"
    path.write_text("# k chi\n0.5 0.1\n1.0 0.2\n2.0 -0.3\n")

    xas = _import_experimental_spectrum_impl(str(path), {"labels": ["k", "chi"], "autobk": False})

    np.testing.assert_allclose(xas.get_array("k"), [0.5, 1.0, 2.0])
    np.testing.assert_allclose(xas.get_array("chi_k"), [0.1, 0.2, -0.3])
    assert xas.base.attributes.get("source_kind") == "experimental"
    assert xas.base.attributes.get("larch_reader") == "read_ascii"


def test_import_energy_mu_runs_larch_autobk(tmp_path):
    """Energy-space data is handed to Larch for background subtraction."""
    from aiida_feff.calcfunctions.experimental import _import_experimental_spectrum_impl

    energy = np.linspace(7000.0, 7350.0, 401)
    mu = 0.2 + 0.0001 * energy + 0.8 / (1 + np.exp(-(energy - 7112.0) / 2.0))
    path = tmp_path / "experimental_mu.dat"
    path.write_text("\n".join(f"{x:.6f} {y:.8f}" for x, y in zip(energy, mu, strict=True)))

    xas = _import_experimental_spectrum_impl(
        str(path), {"labels": ["energy", "mu"], "autobk": True}
    )

    assert "energy" in xas.get_arraynames()
    assert "mu" in xas.get_arraynames()
    assert "k" in xas.get_arraynames()
    assert "chi_k" in xas.get_arraynames()


def test_scaled_chi_arrays_applies_amplitude_and_energy_shift():
    """$S_0^2$ scales amplitude and ΔE₀ shifts k through kinetic energy."""
    from aiida_feff.calcfunctions.experimental import scaled_chi_arrays

    # Shift by exactly hbar^2/2m (3.80998 eV.A^2), asserted as a literal so the
    # test cannot pass with a wrong constant: importing the constant it checks
    # would make any value self-consistent.
    k, chi = scaled_chi_arrays(
        np.array([0.0, 1.0, 2.0]),
        np.array([1.0, 2.0, 3.0]),
        s02=0.5,
        e0_shift=3.8099821,
    )

    # k' = sqrt(k^2 - 1); the k=0 point falls below the shifted threshold.
    np.testing.assert_allclose(k, [0.0, np.sqrt(3.0)], atol=1e-4)
    np.testing.assert_allclose(chi, [1.0, 1.5])


def test_hbar_squared_over_2m_matches_the_literature_value():
    from aiida_feff.constants import HBAR2_OVER_2M_EV_ANGSTROM2

    assert pytest.approx(3.8099821, rel=1e-6) == HBAR2_OVER_2M_EV_ANGSTROM2


class TestScaleSimulatedSpectrum:
    """Every array indexed by k must be masked together with k."""

    @staticmethod
    def _node():
        from aiida_feff.data.xasdata import XasData

        k = np.linspace(0.0, 12.0, 121)
        node = XasData()
        node.set_spectrum(np.linspace(-20.0, 200.0, 50), np.ones(50), e0=7112.0)
        node.set_chi(k, np.sin(2 * k * 2.5))
        node.set_array("chi_k_std", 0.01 * np.ones_like(k))
        return node

    def test_arrays_stay_the_same_length(self, aiida_profile):
        from aiida.orm import Dict

        from aiida_feff.calcfunctions.experimental import scale_simulated_spectrum

        out = scale_simulated_spectrum(self._node().store(), Dict({"e0_shift": 5.0}))
        n_k = len(out.get_array("k"))
        assert n_k < 121, "a positive e0_shift must drop sub-threshold points"
        assert len(out.get_array("chi_k")) == n_k
        assert len(out.get_array("chi_k_std")) == n_k

    def test_energy_grid_is_left_alone(self, aiida_profile):
        from aiida.orm import Dict

        from aiida_feff.calcfunctions.experimental import scale_simulated_spectrum

        out = scale_simulated_spectrum(self._node().store(), Dict({"e0_shift": 5.0}))
        assert len(out.get_array("energy")) == 50

    def test_s02_scales_chi(self, aiida_profile):
        from aiida.orm import Dict

        from aiida_feff.calcfunctions.experimental import scale_simulated_spectrum

        source = self._node().store()
        out = scale_simulated_spectrum(source, Dict({"s02": 0.5}))
        np.testing.assert_allclose(out.get_array("chi_k"), 0.5 * source.get_array("chi_k"))

    def test_scaling_factors_are_recorded_as_attributes(self, aiida_profile):
        from aiida.orm import Dict

        from aiida_feff.calcfunctions.experimental import scale_simulated_spectrum

        out = scale_simulated_spectrum(self._node().store(), Dict({"s02": 0.8, "e0_shift": 2.0}))
        assert out.base.attributes.get("comparison_s02") == pytest.approx(0.8)
        assert out.base.attributes.get("comparison_e0_shift") == pytest.approx(2.0)

    def test_missing_chi_raises(self, aiida_profile):
        from aiida.orm import Dict

        from aiida_feff.calcfunctions.experimental import scale_simulated_spectrum
        from aiida_feff.data.xasdata import XasData

        node = XasData()
        node.set_spectrum(np.linspace(0, 10, 5), np.ones(5))
        with pytest.raises(ValueError, match="no .* arrays to scale"):
            scale_simulated_spectrum(node.store(), Dict({}))
