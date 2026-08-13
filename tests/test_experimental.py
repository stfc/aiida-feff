"""Tests for Larch-backed experimental spectrum imports."""

import numpy as np


def test_import_k_chi_ascii(tmp_path):
    """Larch imports explicitly-labelled χ(k) text data without custom parsing."""
    from aiida_feff.calcfunctions.experimental import _import_experimental_spectrum_impl

    path = tmp_path / "experimental_chi.dat"
    path.write_text("# k chi\n0.5 0.1\n1.0 0.2\n2.0 -0.3\n")

    xas = _import_experimental_spectrum_impl(str(path), {"labels": ["k", "chi"], "autobk": False})

    np.testing.assert_allclose(xas.get_array("k"), [0.5, 1.0, 2.0])
    np.testing.assert_allclose(xas.get_array("chi_k"), [0.1, 0.2, -0.3])
    assert xas.base.extras.get("source_kind") == "experimental"
    assert xas.base.extras.get("larch_reader") == "read_ascii"


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
    from aiida_feff.calcfunctions.experimental import (
        _HBAR2_OVER_2M_ELECTRON_EV_ANGSTROM2,
        scaled_chi_arrays,
    )

    k, chi = scaled_chi_arrays(
        np.array([0.0, 1.0, 2.0]),
        np.array([1.0, 2.0, 3.0]),
        s02=0.5,
        e0_shift=_HBAR2_OVER_2M_ELECTRON_EV_ANGSTROM2,
    )

    np.testing.assert_allclose(k, [0.0, np.sqrt(3.0)])
    np.testing.assert_allclose(chi, [1.0, 1.5])
