"""Tests for larch calcfunctions.

Tests that require larch are marked ``@pytest.mark.requires_larch``
and skipped automatically when xraylarch is not installed.
"""

import numpy as np
import pytest

try:
    import larch  # noqa: F401

    HAS_LARCH = True
except ImportError:
    HAS_LARCH = False

requires_larch = pytest.mark.skipif(not HAS_LARCH, reason="xraylarch not installed")


class TestAverageXasData:
    """Tests for average_xas_data that do NOT need a running AiiDA daemon."""

    def _make_xas(self, offset=0.0):
        """Create a simple in-memory XasData (not stored)."""
        import numpy as np

        from aiida_feff.data.xasdata import XasData

        xas = XasData()
        energy = np.linspace(-20, 200, 100)
        mu = np.exp(-((energy - 30 + offset) ** 2) / 200)
        xas.set_spectrum(energy, mu, e0=7112.0)
        k = np.linspace(0.5, 15.0, 150)
        chi = 0.4 * np.sin(2 * 2.5 * k) * np.exp(-2 * 0.003 * k**2) / (k + 0.1)
        xas.set_chi(k, chi)
        return xas

    def test_average_shape_preserved(self):
        """Output arrays must have same shape as inputs."""
        from aiida_feff.calcfunctions.larch import _average_xas_data_impl

        xas_0 = self._make_xas(0.0)
        xas_1 = self._make_xas(1.0)
        avg = _average_xas_data_impl(snap_0=xas_0, snap_1=xas_1)
        assert avg.get_array("energy").shape == xas_0.get_array("energy").shape
        assert avg.get_array("chi_k").shape == xas_0.get_array("chi_k").shape

    def test_average_of_identical_is_identity(self):
        from aiida_feff.calcfunctions.larch import _average_xas_data_impl

        xas = self._make_xas(0.0)
        avg = _average_xas_data_impl(snap_0=xas, snap_1=xas)
        np.testing.assert_allclose(avg.get_array("chi_k"), xas.get_array("chi_k"), rtol=1e-10)

    def test_std_is_zero_for_identical(self):
        from aiida_feff.calcfunctions.larch import _average_xas_data_impl

        xas = self._make_xas(0.0)
        avg = _average_xas_data_impl(snap_0=xas, snap_1=xas)
        np.testing.assert_allclose(avg.get_array("chi_k_std"), 0.0, atol=1e-14)

    def test_n_snapshots_extra_set(self):
        from aiida_feff.calcfunctions.larch import _average_xas_data_impl

        nodes = {f"snap_{i}": self._make_xas(float(i)) for i in range(5)}
        avg = _average_xas_data_impl(**nodes)
        assert avg.base.extras.get("n_snapshots") == 5

    def test_empty_raises(self):
        from aiida_feff.calcfunctions.larch import _average_xas_data_impl

        with pytest.raises(ValueError, match="No XasData nodes"):
            _average_xas_data_impl()


@requires_larch
class TestChiKToR:
    def test_output_arrays_present(self, generate_xas_data):
        from aiida.orm import Dict

        from aiida_feff.calcfunctions.larch import chi_k_to_r

        xas = generate_xas_data()
        ft_params = Dict({"kmin": 3.0, "kmax": 12.0, "kweight": 2, "rmax": 6.0})
        result = chi_k_to_r(xas_data=xas, ft_params=ft_params)
        for name in ("r", "chir_mag", "chir_re", "chir_im"):
            assert name in result.get_arraynames()
