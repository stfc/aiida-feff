"""Tests for the plotting helpers."""

from __future__ import annotations

import pytest
from aiida.orm import Dict

pytest.importorskip("matplotlib")
pytest.importorskip("larch.xafs")


@pytest.fixture(autouse=True)
def headless_backend():
    import matplotlib

    matplotlib.use("Agg")


class TestPlotMuE:
    def test_returns_a_figure(self, generate_xas_data, aiida_profile):
        from aiida_feff.visualise import plot_mu_e

        assert plot_mu_e(generate_xas_data()).axes

    def test_mu0_is_optional_extra_trace(self, generate_xas_data, aiida_profile):
        from aiida_feff.visualise import plot_mu_e

        with_mu0 = plot_mu_e(generate_xas_data(), show_mu0=True)
        assert len(with_mu0.axes[0].get_lines()) == 2


class TestPlotChiK:
    def test_applies_k_weighting(self, generate_xas_data, aiida_profile):
        import numpy as np

        from aiida_feff.visualise import plot_chi_k

        xas = generate_xas_data()
        fig = plot_chi_k(xas, kweight=2)
        _k, y = fig.axes[0].get_lines()[0].get_data()
        np.testing.assert_allclose(y, xas.k**2 * xas.chi_k)


class TestPlotChiR:
    def test_axis_label_follows_kweight(self, generate_xas_data, aiida_profile):
        from aiida_feff.visualise import plot_chi_r

        # chi(R) from a k^n transform has units A^-(n+1); a hard-coded A^-3
        # label is wrong for every kweight but 2.
        fig = plot_chi_r(generate_xas_data(), ft_params={"kweight": 1})
        assert "-2" in fig.axes[0].get_ylabel()

    def test_reads_kweight_back_off_a_tracked_ft_node(self, generate_xas_data, aiida_profile):
        from aiida_feff.calcfunctions.larch import chi_k_to_r
        from aiida_feff.visualise import plot_chi_r

        chir = chi_k_to_r(generate_xas_data(), Dict({"kweight": 3}))
        fig = plot_chi_r(chir)
        assert "-4" in fig.axes[0].get_ylabel()

    def test_unknown_component_rejected(self, generate_xas_data, aiida_profile):
        from aiida_feff.visualise import plot_chi_r

        with pytest.raises(ValueError, match="component must be"):
            plot_chi_r(generate_xas_data(), component="magnitude")

    def test_unsupported_source_type_rejected(self, aiida_profile):
        from aiida.orm import Int

        from aiida_feff.visualise import plot_chi_r

        with pytest.raises(TypeError, match="Expected XasData or ArrayData"):
            plot_chi_r(Int(1))

    def test_inline_and_tracked_transforms_agree(self, generate_xas_data, aiida_profile):
        """One FT implementation, so a plot cannot disagree with provenance."""
        import numpy as np

        from aiida_feff.calcfunctions.larch import chi_k_to_r
        from aiida_feff.visualise import _xftf_inline

        xas = generate_xas_data()
        ft = {"kmin": 2.0, "kmax": 12.0, "kweight": 2}
        tracked = chi_k_to_r(xas, Dict(ft))
        r, mag, _re, _im, _params = _xftf_inline(xas, ft)
        np.testing.assert_allclose(r, tracked.get_array("r"))
        np.testing.assert_allclose(mag, tracked.get_array("chir_mag"))
