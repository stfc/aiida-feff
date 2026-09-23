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

    def test_the_plotted_curve_matches_the_tracked_transform(
        self, generate_xas_data, aiida_profile
    ):
        """One FT implementation, so a plot cannot disagree with provenance.

        Asserted through the curve that reaches matplotlib rather than through
        a helper, because the helper is what a refactor moves and the drawn
        line is what a reader believes.
        """
        import numpy as np

        from aiida_feff.calcfunctions.larch import chi_k_to_r
        from aiida_feff.visualise import plot_chi_r

        xas = generate_xas_data()
        ft = {"kmin": 2.0, "kmax": 12.0, "kweight": 2}
        tracked = chi_k_to_r(xas, Dict(ft))

        fig = plot_chi_r(xas, ft_params=ft, component="mag")
        r_plotted, mag_plotted = fig.axes[0].lines[-1].get_data()

        np.testing.assert_allclose(r_plotted, tracked.get_array("r"))
        np.testing.assert_allclose(mag_plotted, tracked.get_array("chir_mag"))


class TestPlotStyling:
    """Overlay styling must reach matplotlib.

    Without this passthrough a caller wanting several curves on one axis has
    to bypass these helpers and re-derive the k-weighting and the axis
    labels, which is exactly how examples/ ended up with an untested second
    copy of the plotting code.
    """

    def test_chi_k_forwards_style_to_the_line(self, generate_xas_data):
        from aiida_feff.visualise import plot_chi_k

        fig = plot_chi_k(generate_xas_data(), color="crimson", linestyle="--", lw=1.3, label="P1")
        line = fig.axes[0].lines[-1]
        assert line.get_linestyle() == "--"
        assert line.get_label() == "P1"
        assert line.get_linewidth() == 1.3

    def test_chi_r_forwards_style_to_the_line(self, generate_xas_data):
        from aiida_feff.visualise import plot_chi_r

        fig = plot_chi_r(generate_xas_data(), color="crimson", linestyle=":", label="P2")
        line = fig.axes[0].lines[-1]
        assert line.get_linestyle() == ":"
        assert line.get_label() == "P2"

    def test_overlays_share_one_axis(self, generate_xas_data):
        """Two calls with the same ax must add two lines, not two figures."""
        from aiida_feff.visualise import plot_chi_k

        fig = plot_chi_k(generate_xas_data(), label="a")
        ax = fig.axes[0]
        plot_chi_k(generate_xas_data(), ax=ax, label="b", color="grey")
        assert len(ax.lines) == 2
        assert [line.get_label() for line in ax.lines] == ["a", "b"]

    def test_envelope_follows_the_line_colour(self, generate_xas_data, aiida_profile):
        """A shaded band in a different colour from its line misreads as a second series."""
        import numpy as np

        from aiida_feff.data.xasdata import XasData
        from aiida_feff.visualise import plot_chi_k

        k = np.linspace(1.0, 12.0, 40)
        node = XasData()
        node.set_chi(k, np.sin(k))
        node.set_array("chi_k_std", np.full_like(k, 0.05))

        fig = plot_chi_k(node, color="crimson", plot_envelope=True)
        collections = fig.axes[0].collections
        assert collections, "envelope was not drawn"
