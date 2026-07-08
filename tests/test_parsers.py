"""Tests for FeffParser using mock retrieved FolderData."""

import io
import textwrap

import pytest
from aiida.orm import FolderData

# ---------------------------------------------------------------------------
# Sample output files
# ---------------------------------------------------------------------------

SAMPLE_XMUDA = textwrap.dedent("""\
    # FEFF  xmu.dat
    # Abs   Z=26 Rmt= 1.302 Rnm= 1.420 Fe
    # Pot 1 Z=26 Rmt= 1.302 Rnm= 1.420 Fe
    # kf=1.927  Vint=-13.233  Rs_int=2.012 mu=-0.117  kc=0.000
    # e0 = 7112.00
    #   omega    k       mu      mu0      mu_free   chi  |chi|  phase  amp
       -20.000   0.000   0.4521  0.4521   0.4521   0.000 0.000  0.000  0.000
       -10.000   0.000   0.5312  0.5312   0.5312   0.000 0.000  0.000  0.000
         0.000   0.000   0.8945  0.8945   0.8945   0.000 0.000  0.000  0.000
        10.000   1.620   0.7234  0.6100   0.6100   0.034 0.034  1.212  0.523
        20.000   2.290   0.6012  0.5800   0.5800   0.028 0.028  1.892  0.412
    """)

SAMPLE_CHI = textwrap.dedent("""\
    # FEFF  chi.dat
    #   k         chi(k)    |chi|     phase
      0.500     0.00000   0.00000   0.00000
      1.000     0.04523   0.04523   0.12340
      2.000     0.08912   0.08912   0.34560
      3.000     0.12345   0.12345   0.56780
    """)


@pytest.fixture()
def retrieved_ok():
    """FolderData containing both xmu.dat and chi.dat."""
    folder = FolderData()
    folder.base.repository.put_object_from_filelike(io.BytesIO(SAMPLE_XMUDA.encode()), "xmu.dat")
    folder.base.repository.put_object_from_filelike(io.BytesIO(SAMPLE_CHI.encode()), "chi.dat")
    return folder


class TestFeffParserHelpers:
    """Unit tests for the file-format helper functions (no AiiDA db needed)."""

    def test_parse_chi_shape(self):
        from aiida_feff.parsers.feff import _parse_chi

        k, chi = _parse_chi(SAMPLE_CHI)
        assert k.shape == (4,)
        assert chi.shape == (4,)

    def test_parse_chi_values(self):
        from aiida_feff.parsers.feff import _parse_chi

        k, chi = _parse_chi(SAMPLE_CHI)
        assert k[0] == pytest.approx(0.5)
        assert chi[2] == pytest.approx(0.08912, rel=1e-4)


class TestFeffParserIntegration:
    """Integration tests using aiida-core's parse_retrieved fixture."""

    def test_xas_data_output_present(self, parse_retrieved):
        """Parser must emit an xas_data output for a complete run."""
        result = parse_retrieved(
            entry_point_name="feff.feff",
            retrieved={"xmu.dat": SAMPLE_XMUDA, "chi.dat": SAMPLE_CHI},
        )
        assert "xas_data" in result.outputs

    def test_xas_data_arrays(self, parse_retrieved):
        result = parse_retrieved(
            entry_point_name="feff.feff",
            retrieved={"xmu.dat": SAMPLE_XMUDA, "chi.dat": SAMPLE_CHI},
        )
        xas = result.outputs.xas_data
        assert "energy" in xas.get_arraynames()
        assert "k" in xas.get_arraynames()
        assert "chi_k" in xas.get_arraynames()

    def test_no_chi_exit_ok(self, parse_retrieved):
        """Parser should succeed (exit 0) when chi.dat is absent."""
        result = parse_retrieved(
            entry_point_name="feff.feff",
            retrieved={"xmu.dat": SAMPLE_XMUDA},
        )
        assert result.exit_status == 0
        assert "xas_data" in result.outputs
        xas = result.outputs.xas_data
        assert "chi_k" not in xas.get_arraynames()

    def test_missing_xmuda_returns_error(self, parse_retrieved):
        """Parser must return ERROR_MISSING_XMUDA when xmu.dat is absent."""
        result = parse_retrieved(
            entry_point_name="feff.feff",
            retrieved={},
        )
        assert result.exit_status == 310
