"""Tests for FeffParser using mock retrieved FolderData."""

import io
import textwrap

import pytest
from aiida.orm import FolderData

# ---------------------------------------------------------------------------
# Sample output files
# ---------------------------------------------------------------------------

# Column layout as Feff8L writes it: omega is energy RELATIVE to e0, which is
# stated separately in the header.  The old fixture invented a 9-column format
# no FEFF emits, so it validated the parser against fiction.
SAMPLE_XMUDA = textwrap.dedent("""\
    # Feff8L (EXAFS)  0.1
    # Abs   Z=26 Rmt= 1.302 Rnm= 1.420 Fe
    # Pot 1 Z=26 Rmt= 1.302 Rnm= 1.420 Fe
    # kf=1.927  Vint=-13.233  Rs_int=2.012 mu=-0.117  kc=0.000
    # e0 = 7112.00
    #   omega      e        k        mu       mu0      chi
       -20.000  -20.000   0.000    0.4521   0.4521   0.0000
       -10.000  -10.000   0.000    0.5312   0.5312   0.0000
         0.000    0.000   0.000    0.8945   0.8945   0.0000
        10.000   10.000   1.620    0.7234   0.6100   0.0340
        20.000   20.000   2.290    0.6012   0.5800   0.0280
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


class TestPotentialsOnlyRuns:
    """A CONTROL card with modules 4-6 off legitimately produces no xmu.dat."""

    @staticmethod
    def _parameters(control):
        from aiida_feff.data.parameters import FeffParameters

        return FeffParameters(dict={"edge": "K", "control": control})

    def test_missing_xmu_is_success_for_a_potentials_only_run(self, parse_retrieved):
        from aiida_feff.calculations.feff import CONTROL_POT_ONLY

        result = parse_retrieved(
            entry_point_name="feff.feff",
            retrieved={"log.dat": "Feff8L (EXAFS)  0.1\n"},
            inputs={"parameters": self._parameters(CONTROL_POT_ONLY)},
        )
        # Its deliverable is pot.pad / phase.pad in the remote folder, so no
        # xas_data is emitted and the job is still finished_ok.
        assert result.exit_status == 0
        assert "xas_data" not in result.outputs

    def test_missing_xmu_is_still_an_error_for_a_spectrum_run(self, parse_retrieved):
        from aiida_feff.calculations.feff import CONTROL_NO_POT

        result = parse_retrieved(
            entry_point_name="feff.feff",
            retrieved={"log.dat": "Feff8L (EXAFS)  0.1\n"},
            inputs={"parameters": self._parameters(CONTROL_NO_POT)},
        )
        assert result.exit_status == 310


class TestIsPotentialsOnly:
    @pytest.mark.parametrize(
        ("control", "expected"),
        [
            ("1 1 1 0 0 0", True),
            ("0 0 0 1 1 1", False),
            ("1 1 1 1 1 1", False),
            ("1 1 1 0 0 1", False),
            ("1 1 1", False),  # malformed: too few modules
            ("", False),
            (None, False),
        ],
    )
    def test_detects_spectrum_modules(self, control, expected):
        from aiida_feff.calculations.feff import is_potentials_only

        assert is_potentials_only(control) is expected


class TestParsedMetadata:
    """What the parser records so the spectrum can be reproduced later."""

    def test_energy_is_relative_and_e0_is_absolute(self, parse_retrieved):
        result = parse_retrieved(entry_point_name="feff.feff", retrieved={"xmu.dat": SAMPLE_XMUDA})
        xas = result.outputs.xas_data
        assert xas.e0 == pytest.approx(7112.0)
        # omega runs from -20; absolute_energy puts it back on the real scale.
        assert xas.energy.min() == pytest.approx(-20.0)
        assert xas.absolute_energy.min() == pytest.approx(7092.0)

    def test_e0_is_an_attribute_not_an_extra(self, parse_retrieved):
        """Extras stay mutable after storage and are excluded from the hash."""
        result = parse_retrieved(entry_point_name="feff.feff", retrieved={"xmu.dat": SAMPLE_XMUDA})
        assert result.outputs.xas_data.base.attributes.get("e0") == pytest.approx(7112.0)

    def test_feff_and_library_versions_are_recorded(self, parse_retrieved):
        result = parse_retrieved(
            entry_point_name="feff.feff",
            retrieved={"xmu.dat": SAMPLE_XMUDA, "log.dat": "  Feff 8.50L\n"},
        )
        xas = result.outputs.xas_data
        assert xas.base.attributes.get("feff_version") == "Feff8.50L"
        assert "xraylarch" in xas.base.attributes.get("code_versions")

    def test_chi_source_distinguishes_autobk_from_feff(self, parse_retrieved):
        """larch's spline background and FEFF's mu0 are different definitions."""
        result = parse_retrieved(
            entry_point_name="feff.feff",
            retrieved={"xmu.dat": SAMPLE_XMUDA, "chi.dat": SAMPLE_CHI},
        )
        assert result.outputs.xas_data.base.attributes.get("chi_source") in {
            "larch.autobk",
            "feff.chi.dat",
        }


class TestExcerptTraceback:
    def test_keeps_the_exception_not_the_header(self):
        from aiida_feff.parsers.feff import excerpt_traceback

        tb = "Traceback (most recent call last):\n" + "  frame\n" * 200 + "ValueError: boom"
        excerpt = excerpt_traceback(tb)
        assert excerpt.endswith("ValueError: boom")
        assert "Traceback (most recent" not in excerpt
