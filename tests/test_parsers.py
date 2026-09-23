"""Tests for FeffParser using mock retrieved FolderData."""

import io
import textwrap

import numpy as np
import pytest
from aiida.orm import FolderData

from aiida_feff.calculations.feff import FeffCalculation

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


def _dense_xmuda() -> str:
    """A spectrum dense enough for larch's autobk spline to converge.

    SAMPLE_XMUDA has 8 data points, which is too few, so it exercises only
    the chi.dat fallback. Both branches need to be reachable for the
    chi_source attribute to be testable at all.
    """
    energy = np.arange(6900.0, 7800.0, 1.0)
    k = np.sqrt(np.clip(energy - 7112.0, 0.0, None) * 0.2624684)
    mu = np.where(energy < 7112.0, 0.1, 1.0 + 0.05 * np.sin(2 * k * 2.5) * np.exp(-0.03 * k * k))
    mu0 = np.where(energy < 7112.0, 0.1, 1.0)
    rows = [
        f"{energy[i] - 7112.0:12.4f} {energy[i]:12.4f} {k[i]:10.4f} "
        f"{mu[i]:12.6f} {mu0[i]:12.6f} {0.0:12.6f}"
        for i in range(len(energy))
    ]
    header = [
        "# Feff8L (EXAFS)  0.1",
        "# e0 = 7112.00",
        "#   omega      e        k        mu       mu0      chi",
    ]
    return "\n".join(header + rows) + "\n"


DENSE_XMUDA = _dense_xmuda()


@pytest.fixture()
def retrieved_ok():
    """FolderData containing chi.dat."""
    folder = FolderData()
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
            retrieved={"chi.dat": SAMPLE_CHI},
        )
        assert "xas_data" in result.outputs

    def test_xas_data_arrays(self, parse_retrieved):
        result = parse_retrieved(
            entry_point_name="feff.feff",
            retrieved={"chi.dat": SAMPLE_CHI},
        )
        xas = result.outputs.xas_data
        assert "k" in xas.get_arraynames()
        assert "chi_k" in xas.get_arraynames()
        assert "energy" not in xas.get_arraynames()
        assert "mu" not in xas.get_arraynames()

    def test_missing_chi_returns_error(self, parse_retrieved):
        """Parser must return ERROR_MISSING_CHIDAT when chi.dat is absent."""
        result = parse_retrieved(
            entry_point_name="feff.feff",
            retrieved={},
        )
        assert result.exit_status == FeffCalculation.exit_codes.ERROR_MISSING_CHIDAT.status


class TestPotentialsOnlyRuns:
    """A CONTROL card with modules 4-6 off legitimately produces no chi.dat."""

    @staticmethod
    def _parameters(control):
        from aiida_feff.data.parameters import FeffParameters

        return FeffParameters(dict={"edge": "K", "control": control})

    def test_missing_chi_is_success_for_a_potentials_only_run(self, parse_retrieved):
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

    def test_missing_chi_is_still_an_error_for_a_spectrum_run(self, parse_retrieved):
        from aiida_feff.calculations.feff import CONTROL_NO_POT

        result = parse_retrieved(
            entry_point_name="feff.feff",
            retrieved={"log.dat": "Feff8L (EXAFS)  0.1\n"},
            inputs={"parameters": self._parameters(CONTROL_NO_POT)},
        )
        assert result.exit_status == FeffCalculation.exit_codes.ERROR_MISSING_CHIDAT.status

    def test_a_potentials_run_that_never_started_is_not_success(self, parse_retrieved):
        """An empty log means FEFF never ran, whatever CONTROL asked for."""
        from aiida_feff.calculations.feff import CONTROL_POT_ONLY

        result = parse_retrieved(
            entry_point_name="feff.feff",
            retrieved={"log.dat": ""},
            inputs={"parameters": self._parameters(CONTROL_POT_ONLY)},
        )
        assert result.exit_status == FeffCalculation.exit_codes.ERROR_POTENTIALS_INCOMPLETE.status
        assert "xas_data" not in result.outputs


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

    def test_feff_and_library_versions_are_recorded(self, parse_retrieved):
        result = parse_retrieved(
            entry_point_name="feff.feff",
            retrieved={"chi.dat": SAMPLE_CHI, "log.dat": "  Feff 8.50L\n"},
        )
        xas = result.outputs.xas_data
        assert xas.base.attributes.get("feff_version") == "Feff8.50L"
        assert "xraylarch" in xas.base.attributes.get("code_versions")

    def test_chi_source_records_feff_chi_dat(self, parse_retrieved):
        result = parse_retrieved(
            entry_point_name="feff.feff",
            retrieved={"chi.dat": SAMPLE_CHI},
        )
        xas = result.outputs.xas_data
        assert xas.base.attributes.get("chi_source") == "feff.chi.dat"
        assert xas.get_array("chi_k")[2] == pytest.approx(0.08912, rel=1e-4)


class TestExcerptTraceback:
    def test_keeps_the_exception_not_the_header(self):
        from aiida_feff.parsers.feff import excerpt_traceback

        tb = "Traceback (most recent call last):\n" + "  frame\n" * 200 + "ValueError: boom"
        excerpt = excerpt_traceback(tb)
        assert excerpt.endswith("ValueError: boom")
        assert "Traceback (most recent" not in excerpt
