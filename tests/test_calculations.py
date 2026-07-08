"""Tests for FeffCalculation (no live AiiDA daemon required).

Uses ``generate_calc_job`` from aiida-core's testing utilities to exercise
``prepare_for_submission`` without actually running FEFF.
"""

import pytest
from aiida import orm


@pytest.fixture()
def feff_calc_inputs(generate_structure, generate_feff_parameters):
    """Assemble a minimal input dict for FeffCalculation."""
    return {
        "structure": generate_structure(),
        "parameters": generate_feff_parameters(),
        # code is injected by generate_calc_job
    }


class TestFeffCalculationPrepare:
    """Test prepare_for_submission output without running FEFF."""

    def test_feff_inp_written(self, generate_calc_job, feff_calc_inputs, fixture_sandbox):
        """feff.inp must appear in the sandbox after prepare_for_submission."""
        from aiida_feff.calculations.feff import FEFF_INPUT_FILE

        generate_calc_job(
            folder=fixture_sandbox,
            entry_point_name="feff.feff",
            inputs=feff_calc_inputs,
        )
        assert fixture_sandbox.isfile(FEFF_INPUT_FILE)

    def test_retrieve_list_contains_xmuda(
        self, generate_calc_job, feff_calc_inputs, fixture_sandbox
    ):
        from aiida_feff.calculations.feff import FEFF_XMUDA_FILE

        calc_info = generate_calc_job(
            folder=fixture_sandbox,
            entry_point_name="feff.feff",
            inputs=feff_calc_inputs,
        )
        assert FEFF_XMUDA_FILE in calc_info.retrieve_list

    def test_feff_inp_contains_edge(self, generate_calc_job, feff_calc_inputs, fixture_sandbox):
        from aiida_feff.calculations.feff import FEFF_INPUT_FILE

        generate_calc_job(
            folder=fixture_sandbox,
            entry_point_name="feff.feff",
            inputs=feff_calc_inputs,
        )
        from pathlib import Path

        content = Path(fixture_sandbox.get_abs_path(FEFF_INPUT_FILE)).read_text()
        assert "EDGE" in content
        assert "ATOMS" in content
        assert "POTENTIALS" in content
        assert "END" in content

    def test_verbatim_input_bypasses_generation(self, generate_calc_job, fixture_sandbox):
        """Supplying feff_input_file should skip structure-based generation."""
        import io

        from aiida_feff.calculations.feff import FEFF_INPUT_FILE

        dummy = "* hand-crafted\nEDGE K\nEND\n"
        sfd = orm.SinglefileData(io.BytesIO(dummy.encode()), filename=FEFF_INPUT_FILE)
        inputs = {"feff_input_file": sfd}

        generate_calc_job(
            folder=fixture_sandbox,
            entry_point_name="feff.feff",
            inputs=inputs,
        )
        from pathlib import Path

        content = Path(fixture_sandbox.get_abs_path(FEFF_INPUT_FILE)).read_text()
        assert "hand-crafted" in content


class TestFeffInpGeneration:
    """Unit-test _build_feff_inp independently of CalcJob machinery."""

    def test_absorber_at_origin(self, generate_structure):
        from aiida_feff.calculations.feff import FeffCalculation
        from aiida_feff.data.parameters import FeffParameters

        structure = generate_structure()
        params = FeffParameters(dict={"edge": "K", "absorbing_atom": 0})
        text = FeffCalculation._build_feff_inp(structure, params)

        # The absorber should sit at 0 0 0
        for line in text.splitlines():
            if "Fe0" in line:
                parts = line.split()
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                assert abs(x) < 1e-6
                assert abs(y) < 1e-6
                assert abs(z) < 1e-6
                break

    def test_potentials_block(self, generate_structure, generate_feff_parameters):
        from aiida_feff.calculations.feff import FeffCalculation

        text = FeffCalculation._build_feff_inp(generate_structure(), generate_feff_parameters())
        assert "POTENTIALS" in text
        # ipot 0 must be present (absorber)
        assert "  0 " in text or "   0 " in text

    def test_scf_null_removes_scf_line(self, generate_structure):
        from aiida_feff.calculations.feff import FeffCalculation
        from aiida_feff.data.parameters import FeffParameters

        structure = generate_structure()
        params = FeffParameters(dict={"edge": "K", "scf": None})
        text = FeffCalculation._build_feff_inp(structure, params)
        assert "SCF" not in text


class TestExcludeHydrogen:
    """Tests for exclude_hydrogen behaviour in _build_feff_inp."""

    def test_h_absent_from_atoms_block(self, generate_h_bearing_structure):
        """With exclude_hydrogen=True no H line should appear in ATOMS block."""
        from aiida_feff.calculations.feff import FeffCalculation
        from aiida_feff.data.parameters import FeffParameters

        structure = generate_h_bearing_structure()
        params = FeffParameters(dict={"edge": "K", "absorbing_atom": 1, "exclude_hydrogen": True})
        text = FeffCalculation._build_feff_inp(structure, params)
        assert " H " not in text and not any(
            line.strip().endswith(" H") for line in text.splitlines()
        )

    def test_absorbing_atom_index_remaps_correctly(self, generate_h_bearing_structure):
        """absorbing_atom=1 (Fe at origin, after H at index 0) remaps to 0 and sits at 0 0 0."""
        from aiida_feff.calculations.feff import FeffCalculation
        from aiida_feff.data.parameters import FeffParameters

        structure = generate_h_bearing_structure()
        # H is index 0; Fe-at-origin is index 1. After stripping H, Fe-at-origin → index 0 (absorber).
        params = FeffParameters(dict={"edge": "K", "absorbing_atom": 1, "exclude_hydrogen": True})
        text = FeffCalculation._build_feff_inp(structure, params)
        origin_lines = [
            line
            for line in text.splitlines()
            if line.strip()
            and not line.startswith("*")
            and all(abs(float(p)) < 1e-5 for p in line.split()[:3] if _is_float(p))
            and "Fe" in line
        ]
        assert origin_lines, "Absorber (Fe) not found at origin in ATOMS block"

    def test_absorber_is_h_raises(self, generate_h_bearing_structure):
        """absorbing_atom pointing at an H site with exclude_hydrogen=True must raise."""
        from aiida_feff.calculations.feff import FeffCalculation
        from aiida_feff.data.parameters import FeffParameters

        structure = generate_h_bearing_structure()
        # H is at index 0
        params = FeffParameters(dict={"edge": "K", "absorbing_atom": 0, "exclude_hydrogen": True})
        with pytest.raises(ValueError, match="hydrogen"):
            FeffCalculation._build_feff_inp(structure, params)


def _is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
