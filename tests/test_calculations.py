"""Tests for FeffCalculation (no live AiiDA daemon required).

Uses ``generate_calc_job`` from aiida-core's testing utilities to exercise
``prepare_for_submission`` without actually running FEFF.
"""

import pytest
from aiida import orm

from tests.helpers import atoms_rows, potentials_rows


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

    def test_retrieve_list_contains_chi_not_xmu(
        self, generate_calc_job, feff_calc_inputs, fixture_sandbox
    ):
        from aiida_feff.calculations.feff import FEFF_CHI_FILE, FEFF_XMUDA_FILE

        calc_info = generate_calc_job(
            folder=fixture_sandbox,
            entry_point_name="feff.feff",
            inputs=feff_calc_inputs,
        )
        assert FEFF_CHI_FILE in calc_info.retrieve_list
        assert FEFF_XMUDA_FILE not in calc_info.retrieve_list

    def test_feff_inp_contains_edge(self, generate_calc_job, feff_calc_inputs, fixture_sandbox):
        from aiida_feff.calculations.feff import FEFF_INPUT_FILE

        generate_calc_job(
            folder=fixture_sandbox,
            entry_point_name="feff.feff",
            inputs=feff_calc_inputs,
        )
        from pathlib import Path

        content = Path(fixture_sandbox.get_abs_path(FEFF_INPUT_FILE)).read_text()
        # Assert the card *values*, not just that the words appear: four
        # substring checks would pass on a file containing only those four
        # words concatenated, and would never notice the wrong edge.
        edge_lines = [ln.split() for ln in content.splitlines() if ln.strip().startswith("EDGE")]
        assert edge_lines == [["EDGE", "K"]], f"unexpected EDGE card: {edge_lines}"

        assert atoms_rows(content), "ATOMS block is empty"
        potentials = potentials_rows(content)
        assert potentials, "POTENTIALS block is empty"
        # ipot 0 is the absorber and must be declared exactly once.
        assert [row["ipot"] for row in potentials].count(0) == 1
        assert content.rstrip().endswith("END")

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
        # ...and the generated content is genuinely absent, rather than the
        # verbatim file merely being appended to it.
        assert "POTENTIALS" not in content


class TestFeffCalculationInputValidation:
    """The spec validator is the only place a CalcJob can refuse cleanly.

    ``CalcJob.presubmit`` assigns to ``calc_info.uuid`` and ``ExitCode`` is an
    immutable NamedTuple, so returning an exit code from
    ``prepare_for_submission`` raises ``AttributeError`` inside the daemon
    instead of failing the job. Every refusal below therefore has to happen
    at submission time, before the node exists -- which is what these tests
    pin. The batch calcjob has had an equivalent class since it was written;
    this one did not.
    """

    @staticmethod
    def _validate(**inputs):
        from aiida_feff.calculations.feff import FeffCalculation

        return FeffCalculation._validate_inputs(inputs, None)

    def test_accepts_structure_plus_parameters(self, generate_structure, generate_feff_parameters):
        assert (
            self._validate(structure=generate_structure(), parameters=generate_feff_parameters())
            is None
        )

    def test_accepts_a_verbatim_input_file(self):
        assert self._validate(feff_input_file=object()) is None

    def test_neither_input_route_is_refused(self):
        message = self._validate()
        assert message is not None
        assert "feff_input_file" in message and "structure" in message

    def test_structure_without_parameters_is_refused(self, generate_structure):
        assert self._validate(structure=generate_structure()) is not None

    def test_parameters_without_structure_is_refused(self, generate_feff_parameters):
        assert self._validate(parameters=generate_feff_parameters()) is not None

    def test_path_aggregation_without_python_code_is_refused(
        self, generate_structure, generate_feff_parameters
    ):
        """Aggregation runs under a remote interpreter that has to be supplied.

        Without this check the job queues, runs FEFF, then dies in the
        aggregation step -- after the wait.
        """
        message = self._validate(
            structure=generate_structure(),
            parameters=generate_feff_parameters(),
            path_cw_threshold=orm.Float(0.0),
        )
        assert message is not None
        assert "python_code" in message

    def test_path_aggregation_with_a_verbatim_file_is_refused(self):
        """Aggregation needs the absorbing element, which only parameters carry."""
        message = self._validate(
            feff_input_file=object(),
            python_code=object(),
            path_cw_threshold=orm.Float(0.0),
        )
        assert message is not None
        assert "absorbing element" in message

    def test_negative_threshold_disables_aggregation(
        self, generate_structure, generate_feff_parameters
    ):
        """A negative threshold means "no paths", so python_code is not needed."""
        assert (
            self._validate(
                structure=generate_structure(),
                parameters=generate_feff_parameters(),
                path_cw_threshold=orm.Float(-1.0),
            )
            is None
        )

    def test_none_value_is_tolerated(self):
        """AiiDA calls the validator with None during port introspection."""
        from aiida_feff.calculations.feff import FeffCalculation

        assert FeffCalculation._validate_inputs(None, None) is None


class TestFeffInpGeneration:
    """Unit-test build_feff_inp independently of CalcJob machinery."""

    def test_absorber_at_origin(self, generate_structure):
        """FEFF defines the absorber as ipot 0 sitting at the coordinate origin.

        The previous version of this test looked for the substring "Fe0",
        which pymatgen never emits -- the loop fell through and the test
        asserted nothing at all. It passed against any output, including an
        empty string. Hence the explicit "exactly one" guard below: an
        unfound absorber must fail, not silently skip.
        """
        from aiida_feff.calculations.feff import FeffCalculation
        from aiida_feff.data.parameters import FeffParameters

        structure = generate_structure()
        params = FeffParameters(dict={"edge": "K", "absorbing_atom": 0})
        text = FeffCalculation.build_feff_inp(structure, params)

        absorbers = [row for row in atoms_rows(text) if row["ipot"] == 0]
        assert len(absorbers) == 1, f"expected exactly one ipot-0 site, got {len(absorbers)}"

        absorber = absorbers[0]
        assert absorber["x"] == pytest.approx(0.0, abs=1e-6)
        assert absorber["y"] == pytest.approx(0.0, abs=1e-6)
        assert absorber["z"] == pytest.approx(0.0, abs=1e-6)
        # FEFF also reports the absorber's distance to itself as zero.
        assert absorber["distance"] == pytest.approx(0.0, abs=1e-6)

    def test_every_other_site_is_offset_from_the_absorber(self, generate_structure):
        """Guards against an ATOMS block that collapsed every site onto 0 0 0.

        ``test_absorber_at_origin`` alone would pass such a block.
        """
        from aiida_feff.calculations.feff import FeffCalculation
        from aiida_feff.data.parameters import FeffParameters

        text = FeffCalculation.build_feff_inp(
            generate_structure(), FeffParameters(dict={"edge": "K", "absorbing_atom": 0})
        )
        scatterers = [row for row in atoms_rows(text) if row["ipot"] != 0]
        assert scatterers, "ATOMS block has no scatterers"
        for row in scatterers:
            radius = (row["x"] ** 2 + row["y"] ** 2 + row["z"] ** 2) ** 0.5
            assert radius > 1e-6
            # The tabulated distance column must agree with the coordinates.
            assert radius == pytest.approx(row["distance"], abs=1e-3)

    def test_potentials_block(self, generate_structure, generate_feff_parameters):
        from aiida_feff.calculations.feff import FeffCalculation

        text = FeffCalculation.build_feff_inp(generate_structure(), generate_feff_parameters())
        rows = potentials_rows(text)
        assert rows, "POTENTIALS block is empty"

        # ipot 0 is the absorber. The old assertion was `"  0 " in text`,
        # which matches whitespace anywhere in the file -- including the
        # ATOMS coordinate columns -- so it could not distinguish a missing
        # POTENTIALS entry from a present one.
        by_ipot = {row["ipot"]: row for row in rows}
        assert 0 in by_ipot, f"no absorber potential declared: {sorted(by_ipot)}"
        assert by_ipot[0]["z"] == 26  # Fe
        assert by_ipot[0]["tag"] == "Fe"
        # ipots must be contiguous from 0, which is what FEFF requires.
        assert sorted(by_ipot) == list(range(len(by_ipot)))

    def test_scf_null_removes_scf_line(self, generate_structure):
        from aiida_feff.calculations.feff import FeffCalculation
        from aiida_feff.data.parameters import FeffParameters

        structure = generate_structure()
        params = FeffParameters(dict={"edge": "K", "scf": None})
        text = FeffCalculation.build_feff_inp(structure, params)
        assert "SCF" not in text


class TestExcludeHydrogen:
    """Tests for exclude_hydrogen behaviour in build_feff_inp."""

    def test_h_absent_from_atoms_block(self, generate_h_bearing_structure):
        """With exclude_hydrogen=True no H line should appear in ATOMS block."""
        from aiida_feff.calculations.feff import FeffCalculation
        from aiida_feff.data.parameters import FeffParameters

        structure = generate_h_bearing_structure()
        params = FeffParameters(dict={"edge": "K", "absorbing_atom": 1, "exclude_hydrogen": True})
        text = FeffCalculation.build_feff_inp(structure, params)
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
        text = FeffCalculation.build_feff_inp(structure, params)
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
            FeffCalculation.build_feff_inp(structure, params)


def _is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
