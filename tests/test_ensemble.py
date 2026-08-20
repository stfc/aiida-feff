"""Tests for EnsembleExafsWorkChain helpers.

Running the full outline needs a FEFF binary, so the pieces that can be tested
without one — absorber resolution, frame ordering, dynamic-output harvesting —
are tested directly.
"""

from __future__ import annotations

import pytest
from aiida import orm

from aiida_feff.workflows.ensemble import dynamic_outputs


class TestDynamicOutputs:
    """Harvesting a dynamic namespace must be one pass, not one per key."""

    @pytest.fixture()
    def calc_with_outputs(self, generate_xas_data, aiida_profile_clean):
        from aiida.common.links import LinkType
        from aiida.orm import CalcJobNode

        node = CalcJobNode()
        node.set_process_type("aiida.calculations:feff.feff_batch")
        node.store()
        created = {}
        for label in ("snap_0000_site_0000", "snap_0001_site_0000"):
            out = generate_xas_data().store()
            out.base.links.add_incoming(
                node, link_type=LinkType.CREATE, link_label=f"xas_data__{label}"
            )
            created[label] = out
        other = generate_xas_data().store()
        other.base.links.add_incoming(
            node, link_type=LinkType.CREATE, link_label="path_contributions__snap_0000_site_0000"
        )
        return node, created

    def test_returns_only_the_requested_namespace(self, calc_with_outputs):
        node, created = calc_with_outputs
        assert set(dynamic_outputs(node, "xas_data")) == set(created)

    def test_keys_have_the_namespace_prefix_stripped(self, calc_with_outputs):
        node, created = calc_with_outputs
        harvested = dynamic_outputs(node, "xas_data")
        for label, expected in created.items():
            assert harvested[label].uuid == expected.uuid

    def test_absent_namespace_is_empty_not_an_error(self, calc_with_outputs):
        node, _created = calc_with_outputs
        assert dynamic_outputs(node, "not_a_namespace") == {}


class TestResolveAbsorberSites:
    """Absorber specifications, all validated to a single species."""

    @pytest.fixture()
    def structure(self, aiida_profile):
        s = orm.StructureData(cell=[[8.0, 0, 0], [0, 8.0, 0], [0, 0, 8.0]])
        for i, symbol in enumerate(["Cu", "Cu", "O", "Cu"]):
            s.append_atom(position=(float(i), 0.0, 0.0), symbols=symbol)
        return s

    @pytest.mark.parametrize(
        ("spec", "expected"),
        [
            (0, [0]),
            ([0, 1], [0, 1]),
            ("Cu", [0, 1, 3]),
            ("0,1", [0, 1]),
            ("Cu:0,2", [0, 3]),
        ],
    )
    def test_accepted_forms(self, structure, spec, expected):
        from aiida_feff.workflows.ensemble import _resolve_absorber_sites

        assert _resolve_absorber_sites(structure, spec) == expected

    def test_mixed_species_rejected(self, structure):
        from aiida_feff.workflows.ensemble import _resolve_absorber_sites

        with pytest.raises(ValueError, match="same element"):
            _resolve_absorber_sites(structure, [0, 2])

    def test_out_of_range_rejected(self, structure):
        from aiida_feff.workflows.ensemble import _resolve_absorber_sites

        with pytest.raises(ValueError, match="out of range"):
            _resolve_absorber_sites(structure, [99])

    def test_unknown_element_rejected(self, structure):
        from aiida_feff.workflows.ensemble import _resolve_absorber_sites

        with pytest.raises(ValueError, match="No atoms with element"):
            _resolve_absorber_sites(structure, "Au")

    def test_empty_list_rejected(self, structure):
        from aiida_feff.workflows.ensemble import _resolve_absorber_sites

        with pytest.raises(ValueError, match="must not be empty"):
            _resolve_absorber_sites(structure, [])


class TestControlCards:
    """The two CONTROL strings live in one place, shared by both paths."""

    def test_potentials_and_spectrum_cards_are_complementary(self):
        from aiida_feff.calculations.feff import (
            CONTROL_NO_POT,
            CONTROL_POT_ONLY,
            is_potentials_only,
        )

        assert is_potentials_only(CONTROL_POT_ONLY)
        assert not is_potentials_only(CONTROL_NO_POT)

    def test_batch_and_workchain_use_the_same_constants(self):
        from aiida_feff.calculations import feff, feff_batch
        from aiida_feff.workflows import ensemble

        assert ensemble.CONTROL_NO_POT is feff.CONTROL_NO_POT
        assert feff_batch.CONTROL_NO_POT is feff.CONTROL_NO_POT
