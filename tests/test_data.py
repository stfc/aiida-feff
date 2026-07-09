"""Tests for FeffParameters and XasData data nodes."""

import pytest

from aiida_feff.data.parameters import FeffParameters


class TestFeffParameters:
    def test_valid_minimal(self):
        p = FeffParameters(dict={"edge": "K"})
        p.validate()

    def test_valid_full(self, generate_feff_parameters):
        p = generate_feff_parameters(spectrum_type="EXAFS", radius=7.0, nleg=6)
        p.validate()

    def test_invalid_edge(self):
        with pytest.raises(ValueError, match="edge must be one of"):
            FeffParameters(dict={"edge": "Z99"}).validate()

    def test_missing_edge(self):
        with pytest.raises(Exception):
            FeffParameters(dict={"s02": 0.9}).validate()

    def test_edge_property(self, generate_feff_parameters):
        p = generate_feff_parameters(edge="L2")
        assert p.edge == "L2"

    def test_radius_property(self, generate_feff_parameters):
        p = generate_feff_parameters(radius=6.5)
        assert p.radius == pytest.approx(6.5)

    def test_scf_null_behaviour(self):
        p_none = FeffParameters(
            dict={
                "edge": "K",
                "scf": None,
            }
        )
        tags_none = p_none.to_pymatgen_user_tags()
        assert "SCF" not in tags_none
        assert "SCF" in tags_none.get("_del", [])




class TestXasData:
    def test_set_and_get_spectrum(self, generate_xas_data):
        xas = generate_xas_data()
        assert xas.energy.shape == (200,)
        assert xas.mu.shape == (200,)

    def test_set_and_get_chi(self, generate_xas_data):
        xas = generate_xas_data()
        assert xas.chi_k.shape == (300,)
        assert xas.k.shape == (300,)

    def test_e0_extra(self, generate_xas_data):
        xas = generate_xas_data()
        assert xas.e0 == pytest.approx(7112.0)

    def test_arraynames(self, generate_xas_data):
        xas = generate_xas_data()
        names = xas.get_arraynames()
        assert "energy" in names
        assert "mu" in names
        assert "k" in names
        assert "chi_k" in names


class TestResolveAbsorberSites:
    """Unit tests for _resolve_absorber_sites (no AiiDA DB needed)."""

    @pytest.fixture
    def cu_structure(self, aiida_profile):
        from aiida import orm

        s = orm.StructureData(cell=[[3.6, 0, 0], [0, 3.6, 0], [0, 0, 3.6]])
        for pos in [[0, 0, 0], [1.8, 1.8, 0], [1.8, 0, 1.8], [0, 1.8, 1.8]]:
            s.append_atom(position=pos, symbols="Cu")
        s.store()
        return s

    @pytest.fixture
    def mixed_structure(self, aiida_profile):
        from aiida import orm

        s = orm.StructureData(cell=[[5, 0, 0], [0, 5, 0], [0, 0, 5]])
        s.append_atom(position=[0, 0, 0], symbols="Cu")
        s.append_atom(position=[2.5, 0, 0], symbols="Fe")
        s.append_atom(position=[0, 2.5, 0], symbols="Cu")
        s.store()
        return s

    def test_int_spec(self, cu_structure):
        from aiida_feff.workflows.ensemble import _resolve_absorber_sites

        assert _resolve_absorber_sites(cu_structure, 2) == [2]

    def test_element_string(self, cu_structure):
        from aiida_feff.workflows.ensemble import _resolve_absorber_sites

        result = _resolve_absorber_sites(cu_structure, "Cu")
        assert result == [0, 1, 2, 3]

    def test_explicit_list(self, cu_structure):
        from aiida_feff.workflows.ensemble import _resolve_absorber_sites

        assert _resolve_absorber_sites(cu_structure, [0, 1]) == [0, 1]

    def test_string_indices(self, cu_structure):
        from aiida_feff.workflows.ensemble import _resolve_absorber_sites

        assert _resolve_absorber_sites(cu_structure, "0,1,2") == [0, 1, 2]

    def test_string_single_index(self, cu_structure):
        from aiida_feff.workflows.ensemble import _resolve_absorber_sites

        assert _resolve_absorber_sites(cu_structure, "2") == [2]

    def test_element_relative_indices(self, cu_structure):
        # "Cu:0,2" → 1st and 3rd Cu sites → absolute indices 0 and 2
        from aiida_feff.workflows.ensemble import _resolve_absorber_sites

        assert _resolve_absorber_sites(cu_structure, "Cu:0,2") == [0, 2]

    def test_element_relative_single(self, cu_structure):
        from aiida_feff.workflows.ensemble import _resolve_absorber_sites

        assert _resolve_absorber_sites(cu_structure, "Cu:1") == [1]

    def test_relative_index_out_of_range(self, cu_structure):
        from aiida_feff.workflows.ensemble import _resolve_absorber_sites

        with pytest.raises(ValueError, match="out of range"):
            _resolve_absorber_sites(cu_structure, "Cu:99")

    def test_missing_element(self, cu_structure):
        from aiida_feff.workflows.ensemble import _resolve_absorber_sites

        with pytest.raises(ValueError, match="No atoms with element 'Fe'"):
            _resolve_absorber_sites(cu_structure, "Fe")

    def test_out_of_range(self, cu_structure):
        from aiida_feff.workflows.ensemble import _resolve_absorber_sites

        with pytest.raises(ValueError, match="out of range"):
            _resolve_absorber_sites(cu_structure, 99)

    def test_mixed_species_rejected(self, mixed_structure):
        from aiida_feff.workflows.ensemble import _resolve_absorber_sites

        with pytest.raises(ValueError, match="same element"):
            _resolve_absorber_sites(mixed_structure, [0, 1])

    def test_mixed_species_string_indices_rejected(self, mixed_structure):
        from aiida_feff.workflows.ensemble import _resolve_absorber_sites

        with pytest.raises(ValueError, match="same element"):
            _resolve_absorber_sites(mixed_structure, "0,1")

    def test_empty_list_rejected(self, cu_structure):
        from aiida_feff.workflows.ensemble import _resolve_absorber_sites

        with pytest.raises(ValueError, match="must not be empty"):
            _resolve_absorber_sites(cu_structure, [])
