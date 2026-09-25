"""Tests for FeffParameters and XasData data nodes."""

import pytest
from aiida import orm

from aiida_feff.data.parameters import FeffParameters


class TestFeffParameters:
    def test_valid_minimal(self):
        p = FeffParameters(dict={"edge": "K", "radius": 5.5})
        p.validate()

    def test_missing_radius(self):
        """radius is required: it decides how big the calculation is.

        The key sets the cluster cutoff as well as the RPATH card, so a default
        picks the physics.  One did, invisibly: this node reported 5.5 while
        md-exafs wrote feff.inp at its own 4.0, which for BCC Fe is 59 atoms
        against 15.
        """
        with pytest.raises(ValueError, match="'radius' is required"):
            FeffParameters(dict={"edge": "K"}).validate()

    def test_valid_full(self, generate_feff_parameters):
        p = generate_feff_parameters(spectrum_type="EXAFS", radius=7.0, nleg=6)
        p.validate()

    def test_invalid_edge(self):
        with pytest.raises(ValueError, match="edge must be one of"):
            FeffParameters(dict={"edge": "Z99", "radius": 5.5}).validate()

    def test_missing_edge(self):
        # Bare `pytest.raises(Exception)` would also pass on an ImportError,
        # AttributeError or TypeError -- i.e. on the validator being broken
        # rather than on it rejecting the input.
        with pytest.raises(ValueError, match="edge"):
            FeffParameters(dict={"s02": 0.9}).validate()

    @pytest.mark.parametrize(
        ("params", "match"),
        [
            ({"edge": "K", "radius": 5.5, "spectrum_type": "XANES"}, "spectrum_type"),
            ({"edge": "K", "radius": 0.0}, "radius"),
            ({"edge": "K", "radius": -1.0}, "radius"),
            ({"edge": "K", "radius": 5.5, "s02": -0.1}, "s02"),
        ],
    )
    def test_out_of_range_values_are_rejected(self, params, match):
        """These raises existed but nothing reached them.

        An unvalidated radius of 0 produces a FEFF run with an empty cluster,
        which fails hours later on the cluster rather than at submission.
        """
        with pytest.raises(ValueError, match=match):
            FeffParameters(dict=params).validate()

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
                "radius": 5.5,
                "scf": None,
            }
        )
        tags_none = p_none.to_pymatgen_user_tags()
        assert "SCF" not in tags_none
        assert "SCF" in tags_none.get("_del", [])


class TestXasData:
    def test_set_and_get_spectrum(self):
        """Round-trip actual values, not just shapes taken from a fixture.

        Shape assertions against a fixture that chose the shape pass for any
        implementation that stores an array of the right length -- including
        one that stores the wrong array.
        """
        import numpy as np

        from aiida_feff.data.xasdata import XasData

        energy = np.linspace(-20.0, 200.0, 11)
        mu = np.arange(11.0)
        mu0 = np.full(11, 0.5)

        xas = XasData()
        xas.set_spectrum(energy, mu, mu0, e0=7112.0)

        np.testing.assert_allclose(xas.energy, energy)
        np.testing.assert_allclose(xas.mu, mu)
        # NB: mu0 has no accessor property, unlike mu/k/chi_k.
        np.testing.assert_allclose(xas.get_array("mu0"), mu0)
        # energy is stored relative to E0; absolute_energy adds it back.
        np.testing.assert_allclose(xas.absolute_energy, energy + 7112.0)

    def test_set_and_get_chi(self):
        import numpy as np

        from aiida_feff.data.xasdata import XasData

        k = np.linspace(0.0, 15.0, 7)
        chi = np.sin(k)

        xas = XasData()
        xas.set_chi(k, chi)

        np.testing.assert_allclose(xas.k, k)
        np.testing.assert_allclose(xas.chi_k, chi)

    def test_fixture_shapes(self, generate_xas_data):
        """The shared fixture keeps the shapes other tests rely on."""
        xas = generate_xas_data()
        assert xas.energy.shape == (200,)
        assert xas.mu.shape == (200,)
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

    @pytest.mark.parametrize("spec", [-1, [0, -1], "-1", "0,-2"])
    def test_negative_indices_rejected(self, cu_structure, spec):
        """md-exafs resolves ``-1`` Python-style; a provenance graph must not.

        The specification is what gets stored, so accepting ``-1`` would record an
        input that does not identify the site actually computed.
        """
        from aiida_feff.workflows.ensemble import _resolve_absorber_sites

        with pytest.raises(ValueError, match="non-negative"):
            _resolve_absorber_sites(cu_structure, spec)


class TestFeffParametersKeyValidation:
    """An accepted-but-ignored key produces a default FEFF run silently."""

    def test_unknown_key_rejected(self):
        with pytest.raises(ValueError, match="Unknown FeffParameters key"):
            FeffParameters(dict={"edge": "K", "radius": 5.5, "not_a_card": 1})

    @pytest.mark.parametrize(
        ("wrong", "right"),
        [("calc_mode", "spectrum_type"), ("rpath", "radius")],
    )
    def test_common_mistakes_name_the_right_key(self, wrong, right):
        # These two spellings appeared in this repo's own README and examples,
        # where they silently did nothing.
        with pytest.raises(ValueError, match=right):
            FeffParameters(dict={"edge": "K", "radius": 5.5, wrong: 6.0})

    def test_every_documented_key_is_accepted(self):
        FeffParameters(
            dict={
                "edge": "K",
                "spectrum_type": "EXAFS",
                "radius": 6.0,
                "absorbing_atom": 0,
                "absorbing_atoms": "Fe",
                "exclude_hydrogen": False,
                "s02": 0.9,
                "nleg": 6,
                "scf": "4.0 0 30 0.2 1",
                "exchange": "0 0 0",
                "control": "1 1 1 1 1 1",
                "print": "1 0 0 0 0 3",
                "exafs": 20,
                "criteria": "4.0 2.5",
                "delete_tags": ["COREHOLE"],
            }
        ).validate()


class TestFeffParametersCards:
    """to_feff_cards backs `verdi data feff export`."""

    def test_includes_edge_and_rpath(self):
        cards = FeffParameters(dict={"edge": "L3", "radius": 6.5}).to_feff_cards()
        assert any(card.startswith("EDGE") and "L3" in card for card in cards)
        assert any(card.startswith("RPATH") and "6.5" in card for card in cards)

    def test_includes_user_cards(self):
        cards = FeffParameters(dict={"edge": "K", "radius": 5.5, "s02": 0.85}).to_feff_cards()
        assert any(card.startswith("S02") and "0.85" in card for card in cards)

    def test_deleted_cards_are_shown_as_comments_not_values(self):
        cards = FeffParameters(dict={"edge": "K", "radius": 5.5, "scf": None}).to_feff_cards()
        assert not any(card.startswith("SCF ") for card in cards)
        assert any(card.startswith("*") and "SCF" in card for card in cards)


class TestExafsArchiveData:
    """Tests for ExafsArchiveData node (ADR 0004)."""

    def test_archive_from_batch_shard(self, tmp_path):
        import numpy as np
        from md_exafs.hdf5 import BatchShardWriter
        from md_exafs.paths import PathResult

        from aiida_feff.data.archive import ExafsArchiveData

        shard_path = tmp_path / "batch_shard.h5"
        k_grid = np.linspace(2.0, 15.0, 100)
        chi = np.sin(k_grid)

        p = PathResult(
            frame_idx=0,
            site_idx=0,
            r_eff=2.5,
            nlegs=2,
            degeneracy=12.0,
            scatterer="Cu",
            cw_ratio=100.0,
            k=np.linspace(0.0, 20.0, 20),
            feff_data=np.ones((20, 6)),
        )

        with BatchShardWriter(shard_path, k_grid=k_grid) as w:
            w.add_task_result(frame_idx=0, site_idx=0, absorber_element="Cu", chi=chi, paths=[p])

        node = ExafsArchiveData(file=str(shard_path))
        assert node.is_shard
        assert not node.is_ensemble
        assert np.allclose(node.k, k_grid)
        assert np.allclose(node.chi, chi, atol=1e-5)

        paths = node.iter_paths()
        assert len(paths) == 1
        assert paths[0].scatterer == "Cu"

        xas = node.to_xas_data()
        assert np.allclose(xas.k, k_grid)
        assert np.allclose(xas.chi_k, chi, atol=1e-5)

    def test_archive_reads_after_the_source_file_is_gone(self, tmp_path):
        """A stored node must be readable from the repository alone.

        The node's contents live in the AiiDA repository, so reads have to
        materialise a local copy.  Doing that inside ``as_path()`` and handing the
        path out afterwards leaves a dangling reader, which only shows up once the
        original file no longer exists — as is the case for any node loaded in a
        later session.
        """
        import numpy as np
        from md_exafs.hdf5 import BatchShardWriter

        from aiida_feff.data.archive import ExafsArchiveData

        shard_path = tmp_path / "batch_shard.h5"
        k_grid = np.linspace(2.0, 15.0, 50)
        chi = np.cos(k_grid)
        with BatchShardWriter(shard_path, k_grid=k_grid) as w:
            w.add_task_result(frame_idx=0, site_idx=0, absorber_element="Cu", chi=chi)

        node = ExafsArchiveData(file=str(shard_path)).store()
        shard_path.unlink()

        reloaded = orm.load_node(node.pk)
        assert reloaded.is_shard
        assert np.allclose(reloaded.chi, chi, atol=1e-5)

        # The explicit reader must stay valid for the whole with-block.
        with reloaded.reader() as archive:
            assert np.allclose(archive.k, k_grid)
            assert np.allclose(archive.chi, chi, atol=1e-5)

    def test_archive_extracts_the_repository_file_only_once(self, tmp_path):
        """Reading several attributes must not re-copy the whole HDF5 each time."""
        import numpy as np
        from md_exafs.hdf5 import BatchShardWriter

        from aiida_feff.data.archive import ExafsArchiveData

        shard_path = tmp_path / "batch_shard.h5"
        k_grid = np.linspace(2.0, 15.0, 50)
        with BatchShardWriter(shard_path, k_grid=k_grid) as w:
            w.add_task_result(frame_idx=0, site_idx=0, absorber_element="Cu", chi=np.cos(k_grid))

        node = ExafsArchiveData(file=str(shard_path)).store()

        calls = 0
        original = type(node).as_path

        def counting_as_path(self):
            nonlocal calls
            calls += 1
            return original(self)

        monkeypatched = type(node)
        monkeypatched.as_path = counting_as_path
        try:
            _ = node.k, node.chi, node.is_shard, node.r
        finally:
            monkeypatched.as_path = original

        assert calls == 1, f"archive was extracted {calls} times for four attribute reads"
