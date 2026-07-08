"""Shared pytest fixtures for aiida-feff tests."""

import pytest
from aiida import orm

# aiida-core 2.x ships fixtures as a plugin — just importing them here
# makes them available; the profile/clean-db fixtures are re-exported so
# individual test files can request them by name.
pytest_plugins = ["aiida.tools.pytest_fixtures"]


@pytest.fixture()
def generate_structure():
    """Return a factory that creates a simple StructureData (BCC Fe)."""

    def _generate(symbol="Fe", a=2.87):
        s = orm.StructureData(cell=[[a, 0, 0], [0, a, 0], [0, 0, a]])
        s.append_atom(position=(0.0, 0.0, 0.0), symbols=symbol)
        s.append_atom(position=(a / 2, a / 2, a / 2), symbols=symbol)
        return s

    return _generate


@pytest.fixture()
def generate_h_bearing_structure():
    """Return a factory: H first, then two Fe atoms.

    Atom indices (0-based):
      0 → H  at (0.5, 0, 0)  (hydrogen — bad absorber)
      1 → Fe at origin        (non-H, good absorber)
      2 → Fe at body-centre   (non-H)

    Putting H first means stripping it shifts every subsequent index down by
    one, giving a non-trivial remap to exercise.
    """

    def _generate(a=2.87):
        s = orm.StructureData(cell=[[a, 0, 0], [0, a, 0], [0, 0, a]])
        s.append_atom(position=(0.5, 0.0, 0.0), symbols="H")
        s.append_atom(position=(0.0, 0.0, 0.0), symbols="Fe")
        s.append_atom(position=(a / 2, a / 2, a / 2), symbols="Fe")
        return s

    return _generate


@pytest.fixture()
def generate_feff_parameters():
    """Return a factory that creates a minimal FeffParameters node."""
    from aiida_feff.data.parameters import FeffParameters

    def _generate(**kwargs):
        defaults = {
            "edge": "K",
            "spectrum_type": "EXAFS",
            "radius": 5.5,
            "s02": 1.0,
        }
        defaults.update(kwargs)
        return FeffParameters(dict=defaults)

    return _generate


@pytest.fixture()
def generate_trajectory():
    """Return a factory that creates a small BCC-Fe TrajectoryData node."""
    import numpy as np
    from aiida.orm import TrajectoryData

    def _generate(n_frames: int = 10, n_atoms: int = 2, sigma: float = 0.05, seed: int = 0):
        rng = np.random.default_rng(seed)
        a = 2.87
        # BCC basis: corner + body-centre; tile to reach n_atoms
        bcc_basis = np.array([[0.0, 0.0, 0.0], [a / 2, a / 2, a / 2]])
        reps = int(np.ceil(n_atoms / 2))
        eq = np.tile(bcc_basis, (reps, 1))[:n_atoms]
        positions = eq[np.newaxis] + rng.normal(scale=sigma, size=(n_frames, n_atoms, 3))
        cells = np.tile([[a, 0, 0], [0, a, 0], [0, 0, a]], (n_frames, 1, 1))
        steps = np.arange(n_frames)

        traj = TrajectoryData()
        traj.set_array("positions", positions)
        traj.set_array("cells", cells)
        traj.set_array("steps", steps)
        traj.base.attributes.set("symbols", ["Fe"] * n_atoms)
        return traj

    return _generate


@pytest.fixture()
def generate_xas_data():
    """Return a factory that creates a populated XasData node."""
    import numpy as np

    from aiida_feff.data.xasdata import XasData

    def _generate():
        xas = XasData()
        energy = np.linspace(-20, 200, 200)
        mu = np.exp(-((energy - 30) ** 2) / 200)
        mu0 = np.ones_like(mu) * 0.5
        xas.set_spectrum(energy, mu, mu0, e0=7112.0)
        k = np.linspace(0, 15, 300)
        chi = 0.5 * np.sin(2 * 2.52 * k) * np.exp(-2 * 0.003 * k**2) / (k**2 + 0.1)
        xas.set_chi(k, chi)
        return xas

    return _generate


# ---------------------------------------------------------------------------
# CalcJob / Parser testing helpers (aiida-core 2.x compatible)
# ---------------------------------------------------------------------------


@pytest.fixture()
def fixture_sandbox(tmp_path):
    """Yield an :class:`~aiida.common.folders.SandboxFolder` backed by tmp_path."""
    from aiida.common.folders import SandboxFolder

    yield SandboxFolder(tmp_path / "sandbox")


@pytest.fixture()
def generate_calc_job(fixture_sandbox, aiida_localhost):
    """Return a factory that calls ``prepare_for_submission`` on a CalcJob.

    Usage::

        calc_info = generate_calc_job(
            folder=fixture_sandbox,
            entry_point_name="feff.feff",
            inputs={"structure": s, "parameters": p},
        )
    """
    from aiida.engine.utils import instantiate_process
    from aiida.manage import get_manager
    from aiida.orm import InstalledCode
    from aiida.plugins import CalculationFactory

    def _generate(folder, entry_point_name, inputs=None):
        inputs = dict(inputs or {})

        if "code" not in inputs:
            code = InstalledCode(
                label="feff-test",
                computer=aiida_localhost,
                filepath_executable="/usr/bin/feff",
            ).store()
            inputs["code"] = code

        inputs.setdefault("metadata", {})
        inputs["metadata"].setdefault("options", {})
        inputs["metadata"]["options"].setdefault("resources", {"num_machines": 1})

        manager = get_manager()
        runner = manager.get_runner()
        calc_class = CalculationFactory(entry_point_name)
        process = instantiate_process(runner, calc_class, **inputs)
        return process.prepare_for_submission(folder)

    return _generate


@pytest.fixture()
def parse_retrieved(aiida_profile_clean):
    """Return a factory that runs a Parser against mock output files.

    Usage::

        result = parse_retrieved(
            entry_point_name="feff.feff",
            retrieved={"xmu.dat": xmu_text, "chi.dat": chi_text},
        )
        assert "xas_data" in result.outputs
        assert result.exit_status == 0
    """
    import io as _io

    from aiida.common.links import LinkType
    from aiida.orm import CalcJobNode, FolderData
    from aiida.plugins import ParserFactory

    def _parse(entry_point_name, retrieved):
        folder = FolderData()
        for filename, content in retrieved.items():
            data = content.encode() if isinstance(content, str) else content
            folder.base.repository.put_object_from_filelike(_io.BytesIO(data), filename)
        folder.store()

        node = CalcJobNode()
        node.set_process_type("aiida.calculations:feff.feff")
        node.store()

        folder.base.links.add_incoming(node, link_type=LinkType.CREATE, link_label="retrieved")

        ParserCls = ParserFactory(entry_point_name)
        parser = ParserCls(node)
        exit_code = parser.parse()

        class _Result:
            exit_status = exit_code.status if exit_code else 0
            outputs = parser.outputs

        return _Result()

    return _parse
