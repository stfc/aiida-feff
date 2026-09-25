"""Shared pytest fixtures for aiida-feff tests."""

import json
import shutil
from pathlib import Path

import numpy as np
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


#: Lattice parameter of the BCC-Fe test system (Å).
BCC_FE_A = 2.87
#: Nearest-neighbour distance in that lattice: the body diagonal half-length.
BCC_FE_NN = BCC_FE_A * np.sqrt(3) / 2


def bcc_supercell_positions(reps: int, a: float = BCC_FE_A) -> tuple[np.ndarray, np.ndarray]:
    """Return (equilibrium positions, cell) for a ``reps``³ BCC supercell.

    A genuine supercell, not a tiled 2-atom basis: tiling the basis puts
    several atoms at the *same* coordinates, which makes any distance-based
    test meaningless.
    """
    basis = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    offsets = np.array([[i, j, k] for i in range(reps) for j in range(reps) for k in range(reps)])
    frac = (offsets[:, None, :] + basis[None, :, :]).reshape(-1, 3)
    cell = np.eye(3) * (a * reps)
    return frac * a, cell


@pytest.fixture()
def generate_trajectory():
    """Return a factory that creates a BCC-Fe TrajectoryData node.

    ``reps=2`` by default, giving a 5.74 Å cell whose minimum-image-safe
    cutoff is 2.87 Å — enough for the first two shells.  Distance-based tests
    must keep their cutoff below :func:`_max_safe_mic_cutoff` of the cell they
    use, so raise ``reps`` rather than the cutoff.
    """
    from aiida.orm import TrajectoryData

    def _generate(
        n_frames: int = 10,
        reps: int = 2,
        sigma: float = 0.05,
        seed: int = 0,
        n_atoms: int | None = None,
    ):
        rng = np.random.default_rng(seed)
        eq, cell = bcc_supercell_positions(reps)
        if n_atoms is not None:
            eq = eq[:n_atoms]
        positions = eq[np.newaxis] + rng.normal(scale=sigma, size=(n_frames, len(eq), 3))
        cells = np.tile(cell, (n_frames, 1, 1))

        traj = TrajectoryData()
        traj.set_array("positions", positions)
        traj.set_array("cells", cells)
        traj.set_array("steps", np.arange(n_frames))
        traj.base.attributes.set("symbols", ["Fe"] * len(eq))
        return traj

    return _generate


#: Real Feff8L output checked into the repo; see the README beside it.
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "aggregate_paths"


def import_remote_script(name: str):
    """Import one of the compute-node scripts by path.

    ``_aggregate_paths.py`` and ``_run_batch.py`` are shipped to the cluster
    and run under an interpreter with no aiida-core, so they are imported here
    the same way — by file, not as a package member.
    """
    import importlib.util

    path = Path(__file__).parent.parent / "src" / "aiida_feff" / "calculations" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def feff_workdir(tmp_path):
    """A temporary directory pre-populated with the real FEFF fixtures."""
    for src in [*FIXTURE_DIR.glob("feff????.dat"), FIXTURE_DIR / "files.dat"]:
        shutil.copy(src, tmp_path / src.name)
    return tmp_path


@pytest.fixture()
def aggregated_hdf5(feff_workdir, monkeypatch):
    """Run the real aggregation script over the fixtures; return the HDF5 bytes."""
    pytest.importorskip("larch.xafs.feffdat")
    (feff_workdir / "_feff_aggregate_config.json").write_text(
        json.dumps({"threshold": 0.0, "frame_idx": 3, "site_idx": 1, "absorber_element": "Ti"})
    )
    monkeypatch.chdir(feff_workdir)
    module = import_remote_script("_aggregate_paths.py")
    module.main()
    return (feff_workdir / "contributions_raw.h5").read_bytes()


@pytest.fixture()
def aggregated_node(aggregated_hdf5, aiida_profile):
    """A PathContributionsData built from real FEFF output."""
    from aiida_feff.data.pathcontributions import PathContributionsData

    return PathContributionsData.from_hdf5_bytes(aggregated_hdf5)


@pytest.fixture()
def generate_xas_data():
    """Return a factory that creates a populated XasData node."""
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

    def _parse(entry_point_name, retrieved, inputs=None, ensemble_inputs=True):
        folder = FolderData()
        for filename, content in retrieved.items():
            data = content.encode() if isinstance(content, str) else content
            folder.base.repository.put_object_from_filelike(_io.BytesIO(data), filename)
        folder.store()

        node = CalcJobNode()
        # Follow the entry point given, so this fixture can drive either parser.
        node.set_process_type(f"aiida.calculations:{entry_point_name}")

        # site_idx and frame_idx carry port defaults, and structure /
        # parameters are supplied by every generated (non-verbatim) run, so a
        # real calcjob usually carries all four and FeffParser reads them to
        # label the spectrum.  They are attached by default so the common shape
        # is the one under test.
        #
        # ``ensemble_inputs=False`` gives the other shape that genuinely
        # occurs: a run driven by a verbatim ``feff_input_file``, for which
        # structure and parameters are absent (both ports are required=False
        # and _validate_inputs accepts feff_input_file alone).
        links = dict(inputs or {})
        if entry_point_name == "feff.feff" and ensemble_inputs:
            from aiida_feff.data.parameters import FeffParameters

            structure = orm.StructureData(cell=[[2.87, 0, 0], [0, 2.87, 0], [0, 0, 2.87]])
            structure.append_atom(position=(0.0, 0.0, 0.0), symbols="Fe")
            links.setdefault("structure", structure)
            links.setdefault("site_idx", orm.Int(0))
            links.setdefault("frame_idx", orm.Int(0))
            links.setdefault(
                "parameters",
                FeffParameters(dict={"edge": "K", "radius": 5.5, "absorbing_atom": 0}),
            )

        for label, input_node in links.items():
            stored = input_node if input_node.is_stored else input_node.store()
            node.base.links.add_incoming(stored, link_type=LinkType.INPUT_CALC, link_label=label)
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
