# aiida-feff

AiiDA plugin wrapping FEFF for provenance-tracked EXAFS, plus MD-EXAFS analysis
(ensemble averaging, Debye–Waller / MSRD, per-path χ(k) reconstruction).

`uv` is the runner: `uv run pytest`, `uv run ruff check src tests`,
`uv run mypy src`. CI runs `pre-commit run --all-files` plus pytest, so a clean
`pre-commit` locally means a clean lint job.

## Module map

Non-obvious pieces only; the rest is standard AiiDA plugin layout.

| Path | Runs where |
|---|---|
| `calculations/_run_batch.py` | **remote compute node**, under the HPC's Python |
| `calculations/_aggregate_paths.py` | **remote compute node**, after each FEFF run |
| everything else in `src/` | local AiiDA daemon / interactive session |

The two remote scripts are copied verbatim into the calcjob sandbox and executed
by an interpreter that has no aiida-core. Keep them to stdlib + numpy + h5py +
larch, absolute imports only, and self-contained — an `from aiida...` import
there fails only at runtime, on the cluster, after the queue wait. Tests import
them by file path via `tests/conftest.py::import_remote_script`.

## The code/feff_code inversion

`FeffCalculation.code` is the FEFF binary. `FeffBatchCalculation.code` is the
**Python interpreter**, and FEFF arrives as `feff_code`. `EnsembleExafsWorkChain`
swaps them when dispatching. Any new calcjob or workchain step touching batch
mode inherits this inversion; state which is which in the docstring.

## FEFF8L assumptions

The plugin targets FEFF8L as bundled by xraylarch. Three places bake that in:

- `FEFF_POTENTIAL_FILES` (`calculations/feff.py`) is the FEFF8L JSON + `.pad`
  set. FEFF9/10 write `pot.bin`/`phase.bin` instead.
- `build_feff_inp` deletes `COREHOLE` unconditionally, because FEFF8L rejects
  the card. Consequence: core-hole treatment is currently unreachable.
- `_aggregate_paths.py` parses `files.dat` by fixed column index and detects the
  data block by the literal `"amp ratio"` — the FEFF8L header.

Work that touches output parsing should say which FEFF wrote the fixture it was
validated against.

The generated `feff.inp` also comes from pymatgen's `MPEXAFSSet`, whose defaults
live in an unversioned `MPEXAFSSet.yaml` inside pymatgen. A pymatgen bump can
change every calculation's input; `build_feff_inp` stamps the generating
versions into a comment line so the change is visible in the graph, and diffing
a generated `feff.inp` across a pymatgen change is still worth doing.

## Physical conventions

- Lengths Å, energies eV throughout. Constants live in `constants.py`, derived
  from scipy rather than hard-coded. No Bohr/Rydberg conversion exists anywhere.
- Disorder is reported as **σ² (variance, Å²)**, never σ, and every second
  moment in the package uses the sample estimator (`ddof=1`).
- Multiple-scattering `reff` is half the total path length, per FEFF.
- The 3-body angle in `debye_waller.py` is the **included angle at the first
  scatterer**, i.e. 180° − FEFF's scattering angle. The scatterer–scatterer leg
  is the difference of the two absorber-relative MIC vectors, so the triangle
  closes; minimum-imaging it separately picks images that form no triangle.
- `XasData.energy` holds FEFF's `omega` column, energy **relative to E0**;
  `XasData.absolute_energy` adds `e0` back.
- Fourier-transform defaults, including `window="kaiser"`, live in
  `calcfunctions.larch.FT_DEFAULTS`. `xftf_arrays` is the only call site of
  larch's `xftf`; route new transforms through it rather than adding a fourth.
- χ(R) carries units Å^-(kweight+1), so `kweight` is recorded in the output
  node's `fourier_params` attribute and any axis label must read it back.

## Provenance rules

- Store scientific metadata (E0, k-weight, FT parameters, code versions) in node
  **attributes**, not extras: extras stay mutable after storage and are excluded
  from the node hash, so caching treats nodes with different physics as
  identical.
- Attributes can only be set on **unstored** nodes. A calcjob node is already
  stored inside `prepare_for_submission` and inside the parser, so metadata
  about the run goes onto the output Data nodes (`versions.py` collects it) or
  into a file in the sandbox.
- `CalcJob.presubmit` assigns to `calc_info.uuid`, and `ExitCode` is an
  immutable NamedTuple, so returning an exit code from `prepare_for_submission`
  raises `AttributeError` inside the daemon instead of failing the job cleanly.
  Validate in `spec.inputs.validator` (both calcjobs do) and raise from
  `prepare_for_submission`.
- Dynamic output namespaces are stored with `__`, not `.`: port `xas_data.snap_0`
  becomes link label `xas_data__snap_0`. Use
  `workflows.ensemble.dynamic_outputs`; searching for the dotted form silently
  matches nothing.

## Minimum-image safety

`_max_safe_mic_cutoff` returns the inscribed-sphere radius of the cell. A
neighbour cutoff above it biases distances downward and MSRD by tens of percent,
so `compute_msrd` **raises** rather than returning a quietly wrong number;
`allow_unsafe_cutoff=True` downgrades it to a warning. Any distance-based test
needs a cell large enough for its cutoff — use `conftest.bcc_supercell_positions`
or the `generate_trajectory` fixture's `reps` argument, not a bigger cutoff.

## Silently-ignored keys are errors here

`FeffParameters`, `compute_msrd` and `compute_adp` all reject keys they do not
read, and `resolve_ft_params` rejects unknown FT keys. A key that is accepted
and ignored produces a default run while the caller believes otherwise, and it
still changes the input node's hash. Add new keys to the corresponding
`VALID_KEYS` / `*_PARAM_KEYS` set in the same change that starts reading them.

## Testing

Every test loads a temporary AiiDA profile — `aiida.tools.pytest_fixtures`
installs session-scoped autouse fixtures. The backend is `core.sqlite_dos`, so
no PostgreSQL and no daemon are needed, but no subset runs profile-free.

Assert physics, not shape. `isinstance(result, Dict)` and `sigma2 >= 0` pass for
any implementation. These hold in the existing fixtures and actually constrain
the code:

- uncorrelated Gaussian displacements of width σ_disp give σ²_shell → 2σ_disp²
- the same give B → 8π²σ_disp²
- ⟨r⟩ → r_eq + σ_perp²/r_eq (perpendicular-displacement bias)
- χ(k) = sin(2kR₀) Fourier-transforms to a peak at R₀
- `path_chi` reproduces larch's own `FeffPathGroup` on the golden fixtures

The last one is the pattern to reach for: where an independent implementation of
the same physics exists, test against it rather than against a restatement of
your own formula. A test that imports the constant it checks proves nothing.

`tests/test_ensemble_run.py` runs the whole workchain outline in-process against
a shell script standing in for FEFF, which is how the batch path, potential
reuse and failure handling get covered without a cluster.

Golden fixtures live in `tests/fixtures/aggregate_paths/` — real Feff8L output
from an SrTiO₃ Ti K-edge run. Their provenance and the reason they are
irreplaceable are in the README beside them.

`examples/` is neither executed nor type-checked by CI, so physics belongs in
`src/`, where it is tested, and examples should call it. The EXAFS equation now
lives in `calcfunctions/exafs.py`; the synthetic example imports it.
