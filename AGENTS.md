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
- Multiple-scattering `reff` is half the total path length, per FEFF. The
  larch oracle in `test_exafs.py` covers one MS path (`feff0007.dat`,
  `nlegs=3`) for exactly this reason; keep an MS path in `ORACLE_DATS`.
- A FEFF-produced `XasData` holds χ(k) alone, read from `chi.dat`. `xmu.dat`
  is not retrieved and larch's `autobk` is not run over simulated data, so
  `XasData.energy`, `.mu` and a meaningful `.e0` exist only on experimental
  imports (`calcfunctions.experimental`). Check `get_arraynames()` before
  reaching for μ(E).
- Fourier-transform defaults, including `window="kaiser"`, are re-exported
  as `calcfunctions.larch.FT_DEFAULTS` from `md_exafs.spectra`. `xftf_arrays` is the only call site of
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

## `clean_scratch` must stay lossless

The batch driver can strip each `snap_*` directory on the fly to stop
`feffNNNN.dat` and the potential binaries exhausting scratch on long ensembles.
The keep-set is *derived* from the calcjob's retrieve list
(`feff_batch._scratch_keep_files`), never restated, so what survives on the
cluster is exactly what AiiDA pulls back and the parser may read. Keep it that
way: the moment stripping removes something retrievable, `clean_scratch`
becomes a second physics setting wearing a disk-usage label.

Concretely, letting the parser fall back to `batch_shard.h5` when `snap_*` is
gone looks harmless and is not. The shard holds χ(k) resampled onto md-exafs'
fixed 0.05–19.95 Å⁻¹ grid, which is lossy whenever FEFF's own grid differs,
and it holds no `paths.dat` or `files.dat` for the path machinery to read.
Retrieving the run directory and reading the shard are not interchangeable.
That path was tried and removed.

`tests/test_ensemble_run.py::test_clean_scratch_does_not_change_the_spectrum`
runs the workchain twice and compares every array bit-for-bit. Any test that
asserts only `n_snapshots` or the node type will pass while the spectrum
changes underneath it.

## One χ(k) grid, two ways to fall off it

Every spectrum reaching the archive is resampled onto md-exafs'
`DEFAULT_K_GRID` (0.05–19.95 Å⁻¹), because spectra have to share a grid before
they can be averaged and FEFF's own grid follows the `EXAFS` k_max card. Two
failure modes hide in that resampling and they need opposite treatment, which
is why `md_exafs.spectra.resample_chi` owns it and nothing else calls
`np.interp` on χ(k):

- A destination point beyond the source by **less than half a source step** is
  grid registration, not missing data. FEFF writes `chi.dat` with 400 or 401
  rows depending on whether its grid starts at k=0, and `np.arange(0.05, 20.0,
  0.05)` overshoots its own last point by 3.6e-15. `resample_chi` carries the
  endpoint value across. `DEFAULT_K_GRID` is now built by exact division so
  the overshoot does not arise in the first place.
- Anything further out is a genuinely shorter run. It becomes NaN,
  `average_chi_arrays` masks it, and `n_contributors` records the coverage.

NaN then must not reach larch. `xftf` multiplies by the window, `0 * nan` is
`nan`, and one uncovered point anywhere on the grid turns **every** χ(R) value
into NaN. `xftf_arrays` zero-fills before transforming, which is what a
finite-range Fourier transform does anyway, and warns when the uncovered k
fall inside the window — there the zeros damp |χ(R)| and the answer is to
lower `kmax`, not to ignore it.

`tests/test_ensemble_run.py::TestShortChiDat` runs the workchain against a
stand-in whose `chi.dat` stops at k=14. The default stand-in writes exactly
`DEFAULT_K_GRID`, so it cannot see any of this.

## Averaging χ(k) has one implementation

`md_exafs.spectra.average_chi_arrays` is it. Four call sites used to
interpolate and combine spectra themselves, with three different conventions
for k outside a member's range, and the differences were invisible until an
ensemble came out ragged. Both execution routes now converge on it: the batch
driver writes shards, `create_serial_shard` writes the serial equivalent, and
`merge_exafs_shards` averages either into the `archive`. The `averaged_xas`
outputs are a *projection* of that archive via `archive_to_averaged_xas`, not
a second average over the per-snapshot nodes.

Reimplementing the mean "just for a plot" is how the two answers drift apart;
`aiidalab-exafs` had such a copy and it is gone.

## Minimum-image safety

A neighbour cutoff above the cell's inscribed-sphere radius biases distances
downward and MSRD by tens of percent. The check now lives in md-exafs. Any
distance-based test needs a cell large enough for its cutoff — use
`conftest.bcc_supercell_positions` or the `generate_trajectory` fixture's
`reps` argument, not a bigger cutoff.

## Silently-ignored keys are errors here

`FeffParameters` rejects keys it does not read, and `resolve_ft_params`
(re-exported from md-exafs) rejects unknown FT keys. A key that is accepted
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

### The physics now lives in md-exafs

`src/` re-exports `path_chi`, `xftf_arrays`, `resolve_ft_params` and
`scaled_chi_arrays` from md-exafs as the *same objects*. A test that imports
one symbol from each package and compares them therefore compares a function
to itself and cannot fail — `tests/test_numerical_parity.py` did exactly that
and was deleted. Test the delta this plugin adds (provenance, node
attributes, AiiDA plumbing, negative-index rejection), and keep the larch
oracle as an integration guard against an md-exafs regression. `md-exafs` is
capped at `<0.4` so a physics-affecting bump has to be a deliberate PR here.

### The FEFF stand-in must read its input

`tests/test_ensemble_run.py` runs the whole workchain outline in-process
against a shell script standing in for FEFF. That script parses `feff.inp`
and derives its spectrum from the nearest-neighbour distance, so each
snapshot has a known, distinct, correct answer. An earlier constant version
made every frame and site identical, which silently reduced
`test_batch_and_serial_paths_agree` and the ensemble-average tests to
comparing copies of one array — they could not detect a dropped snapshot, a
duplicated task, or a wrong frame/site assignment. If you touch the stand-in,
keep it input-dependent, and keep the guard assertions that fail when the
snapshots turn out identical.

Workchain children run **concurrently** under `run_get_node`. Do not key test
behaviour on invocation order via a shared counter file; key it on the input,
as `feff_rejecting_short_bonds` does.

Golden fixtures live in `tests/fixtures/aggregate_paths/` — real Feff8L output
from an SrTiO₃ Ti K-edge run. Their provenance and the reason they are
irreplaceable are in the README beside them.

`examples/` is not type-checked, but it *is* executed: the devcontainer CI job
runs `.devcontainer/smoke-examples.sh`, which drives every example against
real `feff@localhost` / `python3@localhost` codes and asserts on their output,
not merely their exit status. Three broken examples reached main before that
existed (an import of the deleted `calcfunctions.debye_waller`, a call to the
removed `store_msrd`, and a `title` key `FeffParameters` rejects), so an
example that no longer matches the API is now a failing check.

Physics still belongs in `src/`, where it is unit-tested, and examples should
call it. Plotting counts as physics here: χ(R) carries units Å^-(kweight+1),
so an example that reimplements the transform gets the axis units wrong
silently. `visualise.plot_chi_k` and `plot_chi_r` take `ax=` and forward
`**plot_kwargs` to `ax.plot` precisely so a caller can overlay several curves
without re-deriving any of it. Wrap raw arrays in an unstored `XasData` and
pass that, rather than dropping down to matplotlib.

## Debye-Waller is deliberately not provenance-tracked

σ² and ADPs are computed by calling md-exafs directly, not through a
`@calcfunction`. They are cheap to recompute and the trajectory they derive
from is already a stored node, so storing them adds graph weight without
adding recoverable information. The `store_msrd` / `store_adp` wrappers were
removed for this reason — do not reintroduce them as a "fix" for the missing
provenance link.

## The devcontainer is part of the build

`.devcontainer/` is exercised by a CI job that builds it and runs the suite
inside. It went unchanged from the second commit of the project while the
package moved underneath it, which is how it accumulated a destructive
database reset, a Podman-only compose key and a venv shared with the host.
Storage there is `core.sqlite_dos`, matching the tests; RabbitMQ is present
only because `examples/example_ensemble.py` calls `submit`. Images are
digest-pinned and `uv` is version-pinned, so changes to either are visible in
a diff.
