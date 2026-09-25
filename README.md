[![Release](https://img.shields.io/github/v/release/stfc/aiida-feff)](https://github.com/stfc/aiida-feff/releases)
[![PyPI](https://img.shields.io/pypi/v/aiida-feff)](https://pypi.org/project/aiida-feff/)
[![Pipeline Status](https://github.com/stfc/aiida-feff/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/stfc/aiida-feff/actions)
[![Coverage Status](https://coveralls.io/repos/github/stfc/aiida-feff/badge.svg?branch=main)](https://coveralls.io/github/stfc/aiida-feff?branch=main)

# aiida-feff

An [AiiDA](https://www.aiida.net) plugin for the
[FEFF](http://feffproject.org) real-space multiple-scattering code,
enabling fully provenance-tracked EXAFS calculations —
including ensemble averaging over MD snapshots via
[larch](https://xraypy.github.io/xraylarch/).

---

## Features

| Component | Description |
|-----------|-------------|
| `FeffCalculation` | CalcJob wrapping a single FEFF run; builds `feff.inp` from `StructureData` + `FeffParameters` |
| `FeffParser` | Parses `chi.dat` into `XasData` output nodes |
| `FeffBatchCalculation` | CalcJob that runs *N* FEFF instances in one Slurm job (one per core); for HPC ensemble runs |
| `FeffBatchParser` | Parses all per-snapshot outputs from a batch job into a dynamic namespace of `XasData` nodes |
| `FeffParameters` | Validated `Dict` subclass for FEFF control cards |
| `XasData` | `ArrayData` subclass storing χ(k), plus μ(E) on experimental imports |
| `EnsembleExafsWorkChain` | Fan-out over MD snapshots → one `ExafsArchiveData` and ensemble-averaged χ(k); single-job and batch modes |
| `calcfunctions` | Larch post-processing (FT), archive projection, experimental imports |
| `visualise` | Matplotlib helpers for χ(k) and χ(R) plots |

## Installation

```bash
pip install aiida-feff
verdi plugin list aiida.calculations   # should show feff.feff and feff.feff_batch
verdi plugin list aiida.workflows      # should show feff.ensemble
```

For plotting (optional):

```bash
pip install aiida-feff[plots]
```

## Quick start

### Experimental spectrum import

`aiida_feff.calcfunctions.experimental.import_experimental_spectrum` adapts
Larch's readers to a provenance-tracked `XasData` node. Pass a stored
`SinglefileData` upload and a `Dict` of Larch import options, such as plain
text column labels or a selected Athena group. The raw file remains an input
node; the plugin does not implement a competing file parser.

`scale_simulated_spectrum` stores a separate $S_0^2$/ΔE₀-adjusted simulated
`XasData` node for reproducible experimental comparisons.

### 1. Register your FEFF code

```bash
verdi code setup \
    --label feff \
    --computer localhost \
    --input-plugin feff.feff \
    --remote-abs-path /path/to/feff
```

### 2. Run a single-site EXAFS calculation

```python
from aiida import load_profile, orm
from aiida.engine import run
from aiida_feff.calculations.feff import FeffCalculation
from aiida_feff.data.parameters import FeffParameters

load_profile()

structure = orm.StructureData(cell=[[2.87,0,0],[0,2.87,0],[0,0,2.87]])
structure.append_atom(position=(0,0,0), symbols='Fe')
structure.append_atom(position=(1.435,1.435,1.435), symbols='Fe')

params = FeffParameters(dict={
    "edge": "K",
    "spectrum_type": "EXAFS",
    "radius": 5.5,           # FEFF RPATH / cluster radius in Å
    "s02": 1.0,
})

result = run(FeffCalculation,
    code=orm.load_code('feff@localhost'),
    structure=structure,
    parameters=params,
    metadata={"options": {
        "resources": {"num_machines": 1},
        "max_wallclock_seconds": 300,
    }},
)

xas = result['xas_data']
print(f"chi(k) shape: {xas.chi_k.shape}")
```

### 3. Ensemble EXAFS from an MD trajectory (localhost / dev)

> **Recommended environment:** the DevContainer in `.devcontainer/` gives you
> a full IDE against a *real* AiiDA profile: SQLite storage, a RabbitMQ broker,
> a running daemon, and both `feff@localhost` and `python3@localhost`
> registered. Calculations you run there produce real nodes and a real
> provenance graph, so `verdi` works as it would anywhere else.
>
> It runs on the host's own architecture, so it is equally at home in
> **GitHub Codespaces** (Code → Codespaces → Create) and on a local Docker or
> Podman. Verified end to end on Apple Silicon: `post-create.sh` completes,
> FEFF8L runs, and the full test suite passes inside the container.
>
> FEFF8L is not downloaded; it ships inside the `xraylarch` dependency and the
> container points a code at it. On arm64 those x86_64 binaries run through
> the runtime's qemu handler, for which post-create installs the x86_64
> loader. No PostgreSQL is involved: storage is `core.sqlite_dos`, the same
> backend the tests use.
>
> **Podman users:** set `dockerComposeFile` in
> `.devcontainer/devcontainer.json` to
> `["docker-compose.yml", "docker-compose.podman.yml"]`. The override adds
> `userns_mode: keep-id`, which Docker Engine rejects and which therefore
> cannot live in the base file.
>
> *Outside the container you need a broker and daemon running, a working FEFF
> executable, and a Python interpreter registered as an installed code
> (`verdi code create core.code.installed ...`, e.g. `python3@localhost`) for
> path aggregation.*

The synthetic example covers the whole pipeline and doubles as the smoke test
CI runs:

```bash
# Serial: one scheduler job per snapshot, with per-path contributions stored
uv run python examples/example_ensemble_synthetic.py \
    --code feff@localhost --python-code python3@localhost \
    --n-snapshots 6 --store-paths --plot-file ensemble.png

# Batch: one scheduler job per chunk, the mode intended for HPC. Produces a
# consolidated ExafsArchiveData instead of per-snapshot shards.
uv run python examples/example_ensemble_synthetic.py \
    --code feff@localhost --python-code python3@localhost \
    --n-snapshots 12 --batch-size 4 --plot-file ensemble.png

# Reuse one set of scattering potentials across every snapshot
uv run python examples/example_ensemble_synthetic.py \
    --code feff@localhost --n-snapshots 6 --precompute-potentials
```

The ensemble runs need roughly 4 GB of RAM; below that the daemon is killed
mid-run and the only symptom is exit 137.

Pass a real `TrajectoryData` node, or use the synthetic-trajectory helper
included in `examples/` to run a quick end-to-end test without any MD data:

```bash
# Quick self-contained demo (generates a synthetic BCC-Fe trajectory)
uv run python examples/example_ensemble_synthetic.py \
    --code feff@localhost \
    --n-snapshots 6 --sigma 0.06 \
    --plot-file ensemble_exafs.png

# Optional: also store and merge per-path FEFF contributions
uv run python examples/example_ensemble_synthetic.py \
    --code feff@localhost \
    --n-snapshots 6 --sigma 0.06 \
    --store-paths --python-code python3@localhost --path-cw-threshold 5 \
    --plot-file ensemble_exafs.png
```

For a real trajectory, pass it via Python:

```python
from aiida_feff.workflows.ensemble import EnsembleExafsWorkChain

wc = submit(EnsembleExafsWorkChain,
    trajectory=trajectory_node,       # TrajectoryData already in DB
    sample_interval=orm.Int(5),        # use every 5th frame
    parameters=params,
    code=orm.load_code('feff@localhost'),
    options=orm.Dict({"resources": {"num_machines": 1}, "max_wallclock_seconds": 600}),
    group_label=orm.Str("md-exafs/Fe-300K"),  # optional: bundle into a named group
)
```

### 4. Ensemble EXAFS on an HPC cluster (batch mode)

For ensembles of hundreds of snapshots, submitting one Slurm job per
`(frame, site)` pair is inefficient: most HPC centres cap queued jobs per user,
and the scheduler overhead for many short serial jobs is significant.

**Batch mode** solves this by packing many FEFF calculations into a single Slurm
job.  The `EnsembleExafsWorkChain` accepts a `batch_size` input; when set, it
groups all `(frame, site)` pairs into chunks of that size and submits one
`FeffBatchCalculation` per chunk instead of one `FeffCalculation` per pair.
With `batch_size = num_cores_per_node`, the whole ensemble runs in as few Slurm
jobs as `ceil(N_frames × N_sites / num_cores)`.

#### Design choices

**Why one CalcJob per chunk, not a Slurm array?**
AiiDA maps each `CalcJob` 1:1 to a scheduler job.  There is no native
job-array fan-out in AiiDA.  The batch CalcJob is the standard pattern for
running many serial tasks inside one allocation.

**How does FEFF's module load get applied?**
The `feff_code` input is an `InstalledCode` whose `prepend_text` typically
contains `module load feff/8.5` (or similar).  At submission time,
`prepare_for_submission` reads `feff_code.prepend_text` and
`feff_code.filepath_executable` and writes them into `batch_config.json`.
On the compute node, the Python driver generates a small `_run_feff.sh`
wrapper per run directory that sources the environment and calls FEFF —
so the module load is applied correctly for every instance.

**Why does batch mode require `python_code`?**
The `python_code` input (a Python interpreter on the HPC) doubles as
the **runner** for the batch driver script.  The driver script is a pure-Python
file with no AiiDA dependency that uses `concurrent.futures.ProcessPoolExecutor`
to launch FEFF instances in parallel.  If you are not storing path
contributions, the aggregation step is skipped, but the Python interpreter is
still needed to run the driver.

**How are precomputed potentials distributed?**
`precompute_potentials=True` triggers the existing potentials-only step —
one `FeffCalculation` per absorber site (same as non-batch mode).  Their
`RemoteData` outputs are then passed to the batch CalcJob as a dynamic input
namespace (`remote_potentials.site_0000`, `remote_potentials.site_0001`, …).
`prepare_for_submission` populates `remote_copy_list` to copy each site's
potential files (`pot.pad`, `phase.pad`, etc.) to `potentials/site_XXXX/` in
the working directory.  The driver copies from there into each run directory
before calling FEFF, so the SCF step is skipped for every snapshot.

**Partial failures are isolated.**
If an individual FEFF run crashes, the driver logs the error and continues with
the remaining runs.  The batch job exits 0.  The parser detects missing
`chi.dat` files and skips those pairs; the workchain counts them as failures
and produces a partial average (exit code 301) rather than aborting entirely.

#### Setup: register codes on the HPC

Register the FEFF executable.  The `--prepend-text` is the environment setup
that must run before FEFF can be called:

```bash
verdi code create core.code.installed \
    --label feff \
    --computer hpc \
    --filepath-executable /path/to/feff8l \
    --prepend-text "module load feff/8.5" \
    --default-calc-job-plugin feff.feff
```

Register the Python interpreter.  This is used both as the batch runner and
(when `path_cw_threshold >= 0`) for path aggregation.  It must have `larch`,
`numpy`, and `h5py` installed:

```bash
verdi code create core.code.installed \
    --label python3 \
    --computer hpc \
    --filepath-executable /path/to/venv/bin/python3 \
    --default-calc-job-plugin feff.feff_batch
```

Verify:

```bash
verdi code list
# feff@hpc (feff.feff)
# python3@hpc (feff.feff_batch)
```

#### Running a batched ensemble

```python
from aiida import orm
from aiida.engine import submit
from aiida_feff.workflows.ensemble import EnsembleExafsWorkChain
from aiida_feff.data.parameters import FeffParameters

load_profile()

params = FeffParameters(dict={
    "edge": "K",
    "spectrum_type": "EXAFS",
    "radius": 6.0,
    "absorbing_atoms": "Cu",   # all Cu sites; or e.g. "Cu:0,1" for a subset
})

# One Slurm job per node; each job runs 64 FEFF instances in parallel.
# Set batch_size = number of cores you want to allocate per job.
CORES_PER_NODE = 64

wc = submit(
    EnsembleExafsWorkChain,
    trajectory=trajectory_node,          # TrajectoryData in DB
    sample_interval=orm.Int(1),
    parameters=params,
    code=orm.load_code("feff@hpc"),
    python_code=orm.load_code("python3@hpc"),
    precompute_potentials=orm.Bool(True),
    batch_size=orm.Int(CORES_PER_NODE),
    n_workers=orm.Int(CORES_PER_NODE),   # omit to fall back to $SLURM_CPUS_ON_NODE
    options=orm.Dict({
        "resources": {
            "num_machines": 1,
            "num_mpiprocs_per_machine": CORES_PER_NODE,
        },
        "max_wallclock_seconds": 3600,
        "queue_name": "regular",
    }),
    group_label=orm.Str("md-exafs/Cu-300K"),
)
print(f"Submitted workchain pk={wc.pk}")
```

**With path contributions** (stores per-path FEFF scattering factors, enables
later σ² fitting):

```python
wc = submit(
    EnsembleExafsWorkChain,
    trajectory=trajectory_node,
    sample_interval=orm.Int(1),
    parameters=params,
    code=orm.load_code("feff@hpc"),
    python_code=orm.load_code("python3@hpc"),
    precompute_potentials=orm.Bool(True),
    path_cw_threshold=orm.Float(5.0),  # keep paths with ≥5% peak amplitude
    path_r_bin=orm.Float(0.15),
    batch_size=orm.Int(CORES_PER_NODE),
    n_workers=orm.Int(CORES_PER_NODE),
    options=orm.Dict({
        "resources": {
            "num_machines": 1,
            "num_mpiprocs_per_machine": CORES_PER_NODE,
        },
        "max_wallclock_seconds": 7200,
        "queue_name": "regular",
    }),
)
```

#### Sizing guide

| N\_frames × N\_sites | Recommended `batch_size` | Slurm jobs |
|---|---|---|
| ≤ 64 | = total pairs | 1 |
| 65 – 512 | = cores per node (e.g. 64 or 128) | 2 – 8 |
| > 512 | = cores per node | `ceil(N / cores)` |

If `n_workers` is omitted, the driver reads `$SLURM_CPUS_ON_NODE` at runtime,
then falls back to `$SLURM_NTASKS`, then 1.  Set it explicitly if your HPC
uses `--cpus-per-task` instead of `--ntasks`.

#### Checking status

```bash
verdi process status <pk>
# EnsembleExafsWorkChain (pk=<pk>) [ProcessState.RUNNING] [3:submit_batch_calculations]
#   ├── FeffCalculation (pk=...) [FINISHED]  ← potentials-only, 1 per site
#   ├── FeffBatchCalculation (pk=...) [RUNNING]  ← batch 0, 64 pairs
#   └── FeffBatchCalculation (pk=...) [CREATED]  ← batch 1, 64 pairs

verdi process report <pk>           # human-readable log
verdi calcjob outputcat <pk> batch.log   # driver stdout for a batch CalcJob
verdi calcjob outputcat <pk> batch_err.log  # per-run FEFF pass/fail summary
```

#### Retrieving results (same as non-batch mode)

```python
from aiida.orm import load_node

wc = load_node(<pk>)
grand_average  = wc.outputs.averaged_xas.all          # XasData
per_site_avg   = wc.outputs.averaged_xas.site_0000    # XasData, one per absorber
n_failed       = wc.outputs.n_failed.value            # int
path_contrib   = wc.outputs.path_contributions        # PathContributionsData (if stored)
```

### 5. Post-processing with larch (optional)

```python
from aiida_feff.calcfunctions.larch import chi_k_to_r
from aiida.orm import Dict

# Fourier transform (provenance-tracked)
chir = chi_k_to_r(xas_data=xas, ft_params=Dict({"kmin":3, "kmax":14, "kweight":2}))
# Defaults, including window="kaiser", are in aiida_feff.calcfunctions.larch.FT_DEFAULTS.
```

### 6. Plotting (optional)

```python
from aiida_feff.visualise import plot_chi_k, plot_chi_r

# k²χ(k)
fig = plot_chi_k(xas, kweight=2)

# χ(R) — pass the output of chi_k_to_r directly (FT already tracked)
fig = plot_chi_r(chir)

# Overlay multiple spectra on one axes
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
for node in ensemble_xas_nodes:
    plot_chi_k(node, ax=ax, label=node.label, plot_envelope=True)
plt.show()
```

### 7. Debye-Waller σ² from an MD trajectory (optional)

Per-path MSRD (σ²) is computed directly from the trajectory by
[md-exafs](https://pypi.org/project/md-exafs/), which this plugin depends on.

**This step is deliberately not provenance-tracked.** σ² is cheap to recompute
and the trajectory it derives from is already a stored node, so recording the
result would add graph weight without adding recoverable information. The
`store_msrd` / `store_adp` calcfunction wrappers that earlier versions shipped
have been removed for that reason; call md-exafs directly.

```python
from md_exafs.debye_waller import calculate_grouped_msrd

from aiida_feff.utils import trajectory_to_structures

structures = [s.get_ase() for s in trajectory_to_structures(traj_node)]

# `cutoff` must stay below the inscribed-sphere radius of the cell, or the
# minimum-image convention picks the wrong neighbour and biases σ² low.
# Build a supercell rather than raising the cutoff.
two_body, three_body = calculate_grouped_msrd(
    structures,
    central_indices=[0],     # zero-based absorber indices
    central_label="Fe",
    cutoff=3.5,              # neighbour search radius in Å
    cutoff_3body=3.0,        # include 3-body paths (omit to skip)
)

for group in sorted(two_body, key=lambda g: g["reff"]):
    print(f"{group['scatterer']}: reff={group['reff']:.3f} Å  σ²={group['sigma2']:.5f} Å²")
# Fe: reff=2.481 Å  σ²=0.00612 Å²
# Fe: reff=4.052 Å  σ²=0.00891 Å²
```

The resulting σ² values can be passed straight to
`aiida_feff.calcfunctions.exafs.total_chi` as a `scatterer -> σ²` mapping.

Per-atom B-factors and full U tensors come from the same module via
`md_exafs.debye_waller.compute_adp_results`.

## CLI

```bash
verdi data feff list                  # list all FeffParameters / XasData nodes
verdi data feff export <PK>           # preview feff.inp from a FeffParameters node
verdi data feff show <PK>             # inspect arrays in an XasData node
```

## Architecture overview

### Single-job mode (localhost / small ensembles)

```
TrajectoryData + FeffParameters
        │
        ▼
EnsembleExafsWorkChain
  ├─ [optional] FeffCalculation × N_sites   ← potentials-only (CONTROL 1 1 1 0 0 0)
  │
  ├─ FeffCalculation × (N_frames × N_sites) ← one Slurm job each
  │    └─ FeffParser → XasData
  │
  ├─ create_serial_shard (calcfunction) → one shard, as the batch driver writes
  │
  └─ merge_exafs_shards (calcfunction) → ExafsArchiveData (archive)
       └─ archive_to_averaged_xas (calcfunction) → averaged XasData per site + grand average
```

### Batch mode (HPC, hundreds of calculations)

```
TrajectoryData + FeffParameters
        │
        ▼
EnsembleExafsWorkChain (batch_size=64)
  ├─ FeffCalculation × N_sites        ← potentials-only, one Slurm job each
  │    └─ RemoteData (pot.pad, phase.pad, …)
  │
  ├─ FeffBatchCalculation             ← ONE Slurm job, 64 FEFF instances in parallel
  │    ├─ snap_0000_site_0000/feff.inp
  │    ├─ snap_0001_site_0000/feff.inp
  │    │   … (up to batch_size pairs)
  │    └─ _run_batch.py (driver, concurrent.futures, 1 worker/core)
  │         └─ FeffBatchParser → xas_data.snap_FFFF_site_SSSS (dynamic namespace)
  │
  ├─ FeffBatchCalculation             ← next chunk, another Slurm job
  │    └─ …
  │
  └─ merge_exafs_shards (calcfunction) → ExafsArchiveData (archive)
       └─ archive_to_averaged_xas (calcfunction) → averaged XasData per site + grand average
```

Both routes end at the same merge, so the ensemble average has one
implementation and `archive` is always produced.

The FEFF environment (module loads, etc.) is read from `feff_code.prepend_text`
at AiiDA submission time and embedded in `batch_config.json`.  On the compute
node, the driver generates a small `_run_feff.sh` bash wrapper per run
directory, so the environment is correctly applied for every FEFF instance
without requiring any extra code to be installed on the HPC.

## Development

```bash
git clone https://github.com/stfc/aiida-feff
cd aiida-feff
uv sync --locked --extra testing --extra plots
uv run pytest tests/

# Lint exactly as CI does
uv run pre-commit run --all-files
```

## Relationship to [larch-cli](https://github.com/stfc/alc-dls-exafs/)

This plugin is designed to supersede a CLI tool built around larch + FEFF.
The key differences:

| larch-cli | aiida-feff |
|-----------|------------|
| Linear script execution | DAG of provenance-tracked nodes |
| Manual file management | AiiDA handles staging to/from HPC |
| Results as files on disk | All inputs/outputs stored in the database |
| Manual batching (N workers per node) | `EnsembleExafsWorkChain(batch_size=N)` |


## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.
