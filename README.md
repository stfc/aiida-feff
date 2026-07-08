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
| `FeffParser` | Parses `xmu.dat` and `chi.dat` into `XasData` output nodes |
| `FeffBatchCalculation` | CalcJob that runs *N* FEFF instances in one Slurm job (one per core); for HPC ensemble runs |
| `FeffBatchParser` | Parses all per-snapshot outputs from a batch job into a dynamic namespace of `XasData` nodes |
| `FeffParameters` | Validated `Dict` subclass for FEFF control cards |
| `XasData` | `ArrayData` subclass storing μ(E) and χ(k) spectra |
| `EnsembleExafsWorkChain` | Fan-out over MD snapshots → ensemble-averaged χ(k); supports both single-job and batch modes |
| `calcfunctions` | Larch post-processing (FT), and Debye-Waller extraction |
| `visualise` | Matplotlib helpers for μ(E), χ(k), and χ(R) plots |

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
    "calc_mode": "EXAFS",
    "rpath": 5.5,
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

> **Recommended Environment:** Running these examples requires AiiDA services (PostgreSQL, RabbitMQ, daemon) and FEFF. The absolute easiest way to run them is to open this project in a **VS Code DevContainer** (`.devcontainer/`). The container fully configures AiiDA, downloads the FEFF8L binary, sets up both the `feff@localhost` and `python3@localhost` codes, and starts the daemon automatically on creation.
>
> *If running outside the DevContainer, you must manually run `verdi` services, have a working FEFF executable, and register a Python executable as an installed code (`verdi code create core.code.installed ...`) named e.g. `python3@localhost` pointing to your virtual environment's Python interpreter.*

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
`xmu.dat` files and skips those pairs; the workchain counts them as failures
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
    "calc_mode": "EXAFS",
    "rpath": 6.0,
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
```

### 6. Plotting (optional)

```python
from aiida_feff.visualise import plot_mu_e, plot_chi_k, plot_chi_r

# μ(E)
fig = plot_mu_e(xas)

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

Compute per-path MSRD (σ²) directly from a `TrajectoryData` node — no
separate DW code needed.

Two variants are provided:

- `compute_msrd` / `compute_adp` — plain Python functions, return plain dicts/arrays.  **Not recorded in the database.**  Use these when exploring cutoffs and tolerances interactively.
- `store_msrd` / `store_adp` — `@calcfunction` wrappers.  Accept and return AiiDA nodes; every call is recorded in the provenance graph.  Use these in workflows.

```python
from aiida_feff.calcfunctions.debye_waller import compute_msrd, store_msrd
from aiida.orm import Dict

params = {
    "absorber_site": "Fe",   # element, "Fe.1" (first Fe), or "3" (1-based index)
    "cutoff": 3.5,           # neighbour search radius in Å
    "cutoff_3body": 3.0,     # include 3-body paths (omit to skip)
    "skip_frames": 50,       # discard first N frames (equilibration)
    "align": True,           # two-pass Kabsch alignment before MSRD
}

# Interactive exploration — no DB writes:
result = compute_msrd(traj_node, params)
for key, val in sorted(result.items(), key=lambda x: x[1]["reff"]):
    print(f"{key}: reff={val['reff']:.3f} Å  σ²={val['sigma2']:.5f} Å²")

# Store in provenance graph when happy with the parameters:
msrd_node = store_msrd(trajectory=traj_node, params=Dict(params))
# Fe-Fe_2p48_2body: reff=2.481 Å  σ²=0.00612 Å²
# Fe-Fe_4p05_2body: reff=4.052 Å  σ²=0.00891 Å²
```

Per-atom B-factors and full U tensors:

```python
from aiida_feff.calcfunctions.debye_waller import compute_adp, store_adp

adp = compute_adp(traj_node, {"skip_frames": 50})
print(adp["b_factors"])    # ndarray, shape (n_atoms,)
print(adp["u_tensor"])     # ndarray, shape (n_atoms, 3, 3)

# Or with provenance:
adp_node = store_adp(trajectory=traj_node, params=Dict({"skip_frames": 50}))
print(adp_node.get_array("b_factors"))
```

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
  └─ average_xas_data (calcfunction)
       └─ averaged XasData per site + grand average
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
  └─ average_xas_data (calcfunction)
       └─ averaged XasData per site + grand average
```

The FEFF environment (module loads, etc.) is read from `feff_code.prepend_text`
at AiiDA submission time and embedded in `batch_config.json`.  On the compute
node, the driver generates a small `_run_feff.sh` bash wrapper per run
directory, so the environment is correctly applied for every FEFF instance
without requiring any extra code to be installed on the HPC.

## Development

```bash
git clone https://github.com/youruser/aiida-feff
cd aiida-feff
pip install -e .[testing]
pytest tests/ -v
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
