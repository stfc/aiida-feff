#!/usr/bin/env python
"""Generate a synthetic BCC-Fe MD ensemble, run FEFF on all snapshots,
ensemble-average the EXAFS, compute Debye-Waller σ² from the trajectory,
store per-path FEFF contributions, and plot χ(k), χ(R), and the top-N path
amplitude envelopes k²·|F(k)|.

Builds a small "trajectory" by displacing atoms from their equilibrium
positions with Gaussian noise (mimicking thermal motion at ~300 K).
No external trajectory file is needed.

Usage::

    uv run python examples/example_ensemble_synthetic.py --code feff@localhost
    uv run python examples/example_ensemble_synthetic.py --code feff@localhost \
        --n-snapshots 10 --sigma 0.08 --top-paths 4 --plot-file /tmp/ensemble_exafs.png
    uv run python examples/example_ensemble_synthetic.py --code feff@localhost \
        --store-paths --python-code python3@localhost --path-cw-threshold 5
"""

from __future__ import annotations

import click
import numpy as np
from aiida import load_profile, orm
from aiida.common.links import LinkType
from aiida.engine import run_get_node

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_bcc_fe_snapshot(
    rng: np.random.Generator,
    a: float = 2.87,
    sigma: float = 0.06,
) -> orm.StructureData:
    """BCC Fe unit cell with random Gaussian displacements on each atom."""
    s = orm.StructureData(cell=[[a, 0, 0], [0, a, 0], [0, 0, a]])
    eq_positions = [
        (0.0, 0.0, 0.0),
        (a / 2, a / 2, a / 2),
    ]
    for pos in eq_positions:
        disp = rng.normal(scale=sigma, size=3)
        s.append_atom(position=tuple(np.array(pos) + disp), symbols="Fe")
    return s


def make_trajectory(
    n_snapshots: int,
    sigma: float,
    seed: int = 42,
) -> orm.TrajectoryData:
    """Build an AiiDA TrajectoryData from synthetic BCC-Fe snapshots."""
    rng = np.random.default_rng(seed)
    a = 2.87
    n_atoms = 2
    positions = np.zeros((n_snapshots, n_atoms, 3))
    eq = np.array([[0, 0, 0], [a / 2, a / 2, a / 2]])
    for i in range(n_snapshots):
        disp = rng.normal(scale=sigma, size=(n_atoms, 3))
        positions[i] = eq + disp

    cells = np.tile([[a, 0, 0], [0, a, 0], [0, 0, a]], (n_snapshots, 1, 1))
    steps = np.arange(n_snapshots)

    traj = orm.TrajectoryData()
    traj.set_array("positions", positions)
    traj.set_array("cells", cells)
    traj.set_array("steps", steps)
    traj.base.attributes.set("symbols", ["Fe", "Fe"])
    traj.label = f"Synthetic BCC Fe trajectory ({n_snapshots} frames, σ={sigma} Å)"
    return traj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@click.command()
@click.option("--code", required=True, help="Code label, e.g. feff@localhost")
@click.option(
    "--n-snapshots",
    default=5,
    show_default=True,
    type=int,
    help="Number of displaced snapshots to run.",
)
@click.option(
    "--sigma",
    default=0.06,
    show_default=True,
    type=float,
    help="RMS atomic displacement in Å (≈0.06 Å for Fe at 300 K).",
)
@click.option(
    "--top-paths",
    default=4,
    show_default=True,
    type=int,
    help="Number of most-important paths to plot in the path panel.",
)
@click.option(
    "--sigma2-dw",
    default=0.005,
    show_default=True,
    type=float,
    help="Debye-Waller σ² (Å²) applied to path contributions. "
    "Controls high-k damping so path lines look like real χ(k).",
)
@click.option(
    "--store-paths/--no-store-paths",
    default=False,
    show_default=True,
    help=("Store per-path FEFF contributions using remote aggregation. Requires --python-code."),
)
@click.option(
    "--python-code",
    default=None,
    help=(
        "Code label for a Python 3 executable on the same computer as FEFF, "
        "e.g. python3@localhost. Required with --store-paths."
    ),
)
@click.option(
    "--path-cw-threshold",
    default=0.0,
    show_default=True,
    type=float,
    help=(
        "Curved-wave amplitude threshold (0-100) for keeping FEFF paths. "
        "Used only when --store-paths is enabled."
    ),
)
@click.option(
    "--path-r-bin",
    default=0.15,
    show_default=True,
    type=float,
    help="Bin width (Å) for grouping equivalent scattering paths across frames.",
)
@click.option(
    "--batch-size",
    default=None,
    type=int,
    help=(
        "Run snapshots through FeffBatchCalculation in chunks of this size, "
        "one scheduler job per chunk, instead of one job per snapshot. "
        "Requires --python-code. This is the mode intended for HPC, where a "
        "job per snapshot means thousands of queue entries."
    ),
)
@click.option(
    "--precompute-potentials/--no-precompute-potentials",
    default=False,
    show_default=True,
    help=(
        "Run FEFF once per absorber site to generate the scattering "
        "potentials, then reuse them for every snapshot. The potentials "
        "depend on the average environment rather than the instantaneous "
        "one, so this is a physical approximation as well as a saving."
    ),
)
@click.option(
    "--plot-file", default=None, help="Save the plot to this path instead of showing interactively."
)
@click.option(
    "--group-label",
    default=None,
    help=(
        "AiiDA Group label. The finished workchain is added to this group "
        "(created if it does not exist). E.g. 'md-exafs/Fe-300K'."
    ),
)
def main(
    code,
    n_snapshots,
    sigma,
    top_paths,
    sigma2_dw,
    store_paths,
    python_code,
    path_cw_threshold,
    path_r_bin,
    batch_size,
    precompute_potentials,
    plot_file,
    group_label,
):
    # Loading the profile here rather than at import keeps --help usable
    # on a machine with no AiiDA profile configured.
    load_profile()

    from aiida_feff.calcfunctions.larch import chi_k_to_r
    from aiida_feff.data.parameters import FeffParameters
    from aiida_feff.workflows.ensemble import EnsembleExafsWorkChain

    code_node = orm.load_code(code)

    # ── 1. Build synthetic trajectory ────────────────────────────────────────
    click.echo(f"Building synthetic trajectory: {n_snapshots} snapshots, σ={sigma} Å")
    traj = make_trajectory(n_snapshots, sigma)
    traj.store()
    click.echo(f"  Stored TrajectoryData pk={traj.pk}")

    # ── 2. Define FEFF parameters ────────────────────────────────────────────
    params = FeffParameters(
        dict={
            "edge": "K",
            "spectrum_type": "EXAFS",
            "s02": 1.0,
            "radius": 5.5,
            "nleg": 4,
        }
    )
    params.store()

    ft_params = orm.Dict({"kmin": 3.0, "kmax": 14.0, "kweight": 2, "dk": 1.0, "rmax": 8.0})

    # ── 3. Run ensemble workflow ─────────────────────────────────────────────
    click.echo(f"Running EnsembleExafsWorkChain over {n_snapshots} snapshots …")
    wc_inputs = {
        "trajectory": traj,
        "parameters": params,
        "code": code_node,
        "options": orm.Dict(
            {
                "resources": {"num_machines": 1, "num_mpiprocs_per_machine": 1},
                "max_wallclock_seconds": 600,
            }
        ),
    }

    if store_paths:
        if not python_code:
            raise click.UsageError("--store-paths requires --python-code (e.g. python3@localhost).")
        wc_inputs["python_code"] = orm.load_code(python_code)
        wc_inputs["path_cw_threshold"] = orm.Float(path_cw_threshold)
        # Same binning here and in the merged node, so the groups this script
        # plots are the groups the provenance graph recorded.
        wc_inputs["path_r_bin"] = orm.Float(path_r_bin)

    if batch_size:
        if not python_code:
            raise click.UsageError("--batch-size requires --python-code (e.g. python3@localhost).")
        # NB the code/feff_code inversion: FeffBatchCalculation runs a Python
        # driver on the compute node that invokes FEFF itself, so its `code`
        # is the interpreter and FEFF arrives as `feff_code`. The workchain
        # performs that swap; here both are simply supplied.
        wc_inputs.setdefault("python_code", orm.load_code(python_code))
        wc_inputs["batch_size"] = orm.Int(batch_size)
        click.echo(f"  batch mode: chunks of {batch_size} snapshot(s) per scheduler job")

    if precompute_potentials:
        wc_inputs["precompute_potentials"] = orm.Bool(True)
        click.echo("  precomputing scattering potentials once per site")

    if group_label:
        wc_inputs["group_label"] = orm.Str(group_label)

    _, wc_node = run_get_node(
        EnsembleExafsWorkChain,
        **wc_inputs,
    )

    if not wc_node.is_finished_ok:
        click.echo(
            f"WorkChain pk={wc_node.pk} finished with exit status "
            f"{wc_node.exit_status}: {wc_node.exit_message}",
            err=True,
        )
        raise SystemExit(1)

    n_failed = wc_node.outputs.n_failed.value
    click.echo(
        f"WorkChain pk={wc_node.pk} finished OK  "
        f"({n_snapshots - n_failed}/{n_snapshots} snapshots succeeded)"
    )

    averaged_xas = wc_node.outputs.averaged_xas.all
    counts = averaged_xas.get_array("chi_k_count")
    click.echo(
        f"  averaged_xas pk={averaged_xas.pk}  "
        f"k-grid: {averaged_xas.get_array('k').shape}  "
        f"n_snapshots={averaged_xas.base.attributes.get('n_snapshots')}"
    )
    # Coverage is per-k, not a single number: a snapshot whose chi.dat stopped
    # early drops out above its own k_max.
    click.echo(f"    contributors per k: min={int(counts.min())} max={int(counts.max())}")

    # The consolidated ensemble archive (ADR 0004), written on both routes.
    archive = wc_node.outputs.archive
    click.echo(
        f"  archive pk={archive.pk}  ensemble={archive.is_ensemble}  "
        f"k-grid: {archive.k.shape}  r-grid: {archive.r.shape}"
    )
    # Counted from the merge calcfunction's own inputs rather than guessed
    # from child labels, so this cannot drift from what was actually merged.
    n_shards = len(archive.creator.base.links.get_incoming(link_type=LinkType.INPUT_CALC).all())
    click.echo(f"    merged from {n_shards} shard(s)")

    # Grab merged path contributions node (present when --store-paths is enabled).
    path_contrib = getattr(wc_node.outputs, "path_contributions", None)
    if path_contrib is not None:
        info = path_contrib.info()
        click.echo(
            f"  path_contributions pk={path_contrib.pk}  "
            f"frames={info['n_frames']}  sites={info['n_sites']}  "
            f"paths/site={info['n_paths'] // max(info['n_sites'], 1)}  "
            f"size={info['file_size_mb']:.2f} MB"
        )

    # ── 4. Fourier transform (provenance-tracked) ────────────────────────────
    click.echo("Running chi_k_to_r calcfunction …")
    chir_node = chi_k_to_r(xas_data=averaged_xas, ft_params=ft_params)
    click.echo(f"  χ(R) node pk={chir_node.pk}  r-grid: {chir_node.get_array('r').shape}")

    # ── 5. Debye-Waller σ² from the MD trajectory ────────────────────────────
    #
    # Deliberately NOT provenance-tracked. σ² is cheap to recompute from the
    # trajectory, which is itself stored, so putting it in the graph adds node
    # weight without adding recoverable information. The calcfunction wrappers
    # that used to live in aiida_feff.calcfunctions.debye_waller were removed
    # for this reason; the physics lives in md-exafs and is called directly.
    click.echo("Computing Debye-Waller σ² from trajectory (not stored) …")
    from md_exafs.debye_waller import calculate_grouped_msrd

    # The cutoff must stay inside the cell's inscribed sphere, or the
    # minimum-image convention aliases neighbours and biases σ² low.
    from aiida_feff.utils import trajectory_to_structures

    structures = [s.get_ase() for s in trajectory_to_structures(traj)]
    two_body, three_body = calculate_grouped_msrd(
        structures,
        central_indices=[0],
        central_label="Fe",
        cutoff=2.7,
    )
    click.echo(f"  {len(two_body)} two-body and {len(three_body)} three-body path groups")
    for group in sorted(two_body, key=lambda g: g["reff"])[:5]:
        click.echo(
            f"    {group['scatterer']:>4s}  reff={group['reff']:.3f} Å"
            f"  σ²={group['sigma2']:.5f} Å²  n={group['count']}"
        )

    # ── 6. Collect individual snapshot XasData for overlay ──────────────────
    snapshot_xas = []
    for child in wc_node.called:
        if child.is_finished_ok and hasattr(child, "outputs") and "xas_data" in child.outputs:
            snapshot_xas.append(child.outputs.xas_data)

    # ── 7. Per-path χ(k) contributions ──────────────────────────────────────
    # The EXAFS equation itself lives in aiida_feff.calcfunctions.exafs, where
    # it is unit-tested against larch's own FeffPathGroup.  Reimplementing it
    # here would be untested physics in a demo script.
    from aiida_feff.calcfunctions.exafs import group_paths_by_key, path_result_chi
    from aiida_feff.calcfunctions.larch import xftf_arrays

    k_avg = averaged_xas.get_array("k")
    ft_dict = ft_params.get_dict()
    ft_kmin = ft_dict.get("kmin", 3.0)
    ft_kmax = ft_dict.get("kmax", 14.0)

    n_total_frames = path_contrib.info()["n_frames"] if path_contrib is not None else 0
    groups = group_paths_by_key(path_contrib, r_bin=path_r_bin) if path_contrib is not None else {}

    def _mean_chi(paths):
        """Ensemble-mean chi(k) for one path key, on the averaged k grid."""
        per_frame: dict[int, list] = {}
        for path in paths:
            chi = path_result_chi(path, k_avg, sigma2=sigma2_dw)
            per_frame.setdefault(path.frame_idx, []).append(chi)
        # Mean within a frame first, then across frames: a frame in which FEFF
        # found two equivalent paths must not outweigh one where it found one.
        frame_means = [np.mean(v, axis=0) for v in per_frame.values()]
        return np.mean(frame_means, axis=0), len(per_frame)

    mask = (k_avg >= ft_kmin) & (k_avg <= ft_kmax)
    scored = []
    for key, paths in groups.items():
        chi_path, n_frames_seen = _mean_chi(paths)
        freq = n_frames_seen / n_total_frames if n_total_frames else 1.0
        k2chi = k_avg**2 * chi_path
        score = freq * np.trapezoid(np.abs(k2chi[mask]), k_avg[mask])
        representative = paths[0]
        scored.append((score, key, representative, chi_path, freq))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_paths]

    def _chi_to_r(k, k2chi):
        """|chi(R)| via the same transform the provenance-tracked path uses."""
        # k2chi already carries the k^2 weighting, so kweight=0 here.
        result = xftf_arrays(k, k2chi, {**ft_dict, "kweight": 0})
        return result["r"], result["chir_mag"]

    click.echo(
        f"\nTop {top_paths} paths (σ²_DW={sigma2_dw} Å², {n_total_frames} frames, "
        f"scored by freq × ∫|k²χ_path|dk):"
    )
    for rank, (score, key, path, _k2chi, freq) in enumerate(top, 1):
        click.echo(
            f"  {rank}. {key}  r={path.r_eff:.3f} Å  "
            f"deg={path.degeneracy:.1f}  scatterer={path.scatterer}  "
            f"freq={freq:.0%}  score={score:.4f}"
        )

    # ── 8. Plot ──────────────────────────────────────────────────────────────
    # Every curve goes through aiida_feff.visualise, which is unit-tested.
    # This script used to build the axes by hand and call xftf_arrays itself,
    # which made it a second, untested implementation of the k-weighting, the
    # Fourier transform and the axis units -- the units in particular are easy
    # to get wrong, since chi(R) carries Angstrom^-(kweight+1).
    import matplotlib

    if plot_file:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from aiida_feff.data.xasdata import XasData
    from aiida_feff.visualise import plot_chi_k, plot_chi_r

    def as_xas(k, chi):
        """Wrap raw arrays as an unstored XasData so visualise can draw them.

        Unstored: these are presentation intermediates, not results, and
        storing them would add nodes to the graph that nothing can be
        recovered from.
        """
        node = XasData()
        node.set_chi(k, chi)
        return node

    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    fig, (ax_k, ax_r) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Ensemble EXAFS — BCC Fe  ({n_snapshots} snapshots, σ_disp={sigma} Å, "
        f"σ²_DW={sigma2_dw} Å²)",
        fontsize=12,
    )

    # Panel 1: k²χ(k) — snapshots, ensemble average with ±1σ, path overlays.
    for i, snap in enumerate(snapshot_xas):
        plot_chi_k(
            snap,
            kweight=2,
            ax=ax_k,
            label="snapshots" if i == 0 else "_nolegend_",
            color="grey",
            lw=0.5,
            alpha=0.35,
        )
    plot_chi_k(
        averaged_xas,
        kweight=2,
        ax=ax_k,
        label="ensemble avg",
        plot_envelope=True,
        color="steelblue",
        lw=2,
        zorder=3,
    )

    # Panel 2: |χ(R)| from the provenance-tracked transform.
    plot_chi_r(
        chir_node, ax=ax_r, label="ensemble avg", rmax=6.0, color="steelblue", lw=2, zorder=3
    )

    # Path contributions, scaled by how often FEFF found each path. Passing
    # freq * chi through the same helpers keeps the weighting and the units
    # consistent with the curves above.
    for rank, (_score, _key, path, chi_path, freq) in enumerate(top):
        nleg_str = "SS" if path.nlegs == 2 else f"MS{path.nlegs}"
        lbl = f"P{rank + 1}: {path.scatterer} r={path.r_eff:.2f}Å {nleg_str} ({freq:.0%})"
        node = as_xas(k_avg, freq * chi_path)
        style = {"color": colors[rank % 10], "lw": 1.3, "ls": "--", "alpha": 0.9, "zorder": 4}
        plot_chi_k(node, kweight=2, ax=ax_k, label=lbl, **style)
        plot_chi_r(node, ft_params=ft_dict, ax=ax_r, label=lbl, rmax=6.0, **style)

    ax_k.set_title("χ(k)  — dashed: freq-weighted path contributions")
    ax_k.set_xlim(k_avg[0], k_avg[-1])
    ax_k.legend(fontsize=7, loc="lower left")
    ax_r.set_title("χ(R)  — dashed: freq-weighted path contributions")
    ax_r.set_xlim(0, 6)
    ax_r.legend(fontsize=7)

    plt.tight_layout()

    if plot_file:
        fig.savefig(plot_file, dpi=150)
        click.echo(f"Plot saved to {plot_file}")
    else:
        plt.show()

    click.echo("\nProvenance summary:")
    click.echo(f"  TrajectoryData       pk={traj.pk}")
    click.echo(f"  EnsembleWorkChain    pk={wc_node.pk}")
    click.echo(f"  averaged_xas         pk={averaged_xas.pk}")
    click.echo(f"  chi_k_to_r output    pk={chir_node.pk}")
    click.echo("  Debye-Waller sigma^2 : computed, not stored (see step 5)")
    if path_contrib is not None:
        click.echo(f"  path_contributions   pk={path_contrib.pk}")
    if archive is not None:
        click.echo(f"  archive              pk={archive.pk}")


if __name__ == "__main__":
    main()
