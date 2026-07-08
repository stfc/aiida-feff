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
from aiida.engine import run_get_node

load_profile()


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
    plot_file,
    group_label,
):
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
            "title": "Synthetic BCC Fe ensemble",
            "edge": "K",
            "calc_mode": "EXAFS",
            "s02": 1.0,
            "rpath": 5.5,
            "nleg": 4,
            "scf_radius": 4.0,
            "fms_radius": 6.0,
            "kmin": 0.0,
            "kmax": 20.0,
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

    averaged_xas = wc_node.outputs.averaged_xas
    click.echo(
        f"  averaged_xas pk={averaged_xas.pk}  "
        f"k-grid: {averaged_xas.get_array('k').shape}  "
        f"n_snapshots={averaged_xas.base.extras.get('n_snapshots')}"
    )

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
    click.echo("Computing Debye-Waller σ² from trajectory …")
    from aiida_feff.calcfunctions.debye_waller import store_msrd

    dw_params = orm.Dict({"absorber_site": "Fe", "cutoff": 3.5, "align": True})
    msrd_node = store_msrd(trajectory=traj, params=dw_params)
    click.echo(f"  store_msrd pk={msrd_node.pk}")
    click.echo("  Path σ² (Å²):")
    for key, val in sorted(msrd_node.get_dict().items(), key=lambda x: x[1]["reff"]):
        click.echo(
            f"    {key:30s}  reff={val['reff']:.3f} Å"
            f"  σ²={val['sigma2']:.5f} Å²  n_paths={val['count']}"
        )

    # ── 6. Collect individual snapshot XasData for overlay ──────────────────
    snapshot_xas = []
    for child in wc_node.called:
        if child.is_finished_ok and hasattr(child, "outputs") and "xas_data" in child.outputs:
            snapshot_xas.append(child.outputs.xas_data)

    # ── 7. Per-path χ(k) contributions ──────────────────────────────────────
    # Each feff????.dat file contains the scattering data needed to reconstruct
    # the path's contribution to χ(k):
    #
    #   k²χ_j(k) = N · red · k · |F(k)| / r²
    #              · exp(−2r/λ(k))          ← mean-free-path damping
    #              · exp(−2σ²k²)             ← Debye-Waller (supplied via --sigma2-dw)
    #              · sin(2k·r_eff + δ_back(k) + 2φ_c(k))   ← oscillation
    #
    # feff_data columns (FEFF_DATA_COLS order):
    #   [0] real_2phc  = 2φ_c(k)   central-atom phase shift
    #   [1] mag        = |F_eff(k)| backscattering amplitude
    #   [2] phase      = δ_back(k)  backscattering phase
    #   [3] red_factor = S₀² and other reduction factors (NOT Debye-Waller)
    #   [4] lambda_mfp = λ(k)       mean free path (Å)
    #
    # Without exp(−2σ²k²), the envelope rises monotonically with k (k·|F|
    # grows faster than exp(−2r/λ) decays at moderate k); with a realistic
    # σ² the curves peak and damp just like the measured χ(k).
    #
    # Frequency: a path that appears in only 3 of 6 snapshots gets weight 0.5.
    # We track frame presence as a SET to avoid counting a path twice if FEFF
    # finds two symmetry-equivalent paths at the same rounded r_eff within
    # one snapshot (this was causing "700%" in the previous version).

    from collections import defaultdict

    n_total_frames = path_contrib.info()["n_frames"] if path_contrib is not None else 0

    path_groups: dict[tuple, dict] = defaultdict(
        lambda: {
            "frame_data": {},  # frame_idx → {"mag","lambda","phase","phc2","red"}
            "k": None,
            "r_eff": 0.0,
            "nlegs": 0,
            "degeneracy": 0.0,
            "scatterer": "",
        }
    )
    if path_contrib is not None:
        for pr in path_contrib.iter_paths():
            key = (round(pr.r_eff, 2), pr.nlegs, pr.scatterer)
            g = path_groups[key]
            fi = pr.frame_idx
            fd = pr.feff_data
            if fi not in g["frame_data"]:
                g["frame_data"][fi] = {
                    "mag": fd[:, 1].copy(),
                    "lambda": fd[:, 4].copy(),
                    "phase": fd[:, 2].copy(),
                    "phc2": fd[:, 0].copy(),
                    "red": fd[:, 3].copy(),
                }
            else:
                # Average when multiple FEFF paths share the same key in one frame.
                prev = g["frame_data"][fi]
                for col_k, col_i in [
                    ("mag", 1),
                    ("lambda", 4),
                    ("phase", 2),
                    ("phc2", 0),
                    ("red", 3),
                ]:
                    prev[col_k] = 0.5 * (prev[col_k] + fd[:, col_i])
            g["k"] = pr.k
            g["r_eff"] = pr.r_eff
            g["nlegs"] = pr.nlegs
            g["degeneracy"] = pr.degeneracy
            g["scatterer"] = pr.scatterer

    k_avg = averaged_xas.get_array("k")
    ft_dict = ft_params.get_dict()
    ft_kmin = ft_dict.get("kmin", 3.0)
    ft_kmax = ft_dict.get("kmax", 14.0)

    def _path_k2chi(g, sigma2, k_out):
        """Mean oscillatory k²χ_path(k) across frames, interpolated onto k_out."""
        k_p = g["k"]
        vals = []
        for fdata in g["frame_data"].values():
            phase_tot = fdata["phase"] + fdata["phc2"]  # δ_back + 2φ_c
            v = (
                g["degeneracy"]
                * fdata["red"]
                * k_p
                * fdata["mag"]
                / g["r_eff"] ** 2
                * np.exp(-2 * g["r_eff"] / fdata["lambda"])
                * np.exp(-2 * sigma2 * k_p**2)
                * np.sin(2 * k_p * g["r_eff"] + phase_tot)
            )
            vals.append(v)
        mean_k2chi = np.mean(vals, axis=0)
        return np.interp(k_out, k_p, mean_k2chi, left=0.0, right=0.0)

    # Use larch xftf for the path Fourier transforms — same windowing and
    # normalisation as the chi_k_to_r calcfunction, so path |χ(R)| lines are
    # on exactly the same scale as the ensemble-average curve with no ad-hoc
    # calibration factor needed.
    import larch as _larch_mod
    from larch.xafs import xftf as _xftf

    _larch_session = _larch_mod.Interpreter()

    def _chi_to_r(k, k2chi, kmin, kmax):
        """Compute |χ(R)| using larch xftf.

        k2chi is k²-weighted (kweight=2 already applied), so we pass
        kweight=0 to avoid double-weighting inside larch.
        """
        grp = _larch_mod.Group(k=k, chi=k2chi)
        _xftf(
            grp,
            kmin=kmin,
            kmax=kmax,
            kweight=0,
            dk=ft_dict.get("dk", 1.0),
            rmax_out=ft_dict.get("rmax", 8.0),
            _larch=_larch_session,
        )
        return grp.r, np.abs(grp.chir)

    chi_avg = averaged_xas.get_array("chi_k")
    chir_larch = chir_node.get_array("chir_mag")
    r_larch = chir_node.get_array("r")

    scored = []
    for _key, g in path_groups.items():
        if not g["frame_data"]:
            continue
        freq = len(g["frame_data"]) / n_total_frames if n_total_frames > 0 else 1.0
        k2chi = _path_k2chi(g, sigma2_dw, k_avg)
        mask = (k_avg >= ft_kmin) & (k_avg <= ft_kmax)
        score = freq * np.trapezoid(np.abs(k2chi[mask]), k_avg[mask])
        scored.append((score, key, g, k2chi, freq))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_paths]

    click.echo(
        f"\nTop {top_paths} paths (σ²_DW={sigma2_dw} Å², {n_total_frames} frames, "
        f"scored by freq × ∫|k²χ_path|dk):"
    )
    for rank, (score, _key, g, _, freq) in enumerate(top, 1):
        nleg_str = "SS" if g["nlegs"] == 2 else f"MS{g['nlegs']}"
        click.echo(
            f"  {rank}. {nleg_str}  r={g['r_eff']:.3f} Å  "
            f"deg={g['degeneracy']:.1f}  scatterer={g['scatterer']}  "
            f"freq={freq:.0%}  score={score:.4f}"
        )

    # ── 8. Plot ──────────────────────────────────────────────────────────────
    import matplotlib

    if plot_file:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Ensemble EXAFS — BCC Fe  ({n_snapshots} snapshots, σ_disp={sigma} Å, "
        f"σ²_DW={sigma2_dw} Å²)",
        fontsize=12,
    )

    # Panel 1: k²χ(k) — snapshots (grey) + ensemble avg (blue) + path contributions
    ax_k = axes[0]
    if "chi_k_std" in averaged_xas.get_arraynames():
        std = averaged_xas.get_array("chi_k_std")
        ax_k.fill_between(
            k_avg,
            k_avg**2 * (chi_avg - std),
            k_avg**2 * (chi_avg + std),
            alpha=0.15,
            color="steelblue",
        )
    for i, snap in enumerate(snapshot_xas):
        k_s = snap.get_array("k")
        chi_s = snap.get_array("chi_k")
        ax_k.plot(
            k_s,
            k_s**2 * chi_s,
            color="grey",
            lw=0.5,
            alpha=0.35,
            label="snapshots" if i == 0 else None,
        )
    ax_k.plot(k_avg, k_avg**2 * chi_avg, color="steelblue", lw=2, label="ensemble avg", zorder=3)
    for rank, (_score, _key, g, k2chi, freq) in enumerate(top):
        nleg_str = "SS" if g["nlegs"] == 2 else f"MS{g['nlegs']}"
        lbl = f"P{rank + 1}: {g['scatterer']} r={g['r_eff']:.2f}Å {nleg_str} ({freq:.0%})"
        ax_k.plot(
            k_avg,
            freq * k2chi,
            color=colors[rank % 10],
            lw=1.3,
            ls="--",
            alpha=0.9,
            label=lbl,
            zorder=4,
        )
    ax_k.set_xlabel("k (Å⁻¹)")
    ax_k.set_ylabel("k²χ(k) (Å⁻²)")
    ax_k.set_title("χ(k)  — dashed: freq-weighted path contributions")
    ax_k.set_xlim(k_avg[0], k_avg[-1])
    ax_k.legend(fontsize=7, loc="lower left")

    # Panel 2: |χ(R)| — ensemble avg (blue) + path |χ(R)| (dashed)
    ax_r = axes[1]
    ax_r.plot(r_larch, chir_larch, color="steelblue", lw=2, label="ensemble avg", zorder=3)
    for rank, (_score, _key, g, k2chi, freq) in enumerate(top):
        nleg_str = "SS" if g["nlegs"] == 2 else f"MS{g['nlegs']}"
        lbl = f"P{rank + 1}: {g['scatterer']} r={g['r_eff']:.2f}Å {nleg_str}"
        r_p, chir_p = _chi_to_r(k_avg, freq * k2chi, ft_kmin, ft_kmax)
        mask_r = r_p <= 6.5
        ax_r.plot(
            r_p[mask_r],
            chir_p[mask_r],
            color=colors[rank % 10],
            lw=1.3,
            ls="--",
            alpha=0.9,
            label=lbl,
            zorder=4,
        )
    ax_r.set_xlabel("R (Å)")
    ax_r.set_ylabel("|χ(R)| (Å⁻³)")
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
    click.echo(f"  store_msrd output    pk={msrd_node.pk}")
    if path_contrib is not None:
        click.echo(f"  path_contributions   pk={path_contrib.pk}")


if __name__ == "__main__":
    main()
