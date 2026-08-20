#!/usr/bin/env python
"""Example: ensemble-averaged EXAFS from a set of MD snapshots.

Demonstrates how to convert a :class:`~aiida.orm.TrajectoryData` into
an ensemble-averaged spectrum using
:class:`~aiida_feff.workflows.ensemble.EnsembleExafsWorkChain`.

Usage::

    verdi run examples/example_ensemble.py --code feff@localhost \
        --trajectory-pk 1234

Where ``1234`` is the PK of a TrajectoryData already in your database.
To load one from an XYZ file you can do::

    from ase.io import read
    from aiida.orm import TrajectoryData
    from aiida_tools import atoms_to_structure  # or build manually
"""

import click
from aiida import load_profile, orm
from aiida.engine import submit


@click.command()
@click.option("--code", required=True, help="Code label for FEFF")
@click.option(
    "--trajectory-pk",
    "traj_pk",
    type=int,
    required=True,
    help="PK of TrajectoryData in the AiiDA db",
)
@click.option(
    "--step-every",
    default=1,
    show_default=True,
    help="Use every N-th snapshot to reduce the number of FEFF jobs.",
)
@click.option("--edge", default="K", show_default=True)
@click.option("--radius", default=5.5, show_default=True, type=float)
def main(code, traj_pk, step_every, edge, radius):
    # Loading the profile here rather than at import keeps --help usable
    # on a machine with no AiiDA profile configured.
    load_profile()

    from aiida_feff.data.parameters import FeffParameters
    from aiida_feff.workflows.ensemble import EnsembleExafsWorkChain

    code_node = orm.load_code(code)
    trajectory = orm.load_node(traj_pk)

    click.echo(f"Loaded trajectory pk={traj_pk}: {len(trajectory.get_array('positions'))} steps")

    n_steps = len(trajectory.get_array("positions"))
    click.echo(f"Using every {step_every} of {n_steps} frames")

    params = FeffParameters(
        dict={
            "edge": edge,
            "spectrum_type": "EXAFS",
            "radius": radius,
            "s02": 1.0,
            "nleg": 4,
        }
    )
    params.store()

    # Pass the trajectory itself, not a pre-split list of structures: the
    # workchain splits it with the split_trajectory calcfunction, so every
    # snapshot keeps a CREATE link back to the trajectory.  Splitting outside
    # the workchain severs that link and the provenance graph with it.
    inputs = {
        "trajectory": trajectory,
        "sample_interval": orm.Int(step_every),
        "parameters": params,
        "code": code_node,
        "options": orm.Dict(
            {
                "resources": {"num_machines": 1, "num_mpiprocs_per_machine": 1},
                "max_wallclock_seconds": 600,
            }
        ),
    }

    wc = submit(EnsembleExafsWorkChain, **inputs)
    click.echo(f"Submitted EnsembleExafsWorkChain pk={wc.pk}")
    click.echo(f"Monitor with:  verdi process list -a -p {wc.pk}")
    click.echo(f"Results:       verdi process show {wc.pk}")


if __name__ == "__main__":
    main()
