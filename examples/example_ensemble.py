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

load_profile()


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
@click.option("--rpath", default=5.5, show_default=True, type=float)
def main(code, traj_pk, step_every, edge, rpath):
    from aiida_feff.data.parameters import FeffParameters
    from aiida_feff.utils import trajectory_to_structures
    from aiida_feff.workflows.ensemble import EnsembleExafsWorkChain

    code_node = orm.load_code(code)
    trajectory = orm.load_node(traj_pk)

    click.echo(f"Loaded trajectory pk={traj_pk}: {len(trajectory.get_array('positions'))} steps")

    # Convert to StructureData list (every N-th frame)
    n_steps = len(trajectory.get_array("positions"))
    step_ids = list(range(0, n_steps, step_every))
    structures = trajectory_to_structures(trajectory, step_ids=step_ids, store=True)
    click.echo(f"Using {len(structures)} snapshots (step_every={step_every})")

    params = FeffParameters(
        dict={
            "edge": edge,
            "calc_mode": "EXAFS",
            "rpath": rpath,
            "s02": 1.0,
            "nleg": 4,
        }
    )
    params.store()

    inputs = {
        "structures": orm.List(list=[s.pk for s in structures]),
        "parameters": params,
        "code": code_node,
        "max_iterations": orm.Int(3),
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
