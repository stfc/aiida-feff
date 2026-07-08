#!/usr/bin/env python
"""Example: run a single FEFF EXAFS calculation on BCC Fe via AiiDA.

Usage::

    verdi run examples/example_single_site.py --code feff@localhost

Prerequisites
-------------
* AiiDA profile set up (``verdi setup``)
* A ``Code`` node for FEFF registered, e.g.::

      verdi code setup --label feff --computer localhost \
          --input-plugin feff.feff \
          --remote-abs-path /usr/local/bin/feff

  then reference it here as ``feff@localhost``.
"""

import click
from aiida import load_profile, orm
from aiida.engine import run_get_node, submit

load_profile()


def get_structure_bcc_fe(a: float = 2.87) -> orm.StructureData:
    """BCC iron unit cell (2 atoms)."""
    s = orm.StructureData(cell=[[a, 0, 0], [0, a, 0], [0, 0, a]])
    s.append_atom(position=(0.0, 0.0, 0.0), symbols="Fe")
    s.append_atom(position=(a / 2, a / 2, a / 2), symbols="Fe")
    s.label = "BCC Fe a=2.87"
    return s


@click.command()
@click.option("--code", required=True, help="Code label, e.g. feff@localhost")
@click.option(
    "--submit",
    "do_submit",
    is_flag=True,
    default=False,
    help="Submit to daemon instead of running inline.",
)
def main(code, do_submit):
    from aiida_feff.calculations.feff import FeffCalculation
    from aiida_feff.data.parameters import FeffParameters

    code_node = orm.load_code(code)

    structure = get_structure_bcc_fe()

    params = FeffParameters(
        dict={
            "title": "BCC Fe K-edge EXAFS example",
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

    inputs = {
        "code": code_node,
        "structure": structure,
        "parameters": params,
        "metadata": {
            "label": "BCC Fe K-edge EXAFS",
            "description": "Single-site example from aiida-feff documentation",
            "options": {
                "resources": {"num_machines": 1, "num_mpiprocs_per_machine": 1},
                "max_wallclock_seconds": 600,
            },
        },
    }

    if do_submit:
        node = submit(FeffCalculation, **inputs)
        click.echo(f"Submitted FeffCalculation pk={node.pk}")
    else:
        _, node = run_get_node(FeffCalculation, **inputs)
        click.echo(f"Finished FeffCalculation pk={node.pk} exit_status={node.exit_status}")
        if node.is_finished_ok:
            xas = node.outputs.xas_data
            click.echo(f"  energy grid: {xas.energy.shape}  e0={xas.e0:.1f} eV")
            click.echo(f"  chi(k) grid: {xas.chi_k.shape}")


if __name__ == "__main__":
    main()
