"""Miscellaneous utilities for aiida-feff.

trajectory_to_structures
    Split an AiiDA TrajectoryData into a list of StructureData nodes.
split_trajectory
    Provenance-tracked calcfunction version of the above.
structures_to_trajectory
    Provenance-tracked calcfunction: pack StructureData nodes into a TrajectoryData.
"""

from __future__ import annotations

import numpy as np
from aiida.engine import calcfunction
from aiida.orm import Dict, StructureData

# ---------------------------------------------------------------------------
# Trajectory → list of StructureData
# ---------------------------------------------------------------------------


def trajectory_to_structures(
    trajectory,
    step_ids: list[int] | None = None,
    store: bool = False,
) -> list[StructureData]:
    """Convert a :class:`~aiida.orm.TrajectoryData` to a list of StructureData.

    Parameters
    ----------
    trajectory:
        AiiDA TrajectoryData node.
    step_ids:
        Indices of steps to extract.  ``None`` → all steps.
    store:
        If ``True``, call ``.store()`` on each StructureData.

    Returns:
    -------
    list[StructureData]
    """
    from aiida.orm import TrajectoryData

    if not isinstance(trajectory, TrajectoryData):
        raise TypeError(f"Expected TrajectoryData, got {type(trajectory)}")

    symbols = trajectory.symbols
    positions_all = trajectory.get_array("positions")  # (nstep, natom, 3)
    cells_all = trajectory.get_array("cells") if "cells" in trajectory.get_arraynames() else None

    steps = step_ids if step_ids is not None else range(len(positions_all))
    structures = []

    for idx in steps:
        pos = positions_all[idx]
        cell = cells_all[idx] if cells_all is not None else np.eye(3) * 20.0
        s = StructureData(cell=cell.tolist())
        for sym, xyz in zip(symbols, pos, strict=False):
            s.append_atom(position=xyz.tolist(), symbols=sym)
        s.label = f"snapshot_{idx}"
        if store:
            s.store()
        structures.append(s)

    return structures


@calcfunction
def split_trajectory(trajectory, params: Dict) -> dict:
    """Split a TrajectoryData into StructureData snapshots (provenance-tracked).

    Parameters are passed via a ``Dict`` node with keys:

    ``step_ids`` : list[int]
        Frame indices to extract.

    Returns a dynamic output namespace ``{'frame_0000': StructureData, ...}``
    so every snapshot has a ``CREATE`` link back to the trajectory in the
    provenance graph.
    """
    step_ids: list[int] = params["step_ids"]
    structures = trajectory_to_structures(trajectory, step_ids=step_ids)
    return {f"frame_{i:04d}": s for i, s in enumerate(structures)}


@calcfunction
def structures_to_trajectory(**structures: StructureData):
    """Pack a keyed set of StructureData snapshots into a TrajectoryData.

    Parameters are passed as keyword arguments ``s0000=StructureData, s0001=…``
    (sorted lexicographically).  Intended for use when the caller has a list of
    structures rather than a TrajectoryData and needs to pass a single node to
    a batch CalcJob.

    Returns:
        TrajectoryData containing all supplied structures in sorted key order.
    """
    from aiida.orm import TrajectoryData

    sorted_structs = [v for _, v in sorted(structures.items())]
    return TrajectoryData(sorted_structs)
