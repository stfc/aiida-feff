"""Miscellaneous utilities for aiida-feff.

trajectory_to_structures
    Split an AiiDA TrajectoryData into a list of StructureData nodes.
split_trajectory
    Provenance-tracked calcfunction version of the above.
structures_to_trajectory
    Provenance-tracked calcfunction: pack StructureData nodes into a TrajectoryData.
"""

from __future__ import annotations

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
    if "cells" not in trajectory.get_arraynames():
        # Inventing a cell would make StructureData periodic (pbc defaults to
        # True) and MPEXAFSSet would build the FEFF cluster from images of a
        # lattice that does not exist — a silent physics error.
        raise ValueError(
            "TrajectoryData has no 'cells' array. FEFF clusters are built from "
            "the periodic cell, so it cannot be guessed; set the array before "
            "splitting the trajectory."
        )
    cells_all = trajectory.get_array("cells")

    if len(symbols) != positions_all.shape[1]:
        raise ValueError(
            f"TrajectoryData has {len(symbols)} symbols but "
            f"{positions_all.shape[1]} atoms per frame."
        )

    steps = list(step_ids) if step_ids is not None else list(range(len(positions_all)))
    structures = []

    for idx in steps:
        s = StructureData(cell=cells_all[idx].tolist())
        for sym, xyz in zip(symbols, positions_all[idx], strict=True):
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

    Returns a dynamic output namespace ``{'frame_{step_id:06d}': StructureData, ...}``
    where ``step_id`` is the original trajectory index passed in ``step_ids``.
    Every snapshot has a ``CREATE`` link back to the trajectory in the
    provenance graph.

    Recover the original order with :func:`sort_frame_labels`; the zero padding
    is a convenience, not the ordering contract.
    """
    step_ids: list[int] = params["step_ids"]
    structures = trajectory_to_structures(trajectory, step_ids=step_ids)
    return {frame_label(step_id): s for step_id, s in zip(step_ids, structures, strict=True)}


def frame_label(step_id: int) -> str:
    """Canonical dynamic-namespace key for trajectory frame *step_id*."""
    return f"frame_{step_id:06d}"


def sort_frame_labels(labels) -> list[str]:
    """Sort namespace keys by their trailing integer, not lexicographically.

    ``frame_10`` sorts before ``frame_2`` as a string, which silently reorders
    an MD trajectory.  Keys without a trailing integer keep their relative
    order after the numbered ones.
    """

    def key(label: str):
        _, _, tail = label.rpartition("_")
        return (0, int(tail), "") if tail.isdigit() else (1, 0, label)

    return sorted(labels, key=key)


@calcfunction
def structures_to_trajectory(**structures: StructureData):
    """Pack a keyed set of StructureData snapshots into a TrajectoryData.

    Parameters are passed as keyword arguments ``s000000=StructureData, …``.
    Intended for use when the caller has a list of structures rather than a
    TrajectoryData and needs to pass a single node to a batch CalcJob.

    Keys are ordered by their trailing integer via :func:`sort_frame_labels`,
    so the frame order survives past 9999 frames.

    Returns:
        TrajectoryData containing all supplied structures in frame order.
    """
    from aiida.orm import TrajectoryData

    ordered = [structures[key] for key in sort_frame_labels(structures)]
    return TrajectoryData(ordered)
