"""Tests for trajectory <-> structure conversion helpers."""

from __future__ import annotations

import numpy as np
import pytest
from aiida.orm import Dict, TrajectoryData

from aiida_feff.utils import (
    frame_label,
    sort_frame_labels,
    split_trajectory,
    structures_to_trajectory,
    trajectory_to_structures,
)


class TestSortFrameLabels:
    """Frame order must survive past 9999 frames."""

    def test_sorts_numerically_not_lexicographically(self):
        labels = ["frame_000010", "frame_000002", "frame_000001"]
        assert sort_frame_labels(labels) == [
            "frame_000001",
            "frame_000002",
            "frame_000010",
        ]

    def test_handles_five_digit_frames(self):
        labels = [frame_label(i) for i in (100000, 9999, 10000)]
        assert sort_frame_labels(labels) == [
            frame_label(9999),
            frame_label(10000),
            frame_label(100000),
        ]

    def test_unnumbered_keys_sort_last_and_stably(self):
        assert sort_frame_labels(["all", "frame_000002", "frame_000001"]) == [
            "frame_000001",
            "frame_000002",
            "all",
        ]


class TestTrajectoryToStructures:
    def test_missing_cells_raises_rather_than_inventing_one(self, aiida_profile):
        """A guessed cell would make FEFF build a cluster from imaginary images."""
        traj = TrajectoryData()
        traj.set_array("positions", np.zeros((2, 1, 3)))
        traj.set_array("steps", np.arange(2))
        traj.base.attributes.set("symbols", ["Fe"])
        with pytest.raises(ValueError, match="no 'cells' array"):
            trajectory_to_structures(traj)

    def test_symbol_count_mismatch_raises(self, aiida_profile):
        traj = TrajectoryData()
        traj.set_array("positions", np.zeros((2, 3, 3)))
        traj.set_array("cells", np.tile(np.eye(3) * 5.0, (2, 1, 1)))
        traj.set_array("steps", np.arange(2))
        traj.base.attributes.set("symbols", ["Fe", "Fe"])
        with pytest.raises(ValueError, match="symbols"):
            trajectory_to_structures(traj)

    def test_uses_each_frames_own_cell(self, aiida_profile):
        traj = TrajectoryData()
        traj.set_array("positions", np.zeros((2, 1, 3)))
        cells = np.stack([np.eye(3) * 5.0, np.eye(3) * 6.0])
        traj.set_array("cells", cells)
        traj.set_array("steps", np.arange(2))
        traj.base.attributes.set("symbols", ["Fe"])
        structures = trajectory_to_structures(traj)
        assert structures[0].cell[0][0] == pytest.approx(5.0)
        assert structures[1].cell[0][0] == pytest.approx(6.0)

    def test_selects_only_requested_steps(self, generate_trajectory, aiida_profile):
        traj = generate_trajectory(n_frames=5, reps=1)
        assert len(trajectory_to_structures(traj, step_ids=[0, 3])) == 2


class TestSplitTrajectory:
    def test_keys_round_trip_through_frame_label(self, generate_trajectory, aiida_profile):
        traj = generate_trajectory(n_frames=4, reps=1).store()
        result = split_trajectory(traj, Dict({"step_ids": [0, 2, 3]}))
        assert set(result) == {frame_label(i) for i in (0, 2, 3)}

    def test_snapshots_link_back_to_the_trajectory(self, generate_trajectory, aiida_profile):
        traj = generate_trajectory(n_frames=3, reps=1).store()
        result = split_trajectory(traj, Dict({"step_ids": [0, 1]}))
        for node in result.values():
            creator = node.base.links.get_incoming().all_nodes()
            assert creator, "each snapshot must have a CREATE link for provenance"


class TestStructuresToTrajectory:
    def test_frame_order_is_numeric(self, generate_structure, aiida_profile):
        # Lexicographic ordering would put frame 10 before frame 2 and silently
        # scramble the trajectory.
        structures = {
            frame_label(i): generate_structure(a=2.8 + 0.1 * i).store() for i in (1, 2, 10)
        }
        traj = structures_to_trajectory(**structures)
        first_axis = traj.get_array("cells")[:, 0, 0]
        assert list(np.round(first_axis, 3)) == [2.9, 3.0, 3.8]
