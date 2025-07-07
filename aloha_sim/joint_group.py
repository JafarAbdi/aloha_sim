import functools
from collections.abc import Callable
from dataclasses import dataclass, field

import mujoco
import numpy as np


@dataclass(frozen=True, slots=True)
class Gripper:
    """A class to represent a gripper joint."""

    qpos_index: int
    qpos_open: float
    qpos_close: float
    ctrl_index: int
    ctrl_open: float
    ctrl_close: float
    qpos_to_ctrl: Callable[[np.ndarray], float] = field(init=False)

    def __post_init__(self) -> None:
        """Initialize the qpos_to_ctrl function."""
        assert self.qpos_close < self.qpos_open
        assert self.ctrl_close < self.ctrl_open
        object.__setattr__(
            self,
            "qpos_to_ctrl",
            functools.partial(
                np.interp,
                # xp has to be sorted
                xp=[self.qpos_close, self.qpos_open],
                fp=[self.ctrl_close, self.ctrl_open],
            ),
        )


@dataclass(frozen=True, slots=True)
class JointGroup:
    """A class to represent a group of joints."""

    model: mujoco.MjModel
    joint_names: list[str]
    gripper: Gripper | None = None
    tcp_name: str | None = None
    joint_indices: list[int] = field(init=False)
    qpos_indices: list[int] = field(init=False)
    qvel_indices: list[int] = field(init=False)
    qpos_range: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        """Initialize joint indices based on joint names."""
        object.__setattr__(
            self,
            "joint_indices",
            [self.model.joint(name).id for name in self.joint_names],
        )
        object.__setattr__(
            self,
            "qpos_indices",
            self.model.jnt_qposadr[self.joint_indices].tolist(),
        )
        object.__setattr__(
            self,
            "qvel_indices",
            self.model.jnt_dofadr[self.joint_indices].tolist(),
        )
        object.__setattr__(
            self,
            "qpos_range",
            self.model.jnt_range[self.joint_indices],
        )

    def open_gripper(self, qpos) -> None:
        """Open the gripper."""
        assert len(qpos) == self.model.nq, "qpos must have the same length as model.nq"
        qpos[self.gripper.qpos_index] = self.gripper.qpos_open
        return qpos[self.qpos_indices]

    def close_gripper(self, qpos) -> None:
        """Close the gripper."""
        assert len(qpos) == self.model.nq, "qpos must have the same length as model.nq"
        qpos[self.gripper.qpos_index] = self.gripper.qpos_close
        return qpos[self.qpos_indices]

    def __hash__(self) -> int:
        """Return a hash of the joint group."""
        return hash(tuple(self.joint_names))
