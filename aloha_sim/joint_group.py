from collections.abc import Callable
from dataclasses import dataclass, field

import mujoco
import numpy as np


@dataclass(frozen=True, slots=True)
class Gripper:
    """A class to represent a gripper joint."""

    joint_index: int
    qpos_to_ctrl: Callable[[np.ndarray], float]


@dataclass(frozen=True, slots=True)
class JointGroup:
    """A class to represent a group of joints."""

    model: mujoco.MjModel
    joint_names: list[str]
    # > gripper: Gripper
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
