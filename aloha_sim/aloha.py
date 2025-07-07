import functools

import mujoco
import numpy as np

from aloha_sim.cartesian_interpolator import cartesian_interpolator
from aloha_sim.ik_solver import IKSolver
from aloha_sim.joint_group import Gripper, JointGroup
from aloha_sim.motion_planner import (
    MotionPlanner,
    Planner,
    generate_time_optimal_trajectory,
)
from aloha_sim.tasks.base.aloha2_task import (
    _ALL_JOINTS,
    LEFT_ARM_JOINTS,
    LEFT_GRIPPER_CTRL_IDX,
    LEFT_GRIPPER_QPOS_IDX,
    LEFT_TCP,
    RIGHT_ARM_JOINTS,
    RIGHT_GRIPPER_CTRL_IDX,
    RIGHT_GRIPPER_QPOS_IDX,
    RIGHT_TCP,
    SIM_GRIPPER_CTRL_CLOSE,
    SIM_GRIPPER_CTRL_OPEN,
    SIM_GRIPPER_QPOS_CLOSE,
    SIM_GRIPPER_QPOS_OPEN,
)

RESAMPLE_DT = 0.01  # Time step for trajectory resampling

GRIPPER_QPOS_TO_CTRL = functools.partial(
    np.interp,
    # xp has to be sorted
    xp=[SIM_GRIPPER_QPOS_CLOSE, SIM_GRIPPER_QPOS_OPEN],
    fp=[SIM_GRIPPER_CTRL_CLOSE, SIM_GRIPPER_CTRL_OPEN],
)


class Aloha:
    """Aloha class for controlling a robot arm in Mujoco simulation."""

    def __init__(
        self,
        model: mujoco.MjModel,
        disable_collisions: set[tuple[str, str]] | None = None,
    ) -> None:
        """Initialize the Aloha class with a Mujoco model."""
        self.model = model
        self.arms = JointGroup(
            model=model,
            joint_names=list(_ALL_JOINTS),
        )
        self.left_arm = JointGroup(
            model=model,
            joint_names=LEFT_ARM_JOINTS,
            gripper=Gripper(
                qpos_index=LEFT_GRIPPER_QPOS_IDX[0],
                qpos_open=SIM_GRIPPER_QPOS_OPEN,
                qpos_close=SIM_GRIPPER_QPOS_CLOSE,
                ctrl_index=LEFT_GRIPPER_CTRL_IDX,
                ctrl_open=SIM_GRIPPER_CTRL_OPEN,
                ctrl_close=SIM_GRIPPER_CTRL_CLOSE,
            ),
            tcp_name=LEFT_TCP,
        )
        self.right_arm = JointGroup(
            model=model,
            joint_names=RIGHT_ARM_JOINTS,
            gripper=Gripper(
                qpos_index=RIGHT_GRIPPER_QPOS_IDX[0],
                qpos_open=SIM_GRIPPER_QPOS_OPEN,
                qpos_close=SIM_GRIPPER_QPOS_CLOSE,
                ctrl_index=RIGHT_GRIPPER_CTRL_IDX,
                ctrl_open=SIM_GRIPPER_CTRL_OPEN,
                ctrl_close=SIM_GRIPPER_CTRL_CLOSE,
            ),
            tcp_name=RIGHT_TCP,
        )
        self._disable_collisions: set[tuple[str, str]] = disable_collisions or set()
        self.motion_planners: dict[JointGroup, MotionPlanner] = {
            self.arms: MotionPlanner(self.arms),
            self.left_arm: MotionPlanner(self.left_arm),
            self.right_arm: MotionPlanner(self.right_arm),
        }
        self.ik_solvers: dict[JointGroup, IKSolver] = {
            self.left_arm: IKSolver(self.left_arm),
            self.right_arm: IKSolver(self.right_arm),
        }

    def disable_collisions(self, bodies: list[tuple[str, str]]) -> None:
        """Disable collision checks between a list of body pairs."""
        for body1, body2 in bodies:
            self.disable_collision(body1, body2)

    def enable_collisions(self, bodies: list[tuple[str, str]]) -> None:
        """Enable collision checks between a list of body pairs."""
        for body1, body2 in bodies:
            self.enable_collision(body1, body2)

    def disable_collision(self, body1: str, body2: str) -> None:
        """Disable collision checks between two bodies."""
        self._disable_collisions.add((body1, body2))

    def enable_collision(self, body1: str, body2: str) -> None:
        """Enable collision checks between two bodies."""
        self._disable_collisions.discard((body1, body2))
        self._disable_collisions.discard((body2, body1))

    def plan_to_qpos(
        self,
        joint_group: JointGroup,
        data: mujoco.MjData,
        goal_qpos: np.ndarray,
    ):
        """Plan a trajectory to a given joint position goal."""
        assert len(joint_group.joint_names) == len(goal_qpos)
        if (
            waypoints := self.motion_planners[joint_group].plan(
                data,
                goal_qpos,
                self._disable_collisions,
            )
        ) is None:
            print("Failed to plan")
            return None
        if (
            trajectory := generate_time_optimal_trajectory(
                waypoints,
                resample_dt=RESAMPLE_DT,
            )
        ) is None:
            print("Failed to parameterize")
            return None
        return trajectory

    def plan_to_pose(
        self,
        joint_group: JointGroup,
        data: mujoco.MjData,
        target_pose: np.ndarray,
        planner: Planner,
    ):
        """Plan a trajectory to a given pose using the specified planner."""
        assert target_pose.shape == (4, 4), "Pose must be a 4x4 matrix."
        ik_solver = self.ik_solvers[joint_group]
        match planner:
            case Planner.OMPL:
                ik_solution = ik_solver.solve(
                    data,
                    target_pose,
                    self._disable_collisions,
                )
                assert ik_solution is not None
                return self.plan_to_qpos(joint_group, data, ik_solution)
            case Planner.CARTESIAN:
                start_pose = np.eye(4)
                tcp_site = data.site(RIGHT_TCP)
                start_pose[:3, :3] = tcp_site.xmat.reshape((3, 3))
                start_pose[:3, 3] = tcp_site.xpos
                poses = cartesian_interpolator(start_pose, target_pose)
                print(f"Cartesian interpolator generated {len(poses)} waypoints")
                waypoints = []
                for pose in poses:
                    ik_solution = ik_solver.solve(
                        data,
                        pose,
                        self._disable_collisions,
                    )
                    assert ik_solution is not None
                    waypoints.append(ik_solution)
                if (
                    trajectory := generate_time_optimal_trajectory(
                        waypoints,
                        resample_dt=RESAMPLE_DT,
                    )
                ) is None:
                    print("Failed to parameterize")
                    return None
                return trajectory
            case _:
                msg = f"Unknown planner: {planner}"
                raise ValueError(msg)
