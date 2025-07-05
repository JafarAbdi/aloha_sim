import mink
import mujoco
import numpy as np

from aloha_sim.joint_group import JointGroup
from aloha_sim.tasks.base.aloha2_task import RIGHT_TCP
from aloha_sim.utils.collision_checking import check_collision

# IK parameters
SOLVER = "daqp"
POSITION_THRESHOLD = 1e-2
ORIENTATIOB_THRESHOLD = 1e-3
MAX_ITERS = 100


class IKSolver:
    """IK solver for a joint group in a Mujoco model."""

    def __init__(self, joint_group: JointGroup) -> None:
        """Initialize the IK solver with the given Mujoco model."""
        self.joint_group = joint_group

    def solve(
        self,
        data: mujoco.MjData,
        target_pose: np.ndarray,
        disable_collisions,
    ) -> np.ndarray | None:
        """Solve inverse kinematics for the robot to reach the target pose."""
        assert target_pose.shape == (4, 4), "Target pose must be a 4x4 matrix."

        configuration = mink.Configuration(self.joint_group.model)

        # Define tasks
        end_effector_task = mink.FrameTask(
            frame_name=RIGHT_TCP,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        posture_task = mink.PostureTask(model=self.joint_group.model, cost=1e-2)
        tasks = [end_effector_task, posture_task]

        configuration.update(data.qpos)
        end_effector_task.set_target(mink.SE3.from_matrix(target_pose))
        posture_task.set_target_from_configuration(configuration)

        dt = 0.01  # Time step for integration
        # The qvel of the joints we want to control
        joint_qvel = np.zeros(self.joint_group.model.nv)
        for _ in range(MAX_ITERS):
            vel = mink.solve_ik(configuration, tasks, dt, SOLVER, 1e-3)
            joint_qvel[self.joint_group.qvel_indices] = vel[
                self.joint_group.qvel_indices
            ]
            configuration.integrate_inplace(joint_qvel, dt)

            err = tasks[0].compute_error(configuration)
            pos_achieved = np.linalg.norm(err[:3]) <= POSITION_THRESHOLD
            ori_achieved = np.linalg.norm(err[3:]) <= ORIENTATIOB_THRESHOLD

            if pos_achieved and ori_achieved:
                if check_collision(
                    self.joint_group.model,
                    configuration.data,
                    disable_collisions,
                ):
                    print(f"IK solution for {RIGHT_TCP} collides with the environment.")
                    check_collision(
                        self.joint_group.model,
                        configuration.data,
                        disable_collisions,
                        verbose=True,
                    )
                    return None
                return configuration.data.qpos[self.joint_group.qpos_indices]
        print(
            f"Failed to find IK solution for {RIGHT_TCP} after {MAX_ITERS} iterations."
            f"Linear error: {np.linalg.norm(err[:3])}, Angular error: {np.linalg.norm(err[3:])}",
        )
        return None
