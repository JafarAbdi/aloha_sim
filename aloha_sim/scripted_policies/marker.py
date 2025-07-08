from enum import IntEnum, auto, unique

import mink
import mujoco
import numpy as np

from aloha_sim.aloha import Aloha
from aloha_sim.motion_planner import Planner
from aloha_sim.tasks.base.aloha2_task import right_qpos_to_ctrl

MARKER_GEOM_NAME: str = "marker//unnamed_geom_0"
MARKER_BODY_NAME: str = "marker//unnamed_body_0"
OBJECT_Z_OFFSET: float = -0.01
APPROACH_Z_OFFSET: float = 0.1
GRIPPER_CLOSE_STEPS: int = 10

# Collision sets
GRIPPER_OBJECT_COLLISIONS = {
    ("right\\left_finger_link", MARKER_BODY_NAME),
    ("right\\right_finger_link", MARKER_BODY_NAME),
}
GRIPPER_TABLE_COLLISIONS = {
    ("table", "right\\right_finger_link"),
    ("table", "right\\left_finger_link"),
}


# Assumptions:
# - TCP (Tool Center Point) Convention:
# X-axis: Forward along tool approach direction
# Y-axis: Along the gripper's open position direction
# - Object's z-axis is aligned with its height direction
def align_pose_to_target(
    target_pose: mink.SE3,
    candidate_pose_1: mink.SE3,
    candidate_pose_2: mink.SE3,
) -> mink.SE3:
    """Align a pose to a target pose by selecting the optimal rotation around the x-axis.

    This function takes two candidate poses and determines which one requires a smaller
    rotation around the target pose's x-axis to align their z-axes. It then applies
    this optimal rotation to achieve alignment.

    The algorithm works by:
    1. Projecting both candidate z-axes onto the target pose's yz-plane
    2. Calculating the signed rotation angle needed for each candidate
    3. Selecting the candidate requiring the smaller rotation
    4. Applying the rotation to achieve z-axis alignment

    Mathematical Background:
    - The rotation is performed around the target pose's x-axis
    - The signed angle is calculated using atan2 for robustness
    - The negative of atan2 is used because we want to rotate FROM candidate TO target

    Args:
        target_pose: The SE3 pose to align to. This defines the target orientation
            and the coordinate frame for the rotation.
        candidate_pose_1: First candidate SE3 pose to consider for alignment.
        candidate_pose_2: Second candidate SE3 pose to consider for alignment.

    Returns:
        An SE3 pose that represents the target pose rotated around its x-axis
        to align with the selected candidate pose's z-axis orientation.

    Raises:
        ValueError: If any of the input poses are invalid or if the rotation
            matrices are not properly normalized.

    Example:
        >>> target = mink.SE3.from_matrix(np.eye(4))
        >>> candidate1 = mink.SE3.from_rotation(mink.SO3.from_z_radians(0.1))
        >>> candidate2 = mink.SE3.from_rotation(mink.SO3.from_z_radians(0.5))
        >>> aligned = align_pose_to_target(target, candidate1, candidate2)
    """
    # Input validation
    if not all(
        isinstance(pose, mink.SE3)
        for pose in [target_pose, candidate_pose_1, candidate_pose_2]
    ):
        msg = "All inputs must be mink.SE3 objects"
        raise ValueError(msg)

    # Extract coordinate frame from target pose
    target_rotation_matrix = target_pose.rotation().as_matrix()
    # > target_x_axis = target_rotation_matrix[:, 0]
    target_y_axis = target_rotation_matrix[:, 1]
    target_z_axis = target_rotation_matrix[:, 2]

    # Extract z-axes from candidate poses
    candidate_1_z_axis = candidate_pose_1.rotation().as_matrix()[:, 2]
    candidate_2_z_axis = candidate_pose_2.rotation().as_matrix()[:, 2]

    def _calculate_x_axis_rotation_angle(candidate_z_axis: np.ndarray) -> float:
        """Calculate the signed angle needed to rotate around target's x-axis.

        This function projects the candidate z-axis onto the target pose's yz-plane
        and calculates the angle needed to align it with the target's z-axis.

        Args:
            candidate_z_axis: The z-axis vector of the candidate pose (shape: (3,))

        Returns:
            The signed rotation angle in radians. Positive angles represent
            counter-clockwise rotation when looking along the positive x-axis.

        Mathematical Details:
        - Projects candidate_z onto the yz-plane of target_pose coordinate frame
        - Uses atan2(y_component, z_component) to get the angle of the projection
        - Negates the result because we want rotation FROM candidate TO target
        """
        # Project candidate z-axis onto target's yz-plane
        y_component = np.dot(candidate_z_axis, target_y_axis)
        z_component = np.dot(candidate_z_axis, target_z_axis)

        # Calculate signed angle using atan2 for numerical stability
        # Negate because we want to rotate FROM candidate TO target orientation
        return -np.arctan2(y_component, z_component)

    # Calculate rotation angles for both candidates
    angle_1 = _calculate_x_axis_rotation_angle(candidate_1_z_axis)
    angle_2 = _calculate_x_axis_rotation_angle(candidate_2_z_axis)

    # Select the candidate requiring the smaller rotation
    optimal_angle = angle_1 if abs(angle_1) < abs(angle_2) else angle_2

    # Apply the optimal rotation around the target's x-axis
    x_axis_rotation = mink.SE3.from_rotation(mink.SO3.from_x_radians(optimal_angle))
    # Align the target pose by applying the rotation
    return target_pose.multiply(x_axis_rotation)


def get_pick_pose(data: mujoco.MjData) -> np.ndarray:
    """Get the pick pose for the marker in the Mujoco simulation."""
    seq = "XYZ"
    ee_quat = np.zeros(4)
    mujoco.mju_euler2Quat(ee_quat, [0, 1.57, 3.14], seq)

    marker_geom = data.geom(MARKER_GEOM_NAME)

    marker_se3 = mink.SE3.from_rotation_and_translation(
        mink.SO3.from_matrix(marker_geom.xmat.reshape((3, 3))),
        marker_geom.xpos,
    )
    s1 = marker_se3.copy()
    s2 = s1.multiply(mink.SE3.from_rotation(mink.SO3.from_x_radians(3.14)))

    seq = "XYZ"
    ee_quat = np.zeros(4)
    mujoco.mju_euler2Quat(ee_quat, [0, 1.57, 3.14], seq)
    ee_pose = mink.SE3.from_rotation_and_translation(
        mink.SO3(wxyz=ee_quat),
        marker_geom.xpos,
    )
    s = align_pose_to_target(ee_pose, s1, s2)

    return s.as_matrix()


@unique
class State(IntEnum):
    """Defines the sequential states of the pick-and-place task."""

    PRE_APPROACH = auto()
    APPROACH = auto()
    CLOSE_GRIPPER = auto()
    RETREAT = auto()
    DONE = auto()


class Policy:
    """Aloha2 policy for picking a marker object in a Mujoco environment."""

    def __init__(self, env):
        """Initialize the policy with the environment and Aloha instance."""
        # Only needed for the env.physics.data.ptr for planning & env.physics.model.ptr to initialize Aloha
        self.env = env
        self.aloha = Aloha(
            self.env.physics.model.ptr,
            disable_collisions={
                ("marker//unnamed_body_0", "cap//unnamed_body_0"),
                ("left\\left_finger_link", "left\\right_finger_link"),
                ("right\\left_finger_link", "right\\right_finger_link"),
                ("table", "marker//unnamed_body_0"),
                ("table", "cap//unnamed_body_0"),
            },
        )

        self._state: State | None = None
        self._current_trajectory: list[np.ndarray] = []
        self._waypoint_index: int = 0

        self._approach_pose: np.ndarray | None = None
        self._object_pose: np.ndarray | None = None
        self._reset_qpos = np.asarray(
            [
                0.0,
                -0.96,
                1.16,
                0.0,
                -0.3,
                0.0,
                self.aloha.right_arm.gripper.qpos_close,
                self.aloha.right_arm.gripper.qpos_close,
            ],
        )

    def reset(self) -> None:
        """Reset the policy to start a new pick and place sequence."""
        print("Resetting policy state.")
        self._state = State.PRE_APPROACH
        self._current_trajectory = []
        self._waypoint_index = 0

        # Calculate task-specific poses based on the new environment state
        object_offset = np.eye(4)
        object_offset[2, 3] = OBJECT_Z_OFFSET
        # Assumes env is already reset and physics data is current
        self._object_pose = object_offset @ get_pick_pose(self.env.physics.data.ptr)

        approach_offset = np.eye(4)
        approach_offset[2, 3] = APPROACH_Z_OFFSET
        self._approach_pose = approach_offset @ self._object_pose

    def set_task_instruction(self, unused_instruction: str) -> None:
        """Set task instruction (not used in this implementation)."""

    def setup(self) -> None:
        """Setup the policy."""

    def _plan_for_current_state(self) -> list[np.ndarray]:
        """Plans a trajectory based on the current state and handles any actions that must occur *before* the trajectory starts."""
        print(f"Planning for state: {self._state.name}")
        state = self._state
        trajectory = None

        if state == State.PRE_APPROACH:
            trajectory = self.aloha.plan_to_pose(
                self.aloha.right_arm,
                self.env.physics.data.ptr,
                self._approach_pose,
                Planner.OMPL,
            )
        elif state == State.APPROACH:
            trajectory = self.aloha.plan_to_pose(
                self.aloha.right_arm,
                self.env.physics.data.ptr,
                self._object_pose,
                Planner.CARTESIAN,
            )
        elif state == State.CLOSE_GRIPPER:
            # Action before trajectory: disable collisions for a clean grasp
            print("Disabling collisions for grasp.")
            self.aloha.disable_collisions(
                GRIPPER_OBJECT_COLLISIONS | GRIPPER_TABLE_COLLISIONS,
            )
            # Trajectory is a list of repeated "close" actions
            close_qpos = self.aloha.right_arm.close_gripper(self.env.physics.data.qpos)
            trajectory = [close_qpos] * GRIPPER_CLOSE_STEPS
        elif state == State.RETREAT:
            trajectory = self.aloha.plan_to_pose(
                self.aloha.right_arm,
                self.env.physics.data.ptr,
                self._approach_pose,
                Planner.CARTESIAN,
            )
        elif state == State.DONE:
            # No new trajectory; stay in the final position
            return [self._reset_qpos]

        if trajectory is None:
            msg = f"Motion planning failed for state: {state.name}"
            raise RuntimeError(msg)

        return trajectory

    def step(self, unused_observation) -> np.ndarray:
        """Returns the next action to take.

        If the current trajectory is finished, it transitions to the next
        state, plans a new trajectory, and returns the first waypoint.
        """
        # Check if the current trajectory is completed or needs to be planned
        if self._waypoint_index >= len(self._current_trajectory):
            if self._state == State.DONE:
                # If done, keep executing the final action (staying still)
                action_qpos = self._reset_qpos
            else:
                # Plan a new trajectory for the current state
                try:
                    self._current_trajectory = self._plan_for_current_state()
                except RuntimeError as e:
                    print(f"Error during planning: {e}")
                    # If planning fails, return the reset position
                    self._waypoint_index = 0
                    self._current_trajectory = []
                    self._state = State.DONE
                    msg = "Planning failed, resetting to done state."
                    raise RuntimeError(msg) from e
                self._waypoint_index = 0

                # Get the first action of the new trajectory
                action_qpos = self._current_trajectory[self._waypoint_index]

                # Advance to the next state for the *next* planning phase
                next_state_index = self._state.value
                self._state = State(next_state_index + 1)
        else:
            # Continue with the current trajectory
            action_qpos = self._current_trajectory[self._waypoint_index]

        self._waypoint_index += 1

        # Convert qpos action to a control signal for the environment
        return right_qpos_to_ctrl(self.env.physics.data.ctrl, action_qpos)

    def is_done(self) -> bool:
        """Check if the task is done."""
        return self._state == State.DONE and self._waypoint_index >= len(
            self._current_trajectory,
        )
