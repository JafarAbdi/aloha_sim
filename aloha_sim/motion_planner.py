"""Motion planner using OMPL."""

import copy
import enum
import logging
import platform

import mujoco
import numpy as np
import ompl
import time_optimal_trajectory_generation_py as totg
from ompl import base as ob
from ompl import geometric as og

from aloha_sim.joint_group import JointGroup
from aloha_sim.utils.collision_checking import check_collision

# Why using ompl.util.LOG_DEBUG doesn't work on MacOS?
logger_module = ompl.base if (system := platform.system()) == "Darwin" else ompl.util
ompl.util.setLogLevel(
    logger_module.LOG_ERROR,
)


class Planner(enum.StrEnum):
    """Enum for available planners."""

    OMPL = "ompl"
    CARTESIAN = "cartesian_interpolator"


LOGGER = logging.getLogger(__name__)


def get_ompl_planners() -> list[str]:
    """Get OMPL planners.

    Returns:
        List of OMPL planners.
    """
    from inspect import isclass

    module = ompl.geometric
    planners = []
    for obj in dir(module):
        planner_name = f"{module.__name__}.{obj}"
        planner = eval(planner_name)  # noqa: S307, PGH001
        if isclass(planner) and issubclass(planner, ompl.base.Planner):
            planners.append(
                planner_name.split("ompl.geometric.")[1],
            )  # Name is ompl.geometric.<planner>
    return planners


class MotionPlanner:
    """A wrapper for OMPL planners."""

    def __init__(
        self,
        joint_group: JointGroup,
        planner=None,
    ) -> None:
        """Initialize the motion planner.

        Args:
            joint_group: The joint group to plan for.
            planner: The planner to use. If None, RRTConnect is used.
        """
        self.joint_group = joint_group
        self._state_validity_checker = None

        self._bounds = ob.RealVectorBounds(len(self.joint_group.joint_names))
        for i, (lower, upper) in enumerate(self.joint_group.qpos_range):
            self._bounds.setLow(i, lower)
            self._bounds.setHigh(i, upper)
        self._space = ob.RealVectorStateSpace(len(self.joint_group.joint_names))
        self._space.setBounds(self._bounds)
        self._setup = og.SimpleSetup(self._space)
        if planner is None:
            planner = "RRTConnect"
        self._setup.setPlanner(self._get_planner(planner))
        self._setup.setStateValidityChecker(
            ob.StateValidityCheckerFn(self.is_state_valid),
        )

    def _get_planner(self, planner):
        try:
            return eval(  # noqa: S307, PGH001
                f"og.{planner}(self._setup.getSpaceInformation())",
            )
        except AttributeError:
            LOGGER.exception(
                f"Planner '{planner}' not found - Available planners: {get_ompl_planners()}",
            )
            raise

    def as_ompl_state(self, joint_positions):
        """Convert joint positions to ompl state."""
        assert len(joint_positions) == self._space.getDimension()
        state = ob.State(self._space)
        for i, joint_position in enumerate(joint_positions):
            state[i] = joint_position
        return state

    def from_ompl_state(self, state):
        """Convert ompl state to joint positions."""
        return [state[i] for i in range(self._space.getDimension())]

    def plan(
        self,
        data: mujoco.MjData,
        goal_joint_positions: list[float],
        disable_collisions: set[tuple[str, str]],
    ) -> list[list[float]]:
        """Plan a trajectory from start to goal.

        Args:
            data: The Mujoco data object containing the current state.
            goal_joint_positions: The target joint positions to reach.
            disable_collisions: A set of tuples of joint names that should not be checked for collisions.

        Returns:
            The trajectory as a list of joint positions or None if no solution was found.
        """
        assert len(goal_joint_positions) == self._space.getDimension()
        start_joint_positions = data.qpos[self.joint_group.qpos_indices]

        # Data used for planning & collision checking.
        planning_data = copy.deepcopy(data)

        # Create a state validity checker that uses the current data
        def state_checker(joint_positions, *, verbose: bool = False) -> bool:
            planning_data.qpos[self.joint_group.qpos_indices] = joint_positions
            return check_collision(
                self.joint_group.model,
                planning_data,
                disable_collisions,
                verbose=verbose,
            )

        self._state_validity_checker = state_checker

        self._setup.clear()
        start = self.as_ompl_state(start_joint_positions)
        # TODO: We need to check the bounds as well
        if self._state_validity_checker(start_joint_positions, verbose=False):
            self._state_validity_checker(start_joint_positions, verbose=True)
            return None
        start.enforceBounds()
        # > if not start.satisfiesBounds():
        # >     LOGGER.error(
        # >         f"Start joint positions ({goal_joint_positions}) are out of bounds. {self._space.settings()}"
        # >     )
        # >     for joint_position, lower, upper in zip(
        # >         goal_joint_positions,
        # >         self._bounds.low,
        # >         self._bounds.high,
        # >         strict=True,
        # >     ):
        # >         if joint_position < lower or joint_position > upper:
        # >             LOGGER.error(
        # >                 f"Joint position {joint_position} is out of bounds "
        # >                 f"({lower}, {upper})",
        # >             )
        # >     return None
        goal = self.as_ompl_state(goal_joint_positions)
        if self._state_validity_checker(goal_joint_positions, verbose=False):
            self._state_validity_checker(goal_joint_positions, verbose=True)
            return None
        goal.enforceBounds()
        # TODO: Should we only enforce bounds if the state is close to the bounds?
        # > if not goal.satisfiesBounds():
        # >     LOGGER.error(
        # >         f"Goal joint positions {goal_joint_positions} are out of bounds. {self._space.settings()}"
        # >     )
        # >     for joint_position, lower, upper in zip(
        # >         goal_joint_positions,
        # >         self._bounds.low,
        # >         self._bounds.high,
        # >         strict=True,
        # >     ):
        # >         if joint_position < lower or joint_position > upper:
        # >             LOGGER.error(
        # >                 f"Joint position {joint_position} is out of bounds "
        # >                 f"({lower}, {upper})",
        # >             )
        # >     return None
        self._setup.setStartAndGoalStates(start, goal)
        solved = self._setup.solve()
        if not solved:
            LOGGER.info("Did not find solution!")
            return None
        path = self._setup.getSolutionPath()
        if not path.check():
            LOGGER.warning("Path fails check!")

        if ob.PlannerStatus.getStatus(solved) == ob.PlannerStatus.APPROXIMATE_SOLUTION:
            LOGGER.warning("Found approximate solution!")

        LOGGER.debug("Simplifying solution..")
        LOGGER.debug(
            f"Path length before simplification: {path.length()} with {len(path.getStates())} states",
        )

        simplified_path = self._setup.getSolutionPath()
        LOGGER.debug(
            f"Path length after simplifySolution: {simplified_path.length()} with {len(simplified_path.getStates())} states",
        )
        path_simplifier = og.PathSimplifier(self._setup.getSpaceInformation())
        path_simplifier.ropeShortcutPath(simplified_path)
        LOGGER.debug(
            f"Simplified path length after ropeShortcutPath: {simplified_path.length()} with {len(simplified_path.getStates())} states",
        )
        path_simplifier.smoothBSpline(simplified_path)
        LOGGER.debug(
            f"Simplified path length after smoothBSpline: {simplified_path.length()} with {len(simplified_path.getStates())} states",
        )

        if not simplified_path.check():
            LOGGER.warning("Simplified path fails check!")

        LOGGER.debug("Interpolating simplified path...")
        simplified_path.interpolate()

        if not simplified_path.check():
            LOGGER.warning("Interpolated simplified path fails check!")

        solution = []
        for state in simplified_path.getStates():
            solution.append(self.from_ompl_state(state))
        LOGGER.info(f"Found solution with {len(solution)} waypoints")
        return solution

    def is_state_valid(self, state):
        """Check if the state is valid, i.e. not in collision or out of bounds.

        Args:
            state: The state to check.

        Returns:
            True if the state is valid, False otherwise.
        """
        return self._setup.getSpaceInformation().satisfiesBounds(
            state,
        ) and not self._state_validity_checker(self.from_ompl_state(state))


def generate_time_optimal_trajectory(
    waypoints: list[list[float]],
    resample_dt,
) -> list[list[float]] | None:
    """Parameterize the trajectory using Time Optimal Trajectory Generation http://www.golems.org/node/1570.

    Args:
        waypoints: The waypoints to parameterize.
        max_velocity: The maximum velocity for each joint.
        max_acceleration: The maximum acceleration for each joint.
        resample_dt: The resampling time step.

    Returns:
        The parameterized trajectory as a list of (qpos, qvel, time_from_start).
    """
    assert len(waypoints) > 0, "Waypoints must not be empty"
    # The intermediate waypoints of the input path need to be blended so that the entire path is differentiable.
    # This constant defines the maximum deviation allowed at those intermediate waypoints, in radians for revolute joints,
    # or meters for prismatic joints.
    max_velocity = np.ones(len(waypoints[0]))
    max_acceleration = np.ones(len(waypoints[0]))
    max_deviation = 0.1
    trajectory = totg.Trajectory(
        totg.Path(
            waypoints,
            max_deviation,
        ),
        max_velocity,
        max_acceleration,
    )
    if not trajectory.isValid():
        LOGGER.error("Failed to parameterize trajectory")
        return None
    duration = trajectory.getDuration()
    parameterized_trajectory = []
    print(f"Gnenerated trajectory duration: {duration:.3f} seconds")
    for t in np.append(np.arange(0.0, duration, resample_dt), duration):
        parameterized_trajectory.append(trajectory.getPosition(t))
    return parameterized_trajectory
