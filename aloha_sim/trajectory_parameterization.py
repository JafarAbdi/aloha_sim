from abc import ABC, abstractmethod

import numpy as np
import ruckig
import time_optimal_trajectory_generation_py as totg
import toppra


class TrajectoryParameterizer(ABC):
    """Abstract base class for trajectory parameterization."""

    @abstractmethod
    def run(self, trajectory):
        """Parameterizes the given trajectory.

        Args:
            trajectory: The trajectory to be parameterized.

        Returns:
            A parameterized representation of the trajectory.
        """


# Based on https://github.com/adlarkin/mjpl/blob/main/src/mjpl/trajectory/toppra_trajectory.py
class ToppraParameterizer(TrajectoryParameterizer):
    """TOPP-RA."""

    def __init__(
        self,
        dt: float,
        max_velocity: np.ndarray,
        max_acceleration: np.ndarray,
    ):
        """Constructor.

        Args:
            dt: Trajectory timestep.
            max_velocity: Maximum allowed velocity of each joint.
            max_acceleration: Maximum allowed acceleration of each joint.
            min_velocity: Minimum allowed velocity of each joint. If this is
                not set, the negative of max_velocity will be used.
            min_acceleration: Minimum allowed acceleration of each joint. If
                this is not set, the negative of max_acceleration will be used.
        """
        velocity_limits = np.stack((-max_velocity, max_velocity)).T
        acceleration_limits = np.stack((-max_acceleration, max_acceleration)).T

        self.dt = dt
        self.velocity_constraint = toppra.constraint.JointVelocityConstraint(
            velocity_limits,
        )
        self.acceleration_constraint = toppra.constraint.JointAccelerationConstraint(
            acceleration_limits,
        )

    def run(self, waypoints: list[np.ndarray]) -> list[np.ndarray] | None:
        """Run TOPP-RA trajectory parameterization."""
        instance = toppra.algorithm.TOPPRA(
            constraint_list=[self.velocity_constraint, self.acceleration_constraint],
            path=toppra.SplineInterpolator(
                np.linspace(0, 1, len(waypoints)),
                waypoints,
            ),
            parametrizer="ParametrizeConstAccel",
        )
        trajectory = instance.compute_trajectory()
        if trajectory is None:
            return None
        print(
            f"TOPP-RA: Generated trajectory duration: {trajectory.duration:.3f} seconds",
        )
        t = np.append(np.arange(0.0, trajectory.duration, self.dt), trajectory.duration)
        # > [velocity for velocity in trajectory(t, order=1)]
        # > [acceleration for acceleration in trajectory(t, order=2)]
        return list(trajectory(t))


class TotgParameterizer(TrajectoryParameterizer):
    """Parameterize the trajectory using Time Optimal Trajectory Generation http://www.golems.org/node/1570."""

    def __init__(
        self,
        dt: float,
        max_velocity: np.ndarray,
        max_acceleration: np.ndarray,
    ):
        """Constructor.

        Args:
            dt: Trajectory timestep (resample_dt: The resampling time step).
            max_velocity: The maximum velocity for each joint.
            max_acceleration: The maximum acceleration for each joint.
        """
        self.dt = dt
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration

    def run(self, waypoints: list[np.ndarray]) -> list[np.ndarray] | None:
        """Run TOTG trajectory parameterization."""
        # The intermediate waypoints of the input path need to be blended so that the entire path is differentiable.
        # This constant defines the maximum deviation allowed at those intermediate waypoints, in radians for revolute joints,
        # or meters for prismatic joints.
        max_deviation = 0.1
        trajectory = totg.Trajectory(
            totg.Path(
                waypoints,
                max_deviation,
            ),
            self.max_velocity,
            self.max_acceleration,
        )
        if not trajectory.isValid():
            return None
        duration = trajectory.getDuration()
        parameterized_trajectory = []
        print(f"TOTG: Gnenerated trajectory duration: {duration:.3f} seconds")
        for t in np.append(np.arange(0.0, duration, self.dt), duration):
            parameterized_trajectory.append(trajectory.getPosition(t))
            # > trajectory.getVelocity(t)
        return parameterized_trajectory


# Based on https://github.com/adlarkin/mjpl/blob/main/src/mjpl/trajectory/ruckig_trajectory.py
class RuckigParameterizer(TrajectoryParameterizer):
    """Ruckig."""

    def __init__(
        self,
        dt: float,
        max_velocity: np.ndarray,
        max_acceleration: np.ndarray,
        max_jerk: np.ndarray,
    ):
        """Constructor.

        Args:
            dt: Trajectory timestep.
            max_velocity: Maximum allowed velocity of each joint.
            max_acceleration: Maximum allowed acceleration of each joint.
            max_jerk: Maximum allowed jerk of each joint.
        """
        self.dt = dt
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.max_jerk = max_jerk

    def run(self, waypoints: list[np.ndarray]) -> list[np.ndarray] | None:
        """Run Ruckig trajectory parameterization."""
        dof = waypoints[0].size
        otg = ruckig.Ruckig(dof, self.dt, len(waypoints))
        inp = ruckig.InputParameter(dof)
        out = ruckig.OutputParameter(dof, len(waypoints))

        inp.current_position = waypoints[0]
        inp.current_velocity = np.zeros(dof)
        inp.current_acceleration = np.zeros(dof)

        # NOTE: If Ruckig community version is installed, using intermediate
        # waypoints invokes Ruckig's cloud API, which slows down trajectory
        # generation time. Pre-processing the path by filtering out some of
        # the waypoints will make trajectory generation faster. For more info:
        # https://docs.ruckig.com/md_pages_2__intermediate__waypoints.html
        inp.intermediate_positions = waypoints[1:-1]

        inp.target_position = waypoints[-1]
        inp.target_velocity = np.zeros(dof)
        inp.target_acceleration = np.zeros(dof)

        inp.max_velocity = self.max_velocity
        inp.min_velocity = -self.max_velocity
        inp.max_acceleration = self.max_acceleration
        inp.min_acceleration = -self.max_acceleration
        inp.max_jerk = self.max_jerk

        positions = []

        res = ruckig.Result.Working
        while res == ruckig.Result.Working:
            res = otg.update(inp, out)
            positions.append(np.array(out.new_position))
            # > velocities.append(np.array(out.new_velocity))
            # > accelerations.append(np.array(out.new_acceleration))
            out.pass_to_input(inp)
        if res != ruckig.Result.Finished:
            return None

        return positions
