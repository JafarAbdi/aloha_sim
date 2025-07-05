"""This module provides a function to interpolate between two poses in Cartesian space."""

import math

import ndcurves
import numpy as np
import pinocchio

# Cartesian interpolation constants
MAX_TRANSLATIONAL_STEP = 0.01  # m
MAX_ANGULAR_STEP = 0.01  # radians
EPSILON = 1e-6


def cartesian_interpolator(
    start: np.ndarray,
    end: np.ndarray,
) -> list[np.ndarray]:
    """Interpolate between two poses.

    Args:
        start: The start pose
        end: The end pose

    Returns:
        A list of poses interpolated between the start and end poses.
        The list will contain at least two poses: the start pose and the end pose.
    """
    assert start.shape == end.shape == (4, 4)
    start_pose = pinocchio.SE3(start)
    end_pose = pinocchio.SE3(end)

    translational_distance = np.linalg.norm(
        end_pose.translation - start_pose.translation,
    )
    angular_distance = pinocchio.Quaternion.angularDistance(
        pinocchio.Quaternion(start_pose.rotation),
        pinocchio.Quaternion(end_pose.rotation),
    )
    if translational_distance < EPSILON and angular_distance < EPSILON:
        return [start_pose.homogeneous, end_pose.homogeneous]

    translational_steps = math.ceil(translational_distance / MAX_TRANSLATIONAL_STEP)
    angular_steps = math.ceil(angular_distance / MAX_ANGULAR_STEP)
    steps = max(translational_steps, angular_steps)

    curve = ndcurves.SE3Curve(start_pose, end_pose, 0, 1)
    return [curve(t) for t in np.linspace(0, 1, steps)]
