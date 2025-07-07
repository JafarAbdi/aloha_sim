import mujoco
import numpy as np

# Colors for X, Y, Z axes (Red, Green, Blue)
FRAME_COLORS = [
    [1.0, 0.0, 0.0, 1.0],  # X-axis: Red
    [0.0, 1.0, 0.0, 1.0],  # Y-axis: Green
    [0.0, 0.0, 1.0, 1.0],  # Z-axis: Blue
]
# Axis parameters
AXIS_LENGTH = 0.2
AXIS_RADIUS = 0.01


def add_frame_to_scene(scene: mujoco.MjvScene, pose: np.ndarray):
    """Visualizes input pose."""
    assert pose.shape == (4, 4), "Pose must be a 4x4 transformation matrix."
    # Extract position and rotation from pose
    position = pose[:3, 3]
    rotation_matrix = pose[:3, :3]

    # Add three cylindrical arrows for X, Y, Z axes
    for idx, frame_color in enumerate(FRAME_COLORS):
        mujoco.mjv_initGeom(
            geom=scene.geoms[scene.ngeom],
            type=mujoco.mjtGeom.mjGEOM_ARROW,
            size=np.zeros(3),
            pos=np.zeros(3),
            mat=np.zeros(9),
            rgba=np.asarray(frame_color).astype(np.float32),
        )
        mujoco.mjv_connector(
            geom=scene.geoms[scene.ngeom],
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,  # Or mujoco.mjtGeom.mjGEOM_ARROW
            width=AXIS_RADIUS,
            from_=position,
            to=position + AXIS_LENGTH * rotation_matrix[:, idx],
        )
        scene.geoms[scene.ngeom].category = mujoco.mjtCatBit.mjCAT_DECOR
        scene.ngeom += 1


def add_frame_to_renderer(
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pose: np.ndarray,
):
    """Visualizes the X, Y, Z axes in the MuJoCo scene."""
    vopt = mujoco.MjvOption()
    pert = mujoco.MjvPerturb()  # Empty MjvPerturb object

    # https://mujoco.readthedocs.io/en/stable/programming/visualization.html#scene-update
    catmask = mujoco.mjtCatBit.mjCAT_DECOR

    add_frame_to_scene(renderer.scene, pose)

    # Add existing geometries
    mujoco.mjv_addGeoms(
        model,
        data,
        vopt,
        pert,
        catmask,
        renderer.scene,
    )
