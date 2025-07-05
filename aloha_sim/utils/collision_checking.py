from collections import Counter

import mujoco


def get_contact_points(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> Counter[tuple[str, str]]:
    """Get contact points between bodies in the simulation."""
    contacts = Counter()

    for contact in data.contact:
        body1_id = model.geom_bodyid[contact.geom1]
        body1_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body1_id,
        )
        body2_id = model.geom_bodyid[contact.geom2]
        body2_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body2_id,
        )
        contacts[(body1_name, body2_name)] += 1

    return contacts


def check_collision(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    disable_collisions: set[tuple[str, str]],
    *,
    verbose: bool = False,
) -> bool:
    """Check if the robot is in collision."""
    mujoco.mj_forward(model, data)

    contacts = get_contact_points(model, data)

    for body1_name, body2_name in disable_collisions:
        contacts.pop((body1_name, body2_name), None)
        contacts.pop((body2_name, body1_name), None)

    if verbose:
        print(f"Contacts: {contacts}")

    return len(contacts) > 0
