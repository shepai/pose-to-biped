"""
The robots this package knows. get_robot("h1") returns its description (a RobotSpec); every
environment and motion tool takes robot="..." (default "g1").

    from G1_tele.robots import ROBOTS, get_robot
    spec = get_robot("t1")
    print(spec.joint_names, spec.xml_path)

To add one, describe it like h1.py and call register_robot(spec) (or add it to ROBOTS below).
"""
from .spec import N_SLOTS, SLOTS, Joint, RobotSpec  # noqa: F401
from .g1 import G1
from .h1 import H1
from .t1 import T1
from .n1 import N1

ROBOTS = {s.name: s for s in (G1, H1, T1, N1)}
DEFAULT_ROBOT = "g1"


def register_robot(spec):
    ROBOTS[spec.name] = spec
    return spec


def get_robot(robot=None):
    """A RobotSpec from a name ("g1"), a RobotSpec (returned as is) or None (the G1)."""
    if robot is None:
        return ROBOTS[DEFAULT_ROBOT]
    if isinstance(robot, RobotSpec):
        return robot
    key = str(robot).lower()
    if key not in ROBOTS:
        raise KeyError(f"unknown robot {robot!r}; known: {sorted(ROBOTS)}")
    return ROBOTS[key]
