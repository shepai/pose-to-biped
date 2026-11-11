"""
Convert a URDF humanoid into a MuJoCo model this package can use (pure Python + numpy, so it also
runs where MuJoCo isn't installed). Used by examples/get_robots.py for robots that only come as
URDF (the iCub).

    write_mjcf(urdf, out_dir, name, mesh_files, base_rotation=None, ...)

What it does
  * one MuJoCo body per moving link: links attached by fixed joints (sensor frames, covers, the
    sole plate) are merged into the moving link they hang from - their geoms move with them and
    their masses and inertias are combined (parallel axis theorem). Frames listed in keep_frames are
    kept as massless child bodies (MuJoCo can then still report where they are),
  * the root link gets a free joint; base_rotation (3x3) turns the whole robot so that it faces +x
    with z up, as this package expects (URDFs from other tool chains often use other conventions),
  * every revolute joint gets a torque motor (ctrlrange = the URDF effort limit, unless the effort
    is a placeholder above max_effort) - the env turns actuators into its own PD control anyway,
  * visual meshes go in group 2 (no collisions), collision geometry in group 3,
  * scene.xml: the robot, a floor plane and a light, like MuJoCo Menagerie's scenes.

Collision meshes are used as MuJoCo uses them (convex hulls). Hulls of neighbouring links that
already overlap in the standing pose would push each other apart in the simulation, so after
conversion examples/get_robots.py loads the model in MuJoCo and adds <exclude> pairs for every
pair of bodies touching in the default pose (add_rest_excludes()).
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- small rotation helpers
def rpy_to_mat(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def mat_to_quat(R):
    """(w, x, y, z), w >= 0."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def _origin(el):
    """(R, p) of a URDF <origin> (identity when missing)."""
    o = el.find("origin") if el is not None else None
    if o is None:
        return np.eye(3), np.zeros(3)
    xyz = np.array([float(v) for v in o.get("xyz", "0 0 0").split()])
    rpy = np.array([float(v) for v in o.get("rpy", "0 0 0").split()])
    return rpy_to_mat(rpy), xyz


def _fmt(v, nd=6):
    return " ".join(f"{float(x):.{nd}g}" if abs(float(x)) > 1e-12 else "0" for x in np.ravel(v))


# ---------------------------------------------------------------- URDF -> body tree
class _Body:
    def __init__(self, name, joint=None):
        self.name, self.joint = name, joint       # joint: dict(name, type, axis, range, effort, damping, friction)
        self.R, self.p = np.eye(3), np.zeros(3)   # pose in the parent body's frame
        self.geoms = []                           # (kind "visual"/"collision", type, size/mesh, R, p, rgba)
        self.inertials = []                       # (mass, com (3,), inertia about com in body axes (3,3))
        self.children = []


def parse_urdf(urdf_path, keep_frames=(), base_rotation=None):
    """Body tree of the robot (root body first), merging fixed links into their moving parents."""
    root_el = ET.parse(urdf_path).getroot()
    links = {lk.get("name"): lk for lk in root_el.findall("link")}
    joints = root_el.findall("joint")
    kids = {}
    for j in joints:
        kids.setdefault(j.find("parent").get("link"), []).append(j)
    child_links = {j.find("child").get("link") for j in joints}
    roots = [n for n in links if n not in child_links]
    if len(roots) != 1:
        raise ValueError(f"expected one root link, found {roots}")
    keep = set(keep_frames)

    def add_link(body, link_name, R, p):
        """Put link link_name (at pose R, p in body's frame) into body, then walk its children."""
        lk = links[link_name]
        inert = lk.find("inertial")
        if inert is not None and inert.find("mass") is not None and float(inert.find("mass").get("value")) > 0:
            Ri, pi = _origin(inert)
            m = float(inert.find("mass").get("value"))
            ie = inert.find("inertia")
            I = np.zeros((3, 3))
            if ie is not None:
                g = lambda k: float(ie.get(k, 0))                       # noqa: E731
                I = np.array([[g("ixx"), g("ixy"), g("ixz")], [g("ixy"), g("iyy"), g("iyz")],
                              [g("ixz"), g("iyz"), g("izz")]])
            Rb = R @ Ri
            body.inertials.append((m, R @ pi + p, Rb @ I @ Rb.T))
        for kind in ("visual", "collision"):
            for e in lk.findall(kind):
                Rg, pg = _origin(e)
                geo = e.find("geometry")[0]
                rgba = None
                if kind == "visual":
                    c = e.find("material/color")
                    rgba = c.get("rgba") if c is not None else None
                body.geoms.append((kind, geo.tag, dict(geo.attrib), R @ Rg, R @ pg + p, rgba))
        for j in kids.get(link_name, []):
            c = j.find("child").get("link")
            Rj, pj = _origin(j)
            Rc, pc = R @ Rj, R @ pj + p
            t = j.get("type")
            if t == "fixed" and c not in keep:
                add_link(body, c, Rc, pc)                               # merge into this body
                continue
            if t == "fixed":
                info = None
            elif t in ("revolute", "continuous"):
                ax = j.find("axis")
                lim = j.find("limit")
                dyn = j.find("dynamics")
                info = dict(name=j.get("name"),
                            axis=np.array([float(v) for v in (ax.get("xyz") if ax is not None else "1 0 0").split()]),
                            range=(float(lim.get("lower")), float(lim.get("upper"))) if (lim is not None and t == "revolute") else None,
                            effort=float(lim.get("effort")) if lim is not None and lim.get("effort") else None,
                            damping=float(dyn.get("damping", 0)) if dyn is not None else 0.0,
                            friction=float(dyn.get("friction", 0)) if dyn is not None else 0.0)
            else:
                raise ValueError(f"joint {j.get('name')}: type {t} is not supported")
            b = _Body(c, info)
            b.R, b.p = Rc, pc
            body.children.append(b)
            add_link(b, c, np.eye(3), np.zeros(3))

    root = _Body(roots[0])
    Rb = np.eye(3) if base_rotation is None else np.asarray(base_rotation, float)
    add_link(root, roots[0], Rb, np.zeros(3))       # the root's contents, turned so the robot faces +x
    return root


def _combined_inertial(inertials):
    """(mass, com, inertia about com) of several point bodies with inertia, all in one frame."""
    m = sum(i[0] for i in inertials)
    c = sum(i[0] * i[1] for i in inertials) / m
    I = np.zeros((3, 3))
    for mi, ci, Ii in inertials:
        d = ci - c
        I += Ii + mi * (d @ d * np.eye(3) - np.outer(d, d))
    return m, c, I


# ---------------------------------------------------------------- MJCF output
def write_mjcf(urdf_path, out_dir, name, mesh_dir="meshes", keep_frames=(), base_rotation=None, max_effort=1000.0,
               joint_armature=None, mesh_lookup=None):
    """Write out_dir/<name>.xml and out_dir/scene.xml. mesh_lookup: URDF mesh filename -> file name
    in out_dir/mesh_dir (default: the file's base name). Returns the path of scene.xml."""
    out_dir = Path(out_dir)
    root = parse_urdf(urdf_path, keep_frames, base_rotation)
    meshes = {}                                      # file name -> (asset name, scale)

    def mesh_name(attrib):
        f = attrib["filename"]
        fname = mesh_lookup(f) if mesh_lookup else Path(f).name
        scale = attrib.get("scale", "1 1 1")
        key = (fname, scale)
        if key not in meshes:
            stem = Path(fname).stem
            n = stem if all(v[0] != stem for v in meshes.values()) else f"{stem}_{len(meshes)}"
            meshes[key] = (n, scale)
        return meshes[key][0]

    actuated = []

    def body_xml(b, parent_el, is_root=False):
        el = ET.SubElement(parent_el, "body", name=b.name)
        if is_root:
            el.set("pos", "0 0 1")
            ET.SubElement(el, "freejoint", name="floating_base")
        else:
            el.set("pos", _fmt(b.p))
            q = mat_to_quat(b.R)
            if not np.allclose(q, [1, 0, 0, 0], atol=1e-9):
                el.set("quat", _fmt(q, 9))
        if b.inertials:
            m, c, I = _combined_inertial(b.inertials)
            ET.SubElement(el, "inertial", pos=_fmt(c), mass=f"{m:.6g}",
                          fullinertia=_fmt([I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]], 6))
        if b.joint is not None:
            j = b.joint
            a = dict(name=j["name"], axis=_fmt(j["axis"] / np.linalg.norm(j["axis"])))
            if j["range"] is not None:
                a["range"] = _fmt(j["range"])
            if j["damping"]:
                a["damping"] = f"{j['damping']:.6g}"
            if j["friction"]:
                a["frictionloss"] = f"{j['friction']:.6g}"
            if joint_armature and j["name"] in joint_armature:
                a["armature"] = f"{joint_armature[j['name']]:.6g}"
            ET.SubElement(el, "joint", **a)
            actuated.append(j)
        for kind, gtype, attrib, Rg, pg, rgba in b.geoms:
            g = dict(**{"class": kind}, pos=_fmt(pg))
            q = mat_to_quat(Rg)
            if not np.allclose(q, [1, 0, 0, 0], atol=1e-9):
                g["quat"] = _fmt(q, 9)
            if gtype == "mesh":
                g.update(type="mesh", mesh=mesh_name(attrib))
            elif gtype == "box":
                g.update(type="box", size=_fmt(np.array([float(v) for v in attrib["size"].split()]) / 2))
            elif gtype == "cylinder":
                g.update(type="cylinder", size=_fmt([float(attrib["radius"]), float(attrib["length"]) / 2]))
            elif gtype == "sphere":
                g.update(type="sphere", size=_fmt([float(attrib["radius"])]))
            else:
                continue
            if kind == "visual" and rgba:
                g["rgba"] = rgba
            ET.SubElement(el, "geom", **g)
        for c in b.children:
            body_xml(c, el)

    mj = ET.Element("mujoco", model=name)
    ET.SubElement(mj, "compiler", angle="radian", meshdir=mesh_dir, inertiafromgeom="false",
                  balanceinertia="true", autolimits="true")
    default = ET.SubElement(mj, "default")
    ET.SubElement(default, "joint", damping="0", armature="0")
    vis = ET.SubElement(default, "default", **{"class": "visual"})
    ET.SubElement(vis, "geom", group="2", contype="0", conaffinity="0", density="0", rgba="0.75 0.75 0.78 1")
    col = ET.SubElement(default, "default", **{"class": "collision"})
    ET.SubElement(col, "geom", group="3", density="0", rgba="0.6 0.3 0.3 1", friction="1 0.005 0.0001")
    asset = ET.SubElement(mj, "asset")
    world = ET.SubElement(mj, "worldbody")
    body_xml(root, world, is_root=True)
    for (fname, scale), (n, _) in meshes.items():
        a = dict(name=n, file=fname)
        if scale != "1 1 1":
            a["scale"] = scale
        ET.SubElement(asset, "mesh", **a)
    act = ET.SubElement(mj, "actuator")
    for j in actuated:
        a = dict(name=j["name"], joint=j["name"])
        if j["effort"] and j["effort"] <= max_effort:
            a["ctrlrange"] = _fmt([-j["effort"], j["effort"]])
        ET.SubElement(act, "motor", **a)
    ET.SubElement(mj, "contact")                    # rest-pose excludes go here (add_rest_excludes)
    ET.indent(mj)
    out_dir.mkdir(parents=True, exist_ok=True)
    robot_xml = out_dir / f"{name}.xml"
    ET.ElementTree(mj).write(robot_xml, encoding="unicode")

    scene = ET.Element("mujoco", model=f"{name} scene")
    ET.SubElement(scene, "include", file=robot_xml.name)
    ET.SubElement(scene, "statistic", center="0 0 0.5", extent="1.2")
    v = ET.SubElement(scene, "visual")
    ET.SubElement(v, "headlight", diffuse="0.6 0.6 0.6", ambient="0.3 0.3 0.3", specular="0 0 0")
    ET.SubElement(v, "global", azimuth="160", elevation="-20", offwidth="1920", offheight="1080")
    a = ET.SubElement(scene, "asset")
    ET.SubElement(a, "texture", type="skybox", builtin="gradient", rgb1="0.3 0.5 0.7", rgb2="0 0 0", width="512", height="3072")
    ET.SubElement(a, "texture", type="2d", name="groundplane", builtin="checker", mark="edge", rgb1="0.2 0.3 0.4",
                  rgb2="0.1 0.2 0.3", markrgb="0.8 0.8 0.8", width="300", height="300")
    ET.SubElement(a, "material", name="groundplane", texture="groundplane", texuniform="true", texrepeat="5 5",
                  reflectance="0.2")
    w = ET.SubElement(scene, "worldbody")
    ET.SubElement(w, "light", pos="0 0 3", dir="0 0 -1", directional="true")
    ET.SubElement(w, "geom", name="floor", size="0 0 0.05", type="plane", material="groundplane")
    ET.indent(scene)
    scene_xml = out_dir / "scene.xml"
    ET.ElementTree(scene).write(scene_xml, encoding="unicode")
    return scene_xml, sorted(meshes)


def add_rest_excludes(robot_xml, scene_xml, qpos):
    """Needs MuJoCo: load the scene at qpos (the standing pose), find robot-robot contacts, and add a
    <contact><exclude> for every pair of bodies that touch - they overlap by construction (convex
    hulls of neighbouring links), not because the robot is in a bad pose. Returns the pairs added."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(scene_xml))
    d = mujoco.MjData(m)
    d.qpos[:] = qpos
    mujoco.mj_forward(m, d)
    pairs = set()
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        if b1 and b2 and b1 != b2:
            pairs.add(tuple(sorted((m.body(b1).name, m.body(b2).name))))
    tree = ET.parse(robot_xml)
    contact = tree.getroot().find("contact")
    have = {(e.get("body1"), e.get("body2")) for e in contact.findall("exclude")}
    for b1, b2 in sorted(pairs):
        if (b1, b2) not in have and (b2, b1) not in have:
            ET.SubElement(contact, "exclude", body1=b1, body2=b2)
    ET.indent(tree.getroot())
    tree.write(robot_xml, encoding="unicode")
    return sorted(pairs)
