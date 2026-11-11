"""
Check a robot description (G1_tele/robots/<name>.py) against its MuJoCo model, and show what the
package measured from it. Run this after adding or changing a robot:

    python examples/RL/check_robot.py h1
    python examples/RL/check_robot.py all
    python examples/RL/check_robot.py t1 --motion      # also make a few seconds of every motion kind
    python examples/RL/check_robot.py t1 --stand       # also hold the default pose for 3 s in the simulator

What it checks
  * every joint, keypoint body, foot and the base exist in the model, and every joint has an actuator
  * PD gains and torque limits (from the description, else from the model's actuators)
  * the measured sole (contact points, centre, size, ankle height) and size (standing height, leg
    length, hip width, scale against the G1)
  * the default pose is inside the joint limits and has the feet level and on the floor
  * the joint signs used by the universal layout (every slot turning the same way on every robot)
  --motion: the motion generators can make every clip kind (feet planted, centre of mass held)
  --stand:  the PD controller holds the default pose without falling (needs real physics)
"""
import argparse
import sys
import time

import numpy as np

from G1_tele.robots import ROBOTS, get_robot

OK, BAD = "ok ", "!! "


def check(name, motion=False, stand=False):
    spec = get_robot(name)
    print(f"\n=== {spec.name}: {spec.xml_path}")
    problems = 0
    if not spec.xml_path.exists():
        print(f"{BAD}model file missing - run  python examples/get_robots.py {spec.name}")
        return 1
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(spec.xml_path))

    def has(kind, n):
        try:
            getattr(m, kind)(n)
            return True
        except KeyError:
            return False
    missing = ([f"joint {j.name}" for j in spec.joints if not has("joint", j.name)]
               + [f"body {b} (keypoint {k})" for k, (b, _) in spec.keypoints.items() if not has("body", b)]
               + [f"foot body {b}" for b in spec.feet if not has("body", b)]
               + [f"base body {b}" for b in (spec.root_body, spec.pelvis_body) if b and not has("body", b)])
    if missing:
        print(f"{BAD}not in the model: {', '.join(missing)}")
        return 1
    print(f"{OK}{spec.n_joints} joints ({len(spec.ctrl_idx)} driven by the policy, locked: "
          f"{[spec.joints[i].name for i in spec.locked_idx] or 'none'}), {len(spec.keypoints)} keypoints")

    from G1_tele.envs.whole_body import HumanoidEnv
    from G1_tele.motion.kinematics import Kinematics
    try:
        env = HumanoidEnv(robot=spec)
    except ValueError as e:
        print(f"{BAD}{e}")
        return 1
    k = Kinematics(spec)
    g = k.geometry
    print(f"{OK}gains (kp / kd / torque limit) per joint:")
    for i, j in enumerate(spec.joints):
        src = "spec" if j.kp is not None else "model"
        print(f"      {j.slot:22s} {j.name:28s} {env.kp[i]:7.1f} {env.kd[i]:6.2f} {env.tau_lim[i]:7.1f}  ({src})"
              + ("  locked" if j.locked else ""))
    print(f"{OK}mass {k.mass:.1f} kg, standing base height {k.stand_height:.3f} m, leg {k.leg_length:.3f} m "
          f"(scale {k.scale:.2f} x G1), hip half width {k.hip_half_width:.3f} m (x{k.lateral_scale:.2f})")
    print(f"{OK}sole: {len(k.sole_points[0])} contact points, centre {np.round(k.foot_center, 3)} m from the ankle, "
          f"half size {np.round(k.foot_half, 3)} m, ankle {k.ankle_height:.3f} m above the floor, "
          f"stance x {k.stance_x:+.3f} y {k.stance_y:.3f} m" + ("" if k.has_ankle_roll else "  (no ankle roll joint)"))
    q0 = k.default_q
    out = [(j.name, q0[i]) for i, j in enumerate(spec.joints) if not (k.q_lo[i] - 1e-6 <= q0[i] <= k.q_hi[i] + 1e-6)]
    if out:
        print(f"{BAD}default pose outside the joint limits: {out}")
        problems += 1
    k.forward(k.standing_qpos())
    for s, side in enumerate(("left", "right")):
        p, r = k.sole_world(s)
        z = p[:, 2] - r
        if z.min() < -1e-3 or np.ptp(z) > 0.01:
            print(f"{BAD}{side} sole in the default pose: lowest {z.min() * 1000:.1f} mm, tilt spread "
                  f"{np.ptp(z) * 1000:.1f} mm (should be level on the floor)")
            problems += 1
    if k.rest_contacts:
        print(f"    note: {len(k.rest_contacts)} collision pair(s) already touch in the default pose (ignored)")
    signs = k.joint_signs()
    flipped = [j.slot for j, s in zip(spec.joints, signs) if s < 0]
    print(f"{OK}universal layout: slots {sorted(set(spec.slots))[:3]}..., sign flipped for {flipped or 'none'}")

    if motion:
        from G1_tele.motion.gait import GaitGenerator
        from G1_tele.motion.synthetic import SyntheticMotionGenerator
        gen = SyntheticMotionGenerator(k, seed=0, verbose=False)
        for kind in ("upper", "whole", "balance", "step"):
            t0 = time.time()
            root, j = gen.clip(kind, 5.0)
            print(f"{OK}motion '{kind}': {len(root) / 50:.1f} s in {time.time() - t0:.0f} s")
        gg = GaitGenerator(k, seed=0, verbose=False)
        for kind in ("turn", "walk"):
            t0 = time.time()
            try:
                gg.clip(kind, 5.0, max_tries=10)
                print(f"{OK}motion '{kind}': 5.0 s in {time.time() - t0:.0f} s")
            except RuntimeError as e:
                print(f"{BAD}motion '{kind}': {e}")
                problems += 1

    if stand:
        env.reset()
        z0 = env.d.qpos[2]
        steps = int(3.0 / env.dt)
        for _ in range(steps):
            env.step_targets(env.default_q)
        s = env.state()
        drop, tilt = z0 - s["base_pos"][2], np.degrees(np.arccos(np.clip(-s["gravity"][2], -1, 1)))
        good = drop < 0.1 * k.scale and tilt < 10
        print(f"{OK if good else BAD}standing 3 s on PD alone: base {drop * 100:+.1f} cm lower, tilted {tilt:.1f} deg")
        problems += int(not good)
    print(("all good" if not problems else f"{problems} problem(s)") + f" for the {spec.name}")
    return problems


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("robot", help=f"one of {sorted(ROBOTS)} or 'all'")
    p.add_argument("--motion", action="store_true", help="also make a few seconds of every motion kind")
    p.add_argument("--stand", action="store_true", help="also hold the default pose in the simulator")
    a = p.parse_args()
    names = sorted(ROBOTS) if a.robot == "all" else [a.robot]
    sys.exit(1 if sum(check(n, a.motion, a.stand) for n in names) else 0)
