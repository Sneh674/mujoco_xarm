"""
board_corner_reach.py
Point the xArm6 (tool facing down) at the FOUR CORNERS OF THE BOARD and find the
largest board size the arm can reach. The board keeps its position and orientation;
only its size changes (scaled from its centre, same length:width ratio).

What it does:
  1. Finds the board box in your scene and its four corners.
  2. Uses inverse kinematics (tool pointing straight down) to test each corner.
  3. Finds, for each corner, how far out along its direction the arm can still reach
     (the arm's max limit in that direction).
  4. Finds the biggest board where ALL four corners are reachable at the same time.
  5. Prints the new board size (and the numbers to put in scene.xml).
  6. Viewer: shows the resized board, then moves the arm to each corner and along
     the board edges with the tool facing down.

Run:   python board_corner_reach.py     (macOS: mjpython board_corner_reach.py)
"""

import time

import numpy as np
import mujoco
import mujoco.viewer

# ------------------------- SETTINGS (edit these) -------------------------
MODEL_PATH = "scene.xml"
EE_SITE = "attachment_site"        # end-effector site (falls back to a body if missing)
EE_BODY_FALLBACK = "link6"
ARM_DOF = 6
BOARD_NAME = ""                    # geom or body name of the board; "" = first one with "board" in its name
TOOL_AXIS_LOCAL = np.array([0.0, 0.0, 1.0])   # tool axis in the site frame (points out of the flange)
TOOL_CLEARANCE = 0.005             # tool target height above the board surface (m)
SAFETY = 0.97                      # use 97% of the max board size so corners are not at the very edge
S_MAX = 3.0                        # search up to 3x the current board size
BINARY_STEPS = 10                  # precision of the size search
N_RANDOM_STARTS = 15               # extra random IK starts when the first try fails
POS_TOL = 0.002                    # IK position tolerance (m)
ORI_TOL = 0.05                     # IK tool-down tolerance (about 3 degrees)
DAMPING = 0.05
SHOW_VIEWER = True
SEED = 0
# --------------------------------------------------------------------------

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
rng = np.random.default_rng(SEED)

# ---- arm joints ----
joint_ids = []
for j in range(model.njnt):
    if model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
        joint_ids.append(j)
joint_ids = joint_ids[:ARM_DOF]

qadr = []
dofadr = []
low = []
high = []
print("Arm joints used:")
for j in joint_ids:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    if model.jnt_limited[j]:
        lo, hi = model.jnt_range[j]
    else:
        lo, hi = -np.pi, np.pi
    qadr.append(model.jnt_qposadr[j])
    dofadr.append(model.jnt_dofadr[j])
    low.append(lo)
    high.append(hi)
    print(f"  {name}: {np.degrees(lo):.0f} to {np.degrees(hi):.0f} deg")
low = np.array(low)
high = np.array(high)
HOME_Q = np.clip(model.qpos0[qadr].copy(), low, high)

# ---- end effector ----
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
body_id = -1
if site_id >= 0:
    print(f"End effector: site '{EE_SITE}'")
else:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_FALLBACK)
    if body_id < 0:
        body_id = model.nbody - 1
    print(f"End effector: site '{EE_SITE}' not found, using body "
          f"'{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)}'")


def set_q(q):
    """Full forward pass (used for display)."""
    data.qpos[qadr] = q
    mujoco.mj_forward(model, data)


def fk_fast(q):
    """Only kinematics (faster, used inside the IK loop)."""
    data.qpos[qadr] = q
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)


def ee_pose():
    if site_id >= 0:
        return data.site_xpos[site_id].copy(), data.site_xmat[site_id].reshape(3, 3).copy()
    return data.xpos[body_id].copy(), data.xmat[body_id].reshape(3, 3).copy()


down_vec = np.array([0.0, 0.0, -1.0])


def solve_ik(target, q_init, max_iter=300):
    """Move the tool to 'target' with the tool pointing straight down. Returns (q, success)."""
    q = np.array(q_init, dtype=float)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    for it in range(max_iter):
        fk_fast(q)
        pos, rot = ee_pose()
        err_pos = target - pos
        axis_world = rot @ TOOL_AXIS_LOCAL
        err_ori = np.cross(axis_world, down_vec)
        if np.linalg.norm(err_pos) < POS_TOL and np.linalg.norm(err_ori) < ORI_TOL:
            return q, True
        if site_id >= 0:
            mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        else:
            mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
        J = np.vstack([jacp[:, dofadr], jacr[:, dofadr]])
        err = np.concatenate([err_pos, err_ori])
        JJt = J @ J.T + (DAMPING ** 2) * np.eye(6)
        dq = J.T @ np.linalg.solve(JJt, err)
        dq = np.clip(dq, -0.3, 0.3)
        q = np.clip(q + dq, low, high)
    return q, False


def reach_point(target):
    """Try to reach a point: first from the home pose, then from random starting poses."""
    q, ok = solve_ik(target, HOME_Q)
    if ok:
        return q, True
    for k in range(N_RANDOM_STARTS):
        q0 = rng.uniform(low, high)
        q, ok = solve_ik(target, q0)
        if ok:
            return q, True
    return None, False


# ---- arm base position ----
set_q(HOME_Q)
base_xy = data.xpos[model.jnt_bodyid[joint_ids[0]]][:2].copy()
print(f"Base position (x, y): {base_xy[0]:.3f}, {base_xy[1]:.3f}")

# ---- find the board (must be a box) ----
board_geom = -1
if BOARD_NAME != "":
    board_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, BOARD_NAME)
    if board_geom < 0:
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BOARD_NAME)
        if b >= 0 and model.body_geomnum[b] > 0:
            board_geom = model.body_geomadr[b]
else:
    for g in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        if gname is not None and "board" in gname.lower():
            board_geom = g
            break
    if board_geom < 0:
        for b in range(model.nbody):
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
            if bname is not None and "board" in bname.lower() and model.body_geomnum[b] > 0:
                board_geom = model.body_geomadr[b]
                break

if board_geom < 0:
    print("\nCould not find the board. Geom names in your scene:")
    for g in range(model.ngeom):
        print("  ", g, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g))
    print("Set BOARD_NAME at the top of the script to the board's geom or body name.")
    raise SystemExit
if model.geom_type[board_geom] != mujoco.mjtGeom.mjGEOM_BOX:
    print("The board geom is not a box. This script needs a box-shaped board.")
    raise SystemExit

board_label = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, board_geom)
center = data.geom_xpos[board_geom].copy()
R = data.geom_xmat[board_geom].reshape(3, 3).copy()
half_x, half_y, half_z = model.geom_size[board_geom]
top_center = center + R[:, 2] * half_z

print(f"\nBoard geom: {board_label}")
print(f"  center (x, y, z): {center[0] * 100:.1f}, {center[1] * 100:.1f}, {center[2] * 100:.1f} cm")
print(f"  current size: {2 * half_x * 100:.1f} x {2 * half_y * 100:.1f} x {2 * half_z * 100:.1f} cm "
      f"(length x width x thickness)")

signs = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
corner_names = ["(-x,-y)", "(+x,-y)", "(+x,+y)", "(-x,+y)"]


def corner_target(i, s):
    """Tool target at corner i of the board scaled by s (from the board centre)."""
    sx, sy = signs[i]
    offset = sx * half_x * s * R[:, 0] + sy * half_y * s * R[:, 1]
    return top_center + offset + np.array([0.0, 0.0, TOOL_CLEARANCE])


def all_reachable(idx_list, s):
    sols = []
    for i in idx_list:
        q, ok = reach_point(corner_target(i, s))
        if not ok:
            return False, None
        sols.append(q)
    return True, sols


def max_scale(idx_list):
    """Largest scale s (0..S_MAX) at which every corner in idx_list is reachable."""
    ok, sols = all_reachable(idx_list, S_MAX)
    if ok:
        return S_MAX
    lo = 0.0
    hi = S_MAX
    for step in range(BINARY_STEPS):
        mid = (lo + hi) / 2
        ok, sols = all_reachable(idx_list, mid)
        if ok:
            lo = mid
        else:
            hi = mid
    return lo


# ---- is the board centre reachable at all? ----
q_center, ok = reach_point(top_center + np.array([0.0, 0.0, TOOL_CLEARANCE]))
if not ok:
    print("\nThe board CENTER is not reachable with the tool facing down.")
    print("The board is too far from the arm base (or too high/low). Nothing to fit.")
    raise SystemExit

# ---- current board: which corners are reachable? ----
print("\nCurrent board corners (tool facing down):")
for i in range(4):
    q, ok = reach_point(corner_target(i, 1.0))
    print(f"  corner {corner_names[i]}: {'reachable' if ok else 'NOT reachable'}")

# ---- per-corner max reach along its own direction ----
print("\nMax reach per corner direction (from the board centre, along the line to that corner):")
per_corner_scale = []
for i in range(4):
    s_i = max_scale([i])
    per_corner_scale.append(s_i)
    t = corner_target(i, s_i)
    from_center = np.linalg.norm((t - top_center)[:2])
    from_base = np.linalg.norm(t[:2] - base_xy)
    note = "  (hit the search cap)" if s_i >= S_MAX else ""
    print(f"  corner {corner_names[i]}: x{s_i:.2f} of current size | "
          f"{from_center * 100:.1f} cm from board centre | "
          f"{from_base * 100:.1f} cm from arm base{note}")

# ---- biggest board where all four corners work together ----
print("\nSearching for the biggest board where ALL four corners are reachable...")
s_all = max_scale([0, 1, 2, 3])
s_final = s_all * SAFETY
new_half_x = half_x * s_final
new_half_y = half_y * s_final

print("\n===== RESULT =====")
if s_all >= S_MAX:
    print(f"  Even x{S_MAX:.1f} of the current size is reachable (search cap). Raise S_MAX to go larger.")
print(f"  Max uniform scale: x{s_all:.2f}   (using x{s_final:.2f} with the {int(SAFETY * 100)}% safety margin)")
print(f"  Current board : {2 * half_x * 100:.1f} x {2 * half_y * 100:.1f} cm")
print(f"  New board     : {2 * new_half_x * 100:.1f} x {2 * new_half_y * 100:.1f} cm "
      f"(thickness unchanged, same center and orientation)")
print(f"  In scene.xml set the board geom size to:  size=\"{new_half_x:.4f} {new_half_y:.4f} {half_z:.4f}\"")
print("  (these are HALF-sizes in meters). Keep any goal marker or obstacles inside the new board.")
print("  New corner positions (world, cm):")
for i in range(4):
    t = corner_target(i, s_final)
    print(f"    {corner_names[i]}: ({t[0] * 100:.1f}, {t[1] * 100:.1f}, {t[2] * 100:.1f})")

ok, corner_solutions = all_reachable([0, 1, 2, 3], s_final)
if not ok:
    print("\n(Could not re-solve the final corners; skipping the viewer tour.)")
    SHOW_VIEWER = False

# ---- plot ----
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def outline(s):
        pts = [corner_target(i, s)[:2] for i in range(4)]
        pts.append(pts[0])
        return np.array(pts)

    plt.figure(figsize=(6, 6))
    old = outline(1.0)
    new = outline(s_final)
    plt.plot(old[:, 0], old[:, 1], "k--", label="current board")
    plt.plot(new[:, 0], new[:, 1], "g-", label="new board (reachable)")
    for i in range(4):
        t = corner_target(i, per_corner_scale[i])
        plt.plot([t[0]], [t[1]], "bo")
    plt.plot([], [], "bo", label="max reach per corner")
    plt.plot([base_xy[0]], [base_xy[1]], "rx", markersize=10, label="arm base")
    plt.gca().set_aspect("equal")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Board fit (top view)")
    plt.legend()
    plt.savefig("board_fit.png", dpi=150)
    print("\nSaved plot: board_fit.png")
except ImportError:
    print("\n(matplotlib not installed, skipping plot)")

# ---- viewer ----
if SHOW_VIEWER:
    # show the resized board (in memory only, scene.xml is not changed)
    model.geom_size[board_geom][0] = new_half_x
    model.geom_size[board_geom][1] = new_half_y

    targets = [corner_target(i, s_final) for i in range(4)]

    def draw_corners(viewer):
        with viewer.lock():
            scn = viewer.user_scn
            scn.ngeom = 0
            for t in targets:
                mujoco.mjv_initGeom(
                    scn.geoms[scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([0.008, 0.0, 0.0]),
                    t,
                    np.eye(3).flatten(),
                    np.array([1.0, 0.0, 0.0, 0.9], dtype=np.float32),
                )
                scn.ngeom += 1

    def move_joint_space(viewer, q_from, q_to, steps=150):
        for i in range(1, steps + 1):
            if not viewer.is_running():
                return
            q = q_from + (q_to - q_from) * (i / steps)
            set_q(q)
            viewer.sync()
            time.sleep(0.01)

    def move_straight(viewer, q_start, p_start, p_end, steps=80):
        """Move the tool in a straight line (tool facing down) using IK at every step."""
        q = q_start
        for i in range(1, steps + 1):
            if not viewer.is_running():
                return q
            p = p_start + (p_end - p_start) * (i / steps)
            q, ok = solve_ik(p, q, max_iter=100)
            set_q(q)
            viewer.sync()
            time.sleep(0.01)
        return q

    set_q(HOME_Q)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        draw_corners(viewer)
        time.sleep(1.0)

        print("\nMoving to corner", corner_names[0])
        move_joint_space(viewer, HOME_Q, corner_solutions[0])
        q_now = corner_solutions[0]
        time.sleep(1.0)

        for k in range(4):
            nxt = (k + 1) % 4
            if not viewer.is_running():
                break
            print(f"Tracing edge {corner_names[k]} -> {corner_names[nxt]}")
            q_now = move_straight(viewer, q_now, targets[k], targets[nxt])
            time.sleep(0.8)

        if viewer.is_running():
            print("Moving to the board center")
            center_target = top_center + np.array([0.0, 0.0, TOOL_CLEARANCE])
            q_now = move_straight(viewer, q_now, targets[0], center_target)

        print("Tour finished. Close the viewer window to exit.")
        while viewer.is_running():
            time.sleep(0.1)