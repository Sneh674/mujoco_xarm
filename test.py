"""
resize_board_to_reach.py
Make the board as BIG as the xArm6 can reach (tool facing down at all four corners)
and WRITE the new size into scene.xml.

  - The board keeps its center and orientation. Only its length and width change.
  - Your scene.xml is backed up first (scene.xml.bak_before_resize, made only once).
  - The script then reloads the edited file, checks the new size, and shows the arm
    going to the four corners and along the edges.

RESIZE_MODE:
  "max_area" : tries several length:width ratios and keeps the board with the biggest area
               where all four corners are reachable (corners end up at the arm's limit)
  "uniform"  : keeps your current length:width ratio and only scales it up/down

Run:   python resize_board_to_reach.py     (macOS: mjpython resize_board_to_reach.py)
"""

import os
import time

import numpy as np
import mujoco
import mujoco.viewer

# ------------------------- SETTINGS (edit these) -------------------------
MODEL_PATH = "scene.xml"
EE_SITE = "attachment_site"
EE_BODY_FALLBACK = "link6"
ARM_DOF = 6
BOARD_NAME = ""                    # geom or body name of the board; "" = first one with "board" in its name
TOOL_AXIS_LOCAL = np.array([0.0, 0.0, 1.0])
TOOL_CLEARANCE = 0.005             # tool target height above the board surface (m)

RESIZE_MODE = "max_area"           # "max_area" or "uniform"
ASPECT_FACTORS = [0.6, 0.75, 0.9, 1.0, 1.1, 1.3, 1.6]   # length:width variations tried in max_area mode
SAFETY = 0.97                      # keep 3% inside the true limit
S_MAX = 4.0                        # search up to 4x the current size
BINARY_STEPS = 9
N_RANDOM_STARTS = 12
POS_TOL = 0.002
ORI_TOL = 0.05
DAMPING = 0.05

WRITE_XML = True                   # write the new size into the xml file
BACKUP_SUFFIX = ".bak_before_resize"
SHOW_VIEWER = True
SEED = 0
# --------------------------------------------------------------------------

rng = np.random.default_rng(SEED)

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

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
if site_id < 0:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_FALLBACK)
    if body_id < 0:
        body_id = model.nbody - 1
    print(f"Site '{EE_SITE}' not found, using body '{model.body(body_id).name}' as the tool.")


def set_q(q):
    data.qpos[qadr] = q
    mujoco.mj_forward(model, data)


def fk_fast(q):
    data.qpos[qadr] = q
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)


def ee_pose():
    if site_id >= 0:
        return data.site_xpos[site_id].copy(), data.site_xmat[site_id].reshape(3, 3).copy()
    return data.xpos[body_id].copy(), data.xmat[body_id].reshape(3, 3).copy()


down_vec = np.array([0.0, 0.0, -1.0])


def solve_ik(target, q_init, max_iter=300):
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
        dq = J.T @ np.linalg.solve(J @ J.T + (DAMPING ** 2) * np.eye(6), err)
        dq = np.clip(dq, -0.3, 0.3)
        q = np.clip(q + dq, low, high)
    return q, False


def reach_point(target):
    q, ok = solve_ik(target, HOME_Q)
    if ok:
        return q, True
    for k in range(N_RANDOM_STARTS):
        q, ok = solve_ik(target, rng.uniform(low, high))
        if ok:
            return q, True
    return None, False


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
    print("Set BOARD_NAME at the top of the script.")
    raise SystemExit
if model.geom_type[board_geom] != mujoco.mjtGeom.mjGEOM_BOX:
    print("The board geom is not a box. This script needs a box-shaped board.")
    raise SystemExit

board_geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, board_geom)
board_body_name = model.body(model.geom_bodyid[board_geom]).name

set_q(HOME_Q)
center = data.geom_xpos[board_geom].copy()
R = data.geom_xmat[board_geom].reshape(3, 3).copy()
half_x, half_y, half_z = model.geom_size[board_geom]
top_center = center + R[:, 2] * half_z

print(f"\nBoard: {board_geom_name or board_body_name}")
print(f"  center (x, y, z): {center[0] * 100:.1f}, {center[1] * 100:.1f}, {center[2] * 100:.1f} cm")
print(f"  CURRENT size: {2 * half_x * 100:.1f} x {2 * half_y * 100:.1f} x {2 * half_z * 100:.1f} cm")

signs = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
corner_names = ["(-x,-y)", "(+x,-y)", "(+x,+y)", "(-x,+y)"]


def corner_target(i, kx, ky):
    """Tool target at corner i when the board half-sizes are multiplied by kx (length) and ky (width)."""
    sx, sy = signs[i]
    offset = sx * half_x * kx * R[:, 0] + sy * half_y * ky * R[:, 1]
    return top_center + offset + np.array([0.0, 0.0, TOOL_CLEARANCE])


def all_reachable(idx_list, kx, ky):
    sols = []
    for i in idx_list:
        q, ok = reach_point(corner_target(i, kx, ky))
        if not ok:
            return False, None
        sols.append(q)
    return True, sols


def max_scale(idx_list, k):
    """Largest s such that board half-sizes (s*k, s/k) times the current ones are reachable."""
    ok, sols = all_reachable(idx_list, S_MAX * k, S_MAX / k)
    if ok:
        return S_MAX
    lo = 0.0
    hi = S_MAX
    for step in range(BINARY_STEPS):
        mid = (lo + hi) / 2
        ok, sols = all_reachable(idx_list, mid * k, mid / k)
        if ok:
            lo = mid
        else:
            hi = mid
    return lo


# ---- is the board centre reachable at all? ----
q_center, ok = reach_point(top_center + np.array([0.0, 0.0, TOOL_CLEARANCE]))
if not ok:
    print("\nThe board CENTER is not reachable with the tool facing down.")
    print("The board is too far from the arm base (or too high/low). Move it closer in scene.xml first.")
    raise SystemExit

# ---- per-corner limit (each corner on its own line from the board centre) ----
print("\nMax reach per corner direction (current board shape, each corner on its own):")
per_corner_scale = []
for i in range(4):
    s_i = max_scale([i], 1.0)
    per_corner_scale.append(s_i)
    t = corner_target(i, s_i, s_i)
    print(f"  corner {corner_names[i]}: x{s_i:.2f} of the current half-size | "
          f"{np.linalg.norm((t - top_center)[:2]) * 100:.1f} cm from the board centre")

# ---- biggest board with all four corners reachable ----
print("\nSearching for the biggest board where ALL four corners are reachable...")
if RESIZE_MODE == "max_area":
    k_list = ASPECT_FACTORS
else:
    k_list = [1.0]

best_area = -1.0
best_k = 1.0
best_s = 0.0
for k in k_list:
    s = max_scale([0, 1, 2, 3], k)
    area_ratio = s * s          # (s*k) * (s/k) = s^2 times the current area
    print(f"  length x{s * k:.2f}, width x{s / k:.2f}  ->  area x{area_ratio:.2f}")
    if area_ratio > best_area:
        best_area = area_ratio
        best_k = k
        best_s = s

kx_final = best_s * best_k * SAFETY
ky_final = best_s / best_k * SAFETY
new_half_x = half_x * kx_final
new_half_y = half_y * ky_final

print("\n===== RESULT =====")
if best_s >= S_MAX:
    print(f"  Hit the search cap (x{S_MAX:.1f}). Raise S_MAX if you want an even larger board.")
print(f"  Current board : {2 * half_x * 100:.1f} x {2 * half_y * 100:.1f} cm")
print(f"  New board     : {2 * new_half_x * 100:.1f} x {2 * new_half_y * 100:.1f} cm "
      f"(same center, same orientation, same thickness)")
size_text = f"{new_half_x:.4f} {new_half_y:.4f} {half_z:.4f}"
print(f"  New geom size : size=\"{size_text}\"   (half-sizes in meters)")
print("  New corner positions (world, cm):")
for i in range(4):
    t = corner_target(i, kx_final, ky_final)
    print(f"    {corner_names[i]}: ({t[0] * 100:.1f}, {t[1] * 100:.1f}, {t[2] * 100:.1f})")

ok, corner_solutions = all_reachable([0, 1, 2, 3], kx_final, ky_final)
if not ok:
    print("\n(Could not re-solve the final corners, so nothing was written.)")
    raise SystemExit


# ---- write the new size into the xml ----
def set_board_size_in_file(path, geom_name, body_name, new_size_text):
    """Replace size="..." of the board geom inside one xml file. Returns True if it worked."""
    f = open(path, "r")
    text = f.read()
    f.close()

    tag_start = -1
    if geom_name:
        for quote in ['"', "'"]:
            key = "name=" + quote + geom_name + quote
            pos = 0
            while tag_start < 0:
                idx = text.find(key, pos)
                if idx < 0:
                    break
                start = text.rfind("<", 0, idx)
                if text[start:start + 5] == "<geom":
                    tag_start = start
                pos = idx + 1
            if tag_start >= 0:
                break
    else:
        for quote in ['"', "'"]:
            key = "name=" + quote + body_name + quote
            idx = text.find(key)
            if idx >= 0:
                start = text.rfind("<", 0, idx)
                if text[start:start + 5] == "<body":
                    tag_start = text.find("<geom", start)
                    break
    if tag_start < 0:
        return False

    tag_end = text.find(">", tag_start)
    tag = text[tag_start:tag_end + 1]
    for quote in ['"', "'"]:
        key = "size=" + quote
        i = tag.find(key)
        if i >= 0:
            v_start = i + len(key)
            v_end = tag.find(quote, v_start)
            new_tag = tag[:v_start] + new_size_text + tag[v_end:]
            new_text = text[:tag_start] + new_tag + text[tag_end + 1:]

            backup = path + BACKUP_SUFFIX
            if not os.path.exists(backup):
                fb = open(backup, "w")
                fb.write(text)
                fb.close()
            fw = open(path, "w")
            fw.write(new_text)
            fw.close()
            return True
    return False


xml_written = False
if WRITE_XML:
    folder = os.path.dirname(os.path.abspath(MODEL_PATH))
    candidates = [MODEL_PATH]
    for fname in sorted(os.listdir(folder)):
        full = os.path.join(folder, fname)
        if fname.lower().endswith(".xml") and os.path.abspath(full) != os.path.abspath(MODEL_PATH):
            candidates.append(full)
    for path in candidates:
        if set_board_size_in_file(path, board_geom_name, board_body_name, size_text):
            print(f"\nWROTE the new board size into: {path}")
            print(f"Backup of the original: {path}{BACKUP_SUFFIX}")
            xml_written = True
            break
    if not xml_written:
        print("\nCould not find a size=\"...\" on the board geom in the xml files, so nothing was written.")
        print(f"Edit it by hand to:  size=\"{size_text}\"")

if xml_written:
    # reload from the edited file to prove the xml change works
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    got = model.geom_size[board_geom]
    print(f"Reloaded {MODEL_PATH}: board is now {2 * got[0] * 100:.1f} x {2 * got[1] * 100:.1f} cm")
    if abs(got[0] - new_half_x) > 1e-3 or abs(got[1] - new_half_y) > 1e-3:
        print("WARNING: the reloaded size does not match. The board may be defined in another file.")
else:
    # show the new size in the viewer anyway (in memory only)
    model.geom_size[board_geom][0] = new_half_x
    model.geom_size[board_geom][1] = new_half_y

# ---- plot ----
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def outline(kx, ky):
        pts = [corner_target(i, kx, ky)[:2] for i in range(4)]
        pts.append(pts[0])
        return np.array(pts)

    plt.figure(figsize=(6, 6))
    old = outline(1.0, 1.0)
    new = outline(kx_final, ky_final)
    plt.plot(old[:, 0], old[:, 1], "k--", label="old board")
    plt.plot(new[:, 0], new[:, 1], "g-", label="new board")
    base_xy = data.xpos[model.jnt_bodyid[joint_ids[0]]][:2]
    plt.plot([base_xy[0]], [base_xy[1]], "rx", markersize=10, label="arm base")
    plt.gca().set_aspect("equal")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Board resize (top view)")
    plt.legend()
    plt.savefig("board_resize.png", dpi=150)
    print("Saved plot: board_resize.png")
except ImportError:
    print("(matplotlib not installed, skipping plot)")

# ---- viewer ----
if SHOW_VIEWER:
    targets = [corner_target(i, kx_final, ky_final) for i in range(4)]

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
            set_q(q_from + (q_to - q_from) * (i / steps))
            viewer.sync()
            time.sleep(0.01)

    def move_straight(viewer, q_start, p_start, p_end, steps=80):
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

        print("Tour finished. Close the viewer window to exit.")
        while viewer.is_running():
            time.sleep(0.1)