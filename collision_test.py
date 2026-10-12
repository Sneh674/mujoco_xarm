"""
collision_test.py
Test collision detection on EVERY part of the xArm6 in MuJoCo, with black obstacles
added on top of your board (scene.xml itself is not changed).

What it does:
  1. Loads scene.xml, finds the board, and adds a few black box obstacles on it.
  2. Reports arm geoms that have collision switched OFF (these would be invisible to checks).
  3. Test 1 - random poses: counts which arm links touch obstacles, the board, the floor
     or other arm links.
  4. Test 2 - probes each obstacle: tool placed INSIDE it (collision expected) and
     ABOVE it (should be clear).
  5. Test 3 - board scan: tool-down at every grid cell, at a LOW height (hits obstacles)
     and a HIGH height (above them). Prints a map of free / colliding / unreachable cells.
     This map is the "validity map" you can later use for the RL grid.
  6. Viewer: replays the low scan; arm geoms that touch something turn red and the
     contact points are drawn.

Needs MuJoCo 3.2 or newer (pip install -U mujoco).
Run:   python collision_test.py        (macOS: mjpython collision_test.py)
"""

import time

import numpy as np
import mujoco
import mujoco.viewer

# ------------------------- SETTINGS (edit these) -------------------------
MODEL_PATH = "scene.xml"
EE_SITE = "attachment_site"
EE_BODY_FALLBACK = "link6"
ARM_DOF = 6
BOARD_NAME = ""                    # "" = first geom/body with "board" in its name
TOOL_AXIS_LOCAL = np.array([0.0, 0.0, 1.0])

# Black obstacles standing on the board, defined in the BOARD frame:
# (x position as fraction of board half-length, y position as fraction of board half-width,
#  half length (x), half width (y), half height (z)) in meters
OBSTACLES = [
    (-0.35,  0.30, 0.070, 0.015, 0.040),   # wall:  14 x 3 x 8 cm
    ( 0.30,  0.35, 0.030, 0.030, 0.050),   # block:  6 x 6 x 10 cm
    ( 0.10, -0.30, 0.015, 0.070, 0.040),   # wall turned 90 degrees: 3 x 14 x 8 cm
    (-0.30, -0.35, 0.025, 0.025, 0.030),   # block:  5 x 5 x 6 cm
]
OBSTACLE_SCALE = 1.0               # multiply all obstacle sizes (use <1 if the board is small)

GRID_NX = 10                       # scan cells along the board x axis
GRID_NY = 10                       # scan cells along the board y axis
GRID_FRACTION = 0.9                # scan the inner 90% of the board
LOW_CLEARANCE = 0.02               # tool height above the board for the LOW scan (m)
HIGH_EXTRA = 0.04                  # HIGH scan height = tallest obstacle + this (m)

N_RANDOM_POSES = 3000              # poses for Test 1
ENABLE_ARM_COLLISION = False       # True = switch collision ON for arm geoms that are OFF
POS_TOL = 0.002
ORI_TOL = 0.05
DAMPING = 0.05
N_RANDOM_STARTS = 10
SHOW_VIEWER = True
SEED = 0
# --------------------------------------------------------------------------

rng = np.random.default_rng(SEED)


def find_board(m):
    g = -1
    if BOARD_NAME != "":
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, BOARD_NAME)
        if g < 0:
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, BOARD_NAME)
            if b >= 0 and m.body_geomnum[b] > 0:
                g = m.body_geomadr[b]
    else:
        for i in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name is not None and "board" in name.lower():
                g = i
                break
        if g < 0:
            for b in range(m.nbody):
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
                if name is not None and "board" in name.lower() and m.body_geomnum[b] > 0:
                    g = m.body_geomadr[b]
                    break
    return g


# ======================= 1. find the board in the original scene =======================
model0 = mujoco.MjModel.from_xml_path(MODEL_PATH)
data0 = mujoco.MjData(model0)
mujoco.mj_forward(model0, data0)

g0 = find_board(model0)
if g0 < 0:
    print("Could not find the board. Geom names in your scene:")
    for g in range(model0.ngeom):
        print("  ", g, mujoco.mj_id2name(model0, mujoco.mjtObj.mjOBJ_GEOM, g))
    print("Set BOARD_NAME at the top of the script.")
    raise SystemExit
if model0.geom_type[g0] != mujoco.mjtGeom.mjGEOM_BOX:
    print("The board geom is not a box. This script needs a box-shaped board.")
    raise SystemExit

center = data0.geom_xpos[g0].copy()
R = data0.geom_xmat[g0].reshape(3, 3).copy()
half_x, half_y, half_z = model0.geom_size[g0]
top_center = center + R[:, 2] * half_z
print(f"Board: {2 * half_x * 100:.1f} x {2 * half_y * 100:.1f} cm, "
      f"center ({center[0] * 100:.1f}, {center[1] * 100:.1f}, {center[2] * 100:.1f}) cm")

# ======================= 2. add black obstacles =======================
spec = mujoco.MjSpec.from_file(MODEL_PATH)
quat = np.zeros(4)
mujoco.mju_mat2Quat(quat, R.flatten())

obstacle_info = []   # (name, board-frame x, board-frame y, half length, half width, half height)
print("\nObstacles added (black):")
for i, item in enumerate(OBSTACLES):
    fx, fy, hl, hw, hh = item
    hl = hl * OBSTACLE_SCALE
    hw = hw * OBSTACLE_SCALE
    hh = hh * OBSTACLE_SCALE
    ox = fx * half_x
    oy = fy * half_y
    pos = top_center + R @ np.array([ox, oy, hh])
    name = f"obstacle_{i + 1}"
    spec.worldbody.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[hl, hw, hh],
        pos=pos.tolist(),
        quat=quat.tolist(),
        rgba=[0.0, 0.0, 0.0, 1.0],
        contype=1,
        conaffinity=1,
    )
    obstacle_info.append((name, ox, oy, hl, hw, hh))
    print(f"  {name}: {2 * hl * 100:.0f} x {2 * hw * 100:.0f} x {2 * hh * 100:.0f} cm "
          f"(length x width x height) at ({pos[0] * 100:.1f}, {pos[1] * 100:.1f}) cm")
    if abs(ox) + hl > half_x or abs(oy) + hw > half_y:
        print(f"    WARNING: {name} sticks out past the board edge. Lower OBSTACLE_SCALE or edit OBSTACLES.")

model = spec.compile()
data = mujoco.MjData(model)

board_geom = find_board(model)
obstacle_geoms = {}
for g in range(model.ngeom):
    gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
    if gname is not None and gname.startswith("obstacle_"):
        obstacle_geoms[g] = gname
max_obstacle_top = max(info[5] * 2 for info in obstacle_info)   # tallest obstacle height

# ======================= 3. arm setup =======================
joint_ids = []
for j in range(model.njnt):
    if model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
        joint_ids.append(j)
joint_ids = joint_ids[:ARM_DOF]

qadr = []
dofadr = []
low = []
high = []
for j in joint_ids:
    if model.jnt_limited[j]:
        lo, hi = model.jnt_range[j]
    else:
        lo, hi = -np.pi, np.pi
    qadr.append(model.jnt_qposadr[j])
    dofadr.append(model.jnt_dofadr[j])
    low.append(lo)
    high.append(hi)
low = np.array(low)
high = np.array(high)
HOME_Q = np.clip(model.qpos0[qadr].copy(), low, high)

site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
body_id = -1
if site_id < 0:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_FALLBACK)
    if body_id < 0:
        body_id = model.nbody - 1
    print(f"\nSite '{EE_SITE}' not found, using body '{model.body(body_id).name}' as the tool.")

# the root body of the arm (the child of the world that holds the joints)
arm_root = model.jnt_bodyid[joint_ids[0]]
while model.body_parentid[arm_root] != 0:
    arm_root = model.body_parentid[arm_root]


def is_arm_body(b):
    while b != 0:
        if b == arm_root:
            return True
        b = model.body_parentid[b]
    return False


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


def reach_point(target, q_hint=None):
    """Reach a tool-down target: try the hint, then home, then random starts."""
    if q_hint is not None:
        q, ok = solve_ik(target, q_hint)
        if ok:
            return q, True
    q, ok = solve_ik(target, HOME_Q)
    if ok:
        return q, True
    for k in range(N_RANDOM_STARTS):
        q, ok = solve_ik(target, rng.uniform(low, high))
        if ok:
            return q, True
    return None, False


# ======================= 4. collision helpers =======================
def geom_label(g):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
    if name:
        return name
    return model.body(model.geom_bodyid[g]).name


# arm geoms with collision switched off
off_geoms = []
for g in range(model.ngeom):
    if is_arm_body(model.geom_bodyid[g]) and model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
        off_geoms.append(g)
arm_geom_count = 0
for g in range(model.ngeom):
    if is_arm_body(model.geom_bodyid[g]):
        arm_geom_count += 1
print(f"\nArm geoms: {arm_geom_count} total, {len(off_geoms)} with collision OFF")
if ENABLE_ARM_COLLISION and len(off_geoms) > 0:
    for g in off_geoms:
        model.geom_contype[g] = 1
        model.geom_conaffinity[g] = 1
    print(f"  ENABLE_ARM_COLLISION = True: switched collision ON for {len(off_geoms)} arm geoms")
elif len(off_geoms) > 0:
    links_off = sorted(set(model.body(model.geom_bodyid[g]).name for g in off_geoms))
    print("  Links containing collision-OFF geoms: " + ", ".join(links_off))
    print("  (Normal if they are visual copies. If a link has ONLY such geoms, it cannot collide;")
    print("   set ENABLE_ARM_COLLISION = True to test with them switched on.)")

# contacts that already exist at the home pose (e.g. the base touching its mount) are ignored
set_q(HOME_Q)
baseline_pairs = set()
for i in range(data.ncon):
    c = data.contact[i]
    baseline_pairs.add((min(c.geom1, c.geom2), max(c.geom1, c.geom2)))
if len(baseline_pairs) > 0:
    print(f"Ignoring {len(baseline_pairs)} contact pair(s) already present at the home pose:")
    for p in baseline_pairs:
        print(f"    {geom_label(p[0])} <-> {geom_label(p[1])}")


def get_contacts():
    """All contact pairs (geom ids) that are not in the home-pose baseline."""
    found = []
    for i in range(data.ncon):
        c = data.contact[i]
        pair = (min(c.geom1, c.geom2), max(c.geom1, c.geom2))
        if pair not in baseline_pairs:
            found.append(pair)
    return found


def categorize(pair):
    """Return (category, arm bodies involved, other geom id)."""
    g1, g2 = pair
    b1 = model.geom_bodyid[g1]
    b2 = model.geom_bodyid[g2]
    a1 = is_arm_body(b1)
    a2 = is_arm_body(b2)
    if a1 and a2:
        return "self (link vs link)", [model.body(b1).name, model.body(b2).name], -1
    if a1 or a2:
        arm_b = b1 if a1 else b2
        other = g2 if a1 else g1
        if other in obstacle_geoms:
            cat = "obstacle"
        elif other == board_geom:
            cat = "board"
        else:
            cat = "floor / other"
        return cat, [model.body(arm_b).name], other
    return "non-arm", [], -1


# ======================= TEST 1: random poses =======================
print(f"\n===== TEST 1: {N_RANDOM_POSES} random poses =====")
pose_cat_counts = {}
link_counts = {}
n_free = 0
for k in range(N_RANDOM_POSES):
    q = rng.uniform(low, high)
    set_q(q)
    pairs = get_contacts()
    if len(pairs) == 0:
        n_free = n_free + 1
        continue
    cats_here = set()
    links_here = set()
    for pair in pairs:
        cat, arm_bodies, other = categorize(pair)
        if cat == "non-arm":
            continue
        cats_here.add(cat)
        for name in arm_bodies:
            links_here.add(name)
    if len(cats_here) == 0:
        n_free = n_free + 1
        continue
    for cat in cats_here:
        pose_cat_counts[cat] = pose_cat_counts.get(cat, 0) + 1
    for name in links_here:
        link_counts[name] = link_counts.get(name, 0) + 1

print(f"  collision-free poses: {n_free} / {N_RANDOM_POSES}")
print("  poses with a collision, by type:")
for cat in sorted(pose_cat_counts):
    print(f"    {cat:22s} {pose_cat_counts[cat]}")
print("  arm links that were involved in a collision (number of poses):")
for name in sorted(link_counts, key=lambda n: -link_counts[n]):
    print(f"    {name:22s} {link_counts[name]}")
if pose_cat_counts.get("obstacle", 0) == 0:
    print("  NOTE: no arm link ever touched an obstacle. Check the collision-OFF report above.")

# ======================= TEST 2: probe each obstacle =======================
print("\n===== TEST 2: tool inside / above each obstacle =====")
set_q(HOME_Q)
last_q = HOME_Q.copy()
for g, gname in obstacle_geoms.items():
    obs_center = data.geom_xpos[g].copy()
    obs_top = obs_center[2] + model.geom_size[g][2]

    q_in, ok_in = reach_point(obs_center, last_q)
    if ok_in:
        last_q = q_in
        set_q(q_in)
        hit = False
        links = []
        for pair in get_contacts():
            cat, arm_bodies, other = categorize(pair)
            if other == g:
                hit = True
                links = links + arm_bodies
        if hit:
            print(f"  {gname}: tool INSIDE  -> collision detected with {', '.join(sorted(set(links)))}  [OK]")
        else:
            print(f"  {gname}: tool INSIDE  -> NO collision detected  [CHECK: tool geoms may have collision off]")
    else:
        print(f"  {gname}: tool INSIDE  -> point not reachable, skipped")

    above = obs_center.copy()
    above[2] = obs_top + 0.03
    q_up, ok_up = reach_point(above, last_q)
    if ok_up:
        last_q = q_up
        set_q(q_up)
        touching = False
        for pair in get_contacts():
            cat, arm_bodies, other = categorize(pair)
            if other == g:
                touching = True
        if touching:
            print(f"  {gname}: tool 3 cm ABOVE -> touching the obstacle (some arm link is in the way)")
        else:
            print(f"  {gname}: tool 3 cm ABOVE -> clear of this obstacle  [OK]")
    else:
        print(f"  {gname}: tool 3 cm ABOVE -> point not reachable, skipped")

# ======================= TEST 3: board scan =======================
fxs = np.linspace(-GRID_FRACTION, GRID_FRACTION, GRID_NX)
fys = np.linspace(-GRID_FRACTION, GRID_FRACTION, GRID_NY)


def under_obstacle(cx, cy):
    for info in obstacle_info:
        name, ox, oy, hl, hw, hh = info
        if abs(cx - ox) <= hl and abs(cy - oy) <= hw:
            return True
    return False


def scan(height, title):
    results = []
    grid = []
    for iy in range(GRID_NY):
        grid.append([" "] * GRID_NX)
    last = HOME_Q.copy()
    cat_cells = {}
    counts = {"free": 0, "collision": 0, "unreachable": 0}
    for iy in range(GRID_NY):
        if iy % 2 == 0:
            ix_order = range(GRID_NX)
        else:
            ix_order = range(GRID_NX - 1, -1, -1)
        for ix in ix_order:
            cx = fxs[ix] * half_x
            cy = fys[iy] * half_y
            target = top_center + R @ np.array([cx, cy, height])
            q, ok = reach_point(target, last)
            if not ok:
                grid[iy][ix] = "x"
                counts["unreachable"] += 1
                continue
            last = q
            set_q(q)
            pairs = get_contacts()
            cats_here = set()
            for pair in pairs:
                cat, arm_bodies, other = categorize(pair)
                if cat != "non-arm":
                    cats_here.add(cat)
            if len(cats_here) == 0:
                grid[iy][ix] = "."
                counts["free"] += 1
                results.append((q, False))
            else:
                if under_obstacle(cx, cy):
                    grid[iy][ix] = "O"
                else:
                    grid[iy][ix] = "#"
                counts["collision"] += 1
                for cat in cats_here:
                    cat_cells[cat] = cat_cells.get(cat, 0) + 1
                results.append((q, True))
    print(f"\n----- {title} (tool {height * 100:.1f} cm above the board) -----")
    print("  Top view in the board frame: columns = board x (left to right), rows = board y (top = +y)")
    for iy in range(GRID_NY - 1, -1, -1):
        print("   " + " ".join(grid[iy]))
    print("  legend: . free   O collision under an obstacle (expected)   "
          "# collision elsewhere (arm body hit something)   x unreachable")
    print(f"  cells: free {counts['free']}, collision {counts['collision']}, "
          f"unreachable {counts['unreachable']}")
    for cat in sorted(cat_cells):
        print(f"    collision type '{cat}': {cat_cells[cat]} cells")
    return results


print("\n===== TEST 3: board scan (tool facing down) =====")
low_height = LOW_CLEARANCE
high_height = max_obstacle_top + HIGH_EXTRA
low_results = scan(low_height, "LOW scan")
scan(high_height, "HIGH scan")
print("\nA cell marked '#' means the tool can get there but another part of the arm hits something.")
print("In the RL grid, treat '#' and 'x' cells (and the cells under obstacles) as blocked.")

# ======================= viewer replay =======================
if SHOW_VIEWER and len(low_results) > 0:
    original_rgba = model.geom_rgba.copy()
    red = np.array([1.0, 0.0, 0.0, 1.0])

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
        set_q(HOME_Q)
        viewer.sync()
        time.sleep(1.5)
        print("\nReplaying the LOW scan. Arm geoms turn red when touching something.")

        changed = []
        for q, collided in low_results:
            if not viewer.is_running():
                break
            for g in changed:
                model.geom_rgba[g] = original_rgba[g]
            changed = []
            set_q(q)
            for pair in get_contacts():
                for g in pair:
                    if is_arm_body(model.geom_bodyid[g]):
                        model.geom_rgba[g] = red
                        changed.append(g)
            viewer.sync()
            if collided:
                time.sleep(0.25)
            else:
                time.sleep(0.04)

        print("Replay finished. Close the viewer window to exit.")
        while viewer.is_running():
            time.sleep(0.1)