"""
play_arm.py  -  STEP 2: let the trained RL brain drive the xArm6 (tool facing down).

  1. Loads q_table.npz (made by train_rl.py)
  2. Picks a start cell and goal cell (random, or set them below)
  3. Follows the best moves from the Q table to get a path of cells
  4. Moves the arm cell by cell with inverse kinematics, tool pointing down

Run:  python play_arm.py        (macOS: mjpython play_arm.py)
"""

import time
import numpy as np
import mujoco
import mujoco.viewer

# ---------------- settings ----------------
MODEL_PATH = "rl_scene.xml"
EE_SITE = "attachment_site"                  # tool tip site
TOOL_AXIS = np.array([0.0, 0.0, 1.0])        # tool axis in the site frame
TOOL_HEIGHT = 0.01                           # tool height above the board surface (m)
START_CELL = None                            # e.g. (0, 0)  or None = random
GOAL_CELL = None                             # e.g. (5, 8)  or None = random
SUBSTEPS = 10                                # small steps between two cells (smooth motion)
# ------------------------------------------

MOVES = [(1, 0), (-1, 0), (0, 1), (0, -1)]   # must match train_rl.py
DOWN = np.array([0.0, 0.0, -1.0])

Q = np.load("q_table.npz")["Q"]
NX = Q.shape[0]
NY = Q.shape[1]

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# ---------- board and grid ----------
board_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "board")
board_pos = model.geom_pos[board_id]
half_x, half_y, half_z = model.geom_size[board_id]
board_top = board_pos[2] + half_z


def cell_to_world(ix, iy, height):
    """Centre of grid cell (ix, iy) in world coordinates."""
    x = board_pos[0] - half_x + (ix + 0.5) * (2 * half_x / NX)
    y = board_pos[1] - half_y + (iy + 0.5) * (2 * half_y / NY)
    return np.array([x, y, board_top + height])


# ---------- arm joints ----------
joint_ids = []
for j in range(model.njnt):
    if model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
        joint_ids.append(j)
joint_ids = joint_ids[:6]

qadr = []
dofadr = []
low = []
high = []
for j in joint_ids:
    qadr.append(model.jnt_qposadr[j])
    dofadr.append(model.jnt_dofadr[j])
    if model.jnt_limited[j]:
        low.append(model.jnt_range[j][0])
        high.append(model.jnt_range[j][1])
    else:
        low.append(-np.pi)
        high.append(np.pi)
low = np.array(low)
high = np.array(high)
HOME_Q = np.clip(model.qpos0[qadr], low, high)

site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
if site_id < 0:
    raise SystemExit(f"Site '{EE_SITE}' not found. Change EE_SITE at the top.")


def solve_ik(target, q_start):
    """Find joint angles so the tool tip is at 'target' with the tool pointing down."""
    q = q_start.copy()
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    for i in range(300):
        data.qpos[qadr] = q
        mujoco.mj_forward(model, data)
        pos = data.site_xpos[site_id]
        axis = data.site_xmat[site_id].reshape(3, 3) @ TOOL_AXIS
        err = np.concatenate([target - pos, np.cross(axis, DOWN)])   # position error + tilt error
        if np.linalg.norm(err[:3]) < 0.002 and np.linalg.norm(err[3:]) < 0.05:
            return q, True
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        J = np.vstack([jacp[:, dofadr], jacr[:, dofadr]])
        dq = J.T @ np.linalg.solve(J @ J.T + 0.05 ** 2 * np.eye(6), err)
        q = np.clip(q + np.clip(dq, -0.3, 0.3), low, high)
    return q, False


def go_to(target, q_start):
    """IK from the current pose; if that fails, try a few random starting poses."""
    q, ok = solve_ik(target, q_start)
    if ok:
        return q, True
    for k in range(20):
        q, ok = solve_ik(target, np.random.uniform(low, high))
        if ok:
            return q, True
    return q_start, False


# ---------- start, goal and path from the Q table ----------
if START_CELL is None:
    START_CELL = (np.random.randint(NX), np.random.randint(NY))
if GOAL_CELL is None:
    GOAL_CELL = (np.random.randint(NX), np.random.randint(NY))
    while GOAL_CELL == START_CELL:
        GOAL_CELL = (np.random.randint(NX), np.random.randint(NY))

x, y = START_CELL
gx, gy = GOAL_CELL
path = [(x, y)]
for step in range(4 * (NX + NY)):
    if x == gx and y == gy:
        break
    action = np.argmax(Q[x, y, gx, gy])
    new_x = x + MOVES[action][0]
    new_y = y + MOVES[action][1]
    if 0 <= new_x < NX and 0 <= new_y < NY:
        x = new_x
        y = new_y
    path.append((x, y))

print(f"Grid {NX} x {NY}, start {START_CELL}, goal {GOAL_CELL}")
print(f"Path has {len(path) - 1} moves, reached goal: {path[-1] == GOAL_CELL}")

# ---------- put the markers on the board ----------
start_mocap = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "start")]
goal_mocap = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "goal")]
data.mocap_pos[start_mocap] = cell_to_world(START_CELL[0], START_CELL[1], 0.015)
data.mocap_pos[goal_mocap] = cell_to_world(GOAL_CELL[0], GOAL_CELL[1], 0.015)

# ---------- move the arm ----------
data.qpos[qadr] = HOME_Q
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    time.sleep(1.0)

    # first go to the start cell
    p_now = cell_to_world(path[0][0], path[0][1], TOOL_HEIGHT)
    q, ok = go_to(p_now, HOME_Q)
    if not ok:
        print("Could not reach the start cell with the tool facing down.")
    data.qpos[qadr] = q
    mujoco.mj_forward(model, data)
    viewer.sync()
    time.sleep(1.0)

    # then follow the path, one cell at a time
    for i in range(1, len(path)):
        p_next = cell_to_world(path[i][0], path[i][1], TOOL_HEIGHT)
        for k in range(1, SUBSTEPS + 1):
            if not viewer.is_running():
                break
            p = p_now + (p_next - p_now) * (k / SUBSTEPS)
            q, ok = solve_ik(p, q)
            data.qpos[qadr] = q
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.01)
        p_now = p_next
        # print the tool position after each move (world coordinates)
        print(f"move {i}: cell {path[i]}  tool at {data.site_xpos[site_id].round(3)}")

    print("Done. Close the viewer window to exit.")
    while viewer.is_running():
        time.sleep(0.1)