"""

"""

import time
import numpy as np
import mujoco
import mujoco.viewer
from grid_utils import model, NX, NY, MOVES, BLOCKED, is_blocked, allowed_scores, cell_to_world
import torch
import torch.nn as nn
from train_dqn import qNN

q_model = qNN()
# q_model.load_state_dict(torch.load("q_model.pth", map_location="cpu", weights_only=True))
q_model.load_state_dict(torch.load("q_model_use.pth", map_location="cpu", weights_only=True))
q_model.eval()

# ---------------- settings ----------------
EE_SITE = "attachment_site"                  # tool tip site (used if it exists)
EE_BODY = "link6"                            # otherwise this body is used as the tool
TOOL_AXIS = np.array([0.0, 0.0, 1.0])        # tool axis in the site/body frame
TOOL_HEIGHT = 0.01                           # tool height above the board surface (m)
START_CELL = None                            # e.g. (0, 0)  or None = new random cell each replay
GOAL_CELL = None                             # e.g. (6, 12) or None = new random cell each replay
SUBSTEPS = 10                                # small steps between two cells (smooth motion)
ENABLE_ARM_COLLISION = False                 # True = also switch collision ON for arm geoms that have it OFF
# ------------------------------------------

DOWN = np.array([0.0, 0.0, -1.0])

# Q = np.load("q_table.npz")["Q"]
# if Q.shape[0] != NX or Q.shape[1] != NY:
#     raise SystemExit("q_table.npz does not match the grid in scene.xml. Run train_rl.py again.")

data = mujoco.MjData(model)

# ---------- arm joints ----------
HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)
SLIDE = int(mujoco.mjtJoint.mjJNT_SLIDE)
joint_ids = []
for j in range(model.njnt):
    if int(model.jnt_type[j]) in (HINGE, SLIDE):
        joint_ids.append(j)
joint_ids = joint_ids[:6]
if len(joint_ids) < 6:
    raise SystemExit(f"Found only {len(joint_ids)} arm joints, expected 6. Check xarm6.xml.")

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

# ---------- the tool (site if it exists, otherwise a body) ----------
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY)
if site_id >= 0:
    print(f"Tool = site '{EE_SITE}'")
elif body_id >= 0:
    print(f"Site '{EE_SITE}' not found, using body '{EE_BODY}' as the tool")
else:
    print("Neither the site nor the body was found. Sites and bodies in your model:")
    for i in range(model.nsite):
        print("  site:", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i))
    for i in range(model.nbody):
        print("  body:", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i))
    raise SystemExit("Set EE_SITE or EE_BODY at the top to one of these names.")


def tool_position():
    """World position of the tool."""
    if site_id >= 0:
        return data.site_xpos[site_id].copy()
    return data.xpos[body_id].copy()


def tool_rotation():
    """3x3 rotation matrix of the tool."""
    if site_id >= 0:
        return data.site_xmat[site_id].reshape(3, 3).copy()
    return data.xmat[body_id].reshape(3, 3).copy()


def solve_ik(target, q_start):
    """Find joint angles so the tool is at 'target' with the tool pointing down."""
    q = q_start.copy()
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    for i in range(300):
        data.qpos[qadr] = q
        mujoco.mj_kinematics(model, data)      # only kinematics (faster than mj_forward)
        mujoco.mj_comPos(model, data)
        pos = tool_position()
        axis = tool_rotation() @ TOOL_AXIS
        err = np.concatenate([target - pos, np.cross(axis, DOWN)])   # position error + tilt error
        if np.linalg.norm(err[:3]) < 0.002 and np.linalg.norm(err[3:]) < 0.05:
            return q, True
        if site_id >= 0:
            mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        else:
            mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
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


def random_free_cell():
    while True:
        x = np.random.randint(NX)
        y = np.random.randint(NY)
        if not is_blocked(x, y):
            return (x, y)


# ---------- collision display (arm parts that touch something turn red) ----------
RED = np.array([1.0, 0.0, 0.0, 1.0])

# the root body of the arm (the child of the world that holds the joints)
arm_root = model.jnt_bodyid[joint_ids[0]]
while model.body_parentid[arm_root] != 0:
    arm_root = model.body_parentid[arm_root]


def is_arm_body(b):
    """True if body b belongs to the arm."""
    while b != 0:
        if b == arm_root:
            return True
        b = model.body_parentid[b]
    return False


# arm geoms with collision switched OFF can never touch anything
off_geoms = []
arm_geom_count = 0
for g in range(model.ngeom):
    if is_arm_body(model.geom_bodyid[g]):
        arm_geom_count = arm_geom_count + 1
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            off_geoms.append(g)
print(f"Arm geoms: {arm_geom_count} total, {len(off_geoms)} with collision OFF")
if ENABLE_ARM_COLLISION:
    for g in off_geoms:
        model.geom_contype[g] = 1
        model.geom_conaffinity[g] = 1
    print(f"  ENABLE_ARM_COLLISION = True: collision switched ON for {len(off_geoms)} arm geoms")

original_rgba = model.geom_rgba.copy()

# contacts that already exist at the home pose (e.g. the base touching its mount) are ignored
data.qpos[qadr] = HOME_Q
mujoco.mj_forward(model, data)
baseline_pairs = set()
for i in range(data.ncon):
    c = data.contact[i]
    baseline_pairs.add((min(c.geom1, c.geom2), max(c.geom1, c.geom2)))


def get_contacts():
    """All contact pairs (geom ids) that are not in the home-pose baseline."""
    found = []
    for i in range(data.ncon):
        c = data.contact[i]
        pair = (min(c.geom1, c.geom2), max(c.geom1, c.geom2))
        if pair not in baseline_pairs:
            found.append(pair)
    return found


def update_colors():
    """Turn arm geoms red while they touch something, normal colour otherwise.
    Call this right after mj_forward (which finds the contacts)."""
    model.geom_rgba[:] = original_rgba
    for pair in get_contacts():
        for g in pair:
            if is_arm_body(model.geom_bodyid[g]):
                model.geom_rgba[g] = RED


# ---------- markers ----------
start_mocap = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "start")]
goal_mocap = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "goal")]


def show_markers(start, goal):
    data.mocap_pos[start_mocap] = cell_to_world(start[0], start[1], 0.015)
    data.mocap_pos[goal_mocap] = cell_to_world(goal[0], goal[1], 0.015)


# ---------- one run: choose cells, build path, move the arm ----------
def pick_cells():
    """Start and goal: the ones set at the top, or random free cells."""
    start = START_CELL
    goal = GOAL_CELL
    if start is None:
        start = random_free_cell()
    if goal is None:
        goal = random_free_cell()
        while goal == start:
            goal = random_free_cell()
    if is_blocked(start[0], start[1]) or is_blocked(goal[0], goal[1]):
        raise SystemExit("START_CELL or GOAL_CELL is inside an obstacle or off the board.")
    return start, goal

def get_state(x, y, gx, gy):
    x_diff = gx - x
    y_diff = gy - y

    blocked = []

    for dx, dy in MOVES:
        nx = x + dx
        ny = y + dy

        if nx < 0 or nx >= NX or ny < 0 or ny >= NY:
            blocked.append(1)
        elif is_blocked(nx, ny):
            blocked.append(1)
        else:
            blocked.append(0)

    return torch.tensor(
        [x, y, x_diff, y_diff] + blocked,
        dtype=torch.float32
    )

def make_path(start, goal):
    """Follow the best moves from the Q table until the goal is reached."""
    x, y = start
    gx, gy = goal
    path = [(x, y)]
    for step in range(1 * (NX * NY)):
        if x == gx and y == gy:
            break
        # action = np.argmax(allowed_scores(Q, x, y, gx, gy))
        x_diff = gx - x
        y_diff = gy - y

        # state = torch.tensor([x, y, x_diff, y_diff], dtype=torch.float32)

        with torch.no_grad():
            state =  get_state(x, y, gx, gy)
            q_values = q_model(state)
            action = torch.argmax(q_values).item()
        x = x + MOVES[action][0]
        y = y + MOVES[action][1]
        path.append((x, y))
    return path


def follow_path(viewer, path):
    """Reset the arm to home, go to the start cell, then move cell by cell."""
    data.qpos[qadr] = HOME_Q
    mujoco.mj_forward(model, data)
    update_colors()
    viewer.sync()
    time.sleep(0.5)

    p_now = cell_to_world(path[0][0], path[0][1], TOOL_HEIGHT)
    q, ok = go_to(p_now, HOME_Q)
    if not ok:
        print("Could not reach the start cell with the tool facing down.")
    data.qpos[qadr] = q
    mujoco.mj_forward(model, data)
    update_colors()
    viewer.sync()
    time.sleep(0.5)

    for i in range(1, len(path)):
        p_next = cell_to_world(path[i][0], path[i][1], TOOL_HEIGHT)
        for k in range(1, SUBSTEPS + 1):
            if not viewer.is_running():
                return
            p = p_now + (p_next - p_now) * (k / SUBSTEPS)
            q, ok = solve_ik(p, q)
            data.qpos[qadr] = q
            mujoco.mj_forward(model, data)
            update_colors()
            viewer.sync()
            time.sleep(0.01)
        p_now = p_next
        print(f"  move {i}: cell {path[i]}  tool at {tool_position().round(3)}")


# ---------- replay only when YOU ask ----------
# Click Reset in the viewer (or press Backspace) -> the arm jumps back to its home pose
# and we notice that. Backup: press the R key.
replay_key = [False]


def key_callback(keycode):
    if keycode == ord("R"):
        replay_key[0] = True


def wait_for_replay(viewer):
    """Wait until Reset is clicked (arm back at home) or R is pressed. False = window closed."""
    with viewer.lock():
        ended_at_home = np.allclose(data.qpos[qadr], HOME_Q, atol=1e-6)
    print("Done. Click Reset (or press Backspace, or press R) to replay. Close the window to exit.")
    while viewer.is_running():
        with viewer.lock():
            is_home = np.allclose(data.qpos[qadr], HOME_Q, atol=1e-6)
        if (is_home and not ended_at_home) or replay_key[0]:
            replay_key[0] = False
            return True
        time.sleep(0.05)
    return False


data.qpos[qadr] = HOME_Q
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1     # draw contact points
    time.sleep(1.0)
    run = 1
    while viewer.is_running():
        start, goal = pick_cells()
        path = make_path(start, goal)
        print(f"Run {run}: start {start}, goal {goal}, {len(path) - 1} moves, "
              f"reached goal: {path[-1] == goal}")
        show_markers(start, goal)
        follow_path(viewer, path)
        run = run + 1
        if not wait_for_replay(viewer):
            break