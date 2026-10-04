import numpy as np
import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

# ------------------------------------------------------------
# SETTINGS (change these)
# ------------------------------------------------------------
# board / grid, in meters, in the MuJoCo world frame
X_MIN, Y_MIN = 0.2, -0.2      # corner of the board
CELL = 0.05                   # cell size
N = 8                         # N x N cells

CARRY_Z = 0.15                # height of the tool tip above the floor

START_CELL = (0, 0)           # path start (column i, row j)
END_CELL = (7, 7)             # path end

# tool definition, in the local frame of link6
BODY_NAME = "link6"
TOOL_AXIS = np.array([0.0, 0.0, 1.0])     # direction the suction cup points
TOOL_OFFSET = np.array([0.0, 0.0, 0.05])  # distance from the flange to the cup tip (m)

DOWN = np.array([0.0, 0.0, -1.0])         # where we want the tool axis to point

# ------------------------------------------------------------
# GRID <-> WORLD
# ------------------------------------------------------------
def cell_to_xy(i, j):
    x = X_MIN + (i + 0.5) * CELL
    y = Y_MIN + (j + 0.5) * CELL
    return x, y

def make_path(start, end):
    """List of cells on a straight line from start to end (plain loop)."""
    si, sj = start
    ei, ej = end
    steps = max(abs(ei - si), abs(ej - sj))
    path = []
    for k in range(steps + 1):
        if steps == 0:
            t = 0
        else:
            t = k / steps
        i = int(round(si + (ei - si) * t))
        j = int(round(sj + (ej - sj) * t))
        path.append((i, j))
    return path

# ------------------------------------------------------------
# ARM HELPERS
# ------------------------------------------------------------
ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BODY_NAME)
joint_low = model.jnt_range[:6, 0]
joint_high = model.jnt_range[:6, 1]

def tool_state(d):
    """Return the tool tip position and the direction the tool points (world frame)."""
    R = d.xmat[ee_id].reshape(3, 3)
    tip = d.xpos[ee_id] + R @ TOOL_OFFSET
    axis = R @ TOOL_AXIS
    return tip, axis

def tilt_degrees(axis):
    c = np.clip(np.dot(axis, DOWN), -1.0, 1.0)
    return np.degrees(np.arccos(c))

def solve_ik(target, q_start):
    """
    Find 6 joint angles so that:
      1) the tool tip is at 'target' (x, y, z)
      2) the tool axis points straight down
    Rotation around the tool axis is left free (fine for a suction cup).
    """
    d = mujoco.MjData(model)          # scratch copy, the real sim is untouched
    d.qpos[:6] = q_start
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    for _ in range(300):
        mujoco.mj_forward(model, d)
        tip, axis = tool_state(d)

        pos_err = target - tip
        rot_err = np.cross(axis, DOWN)   # zero when axis already points down

        if np.linalg.norm(pos_err) < 0.001 and np.linalg.norm(rot_err) < 0.01:
            break

        mujoco.mj_jac(model, d, jacp, jacr, tip, ee_id)
        J = np.vstack([jacp[:, :6], jacr[:, :6]])    # 6 x 6
        err = np.concatenate([pos_err, rot_err])     # 6

        # damped least squares step
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
        d.qpos[:6] = np.clip(d.qpos[:6] + 0.5 * dq, joint_low, joint_high)

    mujoco.mj_forward(model, d)
    tip, axis = tool_state(d)
    pos_error = np.linalg.norm(target - tip)
    return d.qpos[:6].copy(), pos_error, tilt_degrees(axis)

# ------------------------------------------------------------
# PLACE THE ARM AT THE START CELL
# ------------------------------------------------------------
home_q = np.array([0.0, -0.5, 0.5, 0.0, 1.0, 0.0])
sx, sy = cell_to_xy(*START_CELL)
q0, err0, tilt0 = solve_ik(np.array([sx, sy, CARRY_Z]), home_q)
print("start pose: position error %.4f m, tilt %.1f deg" % (err0, tilt0))

data.qpos[:6] = q0
data.ctrl[:6] = q0
mujoco.mj_forward(model, data)

# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:

    def run(steps):
        for _ in range(steps):
            mujoco.mj_step(model, data)
            viewer.sync()

    def move_to_xy(x, y, z=CARRY_Z, waypoints=8):
        time.sleep(1)
        """Move the tool tip in a straight line to (x, y, z), keeping it pointing down."""
        start, _ = tool_state(data)
        goal = np.array([x, y, z])
        q = data.qpos[:6].copy()
        for k in range(1, waypoints + 1):
            point = start + (goal - start) * k / waypoints
            q, err, tilt = solve_ik(point, q)
            if err > 0.01 or tilt > 5.0:
                print("IK failed at", np.round(point, 3),
                      "error %.3f m, tilt %.1f deg" % (err, tilt))
                return False
            data.ctrl[:6] = q
            run(60)
        return True

    run(500)   # let the arm settle at the start pose

    path = make_path(START_CELL, END_CELL)
    print("path:", path)

    for (i, j) in path:
        x, y = cell_to_xy(i, j)
        ok = move_to_xy(x, y)
        tip, axis = tool_state(data)
        print("cell", (i, j),
              "target", (round(x, 3), round(y, 3)),
              "actual", np.round(tip[:2], 3),
              "tilt %.1f deg" % tilt_degrees(axis),
              "ok" if ok else "FAILED")
        if not ok:
            break

    print("done")
    while viewer.is_running():
        run(10)