import numpy as np
import mujoco

model = mujoco.MjModel.from_xml_path("scene.xml")
ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link6")
joint_low = model.jnt_range[:6, 0]
joint_high = model.jnt_range[:6, 1]

X_MIN, Y_MIN, CELL, N = 0.2, -0.2, 0.05, 8
Z = 0.15
TOOL_AXIS = np.array([0.0, 0.0, 1.0])
TOOL_OFFSET = np.array([0.0, 0.0, 0.05])
DOWN = np.array([0.0, 0.0, -1.0])

def try_ik(target, q_start, iters=1000):
    d = mujoco.MjData(model)
    d.qpos[:6] = q_start
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    for _ in range(iters):
        mujoco.mj_forward(model, d)
        R = d.xmat[ee_id].reshape(3, 3)
        tip = d.xpos[ee_id] + R @ TOOL_OFFSET
        axis = R @ TOOL_AXIS
        pos_err = target - tip
        rot_err = np.cross(axis, DOWN)
        if np.linalg.norm(pos_err) < 0.001 and np.linalg.norm(rot_err) < 0.01:
            break
        mujoco.mj_jac(model, d, jacp, jacr, tip, ee_id)
        J = np.vstack([jacp[:, :6], jacr[:, :6]])
        err = np.concatenate([pos_err, rot_err])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
        d.qpos[:6] = np.clip(d.qpos[:6] + 0.5 * dq, joint_low, joint_high)
    mujoco.mj_forward(model, d)
    R = d.xmat[ee_id].reshape(3, 3)
    tip = d.xpos[ee_id] + R @ TOOL_OFFSET
    return np.linalg.norm(target - tip), d.qpos[:6].copy()

starts = [
    np.array([0.0, -0.5, 0.5, 0.0, 1.0, 0.0]),
    np.array([0.0, 0.3, 0.5, 0.0, 0.5, 0.0]),
    np.array([0.0, 0.5, 1.0, 0.0, 0.5, 0.0]),
]

for j in range(N - 1, -1, -1):          # print top row first
    row = ""
    for i in range(N):
        x = X_MIN + (i + 0.5) * CELL
        y = Y_MIN + (j + 0.5) * CELL
        best = 1.0
        for s in starts:
            e, q = try_ik(np.array([x, y, Z]), s)
            best = min(best, e)
        row += " ok " if best < 0.005 else " X  "
    print(row)