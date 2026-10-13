"""
grid_utils.py  -  shared helpers for train_rl.py and play_arm.py

It reads scene.xml and builds the grid:
  - the board is cut into cells
  - every cell touched by an "obstacle..." box is marked BLOCKED
Both scripts import this file, so they always agree on the grid.
"""

import numpy as np
import mujoco

# ---------------- settings ----------------
MODEL_PATH = "scene.xml"
CELL = 0.05        # wanted cell size in meters (5 cm)
MARGIN = 0.01      # extra safety space around obstacles (m)
# ------------------------------------------

# 4 moves: (change in x cell, change in y cell)  ->  +x, -x, +y, -y
MOVES = [(1, 0), (-1, 0), (0, 1), (0, -1)]

model = mujoco.MjModel.from_xml_path(MODEL_PATH)

# ---------- board and grid ----------
board_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "board")
board_pos = model.geom_pos[board_id].copy()
half_x, half_y, half_z = model.geom_size[board_id]
board_top = board_pos[2] + half_z

NX = int(2 * half_x / CELL)        # number of cells along x
NY = int(2 * half_y / CELL)        # number of cells along y
cell_w = 2 * half_x / NX           # real cell size along x
cell_h = 2 * half_y / NY           # real cell size along y


def cell_to_world(ix, iy, height):
    """Centre of grid cell (ix, iy) in world coordinates."""
    x = board_pos[0] - half_x + (ix + 0.5) * cell_w
    y = board_pos[1] - half_y + (iy + 0.5) * cell_h
    return np.array([x, y, board_top + height])


# ---------- obstacles -> blocked cells ----------
def find_blocked_cells():
    blocked = set()
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        if name is None or not name.startswith("obstacle"):
            continue
        ox = model.geom_pos[g][0]       # obstacle centre
        oy = model.geom_pos[g][1]
        hx = model.geom_size[g][0]      # obstacle half sizes
        hy = model.geom_size[g][1]
        for ix in range(NX):
            for iy in range(NY):
                c = cell_to_world(ix, iy, 0)
                # blocked if the cell overlaps the obstacle (plus a small margin)
                if abs(c[0] - ox) < hx + cell_w / 2 + MARGIN and abs(c[1] - oy) < hy + cell_h / 2 + MARGIN:
                    blocked.add((ix, iy))
    return blocked


BLOCKED = find_blocked_cells()


def is_blocked(x, y):
    """True if (x, y) is off the board or inside an obstacle cell."""
    if x < 0 or x >= NX or y < 0 or y >= NY:
        return True
    return (x, y) in BLOCKED


def allowed_scores(Q, x, y, gx, gy):
    """The 4 Q scores for this situation, with blocked moves set to a very low number."""
    scores = Q[x, y, gx, gy].copy()
    for a in range(4):
        if is_blocked(x + MOVES[a][0], y + MOVES[a][1]):
            scores[a] = -1e9
    return scores
