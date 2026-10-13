"""
train_rl.py  -  STEP 1: train the RL "brain" on the board grid (no obstacles yet).

Idea:
  - The board is cut into a grid of cells.
  - State  = (where the tool is, where the goal is)   -> so start/goal can change
  - Action = move one cell: +x, -x, +y, -y
  - Reward = -1 for every move, +100 when the goal cell is reached
  - Q-learning fills a table: Q[state, action] = how good that move is.

This script only trains (no arm, no viewer). It saves q_table.npz.
Run:  python train_rl.py
"""

import numpy as np
import mujoco

# ---------------- settings ----------------
MODEL_PATH = "rl_scene.xml"
CELL = 0.05          # wanted cell size in meters (5 cm)
EPISODES = 40000     # how many practice runs
ALPHA = 0.5          # learning speed
GAMMA = 0.95         # how much future reward counts
EPSILON = 0.2        # chance of a random move (exploration)
# ------------------------------------------

# 4 moves: (change in x cell, change in y cell)
MOVES = [(1, 0), (-1, 0), (0, 1), (0, -1)]

# --- read the board size from scene.xml and make the grid ---
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
board_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "board")
half_x = model.geom_size[board_id][0]
half_y = model.geom_size[board_id][1]

NX = int(2 * half_x / CELL)    # number of cells along x
NY = int(2 * half_y / CELL)    # number of cells along y
print(f"Board {2 * half_x:.2f} x {2 * half_y:.2f} m  ->  grid {NX} x {NY} cells")

MAX_STEPS = 4 * (NX + NY)      # give up on an episode after this many moves

# Q table: Q[x, y, goal_x, goal_y, action]
Q = np.zeros((NX, NY, NX, NY, 4))


def random_cell():
    return np.random.randint(NX), np.random.randint(NY)


def take_step(x, y, action):
    """Move one cell. Walking off the board = stay where you are."""
    dx, dy = MOVES[action]
    new_x = x + dx
    new_y = y + dy
    if new_x < 0 or new_x >= NX or new_y < 0 or new_y >= NY:
        new_x = x
        new_y = y
    return new_x, new_y


def test_policy(n_tests=200):
    """Run the learned policy (no randomness) and count how often it reaches the goal."""
    wins = 0
    for t in range(n_tests):
        x, y = random_cell()
        gx, gy = random_cell()
        for step in range(MAX_STEPS):
            if x == gx and y == gy:
                break
            action = np.argmax(Q[x, y, gx, gy])
            x, y = take_step(x, y, action)
        if x == gx and y == gy:
            wins = wins + 1
    return 100.0 * wins / n_tests


# --- training ---
for episode in range(1, EPISODES + 1):
    x, y = random_cell()          # random start
    gx, gy = random_cell()        # random goal
    while gx == x and gy == y:
        gx, gy = random_cell()

    for step in range(MAX_STEPS):
        # pick an action: sometimes random, otherwise the best known
        if np.random.rand() < EPSILON:
            action = np.random.randint(4)
        else:
            action = np.argmax(Q[x, y, gx, gy])

        new_x, new_y = take_step(x, y, action)

        if new_x == gx and new_y == gy:
            reward = 100
            done = True
        else:
            reward = -1
            done = False

        # Q-learning update
        if done:
            best_next = 0
        else:
            best_next = np.max(Q[new_x, new_y, gx, gy])
        old = Q[x, y, gx, gy, action]
        Q[x, y, gx, gy, action] = old + ALPHA * (reward + GAMMA * best_next - old)

        x = new_x
        y = new_y
        if done:
            break

    if episode % 5000 == 0:
        print(f"episode {episode}: reaches the goal in {test_policy():.0f}% of random tests")

np.savez("q_table.npz", Q=Q)
print("Saved q_table.npz")