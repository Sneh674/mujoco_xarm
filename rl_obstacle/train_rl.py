"""
train_rl.py  -  STEP 1: train the RL "brain" on the board grid (fixed obstacles).

  - State  = (where the tool is, where the goal is)   -> start/goal can change
  - Action = move one cell: +x, -x, +y, -y
  - Reward = -1 for every move, +100 when the goal cell is reached
  - Obstacles are fixed (they come from scene.xml). Moves into an obstacle or off
    the board are REMOVED from the choices (action masking).
  - Exploration is biased: random moves prefer the direction of the goal.

Run:  python train_rl.py        (saves q_table.npz)
"""

import numpy as np
from grid_utils import NX, NY, MOVES, BLOCKED, is_blocked, allowed_scores

# ---------------- settings ----------------
EPISODES = 50000     # how many practice runs
ALPHA = 0.5          # learning speed
GAMMA = 0.95         # how much future reward counts
EPS_START = 0.5      # chance of a random move at the beginning ...
EPS_END = 0.05       # ... going down to this at the end
GOAL_BIAS = 3.0      # a random move toward the goal is this many times more likely
# ------------------------------------------

MAX_STEPS = 6 * (NX + NY)     # give up on an episode after this many moves

# ---------- show the map so you can check it ----------
print(f"Grid {NX} x {NY} cells, {len(BLOCKED)} cells blocked by obstacles")
print("Map (top = +y, left = -x):   .  free    #  blocked")
for y in range(NY - 1, -1, -1):
    row = ""
    for x in range(NX):
        if (x, y) in BLOCKED:
            row = row + "# "
        else:
            row = row + ". "
    print("   " + row)

# ---------- check that all free cells are connected ----------
free_cells = []
for x in range(NX):
    for y in range(NY):
        if not is_blocked(x, y):
            free_cells.append((x, y))

seen = {free_cells[0]}
todo = [free_cells[0]]
while len(todo) > 0:
    cx, cy = todo.pop()
    for dx, dy in MOVES:
        nxt = (cx + dx, cy + dy)
        if not is_blocked(nxt[0], nxt[1]) and nxt not in seen:
            seen.add(nxt)
            todo.append(nxt)
if len(seen) < len(free_cells):
    print("WARNING: some free cells are cut off by obstacles. Move an obstacle in scene.xml.")

# Q table: Q[x, y, goal_x, goal_y, action]
Q = np.zeros((NX, NY, NX, NY, 4))


def random_free_cell():
    while True:
        x = np.random.randint(NX)
        y = np.random.randint(NY)
        if not is_blocked(x, y):
            return x, y


def take_step(x, y, action):
    """Move one cell. (Blocked moves are never chosen, this is just a safety check.)"""
    new_x = x + MOVES[action][0]
    new_y = y + MOVES[action][1]
    if is_blocked(new_x, new_y):
        return x, y
    return new_x, new_y


def pick_random_action(x, y, gx, gy):
    """Random move that prefers the goal direction and never picks a blocked move."""
    weights = [1.0, 1.0, 1.0, 1.0]            # +x, -x, +y, -y
    if gx > x:
        weights[0] = GOAL_BIAS
    if gx < x:
        weights[1] = GOAL_BIAS
    if gy > y:
        weights[2] = GOAL_BIAS
    if gy < y:
        weights[3] = GOAL_BIAS
    for a in range(4):
        if is_blocked(x + MOVES[a][0], y + MOVES[a][1]):
            weights[a] = 0.0                  # remove the blocked move
    total = sum(weights)
    if total == 0:
        return np.random.randint(4)
    probs = [w / total for w in weights]      # the probability is shared by the remaining moves
    return np.random.choice(4, p=probs)


def test_policy(n_tests=200):
    """Follow the best moves (no randomness) and count how often the goal is reached."""
    wins = 0
    for t in range(n_tests):
        x, y = random_free_cell()
        gx, gy = random_free_cell()
        for step in range(MAX_STEPS):
            if x == gx and y == gy:
                break
            action = np.argmax(allowed_scores(Q, x, y, gx, gy))
            x, y = take_step(x, y, action)
        if x == gx and y == gy:
            wins = wins + 1
    return 100.0 * wins / n_tests


# ---------- training ----------
for episode in range(1, EPISODES + 1):
    x, y = random_free_cell()                 # random start
    gx, gy = random_free_cell()               # random goal
    while gx == x and gy == y:
        gx, gy = random_free_cell()

    epsilon = EPS_START - (EPS_START - EPS_END) * episode / EPISODES

    for step in range(MAX_STEPS):
        # pick an action: sometimes a (goal-biased) random one, otherwise the best known
        if np.random.rand() < epsilon:
            action = pick_random_action(x, y, gx, gy)
        else:
            action = np.argmax(allowed_scores(Q, x, y, gx, gy))

        new_x, new_y = take_step(x, y, action)

        if new_x == gx and new_y == gy:
            reward = 100
            done = True
        else:
            reward = -1
            done = False

        # Q-learning update (only allowed moves are considered for the next state)
        if done:
            best_next = 0
        else:
            best_next = np.max(allowed_scores(Q, new_x, new_y, gx, gy))
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
