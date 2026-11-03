"""
Project 2: Q-Learning Based Maze Solver.
A tabular Q-learning agent learns a route from S to G.
Run: python project2_q_learning/maze_q_learning.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

SEED = 42
rng = np.random.default_rng(SEED)
os.makedirs("results", exist_ok=True)
os.makedirs("trained_models", exist_ok=True)

# 0 = open cell, 1 = wall. Start=(0,0), Goal=(4,4).
MAZE = np.array([
    [0, 0, 0, 1, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
], dtype=int)
ROWS, COLS = MAZE.shape
START, GOAL = (0, 0), (4, 4)
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]

def valid_moves(state):
    r, c = state
    return [a for a, (dr, dc) in enumerate(ACTIONS)
            if 0 <= r+dr < ROWS and 0 <= c+dc < COLS
            and MAZE[r+dr, c+dc] == 0]

def step(state, action):
    r, c = state
    dr, dc = ACTIONS[action]
    nr, nc = r+dr, c+dc
    if not (0 <= nr < ROWS and 0 <= nc < COLS) or MAZE[nr, nc] == 1:
        return state, -5, False
    next_state = (nr, nc)
    if next_state == GOAL:
        return next_state, 100, True
    return next_state, -1, False

Q = np.zeros((ROWS, COLS, len(ACTIONS)), dtype=float)
episodes = 5000
alpha, gamma = 0.15, 0.95
epsilon, epsilon_min, epsilon_decay = 1.0, 0.03, 0.997
episode_rewards, episode_steps = [], []

for episode in range(episodes):
    state = START
    total_reward = 0
    for t in range(100):
        legal = valid_moves(state)
        if rng.random() < epsilon:
            action = int(rng.choice(legal))
        else:
            values = Q[state[0], state[1], legal]
            best = np.flatnonzero(values == values.max())
            action = int(rng.choice(np.array(legal)[best]))
        next_state, reward, done = step(state, action)
        next_legal = valid_moves(next_state)
        old = Q[state[0], state[1], action]
        future = 0 if done else (np.max(Q[next_state[0], next_state[1], next_legal])
                                 if next_legal else 0)
        Q[state[0], state[1], action] = old + alpha * (reward + gamma*future - old)
        state = next_state
        total_reward += reward
        if done:
            break
    episode_rewards.append(total_reward)
    episode_steps.append(t+1)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

# Extract greedy route, avoiding loops.
def greedy_route():
    state, route, seen = START, [START], {START}
    for _ in range(ROWS*COLS*2):
        if state == GOAL:
            return route
        legal = valid_moves(state)
        if not legal:
            break
        vals = Q[state[0], state[1], legal]
        # Stable tie-breaking for reproducibility.
        action = legal[int(np.argmax(vals))]
        ns, _, done = step(state, action)
        if ns == state or ns in seen:
            # Pick the best unvisited legal successor.
            choices = sorted(legal, key=lambda a: Q[state[0], state[1], a], reverse=True)
            candidates = []
            for a in choices:
                dr, dc = ACTIONS[a]
                nxt = (state[0]+dr, state[1]+dc)
                if nxt not in seen:
                    candidates.append((a, nxt))
            if not candidates:
                break
            action, ns = candidates[0]
        state = ns
        route.append(state)
        seen.add(state)
        if done:
            return route
    return route

route = greedy_route()
np.save("trained_models/maze_q_table.npy", Q)
with open("results/maze_route.txt", "w") as f:
    f.write("Greedy route from start to goal:\\n")
    f.write(" -> ".join(map(str, route)) + "\\n")
    f.write(f"Reached goal: {route[-1] == GOAL}\\n")
    f.write(f"Number of moves: {len(route)-1}\\n")

# Save training curve and maze route image.
window = 100
smooth = np.convolve(episode_rewards, np.ones(window)/window, mode="valid")
plt.figure(figsize=(8, 5))
plt.plot(episode_rewards, alpha=0.2, label="Episode reward")
plt.plot(np.arange(window-1, episodes), smooth, label="100-episode average")
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.title("Q-Learning Training Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/maze_learning_curve.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 6))
plt.imshow(MAZE, cmap="Greys", vmin=0, vmax=1)
for i, (r, c) in enumerate(route):
    plt.text(c, r, "S" if (r,c)==START else ("G" if (r,c)==GOAL else str(i)),
             ha="center", va="center", color="red", fontsize=9)
plt.xticks(range(COLS)); plt.yticks(range(ROWS))
plt.title("Learned Maze Route (S=start, G=goal)")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/maze_learned_route.png", dpi=150)
plt.close()

print("Reached goal:", route[-1] == GOAL)
print("Route:", route)
print("Moves:", len(route)-1)
print("Saved Q-table to trained_models/maze_q_table.npy")
