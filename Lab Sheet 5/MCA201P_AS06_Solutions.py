
# MCA201P Lab Sheet-05: Reinforcement Learning
# Complete reference solutions for Programs 1-35
# Python 3.11+
#
# Install:
#   pip install -r requirements.txt
#
# Notes:
# - Programs 1-20 use Gymnasium + Q-Learning.
# - Programs 21-35 use PyTorch DQN with CartPole.
# - Run individual sections rather than the whole file at once if desired.

import os
import time
import random
from collections import deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. Install/configure Gymnasium
# ============================================================
# Terminal command:
# pip install gymnasium numpy pandas matplotlib scikit-learn torch
#
# Configuration test:
import gymnasium as gym

print("Gymnasium version:", gym.__version__)


# ============================================================
# 2. Create and execute a simple RL environment
# ============================================================
env = gym.make("FrozenLake-v1", is_slippery=False)
observation, info = env.reset(seed=42)
print("Initial observation:", observation)
print("Info:", info)
env.close()


# ============================================================
# 3. Explore observation space and action space
# ============================================================
env = gym.make("FrozenLake-v1", is_slippery=False)
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
print("Number of states:", env.observation_space.n)
print("Number of actions:", env.action_space.n)
env.close()


# ============================================================
# 4. Display states, actions, rewards and termination
# ============================================================
env = gym.make("FrozenLake-v1", is_slippery=False)
state, info = env.reset(seed=42)

for step in range(10):
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    print(
        f"Step={step+1}, State={state}, Action={action}, "
        f"Next State={next_state}, Reward={reward}, "
        f"Terminated={terminated}, Truncated={truncated}"
    )
    state = next_state
    if terminated or truncated:
        break
env.close()


# ============================================================
# 5. Simulate random actions in FrozenLake
# ============================================================
env = gym.make("FrozenLake-v1", is_slippery=False)
state, info = env.reset(seed=42)
total_reward = 0

for step in range(20):
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break

print("Random-policy total reward:", total_reward)
env.close()


# ============================================================
# 6-9. Q-Learning, Q-table initialization/update/training
# ============================================================
def train_q_learning(
    env_name="FrozenLake-v1",
    episodes=5000,
    alpha=0.8,
    gamma=0.95,
    epsilon=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.999,
    slippery=False,
    seed=42,
):
    env = gym.make(env_name, is_slippery=slippery)
    rng = np.random.default_rng(seed)

    q_table = np.zeros(
        (env.observation_space.n, env.action_space.n),
        dtype=np.float64
    )
    rewards = []

    for episode in range(episodes):
        state, info = env.reset(seed=seed + episode)
        total_reward = 0.0

        while True:
            # Epsilon-greedy action selection
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[state]))

            next_state, reward, terminated, truncated, info = env.step(action)

            # Q-Learning update
            best_next_q = np.max(q_table[next_state])
            target = reward if terminated else reward + gamma * best_next_q
            q_table[state, action] += alpha * (
                target - q_table[state, action]
            )

            state = next_state
            total_reward += reward

            if terminated or truncated:
                break

        rewards.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    env.close()
    return q_table, np.array(rewards)


q_table, rewards = train_q_learning()
print("\nLearned Q-table:")
print(np.round(q_table, 3))


# ============================================================
# 10. Evaluate trained Q-Learning agent
# ============================================================
def evaluate_q_learning(q_table, episodes=100, slippery=False):
    env = gym.make("FrozenLake-v1", is_slippery=slippery)
    scores = []

    for episode in range(episodes):
        state, info = env.reset(seed=1000 + episode)
        total_reward = 0

        while True:
            action = int(np.argmax(q_table[state]))
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        scores.append(total_reward)

    env.close()
    return np.mean(scores), np.sum(scores)

avg_reward, total_reward = evaluate_q_learning(q_table)
print("Average evaluation reward:", avg_reward)
print("Total evaluation reward:", total_reward)


# ============================================================
# 11. Plot cumulative rewards during training
# ============================================================
cumulative_rewards = np.cumsum(rewards)

plt.figure()
plt.plot(cumulative_rewards)
plt.title("Cumulative Rewards During Q-Learning")
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.grid(True)
plt.show()


# ============================================================
# 12. Effect of different learning rates
# ============================================================
learning_rates = [0.1, 0.5, 0.8, 1.0]
lr_results = {}

for lr in learning_rates:
    _, r = train_q_learning(alpha=lr, episodes=3000)
    lr_results[lr] = np.mean(r[-500:])

print("\nLearning-rate comparison:")
print(pd.Series(lr_results, name="Average Last 500 Rewards"))

plt.figure()
plt.bar([str(x) for x in learning_rates], list(lr_results.values()))
plt.title("Effect of Learning Rate")
plt.xlabel("Learning Rate (Alpha)")
plt.ylabel("Average Reward")
plt.show()


# ============================================================
# 13. Effect of different discount factors (Gamma)
# ============================================================
gammas = [0.5, 0.7, 0.9, 0.99]
gamma_results = {}

for g in gammas:
    _, r = train_q_learning(gamma=g, episodes=3000)
    gamma_results[g] = np.mean(r[-500:])

print("\nGamma comparison:")
print(pd.Series(gamma_results, name="Average Last 500 Rewards"))

plt.figure()
plt.bar([str(x) for x in gammas], list(gamma_results.values()))
plt.title("Effect of Discount Factor")
plt.xlabel("Gamma")
plt.ylabel("Average Reward")
plt.show()


# ============================================================
# 14. Compare exploration/exploitation using epsilon values
# ============================================================
epsilons = [0.1, 0.3, 0.5, 0.9]
epsilon_results = {}

for eps in epsilons:
    _, r = train_q_learning(epsilon=eps, episodes=3000,
                             epsilon_min=eps, epsilon_decay=1.0)
    epsilon_results[eps] = np.mean(r[-500:])

print("\nInitial/fixed epsilon comparison:")
print(pd.Series(epsilon_results, name="Average Last 500 Rewards"))

plt.figure()
plt.bar([str(x) for x in epsilons], list(epsilon_results.values()))
plt.title("Exploration vs Exploitation")
plt.xlabel("Epsilon")
plt.ylabel("Average Reward")
plt.show()


# ============================================================
# 15. Epsilon-greedy action selection policy
# ============================================================
def epsilon_greedy_action(q_values, epsilon, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    if rng.random() < epsilon:
        return int(rng.integers(len(q_values)))
    return int(np.argmax(q_values))


example_q_values = np.array([0.1, 0.8, 0.2, 0.5])
print("\nEpsilon-greedy example:")
print("Q-values:", example_q_values)
print("Selected action:", epsilon_greedy_action(example_q_values, 0.2))


# ============================================================
# 16. Design a simple Grid World environment
# ============================================================
class SimpleGridWorld:
    """
    Grid:
    S = Start
    G = Goal
    X = Obstacle

    Actions:
    0 = Up
    1 = Right
    2 = Down
    3 = Left
    """

    def __init__(self):
        self.rows = 4
        self.cols = 4
        self.start = (0, 0)
        self.goal = (3, 3)
        self.obstacles = {(1, 1), (2, 1)}
        self.state = self.start

    @property
    def n_states(self):
        return self.rows * self.cols

    @property
    def n_actions(self):
        return 4

    def state_to_index(self, state):
        return state[0] * self.cols + state[1]

    def reset(self):
        self.state = self.start
        return self.state_to_index(self.state)

    def step(self, action):
        row, col = self.state
        moves = {
            0: (-1, 0),
            1: (0, 1),
            2: (1, 0),
            3: (0, -1),
        }

        dr, dc = moves[action]
        new_state = (row + dr, col + dc)

        if (
            new_state[0] < 0 or new_state[0] >= self.rows or
            new_state[1] < 0 or new_state[1] >= self.cols or
            new_state in self.obstacles
        ):
            new_state = self.state

        self.state = new_state

        if self.state == self.goal:
            return self.state_to_index(self.state), 10, True
        return self.state_to_index(self.state), -1, False

    def render(self):
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                pos = (r, c)
                if pos == self.state:
                    row.append("A")
                elif pos == self.start:
                    row.append("S")
                elif pos == self.goal:
                    row.append("G")
                elif pos in self.obstacles:
                    row.append("X")
                else:
                    row.append(".")
            print(" ".join(row))
        print()


grid = SimpleGridWorld()
grid.render()


# ============================================================
# 17. Train Q-Learning agent in custom Grid World
# ============================================================
def train_grid_world(episodes=3000, alpha=0.8, gamma=0.95,
                     epsilon=1.0, epsilon_min=0.05,
                     epsilon_decay=0.995):
    env = SimpleGridWorld()
    q = np.zeros((env.n_states, env.n_actions))
    rng = np.random.default_rng(42)
    episode_rewards = []

    for ep in range(episodes):
        state = env.reset()
        total = 0

        for _ in range(100):
            action = epsilon_greedy_action(q[state], epsilon, rng)
            next_state, reward, done = env.step(action)

            target = reward if done else reward + gamma * np.max(q[next_state])
            q[state, action] += alpha * (target - q[state, action])

            state = next_state
            total += reward

            if done:
                break

        episode_rewards.append(total)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return env, q, np.array(episode_rewards)


grid_env, grid_q, grid_rewards = train_grid_world()
print("Grid World Q-table:")
print(np.round(grid_q, 2))


# ============================================================
# 18. Visualize optimal path learned by agent
# ============================================================
def get_optimal_path(env, q_table, max_steps=50):
    state = env.reset()
    path = [env.state]

    for _ in range(max_steps):
        action = int(np.argmax(q_table[state]))
        next_state, reward, done = env.step(action)
        path.append(env.state)
        state = next_state

        if done:
            break

    return path


path = get_optimal_path(grid_env, grid_q)
print("Optimal path:", path)

# Display path on grid
print("\nOptimal path visualization:")
for r in range(grid_env.rows):
    row = []
    for c in range(grid_env.cols):
        pos = (r, c)
        if pos == grid_env.goal:
            row.append("G")
        elif pos == grid_env.start:
            row.append("S")
        elif pos in grid_env.obstacles:
            row.append("X")
        elif pos in path:
            row.append("*")
        else:
            row.append(".")
    print(" ".join(row))


# ============================================================
# 19. Compare FrozenLake and Grid World
# ============================================================
fl_avg, _ = evaluate_q_learning(q_table, episodes=100)

grid_scores = []
for _ in range(100):
    state = grid_env.reset()
    total = 0
    for _ in range(50):
        action = int(np.argmax(grid_q[state]))
        state, reward, done = grid_env.step(action)
        total += reward
        if done:
            break
    grid_scores.append(total)

comparison = pd.DataFrame({
    "Environment": ["FrozenLake", "Custom Grid World"],
    "Average Reward": [fl_avg, np.mean(grid_scores)]
})
print("\nEnvironment comparison:")
print(comparison)


# ============================================================
# 20. Analyze convergence behavior
# ============================================================
window = 100
moving_average = pd.Series(rewards).rolling(window).mean()

plt.figure()
plt.plot(moving_average)
plt.title("Q-Learning Convergence")
plt.xlabel("Episode")
plt.ylabel(f"{window}-Episode Moving Average Reward")
plt.grid(True)
plt.show()

print(
    "Convergence observation: the moving average becomes more stable "
    "as the Q-table approaches a useful policy."
)


# ============================================================
# 21. Install required libraries for DQN
# ============================================================
# Terminal:
# pip install torch gymnasium numpy matplotlib
#
# Verify PyTorch:
import torch
print("\nPyTorch version:", torch.__version__)


# ============================================================
# 22-23. Basic DQN + training on CartPole
# ============================================================
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


def train_dqn(
    episodes=300,
    batch_size=64,
    gamma=0.99,
    learning_rate=1e-3,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.995,
    target_update_frequency=10,
):
    env = gym.make("CartPole-v1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    policy_net = DQN(state_size, action_size).to(device)
    target_net = DQN(state_size, action_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    memory = ReplayBuffer(20000)

    epsilon = epsilon_start
    rewards_history = []

    def choose_action(state):
        nonlocal epsilon
        if random.random() < epsilon:
            return env.action_space.sample()

        with torch.no_grad():
            state_tensor = torch.tensor(
                state, dtype=torch.float32, device=device
            ).unsqueeze(0)
            return int(policy_net(state_tensor).argmax(dim=1).item())

    for episode in range(episodes):
        state, info = env.reset(seed=episode)
        total_reward = 0

        for _ in range(500):
            action = choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(memory) >= batch_size:
                states, actions, rewards_b, next_states, dones = memory.sample(
                    batch_size
                )

                states_t = torch.tensor(states, device=device)
                actions_t = torch.tensor(actions, device=device).unsqueeze(1)
                rewards_t = torch.tensor(rewards_b, device=device).unsqueeze(1)
                next_states_t = torch.tensor(next_states, device=device)
                dones_t = torch.tensor(dones, device=device).unsqueeze(1)

                current_q = policy_net(states_t).gather(1, actions_t)

                with torch.no_grad():
                    max_next_q = target_net(next_states_t).max(
                        dim=1, keepdim=True
                    )[0]
                    target_q = rewards_t + gamma * max_next_q * (1 - dones_t)

                loss = nn.MSELoss()(current_q, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        rewards_history.append(total_reward)

        if (episode + 1) % target_update_frequency == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if (episode + 1) % 25 == 0:
            print(
                f"DQN Episode {episode+1}/{episodes}, "
                f"Reward={total_reward:.0f}, Epsilon={epsilon:.3f}"
            )

    env.close()

    return policy_net, target_net, np.array(rewards_history)


# Train the DQN.
policy_net, target_net, dqn_rewards = train_dqn(episodes=300)


# ============================================================
# 24. Plot episode-wise DQN rewards
# ============================================================
plt.figure()
plt.plot(dqn_rewards)
plt.title("DQN Episode-wise Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.show()


# ============================================================
# 25. Evaluate trained DQN
# ============================================================
def evaluate_dqn(model, episodes=20):
    env = gym.make("CartPole-v1")
    device = next(model.parameters()).device
    scores = []

    model.eval()

    for episode in range(episodes):
        state, info = env.reset(seed=2000 + episode)
        total = 0

        for _ in range(500):
            state_tensor = torch.tensor(
                state, dtype=torch.float32, device=device
            ).unsqueeze(0)

            with torch.no_grad():
                action = int(model(state_tensor).argmax(dim=1).item())

            state, reward, terminated, truncated, info = env.step(action)
            total += reward

            if terminated or truncated:
                break

        scores.append(total)

    env.close()
    return np.mean(scores), scores


dqn_avg, dqn_scores = evaluate_dqn(policy_net)
print("DQN average evaluation reward:", dqn_avg)


# ============================================================
# 26. Compare Q-Learning and DQN
# ============================================================
print("\nAlgorithm comparison:")
print("Q-Learning is table-based and works well for small discrete state spaces.")
print("DQN approximates Q-values with a neural network and handles continuous states.")


# ============================================================
# 27. Effect of replay memory on DQN performance
# ============================================================
print("\nReplay memory analysis:")
print(
    "Replay memory stores previous transitions and samples random mini-batches. "
    "This breaks strong temporal correlation and generally stabilizes DQN training."
)


# ============================================================
# 28. Role of target network in DQN
# ============================================================
print("\nTarget network analysis:")
print(
    "The target network provides relatively stable target Q-values. "
    "It is updated periodically from the policy network, reducing oscillation "
    "and instability during learning."
)


# ============================================================
# 29. Save trained DQN model
# ============================================================
MODEL_PATH = "cartpole_dqn.pth"
torch.save(policy_net.state_dict(), MODEL_PATH)
print("Saved DQN model to:", MODEL_PATH)


# ============================================================
# 30. Load saved DQN model and test
# ============================================================
loaded_model = DQN(4, 2)
loaded_model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
loaded_model.eval()

loaded_avg, _ = evaluate_dqn(loaded_model, episodes=10)
print("Loaded model average reward:", loaded_avg)


# ============================================================
# 31. Compare cumulative rewards for RL algorithms
# ============================================================
q_cumulative = np.cumsum(rewards)
dqn_cumulative = np.cumsum(dqn_rewards)

plt.figure()
plt.plot(q_cumulative, label="Q-Learning - FrozenLake")
plt.plot(dqn_cumulative, label="DQN - CartPole")
plt.title("Cumulative Rewards Comparison")
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# 32. Visualize learning curve
# ============================================================
def moving_average(values, window=20):
    return pd.Series(values).rolling(window).mean().to_numpy()


plt.figure()
plt.plot(moving_average(dqn_rewards, 20))
plt.title("DQN Learning Curve")
plt.xlabel("Episode")
plt.ylabel("20-Episode Moving Average Reward")
plt.grid(True)
plt.show()


# ============================================================
# 33. Compare training time and convergence
# ============================================================
start = time.perf_counter()
_, q_time_rewards = train_q_learning(episodes=2000)
q_time = time.perf_counter() - start

start = time.perf_counter()
_, _, dqn_time_rewards = train_dqn(episodes=100)
dqn_time = time.perf_counter() - start

timing = pd.DataFrame({
    "Algorithm": ["Q-Learning", "DQN"],
    "Episodes": [2000, 100],
    "Training Time (seconds)": [q_time, dqn_time],
    "Mean Final Rewards": [
        np.mean(q_time_rewards[-200:]),
        np.mean(dqn_time_rewards[-20:])
    ]
})
print("\nTraining time comparison:")
print(timing)


# ============================================================
# 34. Impact of hyperparameters
# ============================================================
print("\nHyperparameter impact:")
hyperparameter_summary = pd.DataFrame({
    "Hyperparameter": [
        "Learning Rate (alpha)",
        "Discount Factor (gamma)",
        "Exploration (epsilon)",
        "Batch Size",
        "Replay Memory",
        "Target Update Frequency"
    ],
    "Impact": [
        "Controls how strongly new information changes Q-values.",
        "Controls importance of future rewards.",
        "Balances exploration and exploitation.",
        "Controls number of transitions used per DQN update.",
        "Improves stability by decorrelating training samples.",
        "Controls how often DQN target values are refreshed."
    ]
})
print(hyperparameter_summary.to_string(index=False))


# ============================================================
# 35. Comparative report
# ============================================================
report = """
REINFORCEMENT LEARNING COMPARATIVE REPORT

Objective:
To understand RL concepts and implement/evaluate Q-Learning and DQN.

Algorithms:
1. Q-Learning
   - Uses a Q-table.
   - Suitable for small discrete state spaces.
   - Simple and computationally inexpensive.

2. Deep Q-Network (DQN)
   - Uses a neural network to approximate Q-values.
   - Uses replay memory and a target network.
   - Suitable for larger/continuous observation spaces such as CartPole.

Performance Analysis:
- Q-Learning converges efficiently on small discrete environments such as
  deterministic FrozenLake.
- DQN needs more computation and training but can process continuous
  observations such as CartPole.
- Learning rate affects the speed and stability of Q-value updates.
- Gamma controls the importance of future rewards.
- Epsilon controls the exploration/exploitation trade-off.
- Replay memory and a target network improve DQN stability.

Observation:
The learning curve generally improves as the agent receives more useful
experience and updates its policy.

Conclusion:
Q-Learning is appropriate for small tabular RL problems, while DQN is more
appropriate when the state representation is too large or continuous for a
practical Q-table.
"""
print(report)

# End of Lab Sheet-05 solutions.
