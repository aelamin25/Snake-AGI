import random
import multiprocessing as mp
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


# =========================================================
# 0. Device (no print here to avoid spam from workers)
# =========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# 1. Snake Environment (single env, 3 channels + history)
# =========================================================

class SnakeEnv:
    """
    Snake as an RL environment.

    - Grid world (grid_size x grid_size).
    - Observation: stacked 3-channel grids over time:
        ch0: head
        ch1: body (excluding head)
        ch2: food
      with history_len frames → shape (3 * history_len, H, W) flattened.

    - Actions: 0=up, 1=right, 2=down, 3=left

    - Rewards (single scalar objective: maximise cumulative reward):
        +1  when food is eaten  -> length increases
        -1  when snake dies     -> hits wall or itself, episode ends
        small negative step penalty each move (encourages efficient paths)

    - Score we *interpret* = len(self.snake), starting at 1.
    """

    def __init__(self, grid_size: int = 15, history_len: int = 4, seed: Optional[int] = None):
        self.grid_size = grid_size
        self.history_len = history_len
        self.rng = random.Random(seed if seed is not None else random.randint(0, 10_000_000))
        self.reset()

    def reset(self) -> np.ndarray:
        self.direction = 1  # start going right
        cx = self.grid_size // 2
        cy = self.grid_size // 2
        self.snake = [(cx, cy)]  # single head
        self._place_food()
        self.done = False

        base = self._get_base_grid()  # (3, H, W)
        self.history = [base.copy() for _ in range(self.history_len)]
        return self._get_obs()

    def _place_food(self):
        empty_cells = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if (x, y) not in self.snake
        ]
        self.food = self.rng.choice(empty_cells)

    def _get_base_grid(self) -> np.ndarray:
        H = W = self.grid_size
        grid = np.zeros((3, H, W), dtype=np.float32)

        # head
        head_x, head_y = self.snake[-1]
        grid[0, head_y, head_x] = 1.0

        # body (excluding head)
        for (x, y) in self.snake[:-1]:
            grid[1, y, x] = 1.0

        # food
        fx, fy = self.food
        grid[2, fy, fx] = 1.0

        return grid

    def _update_history(self):
        base = self._get_base_grid()
        self.history.pop(0)
        self.history.append(base)

    def _get_obs(self) -> np.ndarray:
        stacked = np.concatenate(self.history, axis=0)  # (3*history_len, H, W)
        return stacked.reshape(-1)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        if self.done:
            raise RuntimeError("Call reset() before step() after episode ends.")

        # Action directly sets direction: no extra steering rules
        self.direction = action

        head_x, head_y = self.snake[-1]
        dx, dy = 0, 0
        if self.direction == 0:   # up
            dy = -1
        elif self.direction == 1: # right
            dx = 1
        elif self.direction == 2: # down
            dy = 1
        elif self.direction == 3: # left
            dx = -1

        new_head = (head_x + dx, head_y + dy)

        # 1) Wall collision
        x, y = new_head
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            self.done = True
            reward = -1.0
            return self._get_obs(), reward, self.done, {"reason": "wall"}

        # 2) Will we grow?
        will_grow = (new_head == self.food)

        # 3) Self-collision:
        #    - If grow: full body collidable (tail stays).
        #    - Else: body except tail collidable (tail moves).
        if will_grow:
            collidable_body = set(self.snake)
        else:
            collidable_body = set(self.snake[1:])

        if new_head in collidable_body:
            self.done = True
            reward = -1.0
            return self._get_obs(), reward, self.done, {"reason": "self"}

        # 4) Move snake
        self.snake.append(new_head)
        STEP_PENALTY = -0.01  # Time/hunger cost: discourages aimless wandering

        if will_grow:
            reward = 1.0 + STEP_PENALTY
            self._place_food()
        else:
            reward = STEP_PENALTY
            self.snake.pop(0)

        self._update_history()
        obs = self._get_obs()
        done = self.done
        return obs, reward, done, {}

    @property
    def length(self) -> int:
        # Current body length (starts at 1)
        return len(self.snake)


# =========================================================
# 2. Parallel Environment (multi-CPU via multiprocessing)
# =========================================================

def _worker(remote, parent_remote, grid_size: int, history_len: int):
    """
    Worker process:

    - Creates its own SnakeEnv inside the child process.
    - Listens for 'reset', 'step', 'close'.
    - Auto-resets env after 'done' so rollouts are continuous.
    """
    parent_remote.close()
    env = SnakeEnv(grid_size=grid_size, history_len=history_len, seed=None)

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                ob = env.reset()
                remote.send(ob)
            elif cmd == "step":
                action = int(data)
                ob, reward, done, info = env.step(action)
                if done:
                    ob = env.reset()
                remote.send((ob, reward, done, info))
            elif cmd == "close":
                remote.close()
                break
            else:
                raise NotImplementedError(f"Unknown command {cmd}")
    finally:
        env = None


class ParallelSnakeEnv:
    """
    Vectorised environment using multiple processes.

    - Each worker owns its own SnakeEnv.
    - reset() returns obs for all envs.
    - step(actions) steps all envs in parallel and auto-resets
      those that are done.
    """

    def __init__(self, n_envs: int, grid_size: int, history_len: int):
        self.n_envs = n_envs
        self.grid_size = grid_size
        self.history_len = history_len

        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(n_envs)])
        self.ps = [
            mp.Process(target=_worker, args=(work_remote, remote, grid_size, history_len))
            for (work_remote, remote) in zip(self.work_remotes, self.remotes)
        ]
        for p in self.ps:
            p.daemon = True
            p.start()
        for work_remote in self.work_remotes:
            work_remote.close()

    def reset(self) -> np.ndarray:
        for remote in self.remotes:
            remote.send(("reset", None))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs, axis=0)  # (n_envs, obs_dim)

    def step(self, actions: np.ndarray):
        for remote, act in zip(self.remotes, actions):
            remote.send(("step", int(act)))
        results = [remote.recv() for _ in self.remotes]
        obs, rewards, dones, infos = zip(*results)
        return (
            np.stack(obs, axis=0),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            infos,
        )

    def close(self):
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()


# =========================================================
# 3. PPO Agent (batch / parallel-friendly)
# =========================================================

class ConvActorCritic(nn.Module):
    def __init__(self, grid_size: int, in_channels: int, n_actions: int = 4):
        super().__init__()
        self.grid_size = grid_size
        self.in_channels = in_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        conv_out_dim = 64 * grid_size * grid_size

        self.fc = nn.Sequential(
            nn.Linear(conv_out_dim, 256),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(256, n_actions)
        self.value_head = nn.Linear(256, 1)

    def forward(self, x_flat: torch.Tensor):
        b, hwc = x_flat.shape
        side = self.grid_size
        c = self.in_channels
        x = x_flat.view(b, c, side, side)
        h = self.conv(x)
        h = h.view(b, -1)
        h = self.fc(h)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value


class PPOAgent:
    def __init__(
        self,
        grid_size: int,
        in_channels: int,
        n_actions: int = 4,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        minibatch_size: int = 1024,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.device = device

        self.net = ConvActorCritic(grid_size, in_channels, n_actions).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    @torch.no_grad()
    def act(self, obs_batch: np.ndarray):
        """
        obs_batch: (N, obs_dim)
        Returns: actions (N,), log_probs (N,), values (N,)
        """
        x = torch.from_numpy(obs_batch).float().to(self.device)
        logits, values = self.net(x)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return (
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            values.cpu().numpy(),
        )

    def update(self, batch):
        obs = torch.from_numpy(batch["obs"]).float().to(self.device)           # (T*N, obs_dim)
        actions = torch.from_numpy(batch["actions"]).long().to(self.device)    # (T*N,)
        old_log_probs = torch.from_numpy(batch["log_probs"]).float().to(self.device)
        returns = torch.from_numpy(batch["returns"]).float().to(self.device)
        advantages = torch.from_numpy(batch["advantages"]).float().to(self.device)

        # Normalise advantages
        if advantages.std() > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        num_samples = obs.size(0)
        indices = np.arange(num_samples)

        total_loss = 0.0
        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, num_samples, self.minibatch_size):
                end = start + self.minibatch_size
                batch_idx = indices[start:end]

                logits, values = self.net(obs[batch_idx])
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs=probs)

                log_probs_new = dist.log_prob(actions[batch_idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_probs_new - old_log_probs[batch_idx])
                adv = advantages[batch_idx]

                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(values, returns[batch_idx])

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += float(loss.item())

        denom = max(1, (self.ppo_epochs * ((num_samples - 1) // self.minibatch_size + 1)))
        return total_loss / denom


# =========================================================
# 4. GAE for parallel rollouts
# =========================================================

def compute_gae_parallel(
    rewards: np.ndarray,  # (T, N)
    values: np.ndarray,   # (T+1, N)
    dones: np.ndarray,    # (T, N)
    gamma: float,
    gae_lambda: float,
):
    T, N = rewards.shape
    advantages = np.zeros((T, N), dtype=np.float32)
    gae = np.zeros(N, dtype=np.float32)

    for t in reversed(range(T)):
        mask = 1.0 - dones[t].astype(np.float32)  # 1 if not done, 0 if done
        delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[t] = gae

    returns = advantages + values[:-1]
    return advantages, returns


# =========================================================
# 5. Multi-CPU PPO Training Loop (with episode memory)
# =========================================================

def train_snake_ppo_parallel(
    total_timesteps: int = 4_000_000,
    n_envs: int = 6,
    rollout_steps: int = 256,
    grid_size: int = 15,
    history_len: int = 4,
):
    """
    Parallel PPO training.

    - Collects T=rollout_steps steps from N=n_envs envs in parallel.
    - That’s T*N transitions per update.
    - Repeats until total_timesteps is reached.

    Reward structure comes entirely from the environment:
    +1 for food (minus a small step penalty), -1 for death, and step penalty otherwise.

    We also store complete episodes in `episode_memory` for later replay.
    """

    vec_env = ParallelSnakeEnv(n_envs, grid_size=grid_size, history_len=history_len)
    obs_dim = 3 * history_len * grid_size * grid_size
    in_channels = 3 * history_len

    agent = PPOAgent(
        grid_size=grid_size,
        in_channels=in_channels,
        n_actions=4,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        ppo_epochs=4,
        minibatch_size=2048,
        device=DEVICE,
    )

    initial_entropy_coef = agent.entropy_coef

    obs = vec_env.reset()  # (N, obs_dim)

    # For logging episode returns/lengths
    ep_returns = np.zeros(n_envs, dtype=np.float32)
    ep_lengths = np.zeros(n_envs, dtype=np.int32)
    completed_returns = []
    completed_lengths = []

    # Episode memory: a list of dicts
    # Each dict: {"obs": (T, obs_dim), "actions": (T,), "rewards": (T,), "dones": (T,)}
    episode_memory: List[dict] = []

    # Per-env buffers for building episodes
    current_obs_traj = [[] for _ in range(n_envs)]
    current_act_traj = [[] for _ in range(n_envs)]
    current_rew_traj = [[] for _ in range(n_envs)]
    current_done_traj = [[] for _ in range(n_envs)]

    timesteps_done = 0
    update_idx = 0

    while timesteps_done < total_timesteps:
        update_idx += 1

        # Rollout buffers
        obs_buf = np.zeros((rollout_steps, n_envs, obs_dim), dtype=np.float32)
        actions_buf = np.zeros((rollout_steps, n_envs), dtype=np.int64)
        logprobs_buf = np.zeros((rollout_steps, n_envs), dtype=np.float32)
        rewards_buf = np.zeros((rollout_steps, n_envs), dtype=np.float32)
        dones_buf = np.zeros((rollout_steps, n_envs), dtype=bool)
        values_buf = np.zeros((rollout_steps + 1, n_envs), dtype=np.float32)

        # Entropy schedule (explore → exploit)
        frac = min(1.0, timesteps_done / total_timesteps)
        agent.entropy_coef = initial_entropy_coef * max(0.3, 1.0 - frac)

        # Collect rollout
        for t in range(rollout_steps):
            obs_buf[t] = obs

            # --- Episode memory: store obs for each env at this step ---
            for i in range(n_envs):
                current_obs_traj[i].append(obs[i].copy())

            actions, log_probs, values = agent.act(obs)
            actions_buf[t] = actions
            logprobs_buf[t] = log_probs
            values_buf[t] = values

            next_obs, rewards, dones, infos = vec_env.step(actions)

            rewards_buf[t] = rewards
            dones_buf[t] = dones

            # --- Episode memory: store action, reward, done ---
            for i in range(n_envs):
                current_act_traj[i].append(int(actions[i]))
                current_rew_traj[i].append(float(rewards[i]))
                current_done_traj[i].append(bool(dones[i]))

                if dones[i]:
                    # Finalise this episode
                    episode_memory.append({
                        "obs": np.stack(current_obs_traj[i], axis=0),
                        "actions": np.array(current_act_traj[i], dtype=np.int64),
                        "rewards": np.array(current_rew_traj[i], dtype=np.float32),
                        "dones": np.array(current_done_traj[i], dtype=bool),
                    })
                    # Reset per-env buffers
                    current_obs_traj[i] = []
                    current_act_traj[i] = []
                    current_rew_traj[i] = []
                    current_done_traj[i] = []

            # Episode bookkeeping for logging
            ep_returns += rewards
            ep_lengths += 1
            for i in range(n_envs):
                if dones[i]:
                    completed_returns.append(ep_returns[i])
                    completed_lengths.append(ep_lengths[i])
                    ep_returns[i] = 0.0
                    ep_lengths[i] = 0

            obs = next_obs
            timesteps_done += n_envs

        # Bootstrap values for last obs
        _, _, last_values = agent.act(obs)
        values_buf[-1] = last_values

        # GAE & returns
        advantages, returns = compute_gae_parallel(
            rewards_buf, values_buf, dones_buf, agent.gamma, agent.gae_lambda
        )

        # Flatten rollout: (T*N, ...)
        T, N = rollout_steps, n_envs
        batch = {
            "obs": obs_buf.reshape(T * N, obs_dim),
            "actions": actions_buf.reshape(T * N),
            "log_probs": logprobs_buf.reshape(T * N),
            "returns": returns.reshape(T * N),
            "advantages": advantages.reshape(T * N),
        }

        loss = agent.update(batch)

        if update_idx % 10 == 0:
            if completed_returns:
                recent_returns = np.array(completed_returns[-100:], dtype=np.float32)
                recent_steps = np.array(completed_lengths[-100:], dtype=np.float32)

                avg_ret = float(recent_returns.mean())           # reward
                avg_steps = float(recent_steps.mean())           # survival time (steps)
                avg_foods = float((recent_returns + 1).mean())   # foods ≈ return + 1
                avg_body_len = float((recent_returns + 2).mean())  # final length ≈ return + 2
            else:
                avg_ret = avg_steps = avg_foods = avg_body_len = 0.0

            print(
                f"Update {update_idx:4d} | steps so far {timesteps_done:7d} | "
                f"avg_return(100)={avg_ret:6.2f} | "
                f"avg_steps(100)={avg_steps:6.2f} | "
                f"avg_foods(100)={avg_foods:5.2f} | "
                f"avg_body_len(100)={avg_body_len:5.2f} | "
                f"loss={loss:.4f} | entropy_coef={agent.entropy_coef:.4f}"
            )

    vec_env.close()
    print(f"Training finished. Episodes recorded in memory: {len(episode_memory)}")
    return agent, episode_memory


# =========================================================
# 6. Replay any stored episode from memory
# =========================================================

def replay_episode(
    episode_memory: List[dict],
    idx: int,
    grid_size: int,
    history_len: int,
    pause: float = 0.15,
):
    """
    Replay a stored episode by index.

    - idx can be 0, 1, 2, ... or negative (-1 = last episode).
    - Uses the stored observations, actions, rewards.
    - Reconstructs the grid from obs (takes the last 3 channels: head/body/food).
    """

    if len(episode_memory) == 0:
        print("No episodes in memory to replay.")
        return

    if idx < 0:
        idx = len(episode_memory) + idx
    if idx < 0 or idx >= len(episode_memory):
        print(f"Episode index {idx} out of range (0..{len(episode_memory)-1}).")
        return

    ep = episode_memory[idx]
    obs_traj = ep["obs"]         # (T, obs_dim)
    rewards = ep["rewards"]      # (T,)
    actions = ep["actions"]      # (T,)
    T = obs_traj.shape[0]

    plt.ion()
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    cmap = ListedColormap(["black", "green", "blue", "yellow"])  # empty, body, head, food

    cum_reward = 0.0
    foods_eaten = 0

    for t in range(T):
        obs_flat = obs_traj[t]
        stacked = obs_flat.reshape(3 * history_len, grid_size, grid_size)
        base = stacked[-3:]  # last frame in history: (3, H, W)

        # Decode base into discrete grid: 0 empty, 1 body, 2 head, 3 food
        head_layer = base[0]
        body_layer = base[1]
        food_layer = base[2]

        grid = np.zeros((grid_size, grid_size), dtype=np.int32)
        grid[body_layer > 0.5] = 1
        grid[head_layer > 0.5] = 2
        grid[food_layer > 0.5] = 3

        r_t = float(rewards[t])
        cum_reward += r_t
        if r_t > 0:
            foods_eaten += 1
        approx_length = 1 + foods_eaten  # 1 start + foods eaten so far

        ax.clear()
        ax.imshow(grid, vmin=0, vmax=3, origin="lower", cmap=cmap, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"Replay ep {idx} | step {t+1}/{T} | a={actions[t]} | "
            f"len~{approx_length} | r_t={r_t:.1f}, R={cum_reward:.1f}",
            fontsize=10,
        )
        plt.pause(pause)

    plt.ioff()
    plt.show()
    print(
        f"[replay] Episode {idx} finished. "
        f"Steps={T}, total_reward={cum_reward:.1f}, foods_eaten={foods_eaten}, "
        f"approx_final_length={1+foods_eaten}"
    )


# =========================================================
# 7. Main
# =========================================================

if __name__ == "__main__":
    # Required for multiprocessing on macOS/Windows
    mp.set_start_method("spawn", force=True)

    print(f"Using device: {DEVICE}")

    # Multi-CPU PPO training on a 15x15 grid
    agent, episode_memory = train_snake_ppo_parallel(
        total_timesteps=4_000_000,
        n_envs=6,          # good for an 8-core machine
        rollout_steps=256,
        grid_size=15,      # 15x15 grid
        history_len=4,
    )

    # Automatically replay the last episode after training:
    replay_episode(episode_memory, idx=-1, grid_size=15, history_len=4)
