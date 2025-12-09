# Snake AGI 🐍🧠

Snake AGI is a reinforcement learning playground where a Snake agent starts with **no built-in rules** other than:

> **Maximise cumulative reward.**

Everything else is discovered from experience.

The environment is a 2D grid world (default **15×15**) with one snake and one food item.  
The agent controls the snake with four actions (up, right, down, left), and learns a policy using **PPO** (Proximal Policy Optimization) with a convolutional actor–critic network and **parallel environments** via `multiprocessing`.

---

## Key ideas

- **No hand-crafted rules** like "avoid walls" or "go towards food".
- The only learning signal is the **reward function**:
  - `+1` when food is eaten (snake length increases), minus a small step penalty.
  - `-1` when the snake dies (hits a wall or its own body).
  - `-0.01` per step to discourage wandering forever without finding food.
- Observations are **image-like**:
  - 3 channels per frame: head, body, food.
  - Stacked over a short history (default `history_len = 4`) to give a sense of motion.
- **PPO** with:
  - CNN encoder → shared MLP → policy & value heads.
  - Advantage normalisation, clipping, entropy bonus.
- **Parallel training**:
  - Multiple Snake environments run in separate processes.
  - Collect rollouts across all envs, then perform PPO updates.
- **Episode memory & replay**:
  - Every finished episode is stored.
  - You can visually replay any episode after training.

---

## Project structure

- `snake_agi.py`  
  Main implementation:
  - `SnakeEnv` – environment.
  - `ParallelSnakeEnv` – multi-process vectorised env wrapper.
  - `ConvActorCritic` – CNN-based actor–critic network.
  - `PPOAgent` – PPO algorithm.
  - `train_snake_ppo_parallel` – training loop.
  - `replay_episode` – visualise a stored episode.
  - `__main__` – runs training on a 15×15 grid and replays the last episode.

- `requirements.txt`  
  Python dependencies.

- `.gitignore`  
  Standard Python ignores.

- `LICENSE`  
  MIT license.

---

## Installation

```bash
git clone https://github.com/<your-username>/snake-agi.git
cd snake-agi

# (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
