# training/train_dqn.py

import os
import yaml
import argparse
import random
import json
from collections import deque
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import imageio

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv

from asteroids_env.env import AsteroidsEnv
from training.dqn_model import DQN
from training.replay_buffer import PrioritizedReplayBuffer
from evaluation.evaluate_policy import evaluate_policy, save_gif, sample_frames
from utils.preprocess import preprocess_frame, stack_frames
from utils.model_io import save_model, load_model

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Config Loading ---
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config/train_config.yaml")
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

# --- Hyperparameters ---
init_model_path = config.get("init_model_path", "")  # optional init model

# Load from checkpoint if available
if init_model_path and os.path.isfile(init_model_path):
    print(f"Loading initial model from {init_model_path}")
    policy_net, checkpoint_config, n_actions = load_model(init_model_path, device)
    print("Loaded config:")
    print(yaml.dump(checkpoint_config, sort_keys=False, default_flow_style=False))
    num_frames = checkpoint_config.get("num_frames", 5)
    channels_per_frame = checkpoint_config.get("channels_per_frame", 3)
    input_shape = tuple([num_frames * channels_per_frame, *checkpoint_config["input_shape_2d"]])
else:
    print("Starting training from scratch.")
    num_frames = config.get("num_frames", 5)
    channels_per_frame = config.get("channels_per_frame", 3)
    input_shape = tuple([num_frames * channels_per_frame, *config["input_shape_2d"]])
    n_actions = config.get("n_actions", 5)
    policy_net = DQN(input_shape, n_actions).to(device)

max_frames = config.get("max_frames", 1000000)
max_steps_per_episode = config.get("max_steps_per_episode", 1000)

gamma = config.get("gamma", 0.99)

batch_size = config.get("batch_size", 32)
learning_rate = config.get("learning_rate", 0.0005)

replay_capacity = config.get("replay_capacity", 2_000)
target_update_interval = config.get("target_update_interval", 10_000)
save_interval = config.get("save_interval", 50_000)
log_interval = config.get("log_interval", 10_000)
frame_skip = config.get("frame_skip", 1)

epsilon_start = config.get("epsilon_start", 1.0)
epsilon_final = config.get("epsilon_final", 0.05)
epsilon_decay = config.get("epsilon_decay", 1_000_000)

num_asteroids = config.get("num_asteroids", 5)
max_asteroid_size = config.get("max_asteroid_size", 90)
max_asteroid_speed_start = config.get("max_asteroid_speed_start", 0.5)
max_asteroid_speed_end = config.get("max_asteroid_speed_end", 2.5)
max_asteroid_speed = max_asteroid_speed_start

alpha = config.get("alpha", 0.6)
beta_start = config.get("beta_start", 0.4)

num_envs = config.get("num_envs", 16)
num_envs = min(num_envs, os.cpu_count() or 1)
print(f"Using {num_envs} environments/cores for training.")

eval_episodes = config.get("eval_episodes", 10)  # number of episodes during eval

# --- Directories ---
os.makedirs("models", exist_ok=True)
os.makedirs("gifs", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# --- Environment ---
current_max_speed = [max_asteroid_speed]  # mutable container for shared reference

def make_env():
    def _init():
        return AsteroidsEnv(
            render_mode="rgb_array",
            width=400,
            height=400,
            max_steps=max_steps_per_episode,
            num_asteroids=num_asteroids,
            max_asteroid_size=max_asteroid_size,
            max_asteroid_speed=current_max_speed[0],
            frame_skip=frame_skip
        )
    return _init
env = AsyncVectorEnv([make_env() for _ in range(num_envs)])
n_actions = env.single_action_space.n

# --- Build target network ---
print("Processing device: ", next(policy_net.parameters()).device)  # Check GPU
target_net = DQN(input_shape, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

replay_buffer = PrioritizedReplayBuffer(replay_capacity, alpha)

# --- Frame-level bookkeeping ---
frame_idx = 0
episode_rewards = []
eval_scores = []

# --- Per-env trackers ---
obs, _ = env.reset()
stacked_frames = [deque(maxlen=num_frames) for _ in range(num_envs)]
states = []
env_returns = np.zeros(num_envs, dtype=np.float32)
completed_returns = []  # stores rewards of finished episodes

for i in range(num_envs):
    s, stacked_frames[i] = stack_frames(stacked_frames[i], obs[i], True, num_frames, (input_shape[1], input_shape[2]))
    states.append(s)
states = np.stack(states)

# --- Metrics logging ---
logging_tds = []
logging_sorted_q = np.zeros(n_actions, dtype=np.float64)
logging_q_count = 0
logging_losses = []
logging_action_counts = np.zeros(n_actions)

# --- Episode tracking ---
completed_lengths = []  # lengths of completed episodes
episode_steps = np.zeros(num_envs, dtype=np.int32)

# --- JSON logging ---
log_history = []

# --- Main training loop (frame-based) ---
print(f"Starting training for {max_frames:,} total frames across {num_envs} envs")

while frame_idx < max_frames:
    # --- Choose actions for all envs ---
    state_tensor = torch.tensor(states, dtype=torch.float32).to(device)
    state_tensor = state_tensor.reshape(num_envs, -1, state_tensor.shape[-2], state_tensor.shape[-1])

    # --- Epsilon-greedy action selection ---
    epsilon = epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1.0 * frame_idx / epsilon_decay)

    with torch.no_grad():
        q_values = policy_net(state_tensor)
        greedy_actions = q_values.argmax(dim=1).cpu().numpy()

    random_actions = np.random.randint(0, n_actions, size=num_envs)
    should_explore = np.random.rand(num_envs) < epsilon
    actions = np.where(should_explore, random_actions, greedy_actions)

    logging_action_counts[greedy_actions] += 1

    # --- Step all environments in parallel, with skips ---
    total_rewards = np.zeros(num_envs)
    dones_any = np.zeros(num_envs, dtype=bool)
    next_obs = states.copy()  # placeholder for last obs after skipping frames

    for skip in range(frame_skip):
        # --- Step all environments in parallel ---
        obs_step, rewards_step, dones_step, truncs_step, infos_step = env.step(actions)
        total_rewards += rewards_step
        dones_any |= dones_step
        next_obs = obs_step  # overwrite with last observation

        # --- Reset done environments mid-skip ---
        if dones_step.any():
            reset_obs, _ = env.reset()
            for i, done in enumerate(dones_step):
                if done:
                    next_obs[i] = reset_obs[i]

    # --- Preprocess next states and push to replay buffer ---
    next_states = []
    for i in range(num_envs):
        ns, stacked_frames[i] = stack_frames(
            stacked_frames[i], next_obs[i], dones_any[i], num_frames, (input_shape[1], input_shape[2])
        )
        replay_buffer.push(states[i], actions[i], total_rewards[i], ns, dones_any[i])
        # --- Update per-episode stats ---
        env_returns[i] += total_rewards[i]
        episode_steps[i] += frame_skip
        if dones_any[i]:
            completed_returns.append(env_returns[i])
            completed_lengths.append(episode_steps[i])
            env_returns[i] = 0.0
            episode_steps[i] = 0

        next_states.append(ns)

    states = np.stack(next_states)  # convert to array at the end
    frame_idx += (frame_skip * num_envs)  # we advanced frame_skip * num_envs frames at once

    # --- Train the DQN ---
    if len(replay_buffer.buffer) > batch_size:
        beta = min(1.0, beta_start + frame_idx * (1.0 - beta_start) / max_frames)
        batch, indices, weights = replay_buffer.sample(batch_size, beta)
        s, a, r, ns, d = batch

        s = torch.tensor(np.array(s), dtype=torch.float32).to(device).view(batch_size, *input_shape)
        ns = torch.tensor(np.array(ns), dtype=torch.float32).to(device).view(batch_size, *input_shape)
        a = torch.tensor(a, dtype=torch.int64).unsqueeze(1).to(device)
        r = torch.tensor(r, dtype=torch.float32).unsqueeze(1).to(device)
        d = torch.tensor(d, dtype=torch.float32).unsqueeze(1).to(device)
        weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(device)

        q_all = policy_net(s).detach().cpu().numpy()
        q_values = policy_net(s).gather(1, a)
        next_actions = policy_net(ns).argmax(1, keepdim=True)
        next_q = target_net(ns).gather(1, next_actions).detach()
        target = r + gamma * next_q * (1 - d)
        td_errors = (q_values - target).detach().cpu().numpy().squeeze()
        loss = (weights * F.smooth_l1_loss(q_values, target, reduction="none")).mean()

        logging_tds.extend(np.abs(td_errors).tolist())
        sorted_qs = np.sort(q_all, axis=1)[:, ::-1]  # shape: (batch_size, n_actions), largest first
        logging_sorted_q += sorted_qs.sum(axis=0)
        logging_q_count += sorted_qs.shape[0]
        logging_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        replay_buffer.update_priorities(indices, np.abs(td_errors) + 1e-6)

    # --- Log metrics ---
    if frame_idx % log_interval < (num_envs * frame_skip) and logging_q_count > 0:
        # --- Metrics ---
        action_probs = logging_action_counts / np.sum(logging_action_counts + 1e-10)
        action_entropy = -np.sum(action_probs * np.log(action_probs + 1e-10))

        mean_td = float(np.mean(logging_tds)) if logging_tds else 0.0
        std_td = float(np.std(logging_tds)) if logging_tds else 0.0
        mean_q = logging_sorted_q / logging_q_count if logging_q_count > 0 else 0.0
        mean_loss = float(np.mean(logging_losses)) if logging_losses else 0.0

        # Episode stats over last 100 episodes or fewer
        recent_rewards = completed_returns[-100:]
        recent_lengths = completed_lengths[-100:]
        avg_recent_reward = float(np.mean(recent_rewards)) if recent_rewards else 0.0
        avg_recent_length = float(np.mean(recent_lengths)) if recent_lengths else 0.0

        # --- Print nicely ---
        print(f"Frame {frame_idx:,} | Loss={mean_loss:.4f} | Q means sorted={mean_q} | "
            f"TD={mean_td:.3f}±{std_td:.3f} | Entropy={action_entropy:.3f} | "
            f"Recent reward={avg_recent_reward:.2f} | Avg ep length={avg_recent_length:.1f}")
        print(f"Curriculum: Current max asteroid speed={current_max_speed[0]:.2f}")

        # --- JSON row ---
        log_row = {
            "frame": frame_idx,
            "mean_loss": mean_loss,
            "mean_q": float(np.mean(mean_q)),
            "td_mean": mean_td,
            "td_std": std_td,
            "action_entropy": action_entropy,
            "avg_recent_reward": avg_recent_reward,
            "avg_recent_length": avg_recent_length,
            "curriculum_max_asteroid_speed": current_max_speed[0],
            "completed_episodes": len(completed_returns),
            "epsilon": epsilon,
        }
        log_history.append(log_row)
        # Dump incremental JSON
        with open("logs/training_results.json", "w") as f:
            json.dump(log_history, f, indent=2)

        # --- Reset logging ---
        logging_tds = []
        logging_sorted_q[:] = 0
        logging_q_count = 0
        logging_losses = []
        logging_action_counts = np.zeros(n_actions)

    # --- Update target network ---
    if frame_idx % target_update_interval < (num_envs * frame_skip):
        target_net.load_state_dict(policy_net.state_dict())

    # --- Update curriculum ---
    if max_asteroid_speed_end > max_asteroid_speed_start:
        progress = min(1.0, frame_idx / max_frames)
        current_max_speed[0] = max_asteroid_speed_start + progress * (max_asteroid_speed_end - max_asteroid_speed_start)
        env.call("set_difficulty", max_asteroid_speed=current_max_speed[0])

    # --- Save model and run evaluation ---
    if frame_idx % save_interval < (num_envs * frame_skip):
        model_path = f"models/policy_net_{frame_idx}.pth"
        save_model(policy_net, config, n_actions, model_path)
        avg_score, best_score, frames, _ = evaluate_policy(
            AsteroidsEnv(
                render_mode="rgb_array",
                width=400,
                height=400,
                max_steps=max_steps_per_episode,
                num_asteroids=num_asteroids,
                max_asteroid_size=max_asteroid_size,
                max_asteroid_speed=current_max_speed[0],
                frame_skip=1
            ),
            policy_net,
            input_shape,
            device,
            num_frames,
            max_steps_per_episode,
            num_eval_episodes=eval_episodes
        )
        gif_path = f"gifs/play_{frame_idx}.gif"
        save_gif(sample_frames(frames, n=500), gif_path)
        eval_scores.append(avg_score)
        print(f"Saved model + eval (avg={avg_score:.2f}, best={best_score:.2f}) → {gif_path}")

# --- Wrap-up ---
env.close()
results = {
    "completed_returns": completed_returns,
    "eval_scores": eval_scores,
    "config": config,
}

def json_compatible(obj):
    if isinstance(obj, np.generic):  # np.float32, np.int64, etc.
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

with open("logs/training_summary.json", "w") as f:
    json.dump(results, f, indent=2, default=json_compatible)

plt.figure()
plt.plot(completed_returns, label="Episode rewards")
if eval_scores:
    plt.plot(np.linspace(0, len(completed_returns), len(eval_scores)), eval_scores, label="Eval avg scores")
plt.xlabel("Episodes (approx)")
plt.ylabel("Reward")
plt.legend()
plt.title("Vectorized DQN Training Progress")
plt.savefig("logs/training_scores.png")
plt.show()
