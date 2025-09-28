# training/train_dqn.py

import os
import yaml
import argparse
import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import imageio

from asteroids_env.env import AsteroidsEnv
from training.dqn_model import DQN
from training.replay_buffer import ReplayBuffer
from utils.preprocess import preprocess_frame, stack_frames

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Config Loading ---
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config/train_config.yaml")
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

# --- Hyperparameters ---
num_frames = config["num_frames"]
input_shape = tuple(config["input_shape"])
num_episodes = config["num_episodes"]
max_steps_per_episode = config["max_steps_per_episode"]

gamma = config["gamma"]

epsilon = config["epsilon"]["start"]
epsilon_decay = config["epsilon"]["decay"]
epsilon_min = config["epsilon"]["min"]

batch_size = config["batch_size"]
learning_rate = config["learning_rate"]

replay_capacity = config["replay_capacity"]
target_update_interval = config["target_update_interval"]
save_interval = config["save_interval"]

# --- Directories ---
os.makedirs("models", exist_ok=True)
os.makedirs("gifs", exist_ok=True)

# --- Environment ---
env = AsteroidsEnv(render_mode="rgb_array", width=512, height=512)
n_actions = env.action_space.n

# --- Networks ---
policy_net = DQN(input_shape, n_actions).to(device)
print("Processing device: ", next(policy_net.parameters()).device) # Check GPU
target_net = DQN(input_shape, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
replay_buffer = ReplayBuffer(replay_capacity)

# --- Reward tracking ---
episode_rewards = []

# --- Frame stack ---
stacked_frames = deque(maxlen=num_frames)

# --- Evaluation function ---
def evaluate_policy(env, policy_net, num_frames=5, max_steps=1000):
    stacked_frames = deque(maxlen=num_frames)
    obs, _ = env.reset()
    state, stacked_frames = stack_frames(stacked_frames, obs, True, num_frames)
    done = False
    total_reward = 0
    steps = 0
    frames = []

    while not done and steps < max_steps:
        state_tensor = torch.tensor(np.array([state]), dtype=torch.float32).to(device)
        action = policy_net(state_tensor).argmax(dim=1).item()
        obs, reward, done, truncated, info = env.step(action)
        state, stacked_frames = stack_frames(stacked_frames, obs, False, num_frames)
        total_reward += reward
        steps += 1
        frames.append(obs)

    return total_reward, steps, frames

# --- GIF saving function ---
def save_gif(frames, filename="play.gif"):
    frames_rgb = [np.array(frame) for frame in frames]
    imageio.mimsave(filename, frames_rgb, fps=30)

# --- Training loop ---
for episode in range(1, num_episodes + 1):
    obs, _ = env.reset()
    state, stacked_frames = stack_frames(stacked_frames, obs, True, num_frames)
    done = False
    total_reward = 0
    step_count = 0

    while not done and step_count < max_steps_per_episode:
        # --- Select action ---
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.tensor(np.array([state]), dtype=torch.float32).to(device)
            action = policy_net(state_tensor).argmax(dim=1).item()

        # --- Step environment ---
        next_obs, reward, done, truncated, info = env.step(action)
        next_state, stacked_frames = stack_frames(stacked_frames, next_obs, False, num_frames)
        total_reward += reward

        # --- Store in replay buffer ---
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state

        # --- Update DQN ---
        if len(replay_buffer) > batch_size:
            s, a, r, ns, d = replay_buffer.sample(batch_size)
            s = torch.tensor(s, dtype=torch.float32).to(device)
            a = torch.tensor(a, dtype=torch.int64).unsqueeze(1).to(device)
            r = torch.tensor(r, dtype=torch.float32).unsqueeze(1).to(device)
            ns = torch.tensor(ns, dtype=torch.float32).to(device)
            d = torch.tensor(d, dtype=torch.float32).unsqueeze(1).to(device)

            q_values = policy_net(s).gather(1, a)
            next_q = target_net(ns).max(1, keepdim=True)[0].detach()
            target = r + gamma * next_q * (1 - d)

            loss = F.mse_loss(q_values, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        step_count += 1

    # --- Update epsilon ---
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    episode_rewards.append(total_reward)

    # --- Update target network ---
    if episode % target_update_interval == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # --- Save model and GIF periodically ---
    if episode % save_interval == 0:
        model_path = f"models/policy_net_{episode}.pth"
        torch.save(policy_net.state_dict(), model_path)
        score, steps_played, frames = evaluate_policy(env, policy_net)
        gif_path = f"gifs/play_episode_{episode}.gif"
        save_gif(frames, gif_path)
        del frames
        print(f"Episode {episode}: reward={total_reward:.2f}, eval_score={score}, gif saved at {gif_path}")
    else:
        print(f"Episode {episode}: reward={total_reward:.2f}")

# --- Plot reward graph ---
plt.figure()
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN Training Progress")
plt.savefig("training_scores.png")
plt.show()
