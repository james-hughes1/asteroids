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

from asteroids_env.env import AsteroidsEnv
from training.dqn_model import DQN
from training.replay_buffer import ReplayBuffer
from evaluation.evaluate_policy import evaluate_policy, save_gif
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

num_episodes = config.get("num_episodes", 1000)
max_steps_per_episode = config.get("max_steps_per_episode", 1000)

gamma = config.get("gamma", 0.99)

epsilon = config["epsilon"].get("start", 1.0)
epsilon_decay = config["epsilon"].get("decay", 0.995)
epsilon_min = config["epsilon"].get("min", 0.1)

batch_size = config.get("batch_size", 32)
learning_rate = config.get("learning_rate", 0.0005)

replay_capacity = config.get("replay_capacity", 2000)
target_update_interval = config.get("target_update_interval", 10)
save_interval = config.get("save_interval", 50)

num_asteroids = config.get("num_asteroids", 5)
max_asteroid_size = config.get("max_asteroid_size", 90)
max_asteroid_speed = config.get("max_asteroid_speed", 0.5)

eval_episodes = config.get("eval_episodes", 10)  # number of episodes during eval

# --- Directories ---
os.makedirs("models", exist_ok=True)
os.makedirs("gifs", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# --- Environment ---
env = AsteroidsEnv(render_mode="rgb_array", width=400, height=400, max_steps=max_steps_per_episode, num_asteroids=num_asteroids, max_asteroid_size=max_asteroid_size, max_asteroid_speed=max_asteroid_speed)
n_actions = env.action_space.n

# --- Build target network ---
print("Processing device: ", next(policy_net.parameters()).device)  # Check GPU
target_net = DQN(input_shape, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
replay_buffer = ReplayBuffer(replay_capacity)

# --- Reward tracking ---
episode_rewards = []
eval_scores = []

# --- Frame stack ---
stacked_frames = deque(maxlen=num_frames)

# --- Training loop ---
for episode in range(1, num_episodes + 1):
    obs, _ = env.reset()
    state, stacked_frames = stack_frames(
        stacked_frames, obs, True, num_frames, (input_shape[1], input_shape[2])
    )
    done = False
    total_reward = 0
    step_count = 0

    while not done and step_count < max_steps_per_episode:
        # --- Select action ---
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.tensor(np.array([state]), dtype=torch.float32).to(device)
            state_tensor = state_tensor.view(1, *input_shape)
            action = policy_net(state_tensor).argmax(dim=1).item()

        # --- Step environment ---
        next_obs, reward, done, truncated, info = env.step(action)
        next_state, stacked_frames = stack_frames(
            stacked_frames, next_obs, False, num_frames, (input_shape[1], input_shape[2])
        )
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
            s = s.view(batch_size, *input_shape)
            ns = ns.view(batch_size, *input_shape)
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

    # --- Save model and run evaluation periodically ---
    if episode % save_interval == 0:
        model_path = f"models/policy_net_{episode}.pth"
        save_model(policy_net, config, n_actions, model_path)

        avg_score, best_score, frames, _ = evaluate_policy(env, policy_net, input_shape, device, num_frames, max_steps_per_episode, num_eval_episodes=eval_episodes)
        gif_path = f"gifs/play_episode_{episode}.gif"
        save_gif(frames, gif_path)

        eval_scores.append(avg_score)
        print(
            f"Episode {episode}: train_reward={total_reward:.2f}, "
            f"eval_avg={avg_score:.2f}, eval_best={best_score:.2f}, "
            f"gif saved at {gif_path}"
        )
    else:
        print(f"Episode {episode}: train_reward={total_reward:.2f}")


# --- Save results ---
results = {
    "episode_rewards": episode_rewards,
    "eval_scores": eval_scores,
    "config": config,
}

with open("logs/training_results.json", "w") as f:
    json.dump(results, f, indent=2)

# --- Plot reward graph ---
plt.figure()
plt.plot(episode_rewards, label="Training rewards")
if eval_scores:
    eval_x = list(range(save_interval, num_episodes + 1, save_interval))
    plt.plot(eval_x, eval_scores, label="Eval avg scores")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.title("DQN Training Progress")
plt.savefig("logs/training_scores.png")
plt.show()
