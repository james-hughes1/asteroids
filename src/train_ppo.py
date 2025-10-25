# training/train_ppo.py

import os
import argparse
import yaml
import shutil
import datetime
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv

from asteroids_env.env import AsteroidsEnv
from utils.preprocess import stack_frames
from utils.model_io import save_model
from evaluation.evaluate_policy import evaluate_policy

from training.ppo_model import PPOActorCritic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 1. Load config
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config/train_config.yaml")
args = parser.parse_args()
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

num_envs = min(config.get("num_envs", 16), os.cpu_count() or 1)
num_frames = config.get("num_frames", 5)
channels_per_frame = config.get("channels_per_frame", 3)
input_shape = (num_frames * channels_per_frame, *config["input_shape_2d"])
n_actions = config.get("n_actions", 5)

# PPO hyperparameters
total_timesteps = config.get("max_frames", 10_000_000)
update_timesteps = config.get("update_timesteps", 4096)
epochs = config.get("ppo_epochs", 4)
mini_batch_size = config.get("mini_batch_size", 256)
gamma = config.get("gamma", 0.99)
gae_lambda = config.get("gae_lambda", 0.95)
clip_coef = config.get("clip_coef", 0.2)
ent_coef = config.get("ent_coef", 0.01)
vf_coef = config.get("vf_coef", 0.5)
max_grad_norm = config.get("max_grad_norm", 0.5)
learning_rate = config.get("learning_rate", 3e-4)
frame_skip = config.get("frame_skip", 1)
max_steps_per_episode = config.get("max_steps_per_episode", 1000)

# Curriculum parameters
num_asteroids = config.get("num_asteroids", 5)
max_asteroid_size = config.get("max_asteroid_size", 90)
min_asteroid_size = config.get("min_asteroid_size", 30)
max_asteroid_speed_start = config.get("max_asteroid_speed_start", 0.5)
max_asteroid_speed_end = config.get("max_asteroid_speed_end", 2.5)
max_asteroid_speed_ramp = config.get("max_asteroid_speed_ramp", 1_000_000)
death_reward = config.get("death_reward", -1.0)
asteroid_destroyed_reward_scalar = config.get("asteroid_destroyed_reward_scalar", 1.0)

# Logging / saving
save_interval = config.get("save_interval", 500_000)
eval_interval = config.get("eval_interval", 200_000)
google_drive_backup = config.get("google_drive_backup", True)
eval_episodes = config.get("eval_episodes", 5)

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir = os.path.join("/content/drive/MyDrive/asteroids_models", f"run_{timestamp}")
if google_drive_backup:
    os.makedirs(run_dir, exist_ok=True)

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("gifs", exist_ok=True)

# -------------------------------
# 2. Build vectorized envs
# -------------------------------
current_max_speed = [max_asteroid_speed_start]

def make_env():
    def _init():
        return AsteroidsEnv(
            render_mode="rgb_array",
            width=400,
            height=400,
            max_steps=max_steps_per_episode,
            num_asteroids=num_asteroids,
            max_asteroid_size=max_asteroid_size,
            min_asteroid_size=min_asteroid_size,
            max_asteroid_speed=current_max_speed[0],
            death_reward=death_reward,
            asteroid_destroyed_reward_scalar=asteroid_destroyed_reward_scalar,
            frame_skip=frame_skip,
        )
    return _init

envs = AsyncVectorEnv([make_env() for _ in range(num_envs)], autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP)
obs, _ = envs.reset()

# -------------------------------
# 3. Initialize PPO model
# -------------------------------
model = PPOActorCritic(input_shape, n_actions).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)

# -------------------------------
# 4. Storage
# -------------------------------
obs_stack = [deque(maxlen=num_frames) for _ in range(num_envs)]
states = []
for i in range(num_envs):
    s, obs_stack[i] = stack_frames(obs_stack[i], obs[i], True, num_frames, config["input_shape_2d"])
    s = s.reshape(-1, *config["input_shape_2d"])  # flatten frames into channels
    states.append(s)
states = np.stack(states).astype(np.uint8)  # shape: (num_envs, C, H, W)

# Rollout buffers
obs_buffer = np.zeros((update_timesteps, num_envs, *input_shape), dtype=np.uint8)
actions_buffer = np.zeros((update_timesteps, num_envs), dtype=np.int32)
logprobs_buffer = np.zeros((update_timesteps, num_envs), dtype=np.float32)
rewards_buffer = np.zeros((update_timesteps, num_envs), dtype=np.float32)
values_buffer = np.zeros((update_timesteps, num_envs), dtype=np.float32)
dones_buffer = np.zeros((update_timesteps, num_envs), dtype=np.float32)

# -------------------------------
# 5. PPO training loop
# -------------------------------
global_step = 0
episode_returns = []

print(f"Starting PPO training for {total_timesteps:,} steps across {num_envs} envs")

while global_step < total_timesteps:
    for t in range(update_timesteps):
        # Prepare state tensor
        s_tensor = torch.tensor(states, dtype=torch.float32, device=device) / 255.0

        # Get action, logprob, value
        with torch.no_grad():
            action, logprob, value = model.get_action_and_value(s_tensor)

        # Step environments
        actions = action.cpu().numpy()
        next_obs, rewards, dones, truncs, infos = envs.step(actions)
        rewards = np.clip(rewards, -10, 10)

        # Store in rollout buffer
        obs_buffer[t] = states
        actions_buffer[t] = actions
        logprobs_buffer[t] = logprob.cpu().numpy()
        rewards_buffer[t] = rewards
        values_buffer[t] = value.cpu().numpy()
        dones_buffer[t] = dones

        # Prepare next states
        next_states = []
        for i in range(num_envs):
            s, obs_stack[i] = stack_frames(obs_stack[i], next_obs[i], dones[i], num_frames, config["input_shape_2d"])
            s = s.reshape(-1, *config["input_shape_2d"])
            next_states.append(s)
        states = np.stack(next_states)
        global_step += num_envs

    # Compute GAE advantages
    with torch.no_grad():
        next_values = model.get_value(torch.tensor(states, dtype=torch.float32, device=device) / 255.0).cpu().numpy()

    advantages = np.zeros_like(rewards_buffer)
    lastgaelam = 0
    for t in reversed(range(update_timesteps)):
        if t == update_timesteps - 1:
            nextnonterminal = 1.0 - dones_buffer[t]
            nextval = next_values
        else:
            nextnonterminal = 1.0 - dones_buffer[t + 1]
            nextval = values_buffer[t + 1]
        delta = rewards_buffer[t] + gamma * nextval * nextnonterminal - values_buffer[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
    returns = advantages + values_buffer

    # Flatten rollout data
    batch_size = num_envs * update_timesteps
    b_obs = torch.tensor(obs_buffer.reshape(batch_size, *input_shape), device=device) / 255.0
    b_actions = torch.tensor(actions_buffer.flatten(), device=device)
    b_logprobs = torch.tensor(logprobs_buffer.flatten(), device=device)
    b_advantages = torch.tensor(advantages.flatten(), device=device)
    b_returns = torch.tensor(returns.flatten(), device=device)

    # PPO update
    inds = np.arange(batch_size)
    for epoch in range(epochs):
        np.random.shuffle(inds)
        for start in range(0, batch_size, mini_batch_size):
            end = start + mini_batch_size
            mb_inds = inds[start:end]

            mb_obs = b_obs[mb_inds]
            mb_actions = b_actions[mb_inds]
            mb_adv = b_advantages[mb_inds]
            mb_logprob_old = b_logprobs[mb_inds]
            mb_returns = b_returns[mb_inds]

            newlogprob, entropy, newvalue = model.evaluate_actions(mb_obs, mb_actions)
            ratio = torch.exp(newlogprob - mb_logprob_old)

            pg_loss = torch.max(-mb_adv * ratio, -mb_adv * torch.clamp(ratio, 1-clip_coef, 1+clip_coef)).mean()
            v_loss = 0.5 * ((newvalue - mb_returns)**2).mean()
            entropy_loss = entropy.mean()
            loss = pg_loss + vf_coef * v_loss - ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

    # Logging, curriculum, saving
    avg_return = np.mean(np.sum(rewards_buffer, axis=0))
    print(f"Step {global_step:,} | Avg return = {avg_return:.3f}")
    episode_returns.append(avg_return)

    progress = min(1.0, global_step / max_asteroid_speed_ramp)
    current_max_speed[0] = max_asteroid_speed_start + progress * (max_asteroid_speed_end - max_asteroid_speed_start)
    envs.call("set_difficulty", max_asteroid_speed=current_max_speed[0])

    if global_step % save_interval < (num_envs * update_timesteps):
        save_model(model, config, n_actions, f"models/ppo_model_{global_step}.pth")
        if google_drive_backup:
            shutil.copy(f"models/ppo_model_{global_step}.pth", os.path.join(run_dir, f"ppo_model_{global_step}.pth"))

    if global_step % eval_interval < (num_envs * update_timesteps):
        avg_score, best_score, worst_score, best_frames, _, complete_rate, _, _ = evaluate_policy(
            AsteroidsEnv(
                render_mode="rgb_array",
                width=400, height=400,
                max_steps=max_steps_per_episode,
                num_asteroids=num_asteroids,
                max_asteroid_size=max_asteroid_size,
                min_asteroid_size=min_asteroid_size,
                max_asteroid_speed=current_max_speed[0],
                death_reward=death_reward,
                asteroid_destroyed_reward_scalar=asteroid_destroyed_reward_scalar,
                frame_skip=1
            ),
            model,
            input_shape,
            device,
            num_frames,
            max_steps_per_episode,
            num_eval_episodes=eval_episodes
        )
        print(f"Eval → avg={avg_score:.3f}, best={best_score:.3f}, complete={complete_rate*100:.1f}%")

envs.close()
print("Training complete.")
