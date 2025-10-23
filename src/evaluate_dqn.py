import argparse
import torch
from training.dqn_model import DQN        # your DQN class
from asteroids_env.env import AsteroidsEnv  # your game environment
from evaluation.evaluate_policy import evaluate_policy, save_gif, sample_frames
from utils.model_io import load_model
import yaml
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("model_path", help="Path to saved DQN model (.pth)")
parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--max_speed", type=float, default=42, help="Specifies difficulty level via max asteroid speed")
args = parser.parse_args()

device = torch.device(args.device)

max_steps = 2000

# # --- Load model ---
env = AsteroidsEnv(render_mode="rgb_array", width=400, height=400, max_steps=max_steps, num_asteroids=4, max_asteroid_size=90, max_asteroid_speed=args.max_speed)
model, config, n_actions = load_model(args.model_path, device)
num_frames = config.get("num_frames", 5)
channels_per_frame = config.get("channels_per_frame", 3)
input_shape = tuple([num_frames * channels_per_frame, *config["input_shape_2d"]])
model.eval()
print("Loaded config:")
print(yaml.dump(config, sort_keys=False, default_flow_style=False))

# --- Evaluate ---
best_reward = -float("inf")
best_frames = None

avg_reward, best_reward, worst_reward, best_frames, worst_frames, complete_rate, cumulative_rewards = evaluate_policy(env, model, input_shape, args.device, num_frames=num_frames, max_steps=max_steps, num_eval_episodes=args.episodes)

print(f"Best episode reward = {best_reward:.2f}, Average reward over {args.episodes} episodes = {avg_reward:.2f}, Worst episode reward = {worst_reward:.2f}")
save_gif(sample_frames(best_frames, n=200), "gifs/best_episode.gif", network_size=(input_shape[1], input_shape[2]), rewards=cumulative_rewards)
save_gif(sample_frames(worst_frames, n=200), "gifs/worst_episode.gif", network_size=(input_shape[1], input_shape[2]))
print(f"Complete rate (all asteroids destroyed): {complete_rate*100:.2f}%")
env.close()
