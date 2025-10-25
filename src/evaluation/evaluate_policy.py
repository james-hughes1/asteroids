import numpy as np
import torch
from collections import deque
import cv2
import imageio
from utils.preprocess import stack_frames, preprocess_frame
from PIL import Image, ImageDraw, ImageFont


def evaluate_policy(
    env,
    policy_net,
    input_shape,
    device,
    num_frames,
    max_steps,
    num_eval_episodes,
    base_seed=42,
    deterministic=True,
    epsilon_eval=0.0,
):
    """
    Evaluate a trained (or training) DQN policy.

    Args:
        env: Gym-like environment
        policy_net: DQN model
        input_shape: Tuple (C, H, W)
        device: torch.device
        num_eval_episodes: Number of episodes to average over
        num_frames: Number of stacked frames per state
        max_steps: Max steps per episode
        base_seed: Seed for reproducibility
        deterministic: If True, uses fixed seeds (for consistent evaluation)
        epsilon_eval: Probability of taking a random action (like ε-greedy during training)

    Returns:
        mean_score: float
        best_score: float
        worst_score: float
        best_frames: list of RGB frames (best-performing episode)
        worst_frames: list of RGB frames (worst-performing episode)
        complete_rate: float (fraction of episodes where all asteroids were destroyed)
        cumulative_rewards: list of cumulative rewards for the best episode
    """
    scores = []
    all_frames = []
    step_counts = []
    complete_rate = 0.0
    n_actions = env.action_space.n
    cumulative_rewards_full = []

    for ep in range(num_eval_episodes):
        stacked_frames = deque(maxlen=num_frames)
        # Optionally seed environment for reproducibility
        obs, _ = env.reset(seed=base_seed + ep if deterministic else None)
        state, stacked_frames = stack_frames(
            stacked_frames, obs, True, num_frames, (input_shape[1], input_shape[2]), rgb=False
        )

        done = False
        cumulative_rewards = []
        total_reward = 0
        steps = 0
        frames = []

        while not done and steps < max_steps:
            state_tensor = torch.tensor(np.array([state]), dtype=torch.float32).to(device)
            state_tensor = state_tensor.view(1, *input_shape)

            # --- ε-greedy action selection for evaluation ---
            with torch.no_grad():
                q_values = policy_net(state_tensor)
                greedy_action = q_values.argmax(dim=1).item()

            if np.random.rand() < epsilon_eval:
                action = np.random.randint(0, n_actions)
            else:
                action = greedy_action
            # -------------------------------------------------

            obs, reward, done, truncated, info = env.step(action)
            state, stacked_frames = stack_frames(
                stacked_frames, obs, False, num_frames, (input_shape[1], input_shape[2]), rgb=False
            )
            total_reward += reward
            cumulative_rewards.append(total_reward)
            steps += 1
            frames.append(obs)

        if len(env.asteroids) == 0:
            complete_rate += 1.0

        cumulative_rewards_full.append(cumulative_rewards)

        scores.append(total_reward)
        all_frames.append(frames)
        step_counts.append(steps)

    complete_rate /= num_eval_episodes

    mean_score = np.sum(scores)/np.sum(step_counts)
    best_idx = int(np.argmax(scores))
    best_score = scores[best_idx]
    worst_idx = int(np.argmin(scores))
    worst_score = scores[worst_idx]

    # Header
    print("\nEvaluation Results:")
    print(f"{'Episode':>8} | {'Score':>10} | {'Steps':>10} | {'Avg. Score':>10}")
    print("-" * 46)

    # Rows
    for i, (s, steps) in enumerate(zip(scores, step_counts), start=1):
        print(f"{i:8d} | {s:10.2f} | {steps:10d} | {s/steps if steps > 0 else 0:10.4f}")

    # Means
    print("-" * 46)
    print(f"{'Mean':>8} | {np.mean(scores):10.2f} | {np.mean(step_counts):10.2f} | {mean_score:10.4f}")

    return mean_score, best_score/len(all_frames[best_idx]), worst_score/len(all_frames[worst_idx]), all_frames[best_idx], all_frames[worst_idx], complete_rate, cumulative_rewards_full[best_idx], cumulative_rewards_full[worst_idx]

# --- GIF saving function ---
def save_gif(frames, filename="gifs/play.gif", network_size=(84,84), scale=4, rewards=None):
    """Save a gif, scaling up with nearest-neighbor so pixels stay blocky."""
    upscaled = []
    font = ImageFont.load_default()
    for i, frame in enumerate(frames):
        # Preprocess frame
        frame = preprocess_frame(frame, network_size=network_size, rgb=True, return_uint8=True)
        enlarged = np.repeat(np.repeat(frame, scale, axis=0), scale, axis=1)

        img = Image.fromarray(enlarged)
        draw = ImageDraw.Draw(img)

        # Overlay reward if provided
        if rewards is not None and i < len(rewards):
            text = f"Reward: {rewards[i]:.1f}"
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            pos = (img.width - text_w - 10, 10)  # top-right
            draw.rectangle(
                [pos[0] - 4, pos[1] - 2, pos[0] + text_w + 4, pos[1] + text_h + 2],
                fill=(0, 0, 0, 180)
            )
            draw.text(pos, text, fill=(255, 255, 255), font=font)

        upscaled.append(np.array(img))
    imageio.mimsave(filename, upscaled, fps=30)
    print(f"Saved GIF: {filename}")

def sample_frames(frames, n=200):
    if len(frames) <= n:
        return frames  # no need to sample if short
    idxs = np.linspace(0, len(frames) - 1, n, dtype=int)
    return [frames[i] for i in idxs]