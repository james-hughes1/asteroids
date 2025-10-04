import numpy as np
import torch
from collections import deque
import cv2
import imageio
from utils.preprocess import stack_frames


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

    Returns:
        mean_score: float
        best_score: float
        best_frames: list of RGB frames (best-performing episode)
        most_steps_frames: list of RGB frames (episode with most steps)
    """
    scores = []
    all_frames = []
    step_counts = []

    for ep in range(num_eval_episodes):
        stacked_frames = deque(maxlen=num_frames)
        # Optionally seed environment for reproducibility
        obs, _ = env.reset(seed=base_seed + ep if deterministic else None)
        state, stacked_frames = stack_frames(
            stacked_frames, obs, True, num_frames, (input_shape[1], input_shape[2])
        )

        done = False
        total_reward = 0
        steps = 0
        frames = []

        while not done and steps < max_steps:
            state_tensor = torch.tensor(np.array([state]), dtype=torch.float32).to(device)
            state_tensor = state_tensor.view(1, *input_shape)
            with torch.no_grad():
                action = policy_net(state_tensor).argmax(dim=1).item()

            obs, reward, done, truncated, info = env.step(action)
            state, stacked_frames = stack_frames(
                stacked_frames, obs, False, num_frames, (input_shape[1], input_shape[2])
            )
            total_reward += reward
            steps += 1
            frames.append(obs)

        scores.append(total_reward)
        all_frames.append(frames)
        step_counts.append(steps)

    best_idx = int(np.argmax(scores))
    mean_score = np.mean(scores)
    best_score = scores[best_idx]
    most_steps_idx = int(np.argmax(step_counts))

    return mean_score, best_score, all_frames[best_idx], all_frames[most_steps_idx]

# --- GIF saving function ---
def save_gif(frames, filename="gifs/play.gif", scale=4):
    """Save a gif, scaling up with nearest-neighbor so pixels stay blocky."""
    upscaled = []
    for frame in frames:
        # frame is HxWx3, resize with NEAREST to keep blocky look
        h, w, _ = frame.shape
        enlarged = cv2.resize(frame, (w*scale, h*scale), interpolation=cv2.INTER_NEAREST)
        upscaled.append(enlarged)
    imageio.mimsave(filename, upscaled, fps=30)