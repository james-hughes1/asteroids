import os
import torch
import yaml
from training.dqn_model import DQN

def save_model(model, config, n_actions, save_path):
    """
    Save model weights, configuration, and metadata in a single .pth file.
    
    Args:
        model (torch.nn.Module): The trained model to save.
        config (dict): The configuration dictionary used for training.
        n_actions (int): Number of actions in the environment.
        save_path (str): File path to save model (e.g., 'models/policy_net_1000.pth').
    """
    # Compute full input shape from config
    num_frames = config["num_frames"]
    channels_per_frame = config["channels_per_frame"]
    input_shape_2d = config["input_shape_2d"]
    input_shape = (num_frames * channels_per_frame, *input_shape_2d)

    model_data = {
        "state_dict": model.state_dict(),
        "config": config,
        "input_shape": input_shape,
        "n_actions": n_actions
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model_data, save_path)
    print(f"✅ Model and config saved to {save_path}")


def load_model(model_path, device="cpu"):
    """
    Load model and its config from a saved .pth file.

    Args:
        model_path (str): Path to the saved .pth file.
        device (str): 'cpu' or 'cuda'.

    Returns:
        model (torch.nn.Module): Loaded model ready for inference.
        config (dict): Configuration dictionary saved with the model.
        n_actions (int): Number of actions in the environment.
    """
    checkpoint = torch.load(model_path, map_location=device)

    # Load model info
    config = checkpoint["config"]
    input_shape = checkpoint["input_shape"]
    n_actions = checkpoint["n_actions"]

    # Recreate model and load weights
    model = DQN(input_shape, n_actions).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    print(f"✅ Loaded model from {model_path}")
    print(f"   - Input shape: {input_shape}")
    print(f"   - Actions: {n_actions}")

    return model, config, n_actions
