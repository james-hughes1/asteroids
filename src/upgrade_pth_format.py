import torch
import argparse
from training.dqn_model import DQN
from asteroids_env.env import AsteroidsEnv

def main():
    parser = argparse.ArgumentParser(description="Upgrade old DQN .pth to include metadata")
    parser.add_argument("--old_path", type=str, required=True, help="Path to old .pth file")
    parser.add_argument("--new_path", type=str, required=True, help="Path to save upgraded .pth file")
    parser.add_argument("--input_shape", type=int, nargs=3, required=True, metavar=("C", "H", "W"),
                        help="Input shape, e.g. --input_shape 15 128 128")
    args = parser.parse_args()

    print(f"Loading old model from: {args.old_path}")
    state_dict = torch.load(args.old_path, map_location="cpu")

    # Get n_actions from environment
    env = AsteroidsEnv()
    n_actions = env.action_space.n

    model_data = {
        "state_dict": state_dict,
        "input_shape": tuple(args.input_shape),
        "n_actions": n_actions,
    }

    torch.save(model_data, args.new_path)
    print(f"Upgraded model saved to {args.new_path}")
    print(f"   • input_shape: {tuple(args.input_shape)}")
    print(f"   • n_actions: {n_actions}")

if __name__ == "__main__":
    main()
