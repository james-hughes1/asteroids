# Asteroids Reinforcement Learning Project

More details to come.

Setup:
1. Run `poetry install` to set up packages
2. Run `poetry run python src/game_test.py`
3. Run `poetry run python src/train_dqn.py --config train_config.yaml` to run the training loop
4. Run `poetry run python src/evaluate_dqn.py models/policy_net_950_1143_10102025.pth --episodes 10` to inspect the behaviour of a saved model.

Learnings:
- Memory leaks and RAM, this code helped: 

import tracemalloc
tracemalloc.start()
# --- Memory usage logging ---
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
print("[Top 5 memory consumers]")
for stat in top_stats[:5]:
    print(stat)