# Asteroids Reinforcement Learning Project

More details to come.

Setup:
1. Run `poetry install` to set up packages
2. Run `poetry run python src/train_dqn.py --config train_config.yaml` to run the training loop
3. Run `poetry run python src/game_test.py`

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