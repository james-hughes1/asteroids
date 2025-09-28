# training/replay_buffer.py
from collections import deque
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Push a transition into the buffer.
        - state and next_state should already be preprocessed NumPy arrays
          (e.g., shape: (num_frames, 84, 84), dtype: float32)
        - action, reward, done can be scalars
        """
        # make copies to avoid accidental modifications outside
        self.buffer.append((
            np.array(state, copy=True, dtype=np.uint8),
            action,
            reward,
            np.array(next_state, copy=True, dtype=np.uint8),
            done
        ))

    def sample(self, batch_size):
        """
        Sample a batch from the buffer.
        Returns NumPy arrays only.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.int64),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)
