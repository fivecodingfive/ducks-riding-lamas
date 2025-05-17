import random
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))

        return (
            tf.convert_to_tensor(state, dtype=tf.float32),
            tf.convert_to_tensor(action, dtype=tf.int32),
            tf.convert_to_tensor(reward, dtype=tf.float32),
            tf.convert_to_tensor(next_state, dtype=tf.float32),
            tf.convert_to_tensor(done, dtype=tf.float32)
        )


    def __len__(self):
        return len(self.buffer)
