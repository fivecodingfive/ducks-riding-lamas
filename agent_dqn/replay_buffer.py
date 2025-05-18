import random
import numpy as np
import tensorflow as tf
from collections import deque

class ReplayBuffer:
    # def __init__(self, capacity):
    #     self.buffer = deque(maxlen=capacity)

    # def add(self, state, action, reward, next_state, done):
    #     self.buffer.append((state, action, reward, next_state, done))

    # def sample(self, batch_size):
    #     batch = random.sample(self.buffer, batch_size)
    #     state, action, reward, next_state, done = map(np.array, zip(*batch))

    #     return (
    #         tf.convert_to_tensor(state, dtype=tf.float32),
    #         tf.convert_to_tensor(action, dtype=tf.int32),
    #         tf.convert_to_tensor(reward, dtype=tf.float32),
    #         tf.convert_to_tensor(next_state, dtype=tf.float32),
    #         tf.convert_to_tensor(done, dtype=tf.float32)
    #     )


    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        # Convert all components to tensors
        state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
        action_tensor = tf.convert_to_tensor(action, dtype=tf.int32)
        reward_tensor = tf.convert_to_tensor(reward, dtype=tf.float32)
        next_state_tensor = tf.convert_to_tensor(next_state, dtype=tf.float32)
        done_tensor = tf.convert_to_tensor(done, dtype=tf.float32)
        self.buffer.append((state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            tf.stack(states),       # Shape: (batch_size, state_dim)
            tf.squeeze(tf.stack(actions)),  # Shape: (batch_size,)
            tf.squeeze(tf.stack(rewards)),  # Shape: (batch_size,)
            tf.stack(next_states),   # Shape: (batch_size, state_dim)
            tf.squeeze(tf.stack(dones))     # Shape: (batch_size,)
        )

    def __len__(self):
        return len(self.buffer)