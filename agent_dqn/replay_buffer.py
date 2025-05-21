import random
import numpy as np
import tensorflow as tf
from collections import deque

class ReplayBuffer:
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

class PrioritizedReplayBuffer:
    def __init__(self, capacity, state_shape, alpha=0.6, epsilon=1e-5):
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = epsilon
        self.pos = 0
        self.size = 0

        self.states = tf.Variable(tf.zeros([capacity, *state_shape], dtype=tf.float32), trainable=False)
        self.actions = tf.Variable(tf.zeros([capacity], dtype=tf.int32), trainable=False)
        self.rewards = tf.Variable(tf.zeros([capacity], dtype=tf.float32), trainable=False)
        self.next_states = tf.Variable(tf.zeros([capacity, *state_shape], dtype=tf.float32), trainable=False)
        self.dones = tf.Variable(tf.zeros([capacity], dtype=tf.float32), trainable=False)
        self.priorities = tf.Variable(tf.zeros([capacity], dtype=tf.float32), trainable=False)

    def add(self, state, action, reward, next_state, done):
        self.states[self.pos].assign(tf.convert_to_tensor(state, dtype=tf.float32))
        self.actions[self.pos].assign(tf.convert_to_tensor(action, dtype=tf.int32))
        self.rewards[self.pos].assign(tf.convert_to_tensor(reward, dtype=tf.float32))
        self.next_states[self.pos].assign(tf.convert_to_tensor(next_state, dtype=tf.float32))
        self.dones[self.pos].assign(tf.convert_to_tensor(done, dtype=tf.float32))

        max_prio = tf.reduce_max(self.priorities[:self.size]) if self.size > 0 else tf.constant(1.0)
        self.priorities[self.pos].assign(max_prio)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta=0.4):
        valid_priorities = self.priorities[:self.size]
        probs = tf.pow(valid_priorities + self.epsilon, self.alpha)
        probs /= tf.reduce_sum(probs)

        indices = tf.random.categorical(tf.math.log([probs]), batch_size)
        indices = tf.squeeze(indices, axis=0)

        sampled_probs = tf.gather(probs, indices)
        total = tf.cast(self.size, tf.float32)
        weights = tf.pow(total * sampled_probs, -beta)
        weights /= tf.reduce_max(weights)

        return (
            tf.gather(self.states, indices),
            tf.gather(self.actions, indices),
            tf.gather(self.rewards, indices),
            tf.gather(self.next_states, indices),
            tf.gather(self.dones, indices),
            indices,
            weights
        )

    def update_priorities(self, indices, td_errors):
        td_errors = tf.abs(td_errors) + self.epsilon
        for i in range(tf.shape(indices)[0]):
            idx = indices[i]
            self.priorities[idx].assign(td_errors[i])

    def __len__(self):
        return self.size
