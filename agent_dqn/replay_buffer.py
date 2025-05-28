import random
import numpy as np
import tensorflow as tf
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity, state_shape):
        self.capacity = capacity
        self.state_shape = state_shape
        self.pos = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.size = tf.Variable(0, dtype=tf.int32, trainable=False)

        self.states = tf.Variable(tf.zeros([capacity, *state_shape], dtype=tf.float32), trainable=False)
        self.actions = tf.Variable(tf.zeros([capacity], dtype=tf.int32), trainable=False)
        self.rewards = tf.Variable(tf.zeros([capacity], dtype=tf.float32), trainable=False)
        self.next_states = tf.Variable(tf.zeros([capacity, *state_shape], dtype=tf.float32), trainable=False)
        self.dones = tf.Variable(tf.zeros([capacity], dtype=tf.float32), trainable=False)

    def add(self, state, action, reward, next_state, done):
        self.states[self.pos].assign(state)
        self.actions[self.pos].assign(action)
        self.rewards[self.pos].assign(reward)
        self.next_states[self.pos].assign(next_state)
        self.dones[self.pos].assign(done)

        self.pos.assign((self.pos + 1) % self.capacity)
        self.size.assign(tf.minimum(self.size + 1, self.capacity))

    def sample(self, batch_size):
        indices = tf.random.shuffle(tf.range(self.size))[:batch_size]

        return (
            tf.gather(self.states, indices),
            tf.gather(self.actions, indices),
            tf.gather(self.rewards, indices),
            tf.gather(self.next_states, indices),
            tf.gather(self.dones, indices),
        )

    def __len__(self):
        return int(self.size.numpy())

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
        indices = tf.cast(indices, dtype=tf.int32)

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
        indices = tf.expand_dims(indices, axis=1)  # [64] → [64,1]
        updates = tf.abs(td_errors) + self.epsilon  # shape: [64]

        # Create updated tensor (new_priorities)
        new_priorities = tf.tensor_scatter_nd_update(self.priorities, indices, updates)

        # assign back to variable
        self.priorities.assign(new_priorities)

    def __len__(self):
        return self.size
