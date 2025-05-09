# This file contains the agent class - select_action, update-weights, etc.
import numpy as np
import random
from collections import deque
import tensorflow as tf
from agents.dqn.model import build_q_network

class DQNAgent:
    def __init__(self, state_size, action_size, config):
        self.state_size = state_size
        self.action_size = action_size

        # Hyperparameter
        self.gamma = config["gamma"]
        self.epsilon = config["epsilon"]
        self.epsilon_min = config.get("epsilon_min", 0.01)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)
        self.batch_size = config["batch_size"]
        self.lr = config["learning_rate"]
        self.target_model_update_freq = config["target_model_update_freq"]
        self.memory = deque(maxlen=config["memory_size"])

        # Modelle
        self.model = build_q_network((state_size,), action_size)
        self.target_model = build_q_network((state_size,), action_size)
        self.target_model.set_weights(self.model.get_weights())
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        self.train_step_counter = 0

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model(state.reshape(1, -1))
        return int(tf.argmax(q_values[0]).numpy())

    def record(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def prepare_batch(self, minibatch):
        states = np.array([t[0] for t in minibatch], dtype=np.float32)
        actions = np.array([t[1] for t in minibatch], dtype=np.int32)
        rewards = np.array([t[2] for t in minibatch], dtype=np.float32)
        next_states = np.array([t[3] for t in minibatch], dtype=np.float32)
        dones = np.array([float(t[4]) for t in minibatch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def update_weights(self, states, actions, rewards, next_states, dones):
        target_q = rewards + self.gamma * np.max(self.target_model(next_states), axis=1) * (1 - dones)

        with tf.GradientTape() as tape:
            q_values = self.model(states)
            q_pred = tf.gather_nd(q_values, indices=np.array(list(enumerate(actions))))
            loss = tf.reduce_mean((target_q - q_pred) ** 2)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.train_step_counter += 1

        if self.train_step_counter % self.target_model_update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())

        return loss.numpy()
