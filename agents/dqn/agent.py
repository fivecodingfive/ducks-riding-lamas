import numpy as np
import tensorflow as tf
from collections import deque
from agents.dqn.model import build_q_network


# AGENT.PY - IMPLEMENTS THE DQN ALGORITHM WITH EXPERIENCE REPLAY, TARGET NETWORK, AND EPSILON-GREEDY ACTIONS.


class DQNAgent:
    def __init__(self, state_size, action_size, config):
        self.state_size = state_size
        self.action_size = action_size

        # Hyperparameters from config
        self.gamma = config["gamma"]
        self.epsilon = config["epsilon"]
        self.epsilon_min = config["epsilon_min"]
        self.epsilon_decay = config["epsilon_decay"]
        self.batch_size = config["batch_size"]
        self.learning_rate = config["learning_rate"]
        self.target_model_update_freq = config["target_model_update_freq"]

        # Memory
        self.memory = deque(maxlen=config["memory_size"])

        # Models
        self.model = build_q_network((self.state_size,), self.action_size)
        self.target_model = build_q_network((self.state_size,), self.action_size)
        self.target_model.set_weights(self.model.get_weights())

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def select_action(self, state):
        q_values = self.model(state)
        return q_values[0].numpy()

    def record(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def prepare_batch(self, minibatch):
        states = np.vstack([sample[0] for sample in minibatch])
        actions = np.array([sample[1] for sample in minibatch], dtype=np.int32)
        rewards = np.array([sample[2] for sample in minibatch], dtype=np.float32)
        next_states = np.vstack([sample[3] for sample in minibatch])
        dones = np.array([sample[4] for sample in minibatch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def update_weights(self, states, actions, rewards, next_states, dones):
        target_qs = self.target_model.predict(next_states)
        targets = rewards + (1 - dones) * self.gamma * np.amax(target_qs, axis=1)

        with tf.GradientTape() as tape:
            q_values = self.model(states)
            indices = np.array([[i, a] for i, a in enumerate(actions)])
            q_selected = tf.gather_nd(q_values, indices)
            loss = tf.reduce_mean((targets - q_selected) ** 2)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss.numpy()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
