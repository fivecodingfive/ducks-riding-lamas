import numpy as np
import random

class Trainer:
    def __init__(self, env, agent, config):
        self.env = env
        self.agent = agent
        self.config = config

    def run_episode(self):
        state = self.env.reset(mode="training")
        total_reward = 0
        episode_losses = []
        episode_q_values = []
        done = False  # NEW: Track episode termination
        step = 0  # NEW: Explicit step counter

        while not done and step < self.config["max_steps"]:  # FIXED: Use while-loop for early termination
            # 1. Action selection
            if np.random.rand() < self.agent.epsilon:
                action = np.random.randint(self.agent.action_size)
                q_values = None  # FIXED: Handle random actions
            else:
                q_values = self.agent.model.predict(state[np.newaxis], verbose=0)
                action = np.argmax(q_values[0])

            # 2. Environment step
            reward, next_state, done = self.env.step(action)  # Fix variable order

            # 3. Store experience
            self.agent.record(state, action, reward, next_state, done)

            # 4. Training
            if len(self.agent.memory) >= self.agent.batch_size:
                minibatch = random.sample(self.agent.memory, self.agent.batch_size)
                states, actions, rewards, next_states, dones = self.agent.prepare_batch(minibatch)
                loss = self.agent.update_weights(states, actions, rewards, next_states, dones)
                episode_losses.append(loss)
                if q_values is not None:  # Only track Q-values for non-random actions
                    episode_q_values.append(np.max(q_values))

            # 5. Target network update
            if step % self.agent.target_model_update_freq == 0:
                self.agent.update_target_model()

            # 6. Epsilon decay
            self.agent.decay_epsilon()

            total_reward += reward
            state = next_state
            step += 1  # FIXED: Increment step counter

        return (
            total_reward,
            step,  # Actual steps taken (not max_steps)
            np.mean(episode_losses) if episode_losses else 0,
            np.mean(episode_q_values) if episode_q_values else 0
        )