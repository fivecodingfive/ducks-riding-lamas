from environment import Environment
import numpy as np
from collections import defaultdict
import random

class TabularQAgent:
    def __init__(self, actions = [0,1,2,3,4], lr=0.5, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1):
        self.actions = actions  # list: [0,1,2,3,4]
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.Q = defaultdict(lambda: np.zeros(len(actions)))  # Q-table：字典，key=state string, value=Q值陣列

    def encode_state(self, obs_tensor):
        obs_np = obs_tensor.numpy() if hasattr(obs_tensor, "numpy") else obs_tensor
        return '_'.join(map(lambda x: f"{x:.1f}", obs_np))

    def select_action(self, obs):
        state_key = self.encode_state(obs)
        if np.random.rand() < self.epsilon:
            return random.choice(self.actions)
        return int(np.argmax(self.Q[state_key]))

    def update(self, obs, action, reward, next_obs, done):
        state_key = self.encode_state(obs)
        next_state_key = self.encode_state(next_obs)

        target = reward
        if not done:
            target += self.gamma * np.max(self.Q[next_state_key])
        
        self.Q[state_key][action] += self.lr * (target - self.Q[state_key][action])

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
    def train(self, env = Environment, mode='training', episodes=500):
        reward_log = []

        for ep in range(episodes):
            obs = env.reset(mode=mode)
            done = False
            total_reward = 0

            while not done:
                action = self.select_action(obs)
                reward, next_obs, done = env.step(action)
                self.update(obs, action, reward, next_obs, done)
                obs = next_obs
                total_reward += reward

            reward_log.append(total_reward)

            # Print the reward every 25 episodes
            if (ep + 1) % 25 == 0:
                avg_reward = sum(reward_log[-25:]) / 25
                print(f"[Episode {ep+1}/{episodes}] Avg Reward (last 25 episodes): {avg_reward:.2f} | Epsilon: {self.epsilon:.3f}")

        # Average reward
        overall_avg = sum(reward_log) / len(reward_log)
        print(f"\n[Training Done] Overall Avg Reward: {overall_avg:.2f}")

        
