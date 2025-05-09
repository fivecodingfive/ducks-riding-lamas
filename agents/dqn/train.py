# This file contains the training loop.
from agents.dqn.agent import DQNAgent
import numpy as np
import random

def train_dqn(env, config):
    obs = env.reset("training")  # needed to define input size
    state_size = len(obs)
    action_size = 5                          # stay, up, right, down, left

    agent = DQNAgent(state_size, action_size, config)

    all_rewards = []

    for episode in range(config["episodes"]):
        obs = env.reset("training")
        total_reward = 0

        for step in range(config["max_steps"]):
            action = agent.select_action(obs)
            reward, next_obs, done = env.step(action)
            agent.record(obs, action, reward, next_obs, done)

            if len(agent.memory) >= config["batch_size"]:
                minibatch = random.sample(agent.memory, config["batch_size"])
                batch = agent.prepare_batch(minibatch)
                loss = agent.update_weights(*batch)

            obs = next_obs
            total_reward += reward

            if done:
                break

        all_rewards.append(total_reward)

        # Epsilon decay
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        print(f"🎓 Episode {episode + 1}/{config['episodes']} | Reward: {total_reward:.1f} | Epsilon: {agent.epsilon:.3f}")

    return all_rewards
