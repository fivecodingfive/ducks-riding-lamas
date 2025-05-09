from lib2to3.fixer_util import does_tree_import

import numpy as np
import random


"""
What this function does
    Loop through episodes
    Call env.reset("training")
    Take steps via env.step(action) → returns reward, next_obs, done
    Store transitions
    Train using minibatches
    Sync target model occasionally
    Decay epsilon
"""

def train_dqn(env, config, agent=None):
    state_size = len(env.get_obs())
    action_size = 5  # actions: 0 (nothing), 1 (up), 2 (right), 3 (down), 4 (left)

    from agents.dqn.agent import DQNAgent  # Avoid circular imports

    if agent is None:
        agent = DQNAgent(state_size, action_size, config)

    rewards_per_episode = []

    for episode in range(config['episodes']):
        state = env.reset("training").reshape(1, -1)
        total_reward = 0

        for step in range(config['max_steps']):
            if np.random.rand() < agent.epsilon:
                action = random.randint(0, action_size - 1)
            else:
                q_values = agent.select_action(state)
                action = np.argmax(q_values)

            reward, next_state, done = env.step(action)
            next_state = next_state.reshape(1, -1)

            agent.record(state, action, reward, next_state, done)

            if len(agent.memory) >= agent.batch_size:
                minibatch = random.sample(agent.memory, agent.batch_size)
                states, actions, rewards, next_states, dones = agent.prepare_batch(minibatch)
                agent.update_weights(states, actions, rewards, next_states, dones)

            if step % agent.target_model_update_freq == 0:
                agent.update_target_model()

            total_reward += reward
            state = next_state

            if done:
                break

        agent.decay_epsilon()
        rewards_per_episode.append(total_reward)
        print(f"Episode {episode + 1}/{config['episodes']} — Reward: {total_reward:.2f} — Epsilon: {agent.epsilon:.3f}")

    return rewards_per_episode