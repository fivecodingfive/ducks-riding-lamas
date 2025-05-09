
import numpy as np
import random
import wandb

# TRAIN.PY - RUNS THE TRAINING LOOP (EPISODES, STEPS, REWARDS) AND UPDATES THE AGENT.

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
    # Initialize W&B
    run = wandb.init(
        project="Basic_DQN_test",
        entity="five_coding_five",
        config=config,
        tags=["DQN", f"variant-{env.variant}", "basic"],
        notes="Initial DQN implementation for gridworld"
    )

    # Environment setup
    dummy_state = env.reset("training")
    state_size = len(dummy_state)
    action_size = 5

    from agents.dqn.agent import DQNAgent
    if agent is None:
        agent = DQNAgent(state_size, action_size, config)

    rewards_per_episode = []

    for episode in range(config['episodes']):
        state = env.reset("training").reshape(1, -1)
        total_reward = 0
        episode_losses = []  # To track average loss per episode

        for step in range(config['max_steps']):
            if np.random.rand() < agent.epsilon:
                action = random.randint(0, action_size - 1)
                q_values = None  # No Q-values for random actions
            else:
                q_values = agent.select_action(state)
                action = np.argmax(q_values)

            reward, next_state, done = env.step(action)
            next_state = next_state.reshape(1, -1)

            agent.record(state, action, reward, next_state, done)

            if len(agent.memory) >= agent.batch_size:
                minibatch = random.sample(agent.memory, agent.batch_size)
                states, actions, rewards, next_states, dones = agent.prepare_batch(minibatch)
                loss = agent.update_weights(states, actions, rewards, next_states, dones)
                episode_losses.append(loss.numpy())

            if step % agent.target_model_update_freq == 0:
                agent.update_target_model()

            total_reward += reward
            state = next_state

            if done:
                break

        # Log metrics at the end of each episode
        avg_loss = np.mean(episode_losses) if episode_losses else None
        wandb.log({
            "episode": episode,
            "reward": total_reward,
            "epsilon": agent.epsilon,
            "loss": avg_loss,
            "avg_q": np.mean(q_values) if q_values is not None else None
        })

        agent.decay_epsilon()
        rewards_per_episode.append(total_reward)
        print(f"Episode {episode + 1}/{config['episodes']} — Reward: {total_reward:.2f} — Epsilon: {agent.epsilon:.3f}")

    run.finish()
    return rewards_per_episode