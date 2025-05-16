import numpy as np
import random
import wandb
from agents.dqn.trainer import Trainer  # Import the Trainer class


def train_dqn(env, config, agent=None):
    # Initialize W&B
    run = wandb.init(
        name="First real 200 episode training run",
        project="Basic_DQN_test",
        entity="five_coding_five-student",
        config=config,
        tags=["DQN", f"variant-{env.variant}", "basic"],
        notes="Initial DQN implementation for gridworld"
    )

    # Environment/Agent setup
    dummy_state = env.reset("training")
    state_size = dummy_state.shape[0]
    action_size = 5

    from agents.dqn.agent import DQNAgent
    if agent is None:
        agent = DQNAgent(state_size, action_size, config)

    # Initialize Trainer
    trainer = Trainer(env, agent, config)
    rewards_per_episode = []

    for episode in range(config['episodes']):
        # Run one episode using the Trainer
        total_reward, steps, avg_loss, avg_q = trainer.run_episode()

        # Log metrics
        wandb.log({
            "reward": total_reward,
            "epsilon": agent.epsilon,
            "loss": avg_loss,
            "avg_q": avg_q,
            "steps": steps
        })

        rewards_per_episode.append(total_reward)
        print(f"Episode {episode + 1}/{config['episodes']} | "
              f"Reward: {total_reward:.1f} | "
              f"Epsilon: {agent.epsilon:.3f} | "
              f"Loss: {avg_loss:.3f}")

    run.finish()
    return rewards_per_episode