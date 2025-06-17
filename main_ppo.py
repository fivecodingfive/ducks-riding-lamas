from agent_ppo.ppo_agent import PPO_Agent
from environment import Environment
import argparse

data_dir = './data'         # specify relative path to data directory (e.g., './data', not './data/variant_0')
variant = args.variant      # specify problem variant (0 for base variant, 1 for first extension, 2 for second extension)
episodes = args.episodes    # specify episodes


if __name__ == "__main__":
    env = Environment(variant=variant, data_dir=data_dir)

    agent = PPO_Agent()

    
    reward_log, _ = agent.train_ppo(agent)