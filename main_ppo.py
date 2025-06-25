from agent_ppo.ppo_agent import PPO_Agent
from environment import Environment
import argparse
from agent_ppo.config import ppo_config
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # to disable GPU usage
seed = 42 
import random
random.seed(seed)
import numpy as np
np.random.seed(seed)
import tensorflow as tf
tf.random.set_seed(seed)

data_dir = './data'        
mode = 'training'  # 'training' or 'validation'
variant = 0     
model_path = './models/ppo_agent_11_reward89.76.keras'  
# episodes defined in ppo_config
# episodes = 200    # specify episodes


if __name__ == "__main__":
    start_time = time.time()

    env = Environment(variant=variant, data_dir=data_dir)

    agent = PPO_Agent(config=ppo_config)
    if mode == 'validation':
        agent.validate_ppo(env, model_path=model_path)
    elif mode == 'training':
        agent.train_ppo(env)

    print(f"Total time taken: {time.time() - start_time:.2f} seconds")
