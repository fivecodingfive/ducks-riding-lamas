# TODO: parse arguments
# Gets interesting later, when we have more models and agents


# set seed
seed = 29

import os
os.environ['PYTHONHASHSEED'] = str(seed)

import random
random.seed(seed)

import numpy as np
np.random.seed(seed)

import tensorflow as tf
tf.random.set_seed(seed)


# initialize environment
from environment import Environment

# initialize dqn
from agents.dqn.train import train_dqn

data_dir = './data'
variant = 0
env = Environment(variant, data_dir)

# --- Define DQN configuration ---
config = {
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.995,
    "batch_size": 32,
    "learning_rate": 0.001,
    "target_model_update_freq": 20,
    "memory_size": 2000,
    "episodes": 500,
    "max_steps": 200
}

# --- Train DQN agent ---
print(f"🚀 Starting DQN training on variant {variant} for {config['episodes']} episodes...")
rewards = train_dqn(env, config)


