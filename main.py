import os
import yaml
import random
import numpy as np
import tensorflow as tf
from environment import Environment
from agents.dqn.train import train_dqn

# Configuration parameters
SEED = 29
VARIANT = 0
CODE = 0
CONFIG_FILE = f"configs/variant{VARIANT}/option{CODE}.yaml"

# CONFIG_FILE = f"configs/variant{VARIANT}/option{CODE}.yaml"


# Set seeds
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# Load config
with open(CONFIG_FILE) as f:
    config = yaml.safe_load(f)

# Initialize environment
env = Environment(VARIANT, './data')

# Run training
print(f"🚀 Starting DQN training for variant {VARIANT}, code {CODE}")
rewards = train_dqn(env, config)
