# TODO: parse arguments
from config import args

# set seed
seed = args.seed # TODO: set seed to allow for reproducibility of results

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

NETWORK_TYPE = args.network
data_dir = './data'         # specify relative path to data directory (e.g., './data', not './data/variant_0')
variant = args.variant      # specify problem variant (0 for base variant, 1 for first extension, 2 for second extension)
episodes = args.episodes    # specify episodes
mode = args.mode            # specify mode of agent with different dataset (training, validation, test)
model_path = args.modelpath # specify path to model parameters in ../models folder
env = Environment(variant=variant, data_dir=data_dir)



# TODO: execute training
# from agent import TabularQAgent
# agent = TabularQAgent()
# agent.train(env=env, mode=mode, episodes=episodes)

from agent_dqn.dqn_agent import DQNAgent
dqn_agent = DQNAgent()
match mode:
    case 'training':
        model_path = dqn_agent.train(env=env,
                                    episodes=episodes,
                                    mode=mode, 
                                    target_update_freq=5)
    case 'validation':
        if model_path:
            dqn_agent.validate(env=env,
                               model_path=model_path)
        else:
            print("No model param to validate")
