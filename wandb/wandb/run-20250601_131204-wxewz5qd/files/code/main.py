# TODO: parse arguments
from config import args
from config import args

# set seed
seed = args.seed # TODO: set seed to allow for reproducibility of results
seed = args.seed # TODO: set seed to allow for reproducibility of results

import os
os.environ['PYTHONHASHSEED'] = str(seed)

import random
random.seed(seed)

import numpy as np
np.random.seed(seed)

import tensorflow as tf
tf.random.set_seed(seed)

import wandb
from datetime import datetime


# initialize environment
from environment import Environment

NETWORK_TYPE = args.network
data_dir = './data'         # specify relative path to data directory (e.g., './data', not './data/variant_0')
variant = args.variant      # specify problem variant (0 for base variant, 1 for first extension, 2 for second extension)
episodes = args.episodes    # specify episodes
mode = args.mode            # specify mode of agent with different dataset (training, validation, test)
model_path = args.modelpath # specify path to model parameters in ../models folder
env = Environment(variant=variant, data_dir=data_dir)


target_update_freq = 5  # Or read from args/config later if you wish

from agent_dqn.dqn_agent import DQNAgent
dqn_agent = DQNAgent()



# WANDB LOGGING FOR DQN


# -------- build flat config from args -------
config = vars(args)

config.update({
    "learning_rate": dqn_agent.learning_rate,
    "gamma": dqn_agent.gamma,
    "batch_size": dqn_agent.batch_size,
    "buffer_size": dqn_agent.buffer_size,
    "epsilon_start": dqn_agent.epsilon,
    "epsilon_min": dqn_agent.epsilon_min,
    "epsilon_decay": dqn_agent.epsilon_decay,
    "target_update_freq": target_update_freq,
    "prioritized_replay": dqn_agent.use_per,
    "network_type": type(dqn_agent.q_network).__name__,
    "device": "gpu" if tf.config.list_physical_devices('GPU') else "cpu",
})

# -------- nest for nicer UI --------
organized_config = {
    "environment": {
        "variant": config["variant"],
        "data_dir": config["data_dir"],
        "mode": config["mode"],
    },
    "model": {
        "network": config["network"],
        "network_type": config["network_type"],
        "target_update_freq": config["target_update_freq"],
        "prioritized_replay": config["prioritized_replay"],
        "device": config["device"],
    },
    "training": {
        "episodes": config["episodes"],
        "seed": config["seed"],
        "learning_rate": config["learning_rate"],
        "gamma": config["gamma"],
        "batch_size": config["batch_size"],
        "buffer_size": config["buffer_size"],
        "epsilon_start": config["epsilon_start"],
        "epsilon_min": config["epsilon_min"],
        "epsilon_decay": config["epsilon_decay"],
    }
}

wandb.init(
    entity="ducks-riding-llamas", 
    project="ride-those-llamas",
    name = f"{args.algorithm}_variant{args.variant}_{datetime.now():%B}{datetime.now().day}",
    group = f"variant{str(args.variant)}_algorithm{str(args.algorithm)}",
    config=organized_config,
    tags=
    [
        f"variant{args.variant}", 
        f"network{args.network}",
        "replay buffer: prioritized" if dqn_agent.use_per else "replay buffer: uniform",
        "cuda" if tf.config.list_physical_devices('GPU') else "cpu",
        f"seed{args.seed}"
    ],
    save_code=True,
    mode="online",
    dir="./wandb",
    # Change when setting things up for the cluster
)






match mode:
    case 'training':
        model_path = dqn_agent.train(
            env=env,
            episodes=episodes,
            mode=mode, 
            target_update_freq=target_update_freq
            )
    case 'validation':
        if model_path:
            dqn_agent.validate(env=env,
                               model_path=model_path)
        else:
            print("No model param to validate")
