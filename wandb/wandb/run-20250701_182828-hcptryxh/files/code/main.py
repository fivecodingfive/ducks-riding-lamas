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

print(">>> [Checkpoint] Script started", flush=True)


# ---------- simple sweep grid (10 combos = array 0-9) ----------
import os, itertools

learning_rates = [1e-3, 5e-4]      # 2 values
batch_sizes    = [64, 128]         # 2 values
eps_decays     = [0.995, 0.98, 0.95]  # 3 values
grid = list(itertools.product(learning_rates, batch_sizes, eps_decays))

task_id = (args.sweep_id if args.sweep_id is not None
           else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))  # 0 when you run locally
lr, bs, eps_decay = grid[task_id]

print(f">>> [Sweep] lr={lr}, batch={bs}, eps_decay={eps_decay}", flush=True)

# overwrite the argparse defaults so the rest of the code sees them
args.learning_rate  = lr
args.batch_size     = bs
args.epsilon_decay  = eps_decay

algo = "dqn"






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

print(f"Args: variant={variant}, episodes={episodes}, mode={mode}, network={NETWORK_TYPE}", flush=True)

print(">>> [Checkpoint] Creating environment", flush=True)
env = Environment(variant=variant, data_dir=data_dir)
print(">>> [Checkpoint] Environment created", flush=True)


target_update_freq = 5  # Or read from args/config later if you wish

from agent_dqn.dqn_agent import DQNAgent
print(">>> [Checkpoint] Creating agent", flush=True)
dqn_agent = DQNAgent(
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    epsilon_decay=args.epsilon_decay,
)
print(">>> [Checkpoint] Agent created", flush=True)



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

print(">>> [Checkpoint] Initializing W&B", flush=True)

run_name = f"{algo}_v{variant}_{datetime.now():%b%d}"

try:
    wandb.init(
        entity="ducks-riding-llamas",
        project="ride-those-llamas",
        name = run_name,
        group = f"v{variant}_{algo}",              # <— unified group with PPO logging
        config = organized_config,
        tags = [
            f"variant{variant}",
            f"algo-{algo}",
            f"lr{lr}",
            f"seed{seed}",
            "cuda" if tf.config.list_physical_devices('GPU') else "cpu",
        ],
        save_code = True,
        dir = os.getenv("WANDB_DIR", "./wandb"),
    )
except Exception as e:
    print(f"[W&B ERROR] Could not initialize W&B: {e}", flush=True)
    os.environ["WANDB_MODE"] = "disabled"




print(">>> [Checkpoint] Entering mode switch", flush=True)

match mode:
    case 'training':
        model_path, reward_log = dqn_agent.train(
            env=env,
            episodes=episodes,
            mode=mode, 
            target_update_freq=target_update_freq
        )
        avg = float(np.mean(reward_log))

        if wandb.run is not None:
            wandb.run.name = (
                f"{algo}_v{variant}-----Rew:{avg:.1f}_{datetime.now():%b%d}"
            )
            wandb.run.save()

    case 'validation':
        if model_path:
            dqn_agent.validate(env=env,
                               model_path=model_path)
        else:
            print("No model param to validate")
