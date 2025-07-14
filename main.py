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

import wandb
from datetime import datetime
from environment import Environment

NETWORK_TYPE = args.network
data_dir = './data'         # specify relative path to data directory (e.g., './data', not './data/variant_0')
variant = args.variant      # specify problem variant (0 for base variant, 1 for first extension, 2 for second extension)
episodes = args.episodes    # specify episodes
mode = args.mode            # specify mode of agent with different dataset (training, validation, test)
model_path = args.modelpath # specify path to model parameters in ../models folder
agent = args.algorithm
per = args.per
alpha = args.alpha

print(f">> [Args] variant={variant}, episodes={episodes}, alpha={alpha}", flush=True)

target_update_freq = 5
print(">>> [Checkpoint] Creating agent", flush=True)
match agent:
    case "dqn":
        from agent_dqn.dqn_agent import DQNAgent
        agent = DQNAgent(
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            epsilon_decay=args.epsilon_decay,
        )
        config = vars(args)
        config.update({
            "learning_rate": agent.learning_rate,
            "gamma": agent.gamma,
            "batch_size": agent.batch_size,
            "buffer_size": agent.buffer_size,
            "epsilon_start": agent.epsilon,
            "epsilon_min": agent.epsilon_min,
            "epsilon_decay": agent.epsilon_decay,
            "target_update_freq": target_update_freq,
            "prioritized_replay": agent.use_per,
            "network_type": type(agent.q_network).__name__,
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

    case "sac":
        from agent_sac.sac_agent import SACAgent
        agent = SACAgent(
            learning_rate=args.learning_rate,
            alpha=args.alpha
        )
        config = vars(args)
        config.update({
            "learning_rate": agent.learning_rate,
            "gamma": agent.gamma,
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
                "device": config["device"],
            },
            "training": {
                "episodes": config["episodes"],
                "seed": config["seed"],
                "learning_rate": config["learning_rate"],
                "gamma": config["gamma"],
            }
        }
    case "a2c":
        from agent_sac.a2c_agent import A2CAgent
        agent = A2CAgent(
            learning_rate=args.learning_rate,
            alpha=args.alpha
        )
        config = vars(args)
        config.update({
            "learning_rate": agent.learning_rate,
            "gamma": agent.gamma,
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
                "device": config["device"],
            },
            "training": {
                "episodes": config["episodes"],
                "seed": config["seed"],
                "learning_rate": config["learning_rate"],
                "gamma": config["gamma"],
            }
        }
print(">>> [Checkpoint] Agent created", flush=True)



# WANDB LOGGING FOR DQN


# -------- build flat config from args -------
print(">>> [Checkpoint] Initializing W&B", flush=True)

run_name = (
    f"var{args.variant}_alpha{args.alpha}_"
    f"decay{args.epsilon_decay}_{datetime.now():%b%d}"
)

try:
    wandb.init(
        entity="ducks-riding-llamas", 
        project="ride-those-llamas",
        name = run_name,
        group=f"variant{str(args.variant)}_algorithm{str(args.algorithm)}",
        config=organized_config,
        tags=[
            f"variant{args.variant}", 
            f"network{args.network}",
            # "replay buffer: prioritized" if dqn_agent.use_per else "replay buffer: uniform",
            "cuda" if tf.config.list_physical_devices('GPU') else "cpu",
            f"seed{args.seed}"
        ],
        save_code=True,
        dir=os.getenv("WANDB_DIR", "./wandb")
    )
except Exception as e:
    print(f"[W&B ERROR] Could not initialize W&B: {e}", flush=True)
    os.environ["WANDB_MODE"] = "disabled"


print(">>> [Checkpoint] Entering mode switch", flush=True)

match mode:
    case 'training':
        train_env = Environment(variant=variant, data_dir=data_dir)
        agent.train(
            env=train_env,
            episodes=episodes,
            target_update_freq=target_update_freq
            )
        agent.save_model(variant)
    case 'validation':
        env = Environment(variant=variant, data_dir=data_dir)
        agent.validate(
            env=env,
            model_path=model_path
        )
