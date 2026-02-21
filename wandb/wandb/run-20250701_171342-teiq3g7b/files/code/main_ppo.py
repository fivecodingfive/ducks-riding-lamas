import os, random, time, argparse
from datetime import datetime

import numpy as np
import tensorflow as tf
import wandb

from environment import Environment
from agent_ppo.ppo_agent import PPO_Agent
from agent_ppo.config import ppo_config

# ─────────────────────────────── CLI ────────────────────────────────────────
parser = argparse.ArgumentParser(description="PPO runner with W&B logging")
parser.add_argument('--variant',   type=int,   default=0,                    help='environment variant to load')
parser.add_argument('--mode',      choices=['training', 'validation'],      default='training')
parser.add_argument('--modelpath', type=str,  default='./models/latest.keras', help='actor model to load when validating')
parser.add_argument('--episodes',  type=int,  default=ppo_config["n_episodes"], help='override episode count in config')
parser.add_argument('--seed',      type=int,  default=42)
args = parser.parse_args()

# ─────────────────────── reproducibility & device ───────────────────────────
os.environ['PYTHONHASHSEED'] = str(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"     # force CPU for fair comparison
random.seed(args.seed)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)

# ─────────────────────────── env & agent setup ─────────────────────────────
ppo_config["n_episodes"] = args.episodes      # allow CLI override
env   = Environment(variant=args.variant, data_dir='./data')
agent = PPO_Agent(config=ppo_config)

# ───────────────────────────── W&B init ─────────────────────────────────────
organized_cfg = {
    "environment": {
        "variant": args.variant,
        "mode":    args.mode,
        "data_dir": "./data",
    },
    "model": {
        "state_size":  ppo_config["state_size"],
        "action_size": ppo_config["action_size"],
    },
    "training": {
        "episodes":         ppo_config["n_episodes"],
        "gamma":            ppo_config["gamma"],
        "lam":              ppo_config["lam"],
        "entropy_start":    ppo_config["entropy"],
        "clip_ratio":       ppo_config["clip_ratio"],
        "policy_lr":        ppo_config["policy_learning_rate"],
        "value_lr":         ppo_config["value_learning_rate"],
    },
    "seed": args.seed,
}

run_name = f"ppo_lr{ppo_config['policy_learning_rate']}_{datetime.now():%b%d}"

try:
    wandb.init(
        entity="ducks-riding-llamas",
        project="ride-those-llamas",
        name=run_name,
        group=f"variant{args.variant}_ppo",
        config=organized_cfg,
        tags=[f"variant{args.variant}", f"mode-{args.mode}", f"seed{args.seed}"],
        save_code=True,
        dir=os.getenv("WANDB_DIR", "./wandb"),
    )
except Exception as e:
    print(f"[W&B WARNING] failed to initialise – logging disabled ({e})", flush=True)
    os.environ["WANDB_MODE"] = "disabled"

# ────────────────────────────── run mode ────────────────────────────────────
start_time = time.time()
if args.mode == "training":
    agent.train_ppo(env)
else:
    agent.validate_ppo(env, model_path=args.modelpath)
print(f"Total runtime: {time.time() - start_time:.2f}s", flush=True)

if wandb.run is not None:
    wandb.finish()





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
