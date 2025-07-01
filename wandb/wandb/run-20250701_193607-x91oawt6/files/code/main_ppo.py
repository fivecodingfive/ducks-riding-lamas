import os, random, time, argparse
from datetime import datetime

import numpy as np
import tensorflow as tf
import wandb

from environment import Environment
from agent_ppo.ppo_agent import PPO_Agent
from agent_ppo.config import ppo_config

# Start timing execution
start_time = time.time()

algo = "ppo"
lr   = ppo_config["policy_learning_rate"]

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
device = "gpu" if tf.config.list_physical_devices('GPU') else "cpu"

organized_cfg = {
    "environment": {
        "variant": args.variant,
        "mode":    args.mode,
        "data_dir": "./data",
    },
    "model": {
        "clip_ratio":       ppo_config["clip_ratio"],
        "entropy_start":    ppo_config["entropy"],
        "entropy_decay":    ppo_config["entropy_decay"],
        "entropy_min":      ppo_config["entropy_min"],
    },
    "training": {
        "episodes":         ppo_config["n_episodes"],
        "gamma":            ppo_config["gamma"],
        "lam":              ppo_config["lam"],
        "policy_lr":        ppo_config["policy_learning_rate"],
        "value_lr":         ppo_config["value_learning_rate"],
        "policy_epochs":    ppo_config["train_policy_epochs"],
        "value_epochs":     ppo_config["train_value_function_epochs"],
        "max_time_steps":   ppo_config["max_time_steps"],
        "seed":             args.seed,
        "device":           device,
    }
}

tags = [
    f"variant{args.variant}",
    f"algo-{algo}",
    f"lr{lr}",
    f"seed{args.seed}",
    device,
]


run_name = f"{algo}_v{args.variant}_{datetime.now():%b%d}"


try:
    run = wandb.init(
        entity="ducks-riding-llamas",
        project="ride-those-llamas",
        name=run_name,
        group = f"v{args.variant}_{algo}",
        config=organized_cfg,
        tags=tags,
        save_code=True,
        dir=os.getenv("WANDB_DIR", "./wandb"),
    )
    if run is not None:
        wandb.define_metric("episode")
        wandb.define_metric("*", step_metric="episode")
except Exception as e:
    print(f"[W&B WARNING] failed to initialise – logging disabled ({e})", flush=True)
    os.environ["WANDB_MODE"] = "disabled"


# ────────────────────────────── run mode ────────────────────────────────────
# (Includes W&B logging)

if args.mode == "training":
    reward_log, _, model_paths = agent.train_ppo(env)       # ← capture return with model paths
    avg = float(np.mean(reward_log))

    # rename the run to include final reward
    if wandb.run is not None:
        wandb.run.name = (
            f"{algo}_v{args.variant}-----Rew:{avg:.1f}_{datetime.now():%b%d}"
        )
        wandb.run.save()
else:
    agent.validate_ppo(env, model_path=args.modelpath)

print(f"Total runtime: {time.time() - start_time:.2f}s", flush=True)

if wandb.run is not None:
    wandb.finish()