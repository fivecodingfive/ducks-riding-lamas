import os
import time
import numpy as np
import tensorflow as tf
import argparse
import wandb
from datetime import datetime

from environment import Environment
from agent_ppo.ppo_agent import PPO_Agent
from agent_ppo.config import ppo_config

start_time = time.time()

algo = "ppo"

# ─────────────────────────── command line args ───────────────────────────────
parser = argparse.ArgumentParser(description="PPO Agent")
parser.add_argument("--mode", type=str, default="training", help="training or validation")
parser.add_argument("--modelpath", type=str, default=None, help="path to model for validation")
parser.add_argument("--episodes", type=int, default=None, help="number of episodes")
parser.add_argument("--seed", type=int, default=None, help="random seed")
parser.add_argument("--variant", type=int, default=0, help="variant number")
parser.add_argument("--sweep_id", type=int, default=None, help="sweep id")

# PPO-specific hyperparameters for sweeping
parser.add_argument("--policy_lr", type=float, default=None, help="policy learning rate")
parser.add_argument("--value_lr", type=float, default=None, help="value function learning rate")
parser.add_argument("--clip", type=float, default=None, help="PPO clip ratio")
parser.add_argument("--entropy", type=float, default=None, help="entropy coefficient")
parser.add_argument("--entropy_decay", type=float, default=None, help="entropy decay rate")
parser.add_argument("--lam", type=float, default=None, help="GAE lambda parameter")
parser.add_argument("--train_policy_epochs", type=int, default=None, help="policy training epochs")
parser.add_argument("--train_value_epochs", type=int, default=None, help="value training epochs")

args = parser.parse_args()

# ─────────────────────────── handle sweep if needed ───────────────────────────
grid_size = 0
if args.sweep_id is not None or "SLURM_ARRAY_TASK_ID" in os.environ:
    try:
        from sweep_grid_ppo import get_sweep_config
        args, grid_size = get_sweep_config(args)
        print(f"Running as part of sweep ({grid_size} total configurations)")
    except ImportError:
        print("Warning: sweep_grid_ppo.py not found. Using fallback sweep configuration.")
        # Fallback sweep configuration if the import fails
        import itertools
        
        # Define a minimal grid with just a few values
        learning_rates = [3e-4, 1e-4]
        value_lrs = [1e-3]
        clips = [0.2]
        entropies = [0.05]
        lams = [0.95]
        
        grid = list(itertools.product(learning_rates, value_lrs, clips, entropies, lams))
        sweep_id = int(os.getenv("SLURM_ARRAY_TASK_ID", args.sweep_id or 0))
        sweep_id = sweep_id % len(grid)
        
        args.policy_lr, args.value_lr, args.clip, args.entropy, args.lam = grid[sweep_id]
        grid_size = len(grid)
        
        print(f"[FALLBACK SWEEP] ID: {sweep_id}/{grid_size-1} | "
              f"Policy LR: {args.policy_lr} | Value LR: {args.value_lr} | "
              f"Clip: {args.clip} | Entropy: {args.entropy} | Lambda: {args.lam}")

# ─────────────────────── reproducibility & device ───────────────────────────
if args.seed is None:
    args.seed = 42
print(f"Using seed: {args.seed}")
np.random.seed(args.seed)
tf.random.set_seed(args.seed)

# ─────────────────────────── env & agent setup ─────────────────────────────
ppo_config["n_episodes"] = args.episodes if args.episodes is not None else ppo_config["n_episodes"]
ppo_config["policy_learning_rate"] = args.policy_lr if args.policy_lr is not None else ppo_config["policy_learning_rate"]
ppo_config["value_learning_rate"] = args.value_lr if args.value_lr is not None else ppo_config["value_learning_rate"]
ppo_config["clip_ratio"] = args.clip if args.clip is not None else ppo_config["clip_ratio"]
ppo_config["entropy"] = args.entropy if args.entropy is not None else ppo_config["entropy"]
ppo_config["entropy_decay"] = args.entropy_decay if args.entropy_decay is not None else ppo_config["entropy_decay"]
ppo_config["lam"] = args.lam if args.lam is not None else ppo_config["lam"]
ppo_config["train_policy_epochs"] = args.train_policy_epochs if args.train_policy_epochs is not None else ppo_config["train_policy_epochs"]
ppo_config["train_value_function_epochs"] = args.train_value_epochs if args.train_value_epochs is not None else ppo_config["train_value_function_epochs"]

env   = Environment(variant=args.variant, data_dir='./data')
agent = PPO_Agent(config=ppo_config)

# ───────────────────────────── W&B init ─────────────────────────────────────
device = "gpu" if tf.config.list_physical_devices('GPU') else "cpu"

# Organize config for W&B
organized_cfg = {
    "env": {
        "variant": args.variant,
    },
    "ppo": {
        "gamma": ppo_config["gamma"],
        "lam": ppo_config["lam"],
        "entropy": ppo_config["entropy"],
        "entropy_decay": ppo_config["entropy_decay"],
        "entropy_min": ppo_config["entropy_min"],
        "clip_ratio": ppo_config["clip_ratio"],
        "train_policy_epochs": ppo_config["train_policy_epochs"],
        "train_value_function_epochs": ppo_config["train_value_function_epochs"],
        "policy_lr": ppo_config["policy_learning_rate"],
        "value_lr": ppo_config["value_learning_rate"],
        "episodes": ppo_config["n_episodes"],
    },
    "system": {
        "device": device,
        "seed": args.seed,
    }
}

# Add sweep info if applicable
if grid_size > 0:
    organized_cfg["sweep"] = {
        "id": args.sweep_id or int(os.getenv("SLURM_ARRAY_TASK_ID", 0)),
        "total_configs": grid_size,
    }

# Set tags for run
tags = [
    f"variant{args.variant}",
    f"algo-{algo}",
    f"policy_lr{ppo_config['policy_learning_rate']}",
    f"value_lr{ppo_config['value_learning_rate']}",
    f"clip{ppo_config['clip_ratio']}",
    f"entropy{ppo_config['entropy']}",
    f"lam{ppo_config['lam']}",
    f"seed{args.seed}",
    device,
]

# Add sweep tag if applicable
if grid_size > 0:
    tags.append("sweep")
    
if "SLURM_JOB_ID" in os.environ:
    tags.append("cluster")

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