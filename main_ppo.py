"""
main_ppo.py – unified entry point for PPO training / validation

• **Local runs** → logs to Weights & Biases (W&B) by default.
• **Cluster runs** → if SLURM script exports TB_LOGDIR and sets WANDB_MODE=disabled,
  W&B is skipped and TensorBoard event files are written to $TB_LOGDIR.

You can combine both: leave WANDB_MODE unset, export TB_LOGDIR → logs to both.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

# Optional W&B – gracefully degrades if unavailable or disabled -----------------
try:
    import wandb  # type: ignore
except ModuleNotFoundError:  # noqa: D401 (flake8-docstrings complain)
    wandb = None  # W&B not installed

# TensorBoard writer (works for both TF & PyTorch) ------------------------------
#from torch.utils.tensorboard import SummaryWriter  # type: ignore

from environment import Environment
from agent_ppo.ppo_agent import PPO_Agent
from agent_ppo.config import ppo_config

# ═══════════════════════════════════════════════════════════════════════════════
# 1 · CLI -----------------------------------------------------------------------
# ═══════════════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(description="PPO Agent runner")
parser.add_argument("--mode", choices=["training", "validation"], default="training")
parser.add_argument("--modelpath", type=str, help="model for validation run")
parser.add_argument("--episodes", type=int, help="override episode count")
parser.add_argument("--seed", type=int, help="random seed (default: 42)")
parser.add_argument("--variant", type=int, default=1, help="env variant")
parser.add_argument("--sweep_id", type=int, help="grid‑sweep index")
parser.add_argument("--train_epochs", type=int, help="override train  epochs")
# Hyper‑parameters (optional; if omitted fall back to config)
parser.add_argument("--policy_lr", type=float)
parser.add_argument("--value_lr", type=float)
parser.add_argument("--clip", type=float)
parser.add_argument("--entropy", type=float)
parser.add_argument("--entropy_decay", type=float)
parser.add_argument("--lam", type=float)
parser.add_argument("--gamma", type=float)
args = parser.parse_args()

# ═══════════════════════════════════════════════════════════════════════════════
# 2 · Sweep grid handling --------------------------------------------------------
# ═══════════════════════════════════════════════════════════════════════════════

grid_size: int = 0
if args.sweep_id is not None or "SLURM_ARRAY_TASK_ID" in os.environ:
    try:
        from sweep_grid_ppo import get_sweep_config  # late import to avoid cost when unused

        args, grid_size = get_sweep_config(args)  # mutates args in‑place
        print(f"[SWEEP] configuration {args.sweep_id} / {grid_size}")
    except ImportError:
        print("[SWEEP] sweep_grid_ppo.py not found – using provided CLI params only")

# ═══════════════════════════════════════════════════════════════════════════════
# 3 · Reproducibility & device ---------------------------------------------------
# ═══════════════════════════════════════════════════════════════════════════════

seed = args.seed or 42
np.random.seed(seed)
tf.random.set_seed(seed)
print(f"Seed set to {seed}")

device_kind = "gpu" if tf.config.list_physical_devices("GPU") else "cpu"
print(f"Running on {device_kind.upper()}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4 · Config patching ------------------------------------------------------------
# ═══════════════════════════════════════════════════════════════════════════════

def override(cfg_key: str, cli_val, default_dict):
    if cli_val is not None:
        default_dict[cfg_key] = cli_val

override("n_episodes", args.episodes, ppo_config)
override("policy_learning_rate", args.policy_lr, ppo_config)
override("value_learning_rate", args.value_lr, ppo_config)
override("clip_ratio", args.clip, ppo_config)
override("entropy", args.entropy, ppo_config)
override("entropy_decay", args.entropy_decay, ppo_config)
override("lam", args.lam, ppo_config)
override("train_epochs", args.train_epochs, ppo_config)
override("gamma", args.gamma, ppo_config)

# ═══════════════════════════════════════════════════════════════════════════════
# 5 · Environment & Agent --------------------------------------------------------
# ═══════════════════════════════════════════════════════════════════════════════

env = Environment(variant=args.variant, data_dir="./data")
agent = PPO_Agent(config=ppo_config)

# ═══════════════════════════════════════════════════════════════════════════════
# 6 · Logger selection (W&B vs TensorBoard) -------------------------------------
# ═══════════════════════════════════════════════════════════════════════════════

USE_TB: bool = bool(os.getenv("TB_LOGDIR"))
USE_WANDB: bool = bool(wandb) and os.getenv("WANDB_MODE", "online") != "disabled"

# ── W&B ------------------------------------------------------------------------
wandb_run = None
if USE_WANDB:
    run_name = f"ppo_v{args.variant}_seed{args.seed}_{datetime.now():%b%d_%H%M%S}"
    try:
        wandb_run = wandb.init(
            entity="ducks-riding-llamas",
            project="ride-those-llamas",
            name=run_name,
            group= "new",
            config={
                "env_variant": args.variant,
                "ppo_config": ppo_config,
                "seed": seed,
                "device": device_kind,
                "sweep_id": args.sweep_id,
            },
            tags=[
                f"variant{args.variant}",
                f"device-{device_kind}",
                "cluster" if "SLURM_JOB_ID" in os.environ else "local",
            ],
            dir=os.getenv("WANDB_DIR", "./wandb"),
            save_code=True,
        )
        # metrics keyed by episode
        wandb.define_metric("episode")
        wandb.define_metric("*", step_metric="episode")
        print("[W&B] run initialised")
    except Exception as exc:
        print(f"[W&B] init failed → disabled ({exc})")
        USE_WANDB = False
        wandb_run = None

# ═══════════════════════════════════════════════════════════════════════════════
# 7 · Train / Validate -----------------------------------------------------------
# ═══════════════════════════════════════════════════════════════════════════════

start_wall = time.time()

if args.mode == "training":
    reward_log, _, _ = agent.train_ppo(env)

    # Log to W&B
    if USE_WANDB and wandb_run is not None:
        for ep, rew in enumerate(reward_log):
            wandb.log({"episode": ep, "reward": rew})
        wandb_run.name = (
            f"ppo_v{args.variant}_avg{float(np.mean(reward_log)):.1f}_{datetime.now():%b%d}"  # rename with final reward
        )
        #wandb_run.save()

else:  # validation mode
    agent.validate_ppo(env, model_path=args.modelpath)

print(f"Total runtime: {time.time() - start_wall:.2f}s", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 8 · Clean‑up -------------------------------------------------------------------
# ═══════════════════════════════════════════════════════════════════════════════

wandb.finish()
