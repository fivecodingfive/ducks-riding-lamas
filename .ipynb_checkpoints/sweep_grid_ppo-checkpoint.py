import itertools
import os

def get_sweep_config(args):
    """
    Configure hyperparameter sweep for PPO based on SLURM_ARRAY_TASK_ID
    Returns modified args with hyperparameters set according to grid position
    """
    # Define hyperparameter grid
    learning_rates = [0.001, 0.0005]
    value_lrs = [0.0008, 0.0012]
    train_epochs = [10, 30]
    clips = [0.3, 0.5]
    entropies = [0]
    entropy_decay = [0.95]
    lams = [0.99, 0.9]
    gamma = [0.95, 0.9]
    seeds = list(range(1))
    
    # Create the full grid of combinations
    grid = list(itertools.product(
        learning_rates, value_lrs, train_epochs,
        clips, entropies, entropy_decay, lams, gamma, seeds))
    
    # Get sweep ID from environment or argument
    sweep_id = int(os.getenv("SLURM_ARRAY_TASK_ID", args.sweep_id or 0))
    
    # Ensure sweep_id is within grid bounds
    if sweep_id >= len(grid):
        print(f"Warning: sweep_id {sweep_id} exceeds grid size {len(grid)}. Using modulo.")
        sweep_id = sweep_id % len(grid)
    
    # Set args based on grid position
    (
        args.policy_lr, args.value_lr, args.train_epochs,
        args.clip, args.entropy, args.entropy_decay, args.lam,
        args.gamma, args.seed                                       
    ) = grid[sweep_id]
    
    # Print sweep configuration
    print(f"[SWEEP] ID: {sweep_id}/{len(grid)-1} | "
          f"Policy LR: {args.policy_lr} | Value LR: {args.value_lr} | "
          f"Clip: {args.clip} | Entropy: {args.entropy} | Lambda: {args.lam} | "
          f"Gamma: {args.gamma} | Seed: {args.seed}")

    return args, len(grid)

