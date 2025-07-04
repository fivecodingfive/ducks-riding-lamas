import itertools
import os

def get_sweep_config(args):
    """
    Configure hyperparameter sweep for PPO based on SLURM_ARRAY_TASK_ID
    Returns modified args with hyperparameters set according to grid position
    """
    # Define hyperparameter grid
    learning_rates = [0.0001]
    value_lrs = [0.001, 0.0005]
    train_policy_epochs = [5, 10, 40]
    clips = [0.1, 0.15, 0.05]
    entropies = [0.1]
    entropy_decay = [0.995, 0.95, 0.9]
    lams = [0.9, 0.95]
    
    # Create the full grid of combinations
    grid = list(itertools.product(learning_rates, value_lrs, train_policy_epochs, clips, entropies, entropy_decay, lams))
    
    # Get sweep ID from environment or argument
    sweep_id = int(os.getenv("SLURM_ARRAY_TASK_ID", args.sweep_id or 0))
    
    # Ensure sweep_id is within grid bounds
    if sweep_id >= len(grid):
        print(f"Warning: sweep_id {sweep_id} exceeds grid size {len(grid)}. Using modulo.")
        sweep_id = sweep_id % len(grid)
    
    # Set args based on grid position
    args.policy_lr, args.value_lr, args.train_policy_epochs, args.clip, args.entropy, args.entropy_decay, args.lam = grid[sweep_id]
    
    # Print sweep configuration
    print(f"[SWEEP] ID: {sweep_id}/{len(grid)-1} | "
          f"Policy LR: {args.policy_lr} | Value LR: {args.value_lr} | "
          f"Clip: {args.clip} | Entropy: {args.entropy} | Lambda: {args.lam}")
    
    return args, len(grid)

