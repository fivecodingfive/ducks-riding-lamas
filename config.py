import argparse

parser = argparse.ArgumentParser(description="Train Tabular Q-learning Agent on GridWorld")

parser.add_argument('--variant', type=int, default=2, choices=[0, 1, 2],
                    help="Environment variant: 0 (base), 1 (extension 1), 2 (extension 2)")

parser.add_argument('--data_dir', type=str, default='./data',
                    help="Path to the data directory (e.g., ./data)")

parser.add_argument('--episodes', type=int, default=300,
                    help="Number of training episodes")

parser.add_argument('--seed', type=int, default=42,
                    help="Random seed for reproducibility")

parser.add_argument('--mode', type=str, default='training', choices=['training', 'validation', 'testing'],
                    help="Run mode for environment")

parser.add_argument('--modelpath', type=str,
                    help="Path to model parameters")

parser.add_argument('--network', type=str, default='mlp', choices=['mlp', 'cnn', 'combine'],
                    help="Type of neural network to use")

parser.add_argument('--algorithm', type=str, default='sac', choices=['dqn', 'ppo', 'a2c', 'sac'],
                    help="Type of agent to use")

parser.add_argument('--per', type=int, default=1, choices=[0, 1],
                    help="Prioritizaed replay buffer")

parser.add_argument('--alpha', type=float, default=0.4,
                    help="Entropy regularization")

parser.add_argument('--sweep_id', type=int, default=None,
                    help="Optional sweep index from SLURM")                    

args = parser.parse_args()