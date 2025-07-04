
ppo_config = {
    "state_size": 10,
    "action_size": 5,

    "gamma": 0.99, # discount factor -> if high, future rewards are more important, and if low, immediate rewards are more important
    "lam": 0.9,
    "entropy": 0.05,
    "entropy_decay": 0.995,  # decays every 5th episode with given factor
    "entropy_min": 0.01,  # minimum entropy value
    "clip_ratio": 0.2,
    "train_policy_epochs": 20,
    "train_value_function_epochs": 20,
    "policy_learning_rate": 0.0005, # Smaller learning rate -> Smaller updates (risk of getting stuck if too small)
    "value_learning_rate": 0.0005,
    "max_time_steps": 200,
    "rollout_steps":   2048,


    "n_episodes": 500	
}

