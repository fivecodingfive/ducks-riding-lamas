
ppo_config = {
    "state_size": 10,
    "action_size": 5,

    "gamma": 0.99,
    "lam": 0.9,
    
    "clip_ratio": 0.2,
    "train_policy_epochs": 40,
    "train_value_function_epochs": 40,
    "policy_learning_rate": 0.0003,
    "value_learning_rate": 0.001,
    "max_time_steps": 200,

    "n_episodes": 200
}

