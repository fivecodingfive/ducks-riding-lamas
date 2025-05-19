# test_vec_env.py
from vec_env import VectorizedEnv

# Passe die Argumente ggf. an deine Environment-Initialisierung an!
N_ENVS = 4
vec_env = VectorizedEnv(n_envs=N_ENVS, variant="0", data_dir='./data')

# Test: Reset
states = vec_env.reset(mode="training")
print("Initial states shape:", states.shape)
print("Initial states:", states.numpy() if hasattr(states, "numpy") else states)

# Test: Schritt mit zufälligen Aktionen
import numpy as np
actions = np.random.randint(0, 5, size=N_ENVS)  # Beispiel: 5 mögliche Aktionen
rewards, next_states, dones = vec_env.step(actions)

print("Rewards:", rewards.numpy() if hasattr(rewards, "numpy") else rewards)
print("Next states shape:", next_states.shape)
print("Dones:", dones.numpy() if hasattr(dones, "numpy") else dones)
