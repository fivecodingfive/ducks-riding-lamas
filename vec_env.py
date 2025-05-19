import tensorflow as tf
from environment import Environment      # dein Original-Env

class VectorizedEnv:
    def __init__(self, n_envs: int, **env_kwargs):
        self.envs   = [Environment(**env_kwargs) for _ in range(n_envs)]
        self.n_envs = n_envs
        self.mode   = None                # wird beim ersten reset gesetzt

    def reset(self, mode="training"):
        self.mode = mode
        states = [env.reset(mode) for env in self.envs]          # tf.Tensor (9,) aus get_obs
        return tf.stack(states)                                  # (n_envs, 9)

    def step(self, actions):
        """actions: 1-D Tensor/ndarray der Länge n_envs"""
        rewards, next_states, dones = [], [], []
        for env, act in zip(self.envs, actions):
            r, next_s, done = env.step(int(act))
            # Wenn Episode vorbei, sofort neue starten,
            # damit Batch-Größe konstant bleibt
            if done:
                next_s = env.reset(self.mode)
            rewards.append(r)
            next_states.append(next_s)
            dones.append(done)
        return (tf.convert_to_tensor(rewards,      dtype=tf.float32),   # (n_envs,)
                tf.stack(next_states),                                  # (n_envs, 9)
                tf.convert_to_tensor(dones,        dtype=tf.float32))   # (n_envs,)