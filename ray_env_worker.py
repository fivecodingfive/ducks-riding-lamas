import ray

@ray.remote
class EnvWorker:
    def __init__(self, env_kwargs):
        from environment import Environment
        self.env = Environment(**env_kwargs)
        self.current_mode = None

    def reset(self, mode="training"):
        self.current_mode = mode
        return self.env.reset(mode)

    def step(self, action):
        r, next_state, done = self.env.step(int(action))
        if done:
            next_state = self.env.reset(self.current_mode)
        return r, next_state, done
