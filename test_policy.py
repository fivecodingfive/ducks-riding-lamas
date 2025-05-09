# compute average reward per test episode with trained policy

"""
from environment import Environment


def test_policy(env):
    test_rew = 0.  # initialize reward tracking

    for i in range(100):  # loop over 100 test episodes
        obs = env.reset('testing')  # get initial obs

        for j in range(200):  # loop over 200 steps per episode
            act = ...  # TODO: get action for the obs from your trained policy
            rew, next_obs, _ = env.step(act)  # take one step in the environment
            test_rew += rew  # track rewards
            obs = next_obs  # continue from the new obs

    avg_test_rew = test_rew / 100  # compute the average reward per episode

    print(avg_test_rew)  # print the result


if __name__ == '__main__':

    data_dir = './data'
    variant = 0
    env = Environment(variant=variant, data_dir=data_dir)  # initialize the environment
    model_path = f"models/dqn_variant{variant}.weights.h5"

    test_policy(env, model_path)  # test the trained policy
"""

import numpy as np
from environment import Environment
from agents.dqn.model import build_q_network

def test_policy(env, model_path):
    # Get observation shape and action space
    obs = env.reset("validation")
    state_size = len(obs)
    action_size = 5  # stay, up, right, down, left

    # Build model and load trained weights
    model = build_q_network((state_size,), action_size)
    model.load_weights(model_path)

    total_reward = 0.0

    # Test for 100 episodes
    for episode in range(100):
        obs = env.reset("validation")
        episode_reward = 0.0

        for _ in range(200):  # max steps per episode
            obs_input = obs.reshape(1, -1)
            q_values = model(obs_input)
            act = int(np.argmax(q_values.numpy()[0]))

            reward, next_obs, done = env.step(act)
            episode_reward += reward
            obs = next_obs

            if done:
                break

        total_reward += episode_reward

    avg_reward = total_reward / 100
    print(f"✅ Average reward over 100 test episodes: {avg_reward:.2f}")


if __name__ == "__main__":
    data_dir = "./data"
    variant = 0
    model_path = f"models/dqn_variant{variant}.weights.h5"

    env = Environment(variant=variant, data_dir=data_dir)
    test_policy(env, model_path)
