"""# compute average reward per test episode with trained policy


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

    data_dir = ...  # TODO: specify relative path to data directory (e.g., './data', not './data/variant_0')
    variant = ...  # TODO: specify problem variant (0 for base variant, 1 for first extension, 2 for second extension)
    env = Environment(variant=variant, data_dir=data_dir)  # initialize the environment

    test_policy(env)  # test the trained policy


"""

from environment import Environment
import numpy as np
from agent_dqn.model import build_q_network


def test_policy(env):
    """Evaluate trained agent on 100 test episodes."""
    test_rew = 0.0
    print_rew = 0.0


    agent = DQNAgent(
        state_dim=9, action_dim=5, learning_rate=0.001,
        gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.9999,
        buffer_size=10000, batch_size=64
    )

    model = build_q_network(9, 5)
    # Build the Q-networ

    model.load_weights(f"models/model_0_reward107.96.keras")  # Load saved weights


    for _ in range(100):  # 100 test episodes
        obs = env.reset(mode='validation')  # Reset environment for new test episode


        for _ in range(200):  # Max 200 steps per episode
            # Get action from trained policy (no exploration)
            q_values = model.predict(obs[np.newaxis], verbose=0)
            act = np.argmax(q_values[0])  # Greedy action selection

            # Environment step
            rew, next_obs, _ = env.step(act)
            test_rew += rew
            print_rew += rew
            obs = next_obs
        print (f"Episode reward: {print_rew:.2f}")
        print_rew = 0.0

    avg_test_rew = test_rew / 100
    print(f"[TEST] Average Reward: {avg_test_rew:.2f}")


if __name__ == '__main__':
    # TODO: Fill these based on your setup
    data_dir = "./data"  # Path to parent data directory (contains variant_0/, variant_1/, etc.)
    variant = 0  # 0=base problem, 1=first extension, 2=second extension

    # Initialize environment and load trained agent
    env = Environment(variant=variant, data_dir=data_dir)

    # --- Load your trained agent here ---
    from agent_dqn.dqn_agent import DQNAgent



    # Run evaluation
    test_policy(env)  # Pass both environment and agent