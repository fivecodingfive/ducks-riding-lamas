import numpy as np
from environment_with_comments import Environment


def test_environment():
    try:
        # Initialize with correct paths
        env = Environment(variant=0, data_dir='./data')

        # Test reset
        obs = env.reset("validation")
        print("Initial observation:", obs)

        # Test step
        action = 2  # Try moving right
        reward, next_obs, done = env.step(action)

        print("After moving right:")
        print("Reward:", reward)
        print("New observation:", next_obs)
        print("Done:", done)

        # Test observation properties
        print("\nObservation details:")
        print("Type:", type(next_obs))
        print("Shape:", next_obs.shape if hasattr(next_obs, 'shape') else "No shape")
        print("Min:", np.min(next_obs))
        print("Max:", np.max(next_obs))

    except Exception as e:
        print("Error:", str(e))
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_environment()


# Outcomes
    # After reset
        # Initial observation: [0.4 0.  0.  0.4 0. ]
            # 0.4 = Agent's row position (row 2: 2/5=0.4)
            # 0.0 = Agent's column position (column 0)
            # 0.0 = Empty inventory (0 items carried)
            # 0.4 = Target row position (same as agent's start)
            # 0.0 = Target column position
    # After moving right:
        # Reward: -1.0
        # New observation: [0.4 0.2 0.  0.4 0. ]
            # ...
            # 0.2 = Agent's new column position (moved right to column 1 (1/5=0.2))
            # ...
        # Done: 0
    # Observation details:
        # Type: <class 'numpy.ndarray'>     # Observation is a NumPy array - Neural networks need input in this format
        # Shape: (5,)                       # The array has 5 numbers total (1. Agent row position, 2. Agent column position, 3. Agent load, 4. Target row position, 5. Target column position)
        # Min: 0.0                          # Min number
        # Max: 0.4                          # Max number