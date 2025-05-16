import random

import numpy as np
import pandas as pd
import tensorflow as tf
from copy import deepcopy
from itertools import compress


# actions: 0 (nothing), 1 (up), 2 (right), 3 (down), 4 (left)

# positions in grid:
# - (0,0) is upper left corner
# - first index is vertical (increasing from top to bottom)
# - second index is horizontal (increasing from left to right)

# if new item appears in a cell into which the agent moves/at which the agent stays in the same time step,
# it is not picked up (if agent wants to pick it up, it has to stay in the cell in the next time step)


# Workflow for an episode
# 1. NN chooses action based on observation
# action = agent_nn.predict(obs)  # e.g., action=2 (move right)
# 2. Execute action in environment
# next_obs, reward, done = env.step(action)
# 3. Store experience (for training)
# memory.append((obs, action, reward, next_obs, done))
# 4. Update observation
# obs = next_obs

# 5. Train NN (After some steps/episodes):
# Sample a batch of past experiences
# batch = memory.sample_batch()
# Adjust NN weights to maximize future rewards
# agent_nn.train(batch)




class Environment(object):
    # No need for the "(object)" in newer Python versions, but it's a good practice for compatibility
    def __init__(self, variant, data_dir):
        # "def" is always used to define a function in Python
        # __init__ is the constructor method
        # "self" is always explicitly required in the constructor
        self.variant = variant
        self.data_dir = data_dir

        # Define the grid size and target location
        self.vertical_cell_count = 5                                                # Number of rows (vertical cells)
        self.horizontal_cell_count = 5                                              # Number of columns (horizontal cells)
        self.vertical_idx_target = 2                                                # Target row (Y-coordinate)
        self.horizontal_idx_target = 0                                              # Target column (X-coordinate)
        self.target_loc = (self.vertical_idx_target, self.horizontal_idx_target)    # Target as (row, col) tuple

        # Episode settings
        self.episode_steps = 200                                                    # Total steps per episode
        self.max_response_time = 15 if self.variant == 2 else 10                    # Time limit to pick up items
        self.reward = 25 if self.variant == 2 else 15                               # Reward value (variant-dependent)

        # Load episode lists from CSV files
        self.training_episodes = pd.read_csv(self.data_dir + f'/variant_{self.variant}/training_episodes.csv')
        self.training_episodes = self.training_episodes.training_episodes.tolist()
        self.validation_episodes = pd.read_csv(self.data_dir + f'/variant_{self.variant}/validation_episodes.csv')
        self.validation_episodes = self.validation_episodes.validation_episodes.tolist()
        self.test_episodes = pd.read_csv(self.data_dir + f'/variant_{self.variant}/test_episodes.csv')
        self.test_episodes = self.test_episodes.test_episodes.tolist()

        # Track remaining episodes and validation counter
        self.remaining_training_episodes = deepcopy(self.training_episodes)
        self.validation_episode_counter = 0

        # Define how many items the agent can hold
        if self.variant == 0 or self.variant == 2:
            self.agent_capacity = 1
        else:
            self.agent_capacity = 3

        # Define which grid cells can contain items
        if self.variant == 0 or self.variant == 1:
            self.eligible_cells = [(0,0), (0,1), (0,2), (0,3), (0,4),
                                   (1,0), (1,1), (1,2), (1,3), (1,4),
                                   (2,0), (2,1), (2,2), (2,3), (2,4),
                                   (3,0), (3,1), (3,2), (3,3), (3,4),
                                   (4,0), (4,1), (4,2), (4,3), (4,4)]
        else:
            self.eligible_cells = [(0,0),        (0,2), (0,3), (0,4),
                                   (1,0),        (1,2),        (1,4),
                                   (2,0),        (2,2),        (2,4),
                                   (3,0), (3,1), (3,2),        (3,4),
                                   (4,0), (4,1), (4,2),        (4,4)]

    # initialize a new episode (specify if training, validation, or testing via the mode argument)
    def reset(self, mode):
        modes = ['training', 'validation', 'testing']

        # 1. Validate mode
        if mode not in modes:
            raise ValueError('Invalid mode. Expected one of: %s' % modes)

        # 2. Reset state
        self.step_count = 0
        self.agent_loc = (self.vertical_idx_target, self.horizontal_idx_target)     # Places agent at target location
        self.agent_load = 0  # Empties the agent's inventory (no carried items)
        # Lists tracking active items and their spawn times are initiated empty at the start of each episode
        self.item_locs = []         # Tracks the locations of items in the grid
        self.item_times = []        # Tracks how many steps ago each item spawned

        # 3. Select episode based on mode
        if mode == "testing":
            # Uses the first episode in self.test_episodes (removes it to avoid reuse).
            episode = self.test_episodes[0]
            self.test_episodes.remove(episode)
        elif mode == "validation":
            # Cycles through self.validation_episodes in order (loops after 100 episodes).
            episode = self.validation_episodes[self.validation_episode_counter]
            self.validation_episode_counter = (self.validation_episode_counter + 1) % 100
        else:
            # Randomly picks an episode from self.remaining_training_episodes. Refills the pool if empty.
            if not self.remaining_training_episodes:
                self.remaining_training_episodes = deepcopy(self.training_episodes)
            episode = random.choice(self.remaining_training_episodes)
            self.remaining_training_episodes.remove(episode)

        # 4. Load episode data from CSV file
        self.data = pd.read_csv(self.data_dir + f'/variant_{self.variant}/episode_data/episode_{episode:03d}.csv',
                                index_col=0)

        # 5. Return observation
        return self.get_obs()


    # take one environment step based on the action act
    def step(self, act):
    # Core function of envrionment: processes the agent's action, updates the environment state, calculates rewards, and checks if the episode is complete

        self.step_count += 1        # Tracks how many steps have been taken in the current episode

        rew = 0                     # Initialize reward tracking

        # Checks if the episode is done when step_count reaches episode_steps
        if self.step_count == self.episode_steps:
            done = 1
        else:
            done = 0

        # agent movement
        if act != 0:
            if act == 1:  # up
                new_loc = (self.agent_loc[0] - 1, self.agent_loc[1])
            elif act == 2:  # right
                new_loc = (self.agent_loc[0], self.agent_loc[1] + 1)
            elif act == 3:  # down
                new_loc = (self.agent_loc[0] + 1, self.agent_loc[1])
            elif act == 4:  # left
                new_loc = (self.agent_loc[0], self.agent_loc[1] - 1)

            # If the new location is valid (eligible_cells), updates agent_loc and applies a penalty (rew -= 1 for moving).
            if new_loc in self.eligible_cells:
                self.agent_loc = new_loc
                rew += -1

        # item pick-up
        if (self.agent_load < self.agent_capacity) and (self.agent_loc in self.item_locs):
        # If the agent has capacity (agent_load < agent_capacity) and is on an item (agent_loc in item_locs)
                self.agent_load += 1                            # Increases the load by 1
                idx = self.item_locs.index(self.agent_loc)      # Finds the index of the item in item_locs
                self.item_locs.pop(idx)                         # Removes the item from item_locs at index found before
                self.item_times.pop(idx)                        # Deletes the timestamp of the picked-up item
                rew += self.reward / 2                          # Adds half the reward for picking up an item (other half granted at drop-off)

        # item drop-off
        if self.agent_loc == self.target_loc:                   # If the agent is at the target location (target_loc)
            rew += self.agent_load * self.reward / 2            # Adds the other half of the reward (multiplied by the number of items in versions beyond the basic one)
            self.agent_load = 0                                 # Resets the agent's load to 0 (all items dropped off)

        # Increments item_times (tracks how long items have existed).
        self.item_times = [i + 1 for i in self.item_times]

        # Removes expired items (those older than max_response_time).
        mask = [i < self.max_response_time for i in self.item_times]
        self.item_locs = list(compress(self.item_locs, mask))
        self.item_times = list(compress(self.item_times, mask))

        # add new items (from CSV data) which appear in the current time step
        new_items = self.data[self.data.step == self.step_count]
        new_items = list(zip(new_items.vertical_idx, new_items.horizontal_idx))
        new_items = [i for i in new_items if i not in self.item_locs]  # not more than one item per cell
        self.item_locs += new_items
        self.item_times += [0] * len(new_items)

        # get new observation
        next_obs = self.get_obs()

        return (
            rew,            # Total reward for this step
            next_obs,       # Updated environment state
            done            # Whether the episode ended
        )



    # TODO: implement function that gives the input features for the neural network(s) based on the current state of the environment

    # The observation is the information you give the agent to make decisions (like video images for a self-driving car).
    # For your grid environment, the agent needs enough information to:
            # Navigate: Know where it is and where it can move.
            # Collect Items: Find items before they expire.
            # Deliver: Remember where the target location is.

    # Key elements to include:
            # Agent State: Current position (agent_loc) and load (agent_load).
            # Item Information: Locations (item_locs) and "urgency" (how long they've been there, derived from item_times).
            # Target Location: Fixed goal (target_loc).
            # Grid Boundaries: Where movement is allowed (eligible_cells).

    # What to avoid
            # Too much data: Don't give the agent every single detail. Just the important stuff.
            # Redundancy: Don't include fixed info (like eligible_cells) in every observation—the agent learns this over time.
            # Raw Data: Preprocess timestamps (e.g., normalize by max_response_time).

    # --> The observation should be a compact representation of the environment state, suitable for input to a neural network.


    def get_obs(self):

        obs = []

        # 1~2: agent position (x, y)
        agent_x, agent_y = self.agent_loc
        obs.append(float(agent_x))
        obs.append(float(agent_y))

        # 3~4: relative position to closest item (dx, dy)
        if self.item_locs:
            closest_item = min(self.item_locs, key=lambda pos: abs(pos[0] - agent_x) + abs(pos[1] - agent_y))
            dx = float(closest_item[0] - agent_x)
            dy = float(closest_item[1] - agent_y)
        else:
            dx, dy = 0.0, 0.0  # 沒 item 就設為 0

        obs.append(dx)
        obs.append(dy)

        # 5: agent load
        obs.append(float(self.agent_load))

        return tf.convert_to_tensor(obs, dtype=tf.float32)  # shape: (5,)



        """
        obs = []

        # One-hot Grid (25 Felder = 5x5)
        grid_size = self.vertical_cell_count * self.horizontal_cell_count

        # 1. Agentenposition (One-hot)
        agent_grid = [0] * grid_size
        agent_idx = self.agent_loc[0] * self.horizontal_cell_count + self.agent_loc[1]
        agent_grid[agent_idx] = 1
        obs.extend(agent_grid)

        # 2. Zielposition (One-hot)
        target_grid = [0] * grid_size
        target_idx = self.target_loc[0] * self.horizontal_cell_count + self.target_loc[1]
        target_grid[target_idx] = 1
        obs.extend(target_grid)

        # 3. Items (Itempositionen mit Alter codiert)
        item_grid = [0] * grid_size
        for loc, time in zip(self.item_locs, self.item_times):
            idx = loc[0] * self.horizontal_cell_count + loc[1]
            # Je älter das Item, desto größer der Wert (optional normalisieren)
            item_grid[idx] = min(1.0, time / self.max_response_time)
        obs.extend(item_grid)

        # 4. Aktuelle Ladung (skalar)
        obs.append(self.agent_load / self.agent_capacity)

        obs_array = np.array(obs, dtype=np.float32)
        assert len(obs_array) == 76, \
            f"Invalid observation size: {len(obs_array)}. Expected 76"
        return obs_array
    """


