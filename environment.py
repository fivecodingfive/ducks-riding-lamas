# actions: 0 (nothing), 1 (up), 2 (right), 3 (down), 4 (left)

# positions in grid:
# - (0,0) is upper left corner
# - first index is vertical (increasing from top to bottom)
# - second index is horizontal (increasing from left to right)

# if new item appears in a cell into which the agent moves/at which the agent stays in the same time step,
# it is not picked up (if agent wants to pick it up, it has to stay in the cell in the next time step)

import random
import pandas as pd
from copy import deepcopy
from itertools import compress
import numpy as np
import tensorflow as tf

class Environment(object):
    def __init__(self, variant, data_dir):
        self.variant = variant
        self.vertical_cell_count = 5
        self.horizontal_cell_count = 5
        self.vertical_idx_target = 2
        self.horizontal_idx_target = 0
        self.target_loc = (self.vertical_idx_target, self.horizontal_idx_target)
        self.episode_steps = 200
        self.max_response_time = 15 if self.variant == 2 else 10
        self.reward = 25 if self.variant == 2 else 15
        self.data_dir = data_dir
        self.state_dim = 0

        self.training_episodes = pd.read_csv(self.data_dir + f'/variant_{self.variant}/training_episodes.csv')
        self.training_episodes = self.training_episodes.training_episodes.tolist()
        self.validation_episodes = pd.read_csv(self.data_dir + f'/variant_{self.variant}/validation_episodes.csv')
        self.validation_episodes = self.validation_episodes.validation_episodes.tolist()
        self.test_episodes = pd.read_csv(self.data_dir + f'/variant_{self.variant}/test_episodes.csv')
        self.test_episodes = self.test_episodes.test_episodes.tolist()

        self.remaining_training_episodes = deepcopy(self.training_episodes)
        self.validation_episode_counter = 0

        if self.variant == 0 or self.variant == 2:
            self.agent_capacity = 1
        else:
            self.agent_capacity = 3

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
        if mode not in modes:
            raise ValueError('Invalid mode. Expected one of: %s' % modes)

        self.step_count = 0
        self.agent_loc = (self.vertical_idx_target, self.horizontal_idx_target)
        self.agent_load = 0  # number of items loaded (0 or 1, except for first extension, where it can be 0,1,2,3)
        self.item_locs = []
        self.item_times = []

        if mode == "testing":
            episode = self.test_episodes[0]
            self.test_episodes.remove(episode)
        elif mode == "validation":
            episode = self.validation_episodes[self.validation_episode_counter]
            self.validation_episode_counter = (self.validation_episode_counter + 1) % 100
        else:
            if not self.remaining_training_episodes:
                self.remaining_training_episodes = deepcopy(self.training_episodes)
            episode = random.choice(self.remaining_training_episodes)
            self.remaining_training_episodes.remove(episode)
        self.data = pd.read_csv(self.data_dir + f'/variant_{self.variant}/episode_data/episode_{episode:03d}.csv',
                                index_col=0)

        return self.get_obs()

    # take one environment step based on the action act
    def step(self, act):
        self.step_count += 1

        rew = 0

        # done signal (1 if episode ends, 0 if not)
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

            if new_loc in self.eligible_cells:
                self.agent_loc = new_loc
                rew += -1

        # item pick-up
        if (self.agent_load < self.agent_capacity) and (self.agent_loc in self.item_locs):
                self.agent_load += 1
                idx = self.item_locs.index(self.agent_loc)
                self.item_locs.pop(idx)
                self.item_times.pop(idx)
                rew += self.reward / 2

        # item drop-off
        if self.agent_loc == self.target_loc:
            rew += self.agent_load * self.reward / 2
            self.agent_load = 0

        # track how long ago items appeared
        self.item_times = [i + 1 for i in self.item_times]

        # remove items for which max response time is reached
        mask = [i < self.max_response_time for i in self.item_times]
        self.item_locs = list(compress(self.item_locs, mask))
        self.item_times = list(compress(self.item_times, mask))

        # add items which appear in the current time step
        new_items = self.data[self.data.step == self.step_count]
        new_items = list(zip(new_items.vertical_idx, new_items.horizontal_idx))
        new_items = [i for i in new_items if i not in self.item_locs]  # not more than one item per cell
        self.item_locs += new_items
        self.item_times += [0] * len(new_items)

        # get new observation
        next_obs = self.get_obs()

        return rew, next_obs, done

    # TODO: implement function that gives the input features for the neural network(s)
    #       based on the current state of the environment


    # IDEEN
        # Early termination of episode
        # Dont walk to closest item always - better:
            # Smallest distance sum of agent - item - target
            # Dont run to item if the item cannot be reached in time
        # Another idea:
            # Include the top 3 closest items (6 elements for dx/dy + time left for each).
            # Add time left for the closest item to the current input (1 extra element).



    def get_obs(self):
        obs = []

        # Agent's current state: x, y, load
        agent_x, agent_y = self.agent_loc
        obs.append(float(agent_x))
        obs.append(float(agent_y))
        obs.append(float(self.agent_load))

        # Target's relative position (dx, dy)
        target_dx = self.target_loc[0] - agent_x
        target_dy = self.target_loc[1] - agent_y
        obs.append(float(target_dx))
        obs.append(float(target_dy))

        # Number of active items
        num_items = len(self.item_locs)
        obs.append(float(num_items))

        # Collect item information
        items_info = []
        for loc, time in zip(self.item_locs, self.item_times):
            dx = loc[0] - agent_x
            dy = loc[1] - agent_y
            time_remaining = self.max_response_time - time
            distance_to_target = (abs(loc[0] - self.target_loc[0]) +
                                  abs(loc[1] - self.target_loc[1]))
            items_info.append((time_remaining, (dx, dy, time_remaining, distance_to_target)))

        # Sort items by time_remaining (ascending), then by Manhattan distance to agent
        items_info.sort(key=lambda x: (x[0], abs(x[1][0]) + abs(x[1][1])))

        # Select top 3 items, pad with zeros if fewer than 3
        top_items = items_info[:3]
        for _ in range(3):
            if top_items:
                _, item_features = top_items.pop(0)
                obs.extend(item_features)
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0])

        return tf.convert_to_tensor(obs, dtype=tf.float32)

    # GET_OBS WITH 28 element vector
    """def get_obs(self):
        obs = []

        # 1~3: Agent's position (x, y) and load
        agent_x, agent_y = self.agent_loc
        obs.append(float(agent_x))
        obs.append(float(agent_y))
        obs.append(float(self.agent_load))

        # 4~28: Grid cell times (25 cells in row-major order)
        for row in range(5):
            for col in range(5):
                cell = (row, col)
                if cell in self.item_locs:
                    idx = self.item_locs.index(cell)
                    time = self.item_times[idx]
                    remaining_time = float(self.max_response_time - time)  # Correct
                    obs.append(remaining_time)
                else:
                    obs.append(0.0)

        return tf.convert_to_tensor(obs, dtype=tf.float32)  # Shape: (28,)"""




    # get_obs from 16.05
    """def get_obs(self):
        obs = []

        # 1~3: Agent state (x, y, load)
        agent_x, agent_y = self.agent_loc
        obs.extend([
            float(agent_x),
            float(agent_y),
            float(self.agent_load)
        ])

        # 4~5: Relative position to target
        target_x, target_y = self.target_loc  # Use instance variable
        obs.extend([
            float(target_x - agent_x),
            float(target_y - agent_y)
        ])

        # 6~8: Closest item info (dx, dy, time_left)
        if self.item_locs:
            # Find closest item using Manhattan distance
            closest_item = min(self.item_locs,
                               key=lambda pos: abs(pos[0] - agent_x) + abs(pos[1] - agent_y))

            # Get its index to find corresponding time
            idx = self.item_locs.index(closest_item)
            item_time = self.item_times[idx]

            obs.extend([
                float(closest_item[0] - agent_x),
                float(closest_item[1] - agent_y),
                float(self.max_response_time - item_time - 1)  # Time remaining
            ])
        else:
            obs.extend([0.0, 0.0, 0.0])

        # 9: Number of active items
        obs.append(float(len(self.item_locs)))

        return tf.convert_to_tensor(obs, dtype=tf.float32)  # Shape: (9,)"""

