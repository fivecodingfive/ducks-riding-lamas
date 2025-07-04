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
from config import args
import numpy as np
import tensorflow as tf
from config import args

_df = pd.read_csv("item_spawn_counts.csv", index_col=0)
_counts = _df.to_numpy(dtype=np.float32)
spawn_distribution = _counts / _counts.sum()
network_type = args.network
_df = pd.read_csv("item_spawn_counts.csv", index_col=0)
_counts = _df.to_numpy(dtype=np.float32)
spawn_distribution = _counts / _counts.sum()
network_type = args.network

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

    def distance(self, loc):
        # distance from agent to target
        dist = abs(self.agent_loc[0] - loc[0]) + abs(self.agent_loc[1] - loc[1])
        return dist



    def distance(self, loc):
        # distance from agent to target
        dist = abs(self.agent_loc[0] - loc[0]) + abs(self.agent_loc[1] - loc[1])
        return dist


    def get_obs(self):
            if network_type == 'cnn':
            
                agent_x, agent_y = self.agent_loc
                target_x, target_y = self.target_loc
                grid0 = np.ones(self.vertical_cell_count * self.horizontal_cell_count, dtype=np.float32)#empty_grid
                grid1 = np.zeros(self.vertical_cell_count * self.horizontal_cell_count, dtype=np.float32)#target
                grid2 = np.zeros(self.vertical_cell_count * self.horizontal_cell_count, dtype=np.float32)#agent
                grid3 = np.zeros(self.vertical_cell_count * self.horizontal_cell_count, dtype=np.float32)#items
            

                grid1[target_x * self.horizontal_cell_count + target_y] = 1.0 
                grid0[target_x * self.horizontal_cell_count + target_y] = 0.0
                if self.agent_load == 0:
                    grid2[agent_x * self.horizontal_cell_count + agent_y] = 1.0
                    grid0[agent_x * self.horizontal_cell_count + agent_y] = 0.0
                else:
                    grid2[agent_x * self.horizontal_cell_count + agent_y] = 0.5
                    grid0[agent_x * self.horizontal_cell_count + agent_y] = 0.0

                for loc, time in zip(self.item_locs, self.item_times):
                    idx = loc[0] * self.horizontal_cell_count + loc[1]
                    dist = self.distance(loc)
                    if time + dist < self.max_response_time:
                        grid3[idx] = 1.0
                        grid0[idx] = 0.0
                        
                grid0 = grid0.reshape(self.vertical_cell_count, self.horizontal_cell_count)
                grid1 = grid1.reshape(self.vertical_cell_count, self.horizontal_cell_count)
                grid2 = grid2.reshape(self.vertical_cell_count, self.horizontal_cell_count)
                grid3 = grid3.reshape(self.vertical_cell_count, self.horizontal_cell_count)
                obs = np.stack([grid0, grid1, grid2, grid3], axis=-1)  # shape: (5, 5, 4)
                obs = tf.convert_to_tensor(obs, dtype=tf.float32)

                return tf.reshape(obs, [-1])

            elif network_type == 'mlp':
                obs = []
                agent_y, agent_x = self.agent_loc
                obs.extend([float(agent_x), float(agent_y)])  # Position
                obs.append(float(self.agent_load))            # Load

                # Default: kein Item → nutze spawn_distribution
                use_distribution = len(self.item_locs) == 0
                dx, dy = 0.0, 0.0

                if not use_distribution:
                # Suche bestes erreichbares Item
                    min_step = float('inf')
                    for i, (iy, ix) in enumerate(self.item_locs):
                        time_left = self.max_response_time - self.item_times[i]
                        step_cost = abs(agent_x - ix) + abs(agent_y - iy)
                        if time_left >= step_cost and step_cost < min_step:
                            dx = ix - agent_x
                            dy = iy - agent_y
                            min_step = step_cost

                obs.extend([dx, dy])  # Richtung zum besten Item oder (0, 0)

                # Entweder echte Verteilung oder Dummy
                if use_distribution:
                    obs.extend(spawn_distribution[0, :].tolist())  # z. B. Zeile 0 nehmen
                else:
                    obs.extend([0.0] * 5)

                return tf.convert_to_tensor(obs, dtype=tf.float32)
            
            elif network_type == 'combine':
                agent_x, agent_y = self.agent_loc
                target_x, target_y = self.target_loc
                grid0 = np.ones(self.vertical_cell_count * self.horizontal_cell_count, dtype=np.float32)#empty_grid
                grid1 = np.zeros(self.vertical_cell_count * self.horizontal_cell_count, dtype=np.float32)#target
                grid2 = np.zeros(self.vertical_cell_count * self.horizontal_cell_count, dtype=np.float32)#agent
                grid3 = np.zeros(self.vertical_cell_count * self.horizontal_cell_count, dtype=np.float32)#items
            

                grid1[target_x * self.horizontal_cell_count + target_y] = 1.0 
                grid0[target_x * self.horizontal_cell_count + target_y] = 0.0
                
                grid2[agent_x * self.horizontal_cell_count + agent_y] = 1.0
                grid0[agent_x * self.horizontal_cell_count + agent_y] = 0.0

                for loc, time in zip(self.item_locs, self.item_times):
                    idx = loc[0] * self.horizontal_cell_count + loc[1]
                    dist = self.distance(loc)
                    if time + dist < self.max_response_time:
                        grid3[idx] = 1.0
                        grid0[idx] = 0.0
                        
                grid0 = grid0.reshape(self.vertical_cell_count, self.horizontal_cell_count)
                grid1 = grid1.reshape(self.vertical_cell_count, self.horizontal_cell_count)
                grid2 = grid2.reshape(self.vertical_cell_count, self.horizontal_cell_count)
                grid3 = grid3.reshape(self.vertical_cell_count, self.horizontal_cell_count)
                obs_cnn = np.stack([grid0, grid1, grid2, grid3], axis=-1)  # shape: (5, 5, 4)
                obs_cnn = tf.convert_to_tensor(obs_cnn, dtype=tf.float32)
                obs_cnn = tf.reshape(obs_cnn, [-1])
                # MLP observation
                obs_mlp = []
                agent_y, agent_x = self.agent_loc
                obs_mlp.append(float(self.agent_load)) 
                # Default: kein Item → nutze spawn_distribution
                use_distribution = len(self.item_locs) == 0
                dx, dy = 0.0, 0.0

                if not use_distribution:
                # Suche bestes erreichbares Item
                    min_step = float('inf')
                    for i, (iy, ix) in enumerate(self.item_locs):
                        time_left = self.max_response_time - self.item_times[i]
                        step_cost = abs(agent_x - ix) + abs(agent_y - iy)
                        if time_left >= step_cost and step_cost < min_step:
                            dx = ix - agent_x
                            dy = iy - agent_y
                            min_step = step_cost

                obs_mlp.extend([dx, dy])  # Richtung zum besten Item oder (0, 0)

                # Entweder echte Verteilung oder Dummy
                if use_distribution:
                    obs_mlp.extend(spawn_distribution[0, :].tolist())  # z. B. Zeile 0 nehmen
                else:
                    obs_mlp.extend([0.0] * 5)
                obs_mlp = tf.convert_to_tensor(obs_mlp, dtype=tf.float32)
                obs = tf.concat([obs_cnn, obs_mlp], axis=0)
                return obs

    def get_loc(self):
    
        return self.agent_loc, self.target_loc, self.item_locs   