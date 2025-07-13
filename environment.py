# # actions: 0 (nothing), 1 (up), 2 (right), 3 (down), 4 (left)

# # positions in grid:
# # - (0,0) is upper left corner
# # - first index is vertical (increasing from top to bottom)
# # - second index is horizontal (increasing from left to right)

# # if new item appears in a cell into which the agent moves/at which the agent stays in the same time step,
# # it is not picked up (if agent wants to pick it up, it has to stay in the cell in the next time step)

# import random
# import pandas as pd
# from copy import deepcopy
# from itertools import compress
# import numpy as np
# import tensorflow as tf
# from config import args

# variant = args.variant
# if variant == 2:
#     _df = pd.read_csv("item_spawn_counts_v2.csv", index_col=0)
#     # _df = pd.read_csv("item_expected_value_var2.csv", index_col=0)
# else:
#     _df = pd.read_csv("item_spawn_counts.csv", index_col=0)

# _counts = _df.to_numpy(dtype=np.float32)
# spawn_distribution = _counts / _counts.sum()
# network_type = args.network
# algorithm = args.algorithm

# class Environment(object):
#     def __init__(self, variant, data_dir):
#         self.variant = variant
#         self.vertical_cell_count = 5
#         self.horizontal_cell_count = 5
#         self.vertical_idx_target = 2
#         self.horizontal_idx_target = 0
#         self.target_loc = (self.vertical_idx_target, self.horizontal_idx_target)
#         self.episode_steps = 200
#         self.max_response_time = 15 if self.variant == 2 else 10
#         self.reward = 25 if self.variant == 2 else 15
#         self.data_dir = data_dir
#         self.state_dim = 0
#         self.state_dim = 0

#         self.training_episodes = pd.read_csv(self.data_dir + f'/variant_{self.variant}/training_episodes.csv')
#         self.training_episodes = self.training_episodes.training_episodes.tolist()
#         self.validation_episodes = pd.read_csv(self.data_dir + f'/variant_{self.variant}/validation_episodes.csv')
#         self.validation_episodes = self.validation_episodes.validation_episodes.tolist()
#         self.test_episodes = pd.read_csv(self.data_dir + f'/variant_{self.variant}/test_episodes.csv')
#         self.test_episodes = self.test_episodes.test_episodes.tolist()

#         self.remaining_training_episodes = deepcopy(self.training_episodes)
#         self.validation_episode_counter = 0

#         if self.variant == 0 or self.variant == 2:
#             self.agent_capacity = 1
#         else:
#             self.agent_capacity = 3

#         if self.variant == 0 or self.variant == 1:
#             self.eligible_cells = [(0,0), (0,1), (0,2), (0,3), (0,4),
#                                    (1,0), (1,1), (1,2), (1,3), (1,4),
#                                    (2,0), (2,1), (2,2), (2,3), (2,4),
#                                    (3,0), (3,1), (3,2), (3,3), (3,4),
#                                    (4,0), (4,1), (4,2), (4,3), (4,4)]
#             self.block_locs = []
#         else:
#             self.eligible_cells = [(0,0),        (0,2), (0,3), (0,4),
#                                    (1,0),        (1,2),        (1,4),
#                                    (2,0),        (2,2),        (2,4),
#                                    (3,0), (3,1), (3,2),        (3,4),
#                                    (4,0), (4,1), (4,2),        (4,4)]
#             self.block_locs = [(0,1),(1,1),(2,1),(1,3),(2,3),(3,3),(4,3)]
#             self.first_sec =   [(0,0),        
#                                 (1,0),        
#                                 (2,0),        
#                                 (3,0), (3,1), (3,2),
#                                 (4,0), (4,1), (4,2)]
#             self.second_sec = [(0,2), (0,3),
#                                (1,2),        
#                                (2,2)]
#             self.third_sec =    [(0,4),
#                                  (1,4),
#                                  (2,4),
#                                  (3,4),
#                                  (4,4)]

#     # initialize a new episode (specify if training, validation, or testing via the mode argument)
#     def reset(self, mode):
#         modes = ['training', 'validation', 'testing']
#         if mode not in modes:
#             raise ValueError('Invalid mode. Expected one of: %s' % modes)

#         self.step_count = 0
#         self.agent_loc = (self.vertical_idx_target, self.horizontal_idx_target)
#         self.agent_load = 0  # number of items loaded (0 or 1, except for first extension, where it can be 0,1,2,3)
#         self.item_locs = []
#         self.item_times = []

#         if mode == "testing":
#             episode = self.test_episodes[0]
#             self.test_episodes.remove(episode)
#         elif mode == "validation":
#             episode = self.validation_episodes[self.validation_episode_counter]
#             self.validation_episode_counter = (self.validation_episode_counter + 1) % 100
#         else:
#             if not self.remaining_training_episodes:
#                 self.remaining_training_episodes = deepcopy(self.training_episodes)
#             episode = random.choice(self.remaining_training_episodes)
#             self.remaining_training_episodes.remove(episode)
#         self.data = pd.read_csv(self.data_dir + f'/variant_{self.variant}/episode_data/episode_{episode:03d}.csv',
#                                 index_col=0)

#         return self.get_obs_pb()

#     # take one environment step based on the action act
#     def step(self, act):
#         self.step_count += 1

#         rew = 0

#         # done signal (1 if episode ends, 0 if not)
#         if self.step_count == self.episode_steps:
#             done = 1
#         else:
#             done = 0

#         # agent movement
#         if act != 0:
#             if act == 1:  # up
#                 new_loc = (self.agent_loc[0] - 1, self.agent_loc[1])
#             elif act == 2:  # right
#                 new_loc = (self.agent_loc[0], self.agent_loc[1] + 1)
#             elif act == 3:  # down
#                 new_loc = (self.agent_loc[0] + 1, self.agent_loc[1])
#             elif act == 4:  # left
#                 new_loc = (self.agent_loc[0], self.agent_loc[1] - 1)

#             if new_loc in self.eligible_cells:
#                 self.agent_loc = new_loc
#                 rew += -1

#         # item pick-up
#         if (self.agent_load < self.agent_capacity) and (self.agent_loc in self.item_locs):
#                 self.agent_load += 1
#                 idx = self.item_locs.index(self.agent_loc)
#                 self.item_locs.pop(idx)
#                 self.item_times.pop(idx)
#                 rew += self.reward / 2

#         # item drop-off
#         if self.agent_loc == self.target_loc:
#             rew += self.agent_load * self.reward / 2
#             self.agent_load = 0

#         # track how long ago items appeared
#         self.item_times = [i + 1 for i in self.item_times]

#         # remove items for which max response time is reached
#         mask = [i < self.max_response_time for i in self.item_times]
#         self.item_locs = list(compress(self.item_locs, mask))
#         self.item_times = list(compress(self.item_times, mask))

#         # add items which appear in the current time step
#         new_items = self.data[self.data.step == self.step_count]
#         new_items = list(zip(new_items.vertical_idx, new_items.horizontal_idx))
#         new_items = [i for i in new_items if i not in self.item_locs]  # not more than one item per cell
#         self.item_locs += new_items
#         self.item_times += [0] * len(new_items)

#         # get new observation
#         next_obs = self.get_obs_pb()

#         return rew, next_obs, done

#     # TODO: implement function that gives the input features for the neural network(s)
#     #       based on the current state of the environment

#     def distance(self, loc):
#         # distance from agent to target
#         dist = abs(self.agent_loc[0] - loc[0]) + abs(self.agent_loc[1] - loc[1])
#         return dist



#     def distance(self, loc):
#         # distance from agent to target
#         dist = abs(self.agent_loc[0] - loc[0]) + abs(self.agent_loc[1] - loc[1])
#         return dist


#     def get_obs(self):
#             if network_type == 'cnn':
            
#                 agent_x, agent_y = self.agent_loc
#                 target_x, target_y = self.target_loc
#                 grid0 = np.ones(self.vertical_cell_count * self.horizontal_cell_count, dtype=np.float32)#empty_grid
#                 grid1 = np.zeros(self.vertical_cell_count * self.horizontal_cell_count, dtype=np.float32)#target
#                 grid2 = np.zeros(self.vertical_cell_count * self.horizontal_cell_count, dtype=np.float32)#agent
#                 grid3 = np.zeros(self.vertical_cell_count * self.horizontal_cell_count, dtype=np.float32)#items
            

#                 grid1[target_x * self.horizontal_cell_count + target_y] = 1.0 
#                 grid0[target_x * self.horizontal_cell_count + target_y] = 0.0
#                 if self.agent_load == 0:
#                     grid2[agent_x * self.horizontal_cell_count + agent_y] = 1.0
#                     grid0[agent_x * self.horizontal_cell_count + agent_y] = 0.0
#                 else:
#                     grid2[agent_x * self.horizontal_cell_count + agent_y] = 0.5
#                     grid0[agent_x * self.horizontal_cell_count + agent_y] = 0.0

#                 for loc, time in zip(self.item_locs, self.item_times):
#                     idx = loc[0] * self.horizontal_cell_count + loc[1]
#                     dist = self.distance(loc)
#                     if time + dist < self.max_response_time:
#                         grid3[idx] = 1.0
#                         grid0[idx] = 0.0
                        
#                 grid0 = grid0.reshape(self.vertical_cell_count, self.horizontal_cell_count)
#                 grid1 = grid1.reshape(self.vertical_cell_count, self.horizontal_cell_count)
#                 grid2 = grid2.reshape(self.vertical_cell_count, self.horizontal_cell_count)
#                 grid3 = grid3.reshape(self.vertical_cell_count, self.horizontal_cell_count)
#                 obs = np.stack([grid0, grid1, grid2, grid3], axis=-1)  # shape: (5, 5, 4)
#                 obs = tf.convert_to_tensor(obs, dtype=tf.float32)

#                 return tf.reshape(obs, [-1])

#             elif network_type == 'mlp':
#                 obs = []
#                 agent_y, agent_x = self.agent_loc
#                 obs.extend([float(agent_x), float(agent_y)])  # Position
#                 obs.append(float(self.agent_load))            # Load

#                 # Default: kein Item → nutze spawn_distribution
#                 use_distribution = len(self.item_locs) == 0
#                 dx, dy = 0.0, 0.0

#                 if not use_distribution:
#                 # Suche bestes erreichbares Item
#                     min_step = float('inf')
#                     for i, (iy, ix) in enumerate(self.item_locs):
#                         time_left = self.max_response_time - self.item_times[i]
#                         step_cost = abs(agent_x - ix) + abs(agent_y - iy)
#                         if time_left >= step_cost and step_cost < min_step:
#                             dx = ix - agent_x
#                             dy = iy - agent_y
#                             min_step = step_cost

#                 obs.extend([dx, dy])  # Richtung zum besten Item oder (0, 0)

#                 # Entweder echte Verteilung oder Dummy
#                 if use_distribution:
#                     obs.extend(spawn_distribution.flatten().tolist())
#                 else:
#                     obs.extend([0.0] * 25)

#                 return tf.convert_to_tensor(obs, dtype=tf.float32)
            
#             elif network_type == 'combine':
#                 agent_x, agent_y = self.agent_loc
#                 target_x, target_y = self.target_loc
#                 grid0 = np.ones(self.vertical_cell_count * self.horizontal_cell_count, dtype=np.float32)#empty_grid
#                 grid1 = np.zeros(self.vertical_cell_count * self.horizontal_cell_count, dtype=np.float32)#target
#                 grid2 = np.zeros(self.vertical_cell_count * self.horizontal_cell_count, dtype=np.float32)#agent
#                 grid3 = np.zeros(self.vertical_cell_count * self.horizontal_cell_count, dtype=np.float32)#items
            

#                 grid1[target_x * self.horizontal_cell_count + target_y] = 1.0 
#                 grid0[target_x * self.horizontal_cell_count + target_y] = 0.0
                
#                 grid2[agent_x * self.horizontal_cell_count + agent_y] = 1.0
#                 grid0[agent_x * self.horizontal_cell_count + agent_y] = 0.0

#                 for loc, time in zip(self.item_locs, self.item_times):
#                     idx = loc[0] * self.horizontal_cell_count + loc[1]
#                     dist = self.distance(loc)
#                     if time + dist < self.max_response_time:
#                         grid3[idx] = 1.0
#                         grid0[idx] = 0.0
                        
#                 grid0 = grid0.reshape(self.vertical_cell_count, self.horizontal_cell_count)
#                 grid1 = grid1.reshape(self.vertical_cell_count, self.horizontal_cell_count)
#                 grid2 = grid2.reshape(self.vertical_cell_count, self.horizontal_cell_count)
#                 grid3 = grid3.reshape(self.vertical_cell_count, self.horizontal_cell_count)
#                 obs_cnn = np.stack([grid0, grid1, grid2, grid3], axis=-1)  # shape: (5, 5, 4)
#                 obs_cnn = tf.convert_to_tensor(obs_cnn, dtype=tf.float32)
#                 obs_cnn = tf.reshape(obs_cnn, [-1])
#                 # MLP observation
#                 obs_mlp = []
#                 agent_y, agent_x = self.agent_loc
#                 obs_mlp.append(float(self.agent_load)) 
#                 # Default: kein Item → nutze spawn_distribution
#                 use_distribution = len(self.item_locs) == 0
#                 dx, dy = 0.0, 0.0

#                 if not use_distribution:
#                 # Suche bestes erreichbares Item
#                     min_step = float('inf')
#                     for i, (iy, ix) in enumerate(self.item_locs):
#                         time_left = self.max_response_time - self.item_times[i]
#                         step_cost = abs(agent_x - ix) + abs(agent_y - iy)
#                         if time_left >= step_cost and step_cost < min_step:
#                             dx = ix - agent_x
#                             dy = iy - agent_y
#                             min_step = step_cost

#                 obs_mlp.extend([dx, dy])  # Richtung zum besten Item oder (0, 0)

#                 # Entweder echte Verteilung oder Dummy
#                 if use_distribution:
#                     obs_mlp.extend(spawn_distribution[0, :].tolist())  # z. B. Zeile 0 nehmen
#                 else:
#                     obs_mlp.extend([0.0] * 5)
#                 obs_mlp = tf.convert_to_tensor(obs_mlp, dtype=tf.float32)
#                 obs = tf.concat([obs_cnn, obs_mlp], axis=0)
#                 return obs

#     def get_loc(self):
    
#         return self.agent_loc, self.target_loc, self.item_locs, self.block_locs, self.agent_load
    
#     def get_obs_pb(self):
#         obs = []
#         agent_y, agent_x = self.agent_loc
#         obs.extend([float(agent_x), float(agent_y)])  # Position
#         obs.append(float(self.agent_load))            # Load
        
#         nearest_item, shortest_dist = dijkstra_nearest_item(
#             agent_load=self.agent_load,
#             agent_capacity=self.agent_capacity,
#             agent_loc=self.agent_loc,
#             item_locs=self.item_locs,
#             block_locs=self.block_locs,
#             target_loc=self.target_loc,
#             grid_shape=(self.vertical_cell_count, self.horizontal_cell_count)
#         )
#         if nearest_item is not None:
#             obs.extend([nearest_item[0], nearest_item[1]])
#             obs.append(shortest_dist)
#         else:
#             obs.extend([float(agent_x), float(agent_y)])
#             obs.append(0.0)

#         locs = np.zeros(3, dtype=np.float32)
#         if len(self.item_locs) != 0:
#             locs = np.array([0.269, 0.245, 0.485])
#         # else:
#         #     locs = np.array([0.0,0.0,0.0])
#         obs.extend(locs)
        
        
#         return tf.convert_to_tensor(obs, dtype=tf.float32)
    
# class TrainEnvironment(object):
#     def __init__(self, variant, data_dir):
#         self.variant = variant
#         self.vertical_cell_count = 5
#         self.horizontal_cell_count = 5
#         self.vertical_idx_target = 2
#         self.horizontal_idx_target = 0
#         self.target_loc = (self.vertical_idx_target, self.horizontal_idx_target)
#         self.episode_steps = 200
#         self.max_response_time = 15 if self.variant == 2 else 10
#         self.reward = 25 if self.variant == 2 else 15
#         self.data_dir = data_dir
#         self.state_dim = 0
#         self.item_cost = 0
#         self.item_expects = np.zeros(self.vertical_cell_count * self.horizontal_cell_count, dtype=np.float32)
#         self.item_counts = np.zeros(self.vertical_cell_count * self.horizontal_cell_count, dtype=np.int32)

#         self.training_episodes = pd.read_csv(self.data_dir + f'/variant_{self.variant}/training_episodes.csv')
#         self.training_episodes = self.training_episodes.training_episodes.tolist()
#         self.validation_episodes = pd.read_csv(self.data_dir + f'/variant_{self.variant}/validation_episodes.csv')
#         self.validation_episodes = self.validation_episodes.validation_episodes.tolist()
#         self.test_episodes = pd.read_csv(self.data_dir + f'/variant_{self.variant}/test_episodes.csv')
#         self.test_episodes = self.test_episodes.test_episodes.tolist()

#         self.remaining_training_episodes = deepcopy(self.training_episodes)
#         self.validation_episode_counter = 0

#         if self.variant == 0 or self.variant == 2:
#             self.agent_capacity = 1
#         else:
#             self.agent_capacity = 3

#         if self.variant == 0 or self.variant == 1:
#             self.eligible_cells = [(0,0), (0,1), (0,2), (0,3), (0,4),
#                                    (1,0), (1,1), (1,2), (1,3), (1,4),
#                                    (2,0), (2,1), (2,2), (2,3), (2,4),
#                                    (3,0), (3,1), (3,2), (3,3), (3,4),
#                                    (4,0), (4,1), (4,2), (4,3), (4,4)]
#             self.block_locs = []
#         else:
#             self.eligible_cells = [(0,0),        (0,2), (0,3), (0,4),
#                                    (1,0),        (1,2),        (1,4),
#                                    (2,0),        (2,2),        (2,4),
#                                    (3,0), (3,1), (3,2),        (3,4),
#                                    (4,0), (4,1), (4,2),        (4,4)]
#             self.block_locs = [(0,1),(1,1),(2,1),(1,3),(2,3),(3,3),(4,3)]
#             self.first_sec =   [(0,0),        
#                                 (1,0),        
#                                 (2,0),        
#                                 (3,0), (3,1), (3,2),
#                                 (4,0), (4,1), (4,2)]
#             self.second_sec = [(0,2), (0,3),
#                                (1,2),        
#                                (2,2)]
#             self.third_sec =    [(0,4),
#                                  (1,4),
#                                  (2,4),
#                                  (3,4),
#                                  (4,4)]

#     # initialize a new episode (specify if training, validation, or testing via the mode argument)
#     def reset(self, mode):
#         modes = ['training', 'validation', 'testing']
#         if mode not in modes:
#             raise ValueError('Invalid mode. Expected one of: %s' % modes)

#         self.step_count = 0
#         self.agent_loc = (self.vertical_idx_target, self.horizontal_idx_target)
#         # self.agent_loc = random.choice(self.eligible_cells)
#         self.agent_load = 0  # number of items loaded (0 or 1, except for first extension, where it can be 0,1,2,3)
#         self.item_locs = []
#         self.item_times = []

#         if mode == "testing":
#             episode = self.test_episodes[0]
#             self.test_episodes.remove(episode)
#         elif mode == "validation":
#             episode = self.validation_episodes[self.validation_episode_counter]
#             self.validation_episode_counter = (self.validation_episode_counter + 1) % 100
#         else:
#             if not self.remaining_training_episodes:
#                 self.remaining_training_episodes = deepcopy(self.training_episodes)
#             episode = random.choice(self.remaining_training_episodes)
#             self.remaining_training_episodes.remove(episode)
#         self.data = pd.read_csv(self.data_dir + f'/variant_{self.variant}/episode_data/episode_{episode:03d}.csv',
#                                 index_col=0)

#         return self.get_obs_pb()

#     # take one environment step based on the action act
#     def step(self, act):
#         self.step_count += 1
#         shaped_reward = 0
#         rew = 0

#         if self.step_count == self.episode_steps:
#             done = 1
#         else:
#             done = 0

#         # --- Agent movement ---
#         ax, ay = self.agent_loc
#         if act != 0:
#             if (ax,ay) in self.third_sec:
#                 shaped_reward+=2
#             if act == 1:  # up
#                 new_loc = (self.agent_loc[0] - 1, self.agent_loc[1])
#                 # if (ax, ay) in self.third_sec: 
#                 #     shaped_reward+=0.2 if self.agent_load==0 else 0.2
#             elif act == 2:  # right
#                 new_loc = (self.agent_loc[0], self.agent_loc[1] + 1)
#             elif act == 3:  # down
#                 new_loc = (self.agent_loc[0] + 1, self.agent_loc[1])
#                 # if (ax, ay) in self.third_sec: 
#                 #     shaped_reward+=0.3 if self.agent_load==0 else 0.2
#             elif act == 4:  # left
#                 new_loc = (self.agent_loc[0], self.agent_loc[1] - 1)

#             if new_loc in self.eligible_cells:
#                 self.agent_loc = new_loc
#                 self.item_cost += 1
#                 rew += -1
#                 if new_loc in [(3,0), (3,1),(3,2)]:
#                     shaped_reward += 1
#             else:
#                 shaped_reward += -1
#         else:
#             if self.agent_load != 0:
#                 shaped_reward += -1

#         # --- Item pickup ---
#         if (self.agent_load < self.agent_capacity) and (self.agent_loc in self.item_locs):
            
#             item_idx = self.agent_loc[0] * self.horizontal_cell_count + self.agent_loc[1]
#             self.item_expects[item_idx] += (self.reward/2 - self.item_cost)
#             self.item_counts[item_idx] += 1
#             self.item_expects[item_idx] /= self.item_counts[item_idx]
#             self.item_cost = 0
            
#             self.agent_load += 1
#             idx = self.item_locs.index(self.agent_loc)
#             self.item_locs.pop(idx)
#             self.item_times.pop(idx)
#             rew += self.reward / 2
#             # shaped_reward += self.item_expects[item_idx]
            
#         # --- Item drop-off ---
#         if self.agent_loc == self.target_loc:
#             self.item_cost = 0
#             rew += self.agent_load * self.reward / 2
#             shaped_reward += self.agent_load * self.reward / 4
#             self.agent_load = 0

#         # --- Update item timers ---
#         self.item_times = [i + 1 for i in self.item_times]
#         mask = [i < self.max_response_time for i in self.item_times]
#         self.item_locs = list(compress(self.item_locs, mask))
#         self.item_times = list(compress(self.item_times, mask))

#         # --- Add new items this step ---
#         new_items = self.data[self.data.step == self.step_count]
#         new_items = list(zip(new_items.vertical_idx, new_items.horizontal_idx))
#         new_items = [i for i in new_items if i not in self.item_locs]
#         self.item_locs += new_items
#         self.item_times += [0] * len(new_items)

#         next_obs = self.get_obs_pb()
#         train_rew = rew + shaped_reward

#         return train_rew, rew, next_obs, done



#     # TODO: implement function that gives the input features for the neural network(s)
#     #       based on the current state of the environment

#     def distance(self, loc):
#         # distance from agent to target
#         dist = abs(self.agent_loc[0] - loc[0]) + abs(self.agent_loc[1] - loc[1])
#         return dist

#     def get_loc(self):
    
#         return self.agent_loc, self.target_loc, self.item_locs, self.block_locs, self.agent_load
    
#     def get_obs(self):
#         if network_type == 'cnn':
#             agent_x, agent_y = self.agent_loc
#             target_x, target_y = self.target_loc

#             grid0 = np.ones((self.vertical_cell_count, self.horizontal_cell_count), dtype=np.float32)  # empty
#             grid1 = np.zeros_like(grid0)  # target
#             grid2 = np.zeros_like(grid0)  # agent
#             grid3 = np.zeros_like(grid0)  # items
#             grid4 = np.zeros_like(grid0)  # reward field ← NEW

#             grid1[target_x, target_y] = 1.0
#             grid0[target_x, target_y] = 0.0

#             if self.agent_load == 0:
#                 grid2[agent_x, agent_y] = 1.0
#                 grid0[agent_x, agent_y] = 0.0
#             else:
#                 grid2[agent_x, agent_y] = 0.5
#                 grid0[agent_x, agent_y] = 0.0

#             for loc, time in zip(self.item_locs, self.item_times):
#                 ix, iy = loc
#                 dist = self.distance(loc)
#                 if time + dist < self.max_response_time:
#                     grid3[ix, iy] = 1.0
#                     grid0[ix, iy] = 0.0

#             obs = np.stack([grid0, grid1, grid2, grid3, grid4], axis=-1)  # shape: (5, 5, 5)
#             obs = tf.convert_to_tensor(obs, dtype=tf.float32)
#             return tf.reshape(obs, [-1])

#         elif network_type == 'mlp':
#             obs = []
#             agent_y, agent_x = self.agent_loc
#             obs.extend([float(agent_x), float(agent_y)])  # Position
#             obs.append(float(self.agent_load))            # Load

#             # Direction to best item (or dummy)
#             use_distribution = len(self.item_locs) == 0
#             dx, dy = 0.0, 0.0
#             if not use_distribution:
#                 min_step = float('inf')
#                 for i, (iy, ix) in enumerate(self.item_locs):
#                     time_left = self.max_response_time - self.item_times[i]
#                     step_cost = abs(agent_x - ix) + abs(agent_y - iy)
#                     if time_left >= step_cost and step_cost < min_step:
#                         dx = ix - agent_x
#                         dy = iy - agent_y
#                         min_step = step_cost
#             obs.extend([dx, dy])  # Directional vector

#             # Spawn distribution dummy if no items
#             if use_distribution:
#                 obs.extend(spawn_distribution.flatten().tolist())
#             else:
#                 obs.extend([0.0] * 25)
                
#             return tf.convert_to_tensor(obs, dtype=tf.float32)
    
#     def get_obs_pb(self):
#         obs = []
#         agent_y, agent_x = self.agent_loc
#         obs.extend([float(agent_x), float(agent_y)])  # Position
#         obs.append(float(self.agent_load))            # Load
        
#         nearest_item, shortest_dist = dijkstra_nearest_item(
#             agent_load=self.agent_load,
#             agent_capacity=self.agent_capacity,
#             agent_loc=self.agent_loc,
#             item_locs=self.item_locs,
#             block_locs=self.block_locs,
#             target_loc=self.target_loc,
#             grid_shape=(self.vertical_cell_count, self.horizontal_cell_count)
#         )
#         if nearest_item is not None:
#             obs.extend([nearest_item[0], nearest_item[1]])
#             obs.append(shortest_dist)
#         else:
#             obs.extend([float(agent_x), float(agent_y)])
#             obs.append(0.0)

#         locs = np.zeros(3, dtype=np.float32)
#         if len(self.item_locs) != 0:
#             locs = np.array([0.269, 0.245, 0.485])
#         # else:
#         #     locs = np.array([0.0,0.0,0.0])
#         obs.extend(locs)
        
        
#         return tf.convert_to_tensor(obs, dtype=tf.float32)

# import heapq

# def dijkstra_nearest_item(agent_load, agent_capacity, agent_loc, item_locs, block_locs, target_loc, grid_shape=(5, 5)):
#     V, H = grid_shape
#     visited = set()
#     blocks = set(block_locs)
#     heap = [(0, agent_loc)]  # (distance, (x, y))
    
#     if agent_load==agent_capacity: 
#         while heap:
#             dist, (x, y) = heapq.heappop(heap)
#             if (x, y) in visited:
#                 continue
#             visited.add((x, y))

#             if (x, y) == target_loc:
#                 return (x, y), dist  # 找到最短距離 item

#             for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
#                 nx, ny = x + dx, y + dy
#                 if (0 <= nx < V and 0 <= ny < H) and (nx, ny) not in blocks:
#                     heapq.heappush(heap, (dist + 1, (nx, ny)))
#     else:
#         while heap:
#             dist, (x, y) = heapq.heappop(heap)
#             if (x, y) in visited:
#                 continue
#             visited.add((x, y))

#             if (x, y) in item_locs:
#                 return (x, y), dist  # 找到最短距離 item

#             for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
#                 nx, ny = x + dx, y + dy
#                 if (0 <= nx < V and 0 <= ny < H) and (nx, ny) not in blocks:
#                     heapq.heappush(heap, (dist + 1, (nx, ny)))

#     return None, float('inf')

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
from dist_matrix import Dist_matrix

_df = pd.read_csv("item_spawn_counts.csv", index_col=0)
_counts = _df.to_numpy(dtype=np.float32)
spawn_distribution = _counts / _counts.sum()
_df2 = pd.read_csv("item_expected_value_var2.csv", index_col=0) 
expected_value = _df2.to_numpy(dtype=np.float32)
network_type = args.network
wall_idx = [(0,1), (1,1), (1,3), (2,1), (2,3), (3,3), (4,3)]

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
        
        self.previous_step = float('inf')


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
            self.block_locs = []
        else:
            self.eligible_cells = [(0,0),        (0,2), (0,3), (0,4),
                                   (1,0),        (1,2),        (1,4),
                                   (2,0),        (2,2),        (2,4),
                                   (3,0), (3,1), (3,2),        (3,4),
                                   (4,0), (4,1), (4,2),        (4,4)]
            self.block_locs = [(0,1),(1,1),(2,1),(1,3),(2,3),(3,3),(4,3)]
        
        self.dist_matrix = Dist_matrix(self.variant)
        self.visited_cells = set() 
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
        self.visited_cells.clear()
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
        shaped_reward = 0
        rew = 0

        if self.step_count == self.episode_steps:
            done = 1
        else:
            done = 0

        if self.variant == 2:
            item_set = set(self.item_locs)
            only_special_items = item_set == {(3, 4), (4, 4)} or item_set == {(4, 4), (3, 4)}
            if act == 0 and len(self.item_locs) != 0 and not only_special_items:
                shaped_reward += -2
        else:
            if act == 0 and len(self.item_locs) != 0:
                shaped_reward += -2
        # --- Agent movement ---
        if act != 0:
            if act == 1:  # up
                new_loc = (self.agent_loc[0] - 1, self.agent_loc[1])
            elif act == 2:  # right
                new_loc = (self.agent_loc[0], self.agent_loc[1] + 1)
            elif act == 3:  # downif act == 0:
        #     shaped_reward += -1.5
                new_loc = (self.agent_loc[0] + 1, self.agent_loc[1])
            elif act == 4:  # left
                new_loc = (self.agent_loc[0], self.agent_loc[1] - 1)

            if new_loc in self.eligible_cells:
                self.agent_loc = new_loc
                rew += -1  # movement penalty
                if self.variant == 2:
                    if new_loc not in self.visited_cells:
                        self.visited_cells.add(new_loc)
                        shaped_reward += 10
            else:
                shaped_reward += -0.5
        
        if self.variant == 2:
            if self.agent_load < self.agent_capacity:
                distance_to_y4 = abs(self.agent_loc[1] - 3)
                shaped_reward += - distance_to_y4 * 1
                        
                    
            if self.agent_load == self.agent_capacity:
                if self.agent_loc in [(4, 0), (4, 1), (4, 2), (0,4), (1, 4), (2, 4), (3, 4), (4, 4)]:
                    shaped_reward += -2
            else:
                if self.agent_loc in [(4, 0), (4, 1), (4, 2), (3, 4), (4, 4)]:
                    shaped_reward += -1

        # --- Item pickup ---
        if (self.agent_load < self.agent_capacity) and (self.agent_loc in self.item_locs):
            self.agent_load += 1
            
            idx = self.item_locs.index(self.agent_loc)
            self.item_locs.pop(idx)
            self.item_times.pop(idx)
            
            rew += self.reward / 2
            
        # --- Item drop-off ---
        if self.agent_load == self.agent_capacity :
            step_to_target = self.dist_matrix.get_dist_from_coord(self.agent_loc, self.target_loc)
            # if step_to_target >= self.previous_step :
            #     shaped_reward += -2
            #     self.previous_step = step_to_target
            
        if self.agent_loc == self.target_loc:
            
            rew += self.agent_load * self.reward / 2
            self.agent_load = 0
            #self.previous_step = float('inf')
            
            

        # --- Update item timers ---
        self.item_times = [i + 1 for i in self.item_times]
        mask = [i < self.max_response_time for i in self.item_times]
        self.item_locs = list(compress(self.item_locs, mask))
        self.item_times = list(compress(self.item_times, mask))

        # --- Add new items this step ---
        new_items = self.data[self.data.step == self.step_count]
        new_items = list(zip(new_items.vertical_idx, new_items.horizontal_idx))
        new_items = [i for i in new_items if i not in self.item_locs]
        self.item_locs += new_items
        self.item_times += [0] * len(new_items)

        next_obs = self.get_obs()
        train_rew = shaped_reward + rew

        return train_rew, rew, next_obs, done



    # TODO: implement function that gives the input features for the neural network(s)
    #       based on the current state of the environment


    def get_loc(self):
    
        return self.agent_loc, self.target_loc, self.item_locs, self.block_locs, self.agent_load
    
    def get_obs(self):
        if network_type == 'cnn':
            agent_x, agent_y = self.agent_loc
            target_x, target_y = self.target_loc

            grid0 = np.ones((self.vertical_cell_count, self.horizontal_cell_count), dtype=np.float32)  # empty
            grid1 = np.zeros_like(grid0)  # target
            grid2 = np.zeros_like(grid0)  # agent
            grid3 = np.zeros_like(grid0)  # items
            grid4 = np.zeros_like(grid0)  #  wall

            grid1[target_x, target_y] = 1.0
            grid0[target_x, target_y] = 0.0

            if self.agent_load == 0:
                grid2[agent_x, agent_y] = 1.0
                grid0[agent_x, agent_y] = 0.0
            else:
                grid2[agent_x, agent_y] = 1.0 - self.agent_load / self.agent_capacity + 0.5
                grid0[agent_x, agent_y] = 0.0

            # Populate grid3 (items) and grid4 (reward field)
            # field_strength = 1.0
            # max_range = 3
            # epsilon = 1e-3

            for loc, time in zip(self.item_locs, self.item_times):
                ix, iy = loc
                dist = self.dist_matrix.get_dist_from_coord(self.agent_loc, loc)
                if time + dist < self.max_response_time:
                    grid3[ix, iy] = 1.0
                    grid0[ix, iy] = 0.0

            if self.variant == 2 :
                    for wall_x, wall_y in wall_idx:
                        grid0[wall_x, wall_y] = 0
                        grid4[wall_x, wall_y] = 1
                    obs = np.stack([grid1, grid2, grid3, grid4], axis=-1)  # shape: (5, 5, 5)
                    obs = tf.convert_to_tensor(obs, dtype=tf.float32)
                    return tf.reshape(obs, [-1])
            
            obs = np.stack([grid0, grid1, grid2, grid3], axis=-1)  # shape: (5, 5, 4
            obs = tf.convert_to_tensor(obs, dtype=tf.float32)
            return tf.reshape(obs, [-1])

        elif network_type == 'mlp':
            if self.variant == 2:
                obs = []
                agent_y, agent_x = self.agent_loc
                obs.extend([float(agent_x), float(agent_y)])  # Position
                obs.append(float(self.agent_load))            # Load

                # Direction to best item (or dummy)
                item_set = set(self.item_locs)
                only_special_items = item_set == {(3, 4), (4, 4)} or item_set == {(4, 4), (3, 4)}

                use_distribution = len(self.item_locs) == 0 or only_special_items
                dx, dy = 1.0, 2.0  # Default dummy values which have largest expected value

                if not use_distribution:
                    # min_step = float('inf')
                    # min_step_to_target = float('inf')
                    max_reward = 0.0
                    best_item = None
                    best_item_index = None
                    best_item_timeleft = None

                    # Step 1: Find original best item
                    for i, (iy, ix) in enumerate(self.item_locs):
                        time_left = self.max_response_time - self.item_times[i]
                        step_cost = self.dist_matrix.get_dist_from_coord(self.agent_loc, (iy, ix))
                        step_cost_to_target = self.dist_matrix.get_dist_from_coord((iy, ix), self.target_loc)
                        item_reward = 15 - step_cost_to_target - step_cost
                        if time_left >= step_cost and item_reward > max_reward:
                            dx, dy = ix, iy
                            # min_step = step_cost
                            # min_step_to_target = step_cost_to_target
                            max_reward = item_reward
                            best_item = (iy, ix)
                            best_item_index = i
                            best_item_timeleft = time_left

                    if best_item is not None:
                    # Step 2: Look for alternative "detour" item
                        for i, (iy, ix) in enumerate(self.item_locs):
                            if i == best_item_index:
                                continue  # Skip the original best item

                            time_left = self.max_response_time - self.item_times[i]
                            step_to_item = self.dist_matrix.get_dist_from_coord(self.agent_loc, (iy, ix))
                            step_item_to_target = self.dist_matrix.get_dist_from_coord((iy, ix), self.target_loc)
                            step_target_to_best_item = self.dist_matrix.get_dist_from_coord(self.target_loc, best_item)
                            total_steps = step_to_item + step_item_to_target + step_target_to_best_item

                            time_left_best_item = self.max_response_time - self.item_times[best_item_index]

                            # Check feasibility: pick up alternative item, deliver to target, return to original best item before it disappears
                            if (time_left >= step_to_item and
                                time_left_best_item >= total_steps and
                                best_item_timeleft >= time_left):
                                
                                # Update dx, dy to go for this more efficient item
                                dx, dy = ix, iy
                                break  # You could continue looping to find the absolute best detour, or stop at first feasible   item 

                if self.agent_load == 0:
                    obs.extend([dx, dy])  # Directional vector
                else:   
                    obs.extend([float(self.target_loc[0]), float(self.target_loc[1])])
                
                if use_distribution:
                    obs.extend(expected_value.flatten().tolist())
                else:
                    obs.extend([0.0] * 25)
                obs = tf.convert_to_tensor(obs, dtype=tf.float32)
                #print(f"[DEBUG] get_obs() -> shape: {obs.shape}, variant: {self.variant}, network_type: {network_type}")
                return obs

            else:
                obs = []
                agent_y, agent_x = self.agent_loc
                obs.extend([float(agent_x), float(agent_y)])  # Position
                obs.append(float(self.agent_load))            # Load

                # Direction to best item (or dummy)
                use_distribution = len(self.item_locs) == 0
                dx, dy = 2.0, 2.0
                
                if not use_distribution:
                    max_reward = 0.0
                    best_item = None
                    best_item_index = None
                    best_item_timeleft = None

                    # Step 1: Find original best item
                    for i, (iy, ix) in enumerate(self.item_locs):
                        time_left = self.max_response_time - self.item_times[i]
                        step_cost = self.dist_matrix.get_dist_from_coord(self.agent_loc, (iy, ix))
                        step_cost_to_target = self.dist_matrix.get_dist_from_coord((iy, ix), self.target_loc)
                        item_reward = 15 - step_cost_to_target - step_cost
                        if time_left >= step_cost and item_reward > max_reward:
                            dx, dy = ix, iy
                            min_step = step_cost
                            min_step_to_target = step_cost_to_target
                            best_item = (iy, ix)
                            best_item_index = i
                            best_item_timeleft = time_left

                    if best_item is not None:
                    # Step 2: Look for alternative "detour" item
                        for i, (iy, ix) in enumerate(self.item_locs):
                            if i == best_item_index:
                                continue  # Skip the original best item

                            time_left = self.max_response_time - self.item_times[i]
                            step_to_item = self.dist_matrix.get_dist_from_coord(self.agent_loc, (iy, ix))
                            step_item_to_target = self.dist_matrix.get_dist_from_coord((iy, ix), self.target_loc)
                            step_target_to_best_item = self.dist_matrix.get_dist_from_coord(self.target_loc, best_item)
                            total_steps = step_to_item + step_item_to_target + step_target_to_best_item

                            time_left_best_item = self.max_response_time - self.item_times[best_item_index]

                            # Check feasibility: pick up alternative item, deliver to target, return to original best item before it disappears
                            if (time_left >= step_to_item and
                                time_left_best_item >= total_steps and
                                best_item_timeleft >= time_left):
                                
                                # Update dx, dy to go for this more efficient item
                                dx, dy = ix, iy
                                break  # You could continue looping to find the absolute best detour, or stop at first feasible   item

                if self.agent_load == 0:
                    obs.extend([dy, dx])  # Directional vector
                else:   
                    obs.extend([float(self.target_loc[0]), float(self.target_loc[1])])
                # Spawn distribution dummy if no items
                if use_distribution:
                    obs.extend(spawn_distribution[0, :].tolist())
                else:
                    obs.extend([0.0] * 5)
                
                obs = tf.convert_to_tensor(obs, dtype=tf.float32)
                print(f"[DEBUG] get_obs() -> shape: {obs.shape}, variant: {self.variant}, network_type: {network_type}")
                return obs