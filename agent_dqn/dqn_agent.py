import os

# # decide backend FIRST
# if os.environ.get("MPLBACKEND") == "Agg":
#     import matplotlib
#     matplotlib.use("Agg")        # head-less

# import matplotlib.pyplot as plt  # safe to import now
import numpy as np
import tensorflow as tf
from .model import build_cnn_network, build_mlp_network, build_combine_network
from .visualizer import GridVisualizer
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from config import args
import wandb 

if args.network == 'cnn':
    STATE_DIM = 100
elif args.network == 'mlp':
    STATE_DIM = 10
elif args.network == 'combine':
    STATE_DIM = 108

PLOT_INTERVAL = 100

class DQNAgent:
    def __init__(self, state_dim=STATE_DIM, action_dim=5, learning_rate=0.001,
                 gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64,
                 prioritized_replay=True, alpha=0.6, beta=0.4,
                 network_type=args.network):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.beta = beta
        self.network_type = network_type
        self.global_step = 0

        ## Q-network
        if  network_type == 'cnn':
            print("Using CNN ...")
            self.q_network = build_cnn_network(self.state_dim, action_dim)
            self.target_network = build_cnn_network(self.state_dim, action_dim)
        elif network_type == 'mlp':
            print("Using MLP...")
            self.q_network = build_mlp_network(self.state_dim, action_dim)
            self.target_network = build_mlp_network(self.state_dim, action_dim)
        elif network_type == 'combine':
            print("Using COMBINATION...")
            self.q_network = build_combine_network(self.state_dim, action_dim)
            self.target_network = build_combine_network(self.state_dim, action_dim)
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        
        ## ReplayBuffer
        if prioritized_replay:
            print("Using prioritized replay buffer...")
            self.replay_buffer = PrioritizedReplayBuffer(capacity=buffer_size, state_shape=(self.state_dim,), alpha=alpha)
            self.use_per = True
            self.beta = beta
        else:
            self.replay_buffer = ReplayBuffer(capacity=buffer_size, state_shape=(self.state_dim,))
            self.use_per = False
        
        # self.q_log = {
        #     'avg_q': [],
        #     'max_q': []
        # }
        # self.loss_log = {
        #     'loss':[]
        # }
        
        self.update_target_network()

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        q_values = self.q_network(tf.convert_to_tensor([state], dtype=tf.float32))
        return int(tf.argmax(q_values[0]))

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, STATE_DIM], dtype=tf.float32),
            tf.TensorSpec(shape=[None], dtype=tf.int32),
            tf.TensorSpec(shape=[None], dtype=tf.float32),
            tf.TensorSpec(shape=[None, STATE_DIM], dtype=tf.float32),
            tf.TensorSpec(shape=[None], dtype=tf.float32),
            tf.TensorSpec(shape=[None], dtype=tf.int32),
            tf.TensorSpec(shape=[None], dtype=tf.float32)
        ],
        jit_compile=True
    )
    
    def train_iterate(self, states, actions, rewards, next_states, dones, indices, weights) -> None:
        next_action = tf.argmax(self.q_network(next_states), axis=1)
        next_action = tf.cast(next_action, tf.int32)     # make dtypes match
        
        batch_idx   = tf.range(tf.shape(next_states)[0], dtype=tf.int32)
        idx_pairs   = tf.stack([batch_idx, next_action], axis=1)
        next_q_target  = tf.gather_nd(self.target_network(next_states), idx_pairs)
        targets = rewards + (1-dones) * self.gamma * next_q_target

        with tf.GradientTape() as tape:
            q_values = self.q_network(states)
            q_pred = tf.reduce_sum(q_values * tf.one_hot(actions, self.action_dim), axis=1)
            
            q_mean = tf.reduce_mean(q_values)
            q_max = tf.reduce_max(q_values)

            td_errors = targets - q_pred
            clip_val = 25
            td_errors = tf.clip_by_value(td_errors, clip_value_min=-clip_val, clip_value_max=clip_val)
            loss = tf.reduce_mean(weights * tf.square(td_errors))

        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

        # Update priority
        if self.use_per:
            self.replay_buffer.update_priorities(indices, td_errors)
        
        
        return q_mean, q_max, loss
    
    def train(self, env, episodes=int, mode=str, target_update_freq=int) -> None:
        """
        Training process of the DQN agent and produce the log file of the training reward log if file path is given.

        Args:
            episodes (int, optional): episode number. Defaults to int.
            mode (str, optional): [training, validation, testing] Defaults to 'training'
            target_update_freq (int, optional): update frequency of target Q-network.
            log_file (str, optional): path to log file. 
        """
        print(">>> [DQNAgent] Entered train()", flush=True)
        print(f">>> [DQNAgent] Episodes: {episodes}, Mode: {mode}, Update freq: {target_update_freq}", flush=True)
        reward_log = []
        step = 0
        total_steps = episodes*200
        
        for episode in range(1, episodes + 1):
            obs = env.reset(mode=mode)
            state = obs.numpy() if hasattr(obs, "numpy") else obs
            total_reward = 0
            done = False
            visualizer = GridVisualizer() if episode % PLOT_INTERVAL==(PLOT_INTERVAL-1) else None
            
            while not done:
                action = self.act(state)
                reward, next_obs, done = env.step(action)
                next_state = next_obs.numpy() if hasattr(next_obs, "numpy") else next_obs
                self.remember(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward

                self.global_step += 1
                
                # Sampling from replay buffer
                if self.use_per:
                    states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(self.batch_size)
                else:
                    states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
                    weights = tf.ones(self.batch_size, dtype=tf.float32)
                    indices = tf.ones(self.batch_size, dtype=tf.int32)
                
                if len(self.replay_buffer) >= self.batch_size:
                    q_mean, q_max, loss = self.train_iterate(states, actions, rewards, next_states, dones, indices, weights)
                    # Soft update in target network
                    tau = 0.005
                    for w, w_t in zip(self.q_network.weights, self.target_network.weights):
                        w_t.assign((1 - tau) * w_t + tau * w)
                        
                    self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
                    
                    ## Decaying learning rate
                    progress = self.global_step / total_steps
                    new_lr = self.learning_rate * (1 - progress) + 1e-5 * progress
                    self.optimizer.learning_rate.assign(new_lr)
                    step += 1
                    
                    avg_q = q_mean.numpy()
                    max_q = q_max.numpy()
                    loss_val = loss.numpy()

                    # self.q_log['avg_q'].append(avg_q)
                    # self.q_log['max_q'].append(max_q)
                    # self.loss_log['loss'].append(loss_val)


                    if wandb.run:
                        wandb.log(
                        {
                            "train/avg_q":   avg_q,       # ✅ Python float
                            "train/max_q":   max_q,       # ✅ Python float
                            "train/loss":    loss_val,    # ✅ Python float
                            "train/epsilon": self.epsilon,
                        },
                        step=self.global_step,
                        commit=False
                        )
                

                # Visualizer update
                if visualizer is not None:
                    agent, target, items = env.get_loc()
                    visualizer.update(agent_loc=agent, target_loc=target, item_locs=items, reward=total_reward)
                
            if visualizer is not None:    
                visualizer.close()

            if wandb.run:
                wandb.log(
                {
                    "episode/reward": total_reward,
                    "episode":        episode,
                },
                step=self.global_step   # keeps charts aligned
            )    
                
            reward_log.append(total_reward)

            if episode % target_update_freq == 0:
                self.update_target_network()
                # print the avg reward of each target network update
                recent_avg = sum(reward_log[-target_update_freq:]) / target_update_freq
                print(f"[Training][Episode {episode}/{episodes}] Avg reward: {recent_avg:.2f} | Epsilon: {self.epsilon:.3f}")
                
        overall_avg = sum(reward_log) / len(reward_log)
        last_avg = sum(reward_log[-50:]) / 50
        print(f"\n[Training Done] Overall Avg Reward: {overall_avg:.2f}")
        
        
        # plt.plot(self.q_log['avg_q'], label="Avg Q-value")
        # plt.plot(self.q_log['max_q'], label="Max Q-value", alpha=0.6)
        # plt.xlabel("Training steps")
        # plt.ylabel("Q-value")
        # plt.title(f"Q-value Convergence \n Last 50 Avg Reward={last_avg:.2f}")
        # plt.legend()
        # plt.grid(True)
        # plots_dir = os.getenv("PLOTS_DIR", "plots")
        # os.makedirs(plots_dir, exist_ok=True)

        # if os.environ.get("MPLBACKEND") == "Agg":
        #     # cluster run – save and close
        #     plt.savefig(os.path.join(plots_dir, f"qvals_{self.global_step}.png"))
        #     plt.close()
        # else:
        #     # local run – interactive
        #     plt.show()
        
        # plt.plot(self.loss_log['loss'], label="Weighted Loss")
        # plt.xlabel("Training steps")
        # plt.ylabel("Loss")
        # plt.title(f"Loss")
        # plt.legend()
        # plt.grid(True)
        # plots_dir = os.getenv("PLOTS_DIR", "plots")
        # os.makedirs(plots_dir, exist_ok=True)

        # if os.environ.get("MPLBACKEND") == "Agg":
        #     # cluster run – save and close
        #     plt.savefig(os.path.join(plots_dir, f"loss_{self.global_step}.png"))
        #     plt.close()
        # else:
        #     # local run – interactive
        #     plt.show()

        
        model_path = self.save_model(overall_avg)
        return model_path, reward_log 
    
    def validate(self, env, model_path=str, episodes=100, mode='validation'):
        """Validate the model parameter with the given path.(No exploring)

        Args:
            env (Environment): _description_
            model_path (str): _description_
            episodes (int): _description_. Defaults to 100.
            mode (str): _description_. Defaults to 'validation'.

        Returns:
            None
        """
        # examine if the model exist
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return None

        # load model
        print(f"Loading model from: {model_path}")
        self.q_network = tf.keras.models.load_model(model_path)
        #self.update_target_network()
        original_epsilon = self.epsilon
        self.epsilon = 0.0  # no exploring when validating model

        # validate
        reward_log = []
        total_reward = 0
            
        for episode in range(episodes):
            obs = env.reset(mode=mode)
            state = obs.numpy() if hasattr(obs, "numpy") else obs
            done = False
            total_reward = 0
            
            # Visualizer
            visualizer = None
            if os.environ.get("MPLBACKEND") != "Agg" and episode % PLOT_INTERVAL == PLOT_INTERVAL - 1:
                visualizer = GridVisualizer()
            
            while not done:
                action = self.act(state)
                reward, next_obs, done = env.step(action)
                next_state = next_obs.numpy() if hasattr(next_obs, "numpy") else next_obs
                state = next_state
                total_reward += reward
                
                # Visualizer update
                if visualizer is not None:
                    agent, target, items = env.get_loc()
                    visualizer.update(agent_loc=agent, target_loc=target, item_locs=items, reward=total_reward)

            if visualizer is not None:
                visualizer.close()
                
            reward_log.append(total_reward)
            
            if episode % 5 == 0:
                recent_avg = sum(reward_log[-5:]) / 5
                print(f"[Validating][Episode {episode}/{episodes}] Avg reward: {recent_avg:.2f}")
                
        self.epsilon = original_epsilon

        avg_reward = sum(reward_log) / len(reward_log)
        print(f"\n[Validation Done] Avg Reward: {avg_reward:.2f} | Model: {os.path.basename(model_path)}")
        
    def save_model(self, avg_reward, base_dir='models'):
        """
        Saving the trained parameter to a .h file starts with 'model' and ends with average reward value.

        Args:
            avg_reward (_type_): average reward in training section
            base_dir (str, optional): _description_. Defaults to 'models'.

        Returns:
            _type_: _description_
        """
        os.makedirs(base_dir, exist_ok=True)

        existing = [f for f in os.listdir(base_dir) if f.startswith("DQNmodel_") and f.endswith(".keras")]

        index = len(existing)
        file_name = f"DQNmodel_{index}_reward{avg_reward:.2f}.keras"
        full_path = os.path.join(base_dir, file_name)

        while os.path.exists(full_path):
            index += 1
            file_name = f"DQNmodel_{index}_reward{avg_reward:.2f}.keras"
            full_path = os.path.join(base_dir, file_name)

        self.q_network.save(full_path)
        print(f"DQN Model saved: {file_name} in {full_path}")
        
        return full_path
