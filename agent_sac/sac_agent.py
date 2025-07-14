import os
import wandb
import tensorflow as tf
import numpy as np
from config import args
from .model import build_cnn_network, build_mlp_network, build_combine_network
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from .visualizer import GridVisualizer

if args.network == 'cnn':
    STATE_DIM = 100
elif args.network == 'mlp':
    STATE_DIM = 9
elif args.network == 'combine':
    STATE_DIM = 108

PLOT_INTERVAL = 50

class SACAgent:
    def __init__(self, state_dim=STATE_DIM, action_dim=5, learning_rate=0.001, 
                 gamma=0.99, tau=0.005, alpha=0.3, use_per=False,
                 buffer_size=50000, batch_size=64, network_type=args.network):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.use_per = use_per
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.global_step = 0
        
        ## Double Q network to avoid overestimation
        if network_type == 'cnn':
            self.actor = build_cnn_network(state_dim, action_dim, output_activation='softmax')
            self.q1 = build_cnn_network(state_dim, action_dim)
            self.q2 = build_cnn_network(state_dim, action_dim)
            self.target_q1 = build_cnn_network(state_dim, action_dim)
            self.target_q2 = build_cnn_network(state_dim, action_dim)
        elif network_type == 'mlp':
            self.actor = build_mlp_network(state_dim, action_dim, output_activation='softmax')
            self.q1 = build_mlp_network(state_dim, action_dim)
            self.q2 = build_mlp_network(state_dim, action_dim)
            self.target_q1 = build_mlp_network(state_dim, action_dim)
            self.target_q2 = build_mlp_network(state_dim, action_dim)
        elif network_type == 'combine':
            self.actor = build_combine_network(state_dim, action_dim, output_activation='softmax')
            self.q1 = build_combine_network(state_dim, action_dim)
            self.q2 = build_combine_network(state_dim, action_dim)
            self.target_q1 = build_combine_network(state_dim, action_dim)
            self.target_q2 = build_combine_network(state_dim, action_dim)

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.q1_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.q2_optimizer = tf.keras.optimizers.Adam(learning_rate)

        self.update_target_networks(tau=1.0)
        
        ## ReplayBuffer
        self.replay_buffer = PrioritizedReplayBuffer(capacity=buffer_size, state_shape=(self.state_dim,)) if use_per else ReplayBuffer(capacity=buffer_size, state_shape=(self.state_dim,))
        print(f">>> Using per:{use_per}")
        
    def act(self, state, deterministic=False):
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        probs = self.actor(state_tensor)[0].numpy()
        if deterministic:
            return np.argmax(probs)
        return np.random.choice(self.action_dim, p=probs)

    def update_target_networks(self, tau=0.005):
        tau = self.tau if tau is None else tau
        for var, target_var in zip(self.q1.trainable_variables, self.target_q1.trainable_variables):
            target_var.assign(tau * var + (1 - tau) * target_var)
        for var, target_var in zip(self.q2.trainable_variables, self.target_q2.trainable_variables):
            target_var.assign(tau * var + (1 - tau) * target_var)
    
    def update_learning_rate(self, current_step, episodes):
        ## Decaying learning rate
        total_steps = episodes * 200
        progress = current_step / total_steps
        new_lr = self.learning_rate * (1 - progress) + 1e-5 * progress
        self.q1_optimizer.learning_rate.assign(new_lr)
        self.q2_optimizer.learning_rate.assign(new_lr)
        self.actor_optimizer.learning_rate.assign(new_lr)

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
    def train_iterate(self, states, actions, rewards, next_states, dones, indices, weights, min_alpha=0.2):
        # Target Q calculation
        next_probs = self.actor(next_states)
        next_log_probs = tf.math.log(tf.clip_by_value(next_probs, 1e-8, 1.0))

        target_q1_vals = self.target_q1(next_states)
        target_q2_vals = self.target_q2(next_states)
        min_target_q = tf.reduce_sum(next_probs * tf.minimum(target_q1_vals, target_q2_vals), axis=1)

        entropy_term = -tf.reduce_sum(next_probs * next_log_probs, axis=1)
        self.alpha = max(self.alpha * 0.995, min_alpha)
        targets = rewards + self.gamma * (1 - dones) * (min_target_q + self.alpha * entropy_term)

        # Q1, Q2 Losses
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            q1_vals = tf.reduce_sum(self.q1(states) * tf.one_hot(actions, self.action_dim), axis=1)
            q2_vals = tf.reduce_sum(self.q2(states) * tf.one_hot(actions, self.action_dim), axis=1)
            td1 = targets - q1_vals
            td2 = targets - q2_vals
            q1_loss = tf.reduce_mean(weights * tf.square(td1))
            q2_loss = tf.reduce_mean(weights * tf.square(td2))
            
        # Two Q networks to solve over-estimation problem
        grads1 = tape1.gradient(q1_loss, self.q1.trainable_variables)
        grads2 = tape2.gradient(q2_loss, self.q2.trainable_variables)
        self.q1_optimizer.apply_gradients(zip(grads1, self.q1.trainable_variables))
        self.q2_optimizer.apply_gradients(zip(grads2, self.q2.trainable_variables))

        # Actor Loss
        with tf.GradientTape() as tape_actor:
            probs = self.actor(states)
            log_probs = tf.math.log(tf.clip_by_value(probs, 1e-8, 1.0))
            q_vals = tf.minimum(self.q1(states), self.q2(states))
            actor_loss = -tf.reduce_mean(tf.reduce_sum(probs * (q_vals - self.alpha * log_probs), axis=1))

        actor_grads = tape_actor.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        q_val = tf.reduce_mean(q_vals)

        if self.global_step % 5 == 0:
            self.update_target_networks()
        if self.use_per:
            self.replay_buffer.update_priorities(indices, tf.maximum(td1,td2)) 
            
        return q_val, q1_loss, q2_loss, actor_loss

    def train(self, env, episodes=400, target_update_freq = int):
        print(">>> [SACAgent] Entered train()", flush=True)
        print(f">>> [SACAgent] Episodes: {episodes}, Mode: training", flush=True)
        ## local log
        reward_log = []
        critic_loss_log =[]

        for episode in range(1, episodes + 1):
            obs = env.reset(mode='training', random_start=True) if episode % 19 == 0 else env.reset(mode='training')
            state = obs.numpy() if hasattr(obs, "numpy") else obs
            total_reward = 0
            done = False
            visualizer = GridVisualizer() if episode % PLOT_INTERVAL == (PLOT_INTERVAL-1) else None
            
            while not done:
                action = self.act(state)
                train_reward, reward, next_obs, done = env.step(action)
                next_state = next_obs.numpy() if hasattr(next_obs, "numpy") else next_obs
                self.remember(state, action, train_reward, next_state, done)
                
                state = next_state
                total_reward += reward
               
                if len(self.replay_buffer) >= self.batch_size:
                    # Sampling from replay buffer
                    if self.use_per:
                        states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(self.batch_size)
                    else:
                        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
                        weights = tf.ones(self.batch_size, dtype=tf.float32)
                        indices = tf.ones(self.batch_size, dtype=tf.int32)
                        
                    q_val, q1_loss, q2_loss, actor_loss = self.train_iterate(states, actions, rewards, next_states, dones, indices, weights)
    
                    critic_loss_log.append((q1_loss+q2_loss)/2)
                    
                    self.update_learning_rate(self.global_step, episodes)

                    ## wandb recording
                    self.global_step += 1
                    if wandb.run:
                        wandb.log({
                            "train/q_value": q_val,
                            "train/q_loss": (q1_loss+q2_loss)/2,
                            "train/over_estimation": (q1_loss-q2_loss),
                            "train/actor_loss": actor_loss
                        }, 
                        step=self.global_step,
                        commit=False
                        )

                # Visualizer update
                if visualizer is not None:
                    agent, target, items, blocks, load = env.get_loc()
                    visualizer.update(agent_loc=agent, target_loc=target, item_locs=items, block_locs=blocks, reward=total_reward, load=load)

            if visualizer is not None:
                visualizer.close()

            reward_log.append(total_reward)

            if episode % target_update_freq == 0:
                avg_recent = sum(reward_log[-target_update_freq:]) / target_update_freq
                loss_recent = sum(critic_loss_log[-target_update_freq*200:]) / (target_update_freq*200)
                print(f"[Training][Episode {episode}/{episodes}] Avg Reward: {avg_recent:.2f}; Avg Loss: {loss_recent:.3f}")
                
            # if episode % 9 == 0:
            #     state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
            #     probs = self.actor(state_tensor)[0].numpy()

            #     import matplotlib.pyplot as plt
            #     plt.figure(figsize=(6, 4))
            #     plt.bar(range(self.action_dim), probs)
            #     plt.xlabel("Action")
            #     plt.ylabel("Probability")
            #     plt.title(f"Policy Distribution (Episode {episode})")
            #     plt.grid(True)
            #     plt.tight_layout()
            #     plt.show()

            if wandb.run:
                wandb.log(
                    {
                        "episode/reward": total_reward,
                        "episode":        episode,
                    },
                    step=self.global_step
                )
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 5))
        plt.imshow(env.get_agent_heatmap(), cmap='hot', interpolation='nearest')
        plt.colorbar(label='Visit Count')
        plt.title('Agent Movement Heatmap')
        plt.xlabel('Y-axis')
        plt.ylabel('X-axis')
        plt.xticks(np.arange(env.horizontal_cell_count))
        plt.yticks(np.arange(env.vertical_cell_count))
        plt.grid(True)
        plt.show()


        overall_avg = sum(reward_log) / len(reward_log)
        print(f"\n[Training Done] Overall Avg Reward: {overall_avg:.2f}")

    def validate(self, env, episodes=200, model_path=str):
        print(f">>> [SACAgent] Validating over {episodes} episodes")
        # examine if the model exist
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return None

        # load model
        print(f"Loading model from: {model_path}")
        self.actor = tf.keras.models.load_model(model_path)

        reward_log = []
        for episode in range(episodes):
            obs = env.reset(mode='validation')
            state = obs.numpy() if hasattr(obs, "numpy") else obs
            done = False
            total_reward = 0

            visualizer = GridVisualizer() if episode % PLOT_INTERVAL == (PLOT_INTERVAL - 1) else None

            while not done:
                action = self.act(state, deterministic=True)

                _, reward, next_obs, done = env.step(action)
                next_state = next_obs.numpy() if hasattr(next_obs, "numpy") else next_obs

                state = next_state
                total_reward += reward

                if visualizer is not None:
                    agent, target, items, blocks, load = env.get_loc()
                    visualizer.update(agent_loc=agent, target_loc=target, item_locs=items, block_locs=blocks, reward=total_reward, load=load)

            if visualizer is not None:
                visualizer.close()

            reward_log.append(total_reward)

            if episode % 5 == 0:
                avg_recent = sum(reward_log[-5:]) / min(5, len(reward_log))
                print(f"[Validating][Episode {episode}/{episodes}] Avg reward (last 5): {avg_recent:.2f}")

        avg_reward = sum(reward_log) / len(reward_log)
        print(f"\n[Validation Done] Avg Reward: {avg_reward:.2f}")

        
    def save_model(self, variant):
        base_dir='models'
        os.makedirs(base_dir, exist_ok=True)

        existing = [f for f in os.listdir(base_dir) if f.startswith("SACmodel_") and f.endswith(".keras")]

        index = len(existing)
        file_name = f"SACmodel_{index}_v{variant}.keras"
        full_path = os.path.join(base_dir, file_name)

        while os.path.exists(full_path):
            index += 1
            file_name = f"SACmodel_{index}_v{variant}.keras"
            full_path = os.path.join(base_dir, file_name)

        self.actor.save(full_path)
        print(f"SAC Model saved: {file_name} in {full_path}")
        
        return full_path