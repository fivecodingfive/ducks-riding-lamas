import os
import wandb
import tensorflow as tf
import numpy as np
from config import args
from .model import build_cnn_network, build_mlp_network, build_combine_network
from .visualizer import GridVisualizer

if args.network == 'cnn':
    STATE_DIM = 100
elif args.network == 'mlp':
    STATE_DIM = 10
elif args.network == 'combine':
    STATE_DIM = 108
    
PLOT_INTERVAL = 99
    
class A2CAgent:
    def __init__(self, state_dim=STATE_DIM, action_dim=5, 
                 learning_rate=0.001, gamma=0.99, network_type=args.network):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.network_type = network_type
        self.global_step = 0

        if network_type == 'cnn':
            self.actor = build_cnn_network(state_dim, action_dim, output_activation='softmax')
            self.critic = build_cnn_network(state_dim, 1)
        elif network_type == 'mlp':
            self.actor = build_mlp_network(state_dim, action_dim, output_activation='softmax')
            self.critic = build_mlp_network(state_dim, 1)
        elif network_type == 'combine':
            self.actor = build_combine_network(state_dim, action_dim, output_activation='softmax')
            self.critic = build_combine_network(state_dim, 1)

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate)

    def act(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        probs = self.actor(state)[0]
        return tf.random.categorical(tf.math.log([probs]), num_samples=1)[0, 0].numpy()
    
    # @tf.function(input_signature=[
    #     tf.TensorSpec(shape=[None, STATE_DIM], dtype=tf.float32),
    #     tf.TensorSpec(shape=[None], dtype=tf.int32),
    #     tf.TensorSpec(shape=[None], dtype=tf.float32),
    #     tf.TensorSpec(shape=[None, STATE_DIM], dtype=tf.float32),
    #     tf.TensorSpec(shape=[None], dtype=tf.float32),
    # ]
    #     ,jit_compile=True
    # )


    # with Entropy
    # def train_iterate(self, states, actions, rewards, next_states, dones, entropy_beta=0.001):
    #     states      = tf.convert_to_tensor(states, dtype=tf.float32)
    #     actions     = tf.convert_to_tensor(actions, dtype=tf.int32)
    #     rewards     = tf.convert_to_tensor(rewards, dtype=tf.float32)
    #     next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
    #     dones       = tf.convert_to_tensor(dones, dtype=tf.float32)
        
    #     # --- Critic update ---
    #     with tf.GradientTape() as tape_critic:
    #         values      = tf.squeeze(self.critic(states), axis=1)
    #         next_values = tf.squeeze(self.critic(next_states), axis=1)
    #         targets     = rewards + self.gamma * next_values * (1.0 - dones)
    #         advantages  = targets - values
    #         critic_loss = tf.reduce_mean(tf.square(advantages))
    #     critic_grads = tape_critic.gradient(critic_loss, self.critic.trainable_variables)
    #     self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

    #     # --- Actor update ---
    #     with tf.GradientTape() as tape_actor:
    #         probs = self.actor(states)  # shape: (batch_size, action_dim)
    #         clipped_probs = tf.clip_by_value(probs, 1e-8, 1.0)
    #         action_probs = tf.gather(clipped_probs, actions[:, None], batch_dims=1)[:, 0]
    #         log_probs = tf.math.log(action_probs)
    #         entropy = -tf.reduce_sum(clipped_probs * tf.math.log(clipped_probs), axis=1)
    #         actor_loss = -tf.reduce_mean(log_probs * advantages + entropy_beta * entropy)

    #     actor_grads = tape_actor.gradient(actor_loss, self.actor.trainable_variables)
    #     self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

    #     return tf.reduce_mean(values), critic_loss

    def train_iterate(self, states, actions, rewards, next_states, dones):
        """
        states:       Tensor [T, D]
        actions:      Tensor [T]
        rewards:      Tensor [T]
        next_states:  Tensor [T, D]
        dones:        Tensor [T]
        """
        states      = tf.convert_to_tensor(states,      dtype=tf.float32)  # shape [T,10]
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)  # shape [T,10]

        T = tf.shape(rewards)[0]
        v_end = tf.squeeze(self.critic(next_states[-1:]), axis=1)[0]  
        returns = tf.TensorArray(dtype=tf.float32, size=T)
        g = v_end * (1.0 - dones[-1])
        # loop from t = T-1 down to 0
        for i in tf.range(T - 1, -1, -1):
            g = rewards[i] + self.gamma * g * (1.0 - dones[i])
            returns = returns.write(i, g)
        returns = returns.stack()        # shape [T]

        # 3) Critic loss & update
        with tf.GradientTape() as tape_critic:
            values = tf.squeeze(self.critic(states), axis=1)    # [T]
            critic_loss = tf.reduce_mean(tf.square(returns - values))
        grads_critic = tape_critic.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads_critic, self.critic.trainable_variables))

        # 4) Actor loss & update (Advantage = G_t - V(s_t))
        adv = returns - values
        with tf.GradientTape() as tape_actor:
            probs = self.actor(states) 
            idx = tf.stack([tf.range(T), actions], axis=1)
            pi_a = tf.gather_nd(probs, idx)
            logp = tf.math.log(tf.clip_by_value(pi_a, 1e-8, 1.0))
            actor_loss = -tf.reduce_mean(logp * tf.stop_gradient(adv))
        grads_actor = tape_actor.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads_actor, self.actor.trainable_variables))

        return tf.reduce_mean(values), critic_loss

    def train(self, env, episodes=400, mode='training', target_update_freq = int):
        print(">>> [A2CAgent] Entered train()", flush=True)
        print(f">>> [A2CAgent] Episodes: {episodes}, Mode: {mode}", flush=True)
        reward_log = []
        critic_loss_log =[]
        total_steps = episodes * 200

        for episode in range(1, episodes + 1):
            obs = env.reset(mode='training')
            state = obs.numpy() if hasattr(obs, "numpy") else obs
            total_reward = 0
            done = False
            visualizer = GridVisualizer() if episode % PLOT_INTERVAL == (PLOT_INTERVAL-1) else None
            
            states, actions, rewards, next_states, dones = [], [], [], [], []

            while not done:
                self.global_step += 1
                action = self.act(state)
                train_reward, reward, next_obs, done = env.step(action)
                next_state = next_obs.numpy() if hasattr(next_obs, "numpy") else next_obs
                
                # Store transition
                states.append(state)
                actions.append(action)
                # rewards.append(train_reward)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(float(done))

                # Perform one-step update
                # q_value, loss = self.train_iterate(state, action, train_reward, next_state, done)
                # critic_loss_log.append(loss)

                state = next_state
                total_reward += reward
                
                ## Decaying learning rate
                progress = self.global_step / total_steps
                new_lr = self.learning_rate * (1 - progress) + 1e-5 * progress
                self.actor_optimizer.learning_rate.assign(new_lr)
                self.critic_optimizer.learning_rate.assign(new_lr)
               
                # When buffer full or episode ends → train
                if len(states) >= 10 or done:
                    q_val, loss = self.train_iterate(states, actions, rewards, next_states, dones)
                    states, actions, rewards, next_states, dones = [], [], [], [], []
                    critic_loss_log.append(loss)

                if wandb.run:
                    wandb.log({
                        "train/q_value": q_val.numpy(),
                        "train/loss": loss.numpy()
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

            reward_log.append(total_reward)

            if episode % target_update_freq == 0:
                avg_recent = sum(reward_log[-target_update_freq:]) / target_update_freq
                loss_recent = sum(critic_loss_log[-target_update_freq*200:]) / (target_update_freq*200)
                print(f"[Training][Episode {episode}/{episodes}] Avg Reward: {avg_recent:.2f}; Avg Loss: {loss_recent}")
            if episode % 20 == 0:
                state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
                probs = self.actor(state_tensor)[0].numpy()

                import matplotlib.pyplot as plt
                plt.figure(figsize=(6, 4))
                plt.bar(range(self.action_dim), probs)
                plt.xlabel("Action")
                plt.ylabel("Probability")
                plt.title(f"Policy Distribution (Episode {episode})")
                plt.grid(True)
                plt.tight_layout()
                plt.show()


            if wandb.run:
                wandb.log(
                    {
                        "episode/reward": total_reward,
                        "episode":        episode,
                    },
                    step=episode
                )

        overall_avg = sum(reward_log) / len(reward_log)
        print(f"\n[Training Done] Overall Avg Reward: {overall_avg:.2f}")
        self.save_model(overall_avg)

    def validate(self, env, episodes=100, mode='validation'):
        print(f">>> [A2CAgent] Validating over {episodes} episodes (mode={mode})")
        reward_log = []

        for episode in range(episodes):
            obs = env.reset(mode=mode)
            state = obs.numpy() if hasattr(obs, "numpy") else obs
            done = False
            total_reward = 0

            visualizer = None
            if episode % PLOT_INTERVAL == (PLOT_INTERVAL-1):
                visualizer = GridVisualizer()

            while not done:
                # Deterministic action: pick argmax
                state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
                probs = self.actor(state_tensor)[0].numpy()
                action = int(np.argmax(probs))

                reward, next_obs, done = env.step(action)
                next_state = next_obs.numpy() if hasattr(next_obs, "numpy") else next_obs

                state = next_state
                total_reward += reward

                if visualizer is not None:
                    agent, target, items = env.get_loc()
                    visualizer.update(agent_loc=agent, target_loc=target, item_locs=items, reward=total_reward)

            if visualizer is not None:
                visualizer.close()

            reward_log.append(total_reward)

            if episode % 5 == 0:
                avg_recent = sum(reward_log[-5:]) / 5
                print(f"[Validating][Episode {episode}/{episodes}] Avg reward (last 5): {avg_recent:.2f}")

        avg_reward = sum(reward_log) / len(reward_log)
        print(f"\n[Validation Done] Avg Reward: {avg_reward:.2f}")
        
    def save_model(self, avg_reward, base_dir='models'):
        os.makedirs(base_dir, exist_ok=True)

        existing = [f for f in os.listdir(base_dir) if f.startswith("A2Cmodel_") and f.endswith(".keras")]

        index = len(existing)
        file_name = f"A2Cmodel_{index}_reward{avg_reward:.2f}.keras"
        full_path = os.path.join(base_dir, file_name)

        while os.path.exists(full_path):
            index += 1
            file_name = f"A2Cmodel_{index}_reward{avg_reward:.2f}.keras"
            full_path = os.path.join(base_dir, file_name)

        self.actor.save(full_path)
        print(f"A2C Model saved: {file_name} in {full_path}")
        
        return full_path