from .model import build_critic_model, build_actor_model
from .buffers import initialize_buffers
from .config import ppo_config  

import scipy.signal 
import tensorflow as tf
import numpy as np
import os
from environment import Environment

# Initialize agent with config
class PPO_Agent:
    def __init__(self, config=ppo_config):
        self.state_size = config["state_size"] 
        self.action_size = config["action_size"]

        # Load config
        self.gamma = config["gamma"]
        self.lam = config["lam"]
        self.entropy = config["entropy"]
        self.entropy_decay = config["entropy_decay"]
        self.entropy_min = config["entropy_min"]
        self.clip_ratio = config["clip_ratio"]
        self.train_policy_epochs = config["train_policy_epochs"]
        self.train_value_function_epochs = config["train_value_function_epochs"]
        self.no_episodes = config["n_episodes"]
        self.max_time_steps = config["max_time_steps"]

        # Initialize networks
        self.critic_network = build_critic_model(self.state_size, 1) # Create critic network
        self.actor_network = build_actor_model(self.state_size, self.action_size) # Create actor network

        # Define Optimizers
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=config["policy_learning_rate"])
        self.value_optimizer = tf.keras.optimizers.Adam(learning_rate=config["value_learning_rate"])

        # Initialize buffers
        initialize_buffers(self)
        

    
    def train_ppo(self, env):
        rew, total_episodes = self.run_ppo(training=True, env=env, model_path=None)
        return rew, total_episodes

    def validate_ppo(self, env, model_path):
        rew, total_episodes = self.run_ppo(training=False, env=env, model_path=model_path)
        return rew, total_episodes

    # Run loop
    def run_ppo(self, training, env, model_path=None):
        number_episodes = self.no_episodes  # use the one from config
        reward_log = []

        if training is True:
            mode = 'training'
            print("Initial action probs:", [f"{p:.2f}" for p in tf.nn.softmax(self.actor_network(tf.expand_dims(env.reset(mode=mode), 0)))[0].numpy()], "Entropy:", f"{self.entropy:.2f}")
        else:
            mode = 'validation'
            # examine if the model exist
            if not os.path.exists(model_path):
                print(f"PPO Agent file not found: {model_path}")
                return None

            # load model
            print(f"Loading ppo agent from: {model_path}")
            self.actor_network = tf.keras.models.load_model(model_path)
            self.entropy = 0.0
            self.entropy_decay = 0.0
            self.entropy_min = 0.0

        for episode in range(1, number_episodes + 1):
            obs = env.reset(mode=mode)
            state = obs
            total_reward = 0
            # episodic step count
            step_count = 0
            done = False
            
            # STEP LOOP
            while not done: 
                # encoded_state = tf.expand_dims(state, axis=0)
                encoded_state = tf.convert_to_tensor([state], dtype=tf.float32)
                logit, action = self.action_selection(encoded_state) # logit refers to the raw, unnormalized output of the actor neural network — before applying softmax. It is input into the logorbabilities function to calculate the log-probabilities of the actions taken

                # Get new transition
                reward, next_obs, done = env.step(action)
                next_state = next_obs
                
                if training:
                    value_from_critic = self.critic_network(encoded_state) # Get value of current state
                    logprobability = self.calc_logprobability(logit, action) # Transform output of NN into log-probability for numerical stability
                    if step_count >= self.max_time_steps:
                        print(f"[{episode}] Max time steps reached!")
                    self.store_transition(state, action, reward, value_from_critic, logprobability, step_count) # Record transition for usage in weight update
                
                state = next_state
                total_reward += reward
                step_count += 1

                if done: 
                    if training:
                        self.calc_advantage(step_count) # Calculate advantages
                    break
            reward_log.append(total_reward)
        

            # WEIGHT UPDATE LOOP
        
            if training:
                # Prepare data to be used in weight update
                reward_buffer,states_buffer,action_buffer,logprobability_buffer,advantages_buffer,return_buffer = self.prepare_training_data(step_count)

                # Calculate weight updates for policy and value network
                for _ in range(self.train_policy_epochs):
                    self.train_policy(states_buffer,action_buffer,logprobability_buffer,advantages_buffer)
                for _ in range(self.train_value_function_epochs):
                    self.train_value_function(states_buffer,return_buffer)
            
            # Print average every fifth episode
            if episode % 5 == 0:
                avg_reward = np.mean(reward_log[-5:])  # average over the last 5 episodes
                self.entropy = max(self.entropy * self.entropy_decay, self.entropy_min)
                print(f"[{episode}/{number_episodes}] Average Reward: {avg_reward:.2f}")
                print("Action probabilities:", [f"{p:.2f}" for p in tf.nn.softmax(self.actor_network(tf.convert_to_tensor([obs], dtype=tf.float32)))[0].numpy()], "Entropy:", f"{self.entropy:.2f}")
                
        overall_avg = sum(reward_log) / len(reward_log)
        if training:
            print(f"\n[Training Done] Overall Avg Reward: {overall_avg:.2f}")
            print("Saving model...")
            self.save_model(overall_avg)
            return reward_log, number_episodes
        else:
            print(f"[Validation Done] Overall Avg Reward: {overall_avg:.2f}")
            return reward_log, number_episodes
            # OLD RETURN: rew[:,:episode+1],episode
            

    @tf.function
    def action_selection(self,state):
        logit = self.actor_network(state)
        action = int(tf.random.categorical(logit, 1)[0, 0])
        return logit, action

    def store_transition(self, state, action, reward, value_from_critic, logprobability, time_step):
        self.state_buffer[time_step, :].assign(state)
        self.action_buffer[time_step].assign(action)
        self.reward_buffer[time_step].assign(reward)
        self.value_from_critic_buffer[time_step].assign(value_from_critic)
        self.logprobability_buffer[time_step].assign(logprobability)

    def prepare_training_data(self, step_count):
        states_buffer       = self.state_buffer[:step_count]
        action_buffer       = self.action_buffer[:step_count]
        logprobability_buffer = self.logprobability_buffer[:step_count]
        advantages_buffer   = self.advantages_buffer[:step_count]
        return_buffer       = self.return_buffer[:step_count]
        reward_buffer       = self.reward_buffer[:step_count]
        return reward_buffer, states_buffer, action_buffer, logprobability_buffer, advantages_buffer, return_buffer 

    # Compute the log-probabilities of taking actions a by using the outputs of actor NN, mainly done for numerical stability purposes
    @tf.function
    def calc_logprobability(self, logit, action):
        logprobabilities_for_all_actions = tf.nn.log_softmax(logit) # Returns a tensor containing the log-probabilities of each action
        logprobability = tf.reduce_sum(tf.one_hot(action, self.action_size) * logprobabilities_for_all_actions, axis=1)
        return logprobability # Returns a tensor of shape [1] containing the log-probability of the action taken


    def calc_advantage(self, time_step):
        # Prepare value estimates
        v = self.value_from_critic_buffer[:time_step]  # shape (T,)
        # v_next: shift left by 1, pad 0 at the end
        v_next = tf.concat([v[1:], tf.zeros([1], dtype=v.dtype)], axis=0)  # shape (T,)
        r = self.reward_buffer[:time_step]  # shape (T,)

        # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        deltas = r + self.gamma * v_next - v  # shape (T,)

        # GAE: discounted cumulative sum of deltas
        advantages = self.discounted_cumulative_sums(deltas, self.gamma * self.lam)  # shape (T,)

        self.advantages_buffer.assign(tf.concat([
            tf.cast(advantages, tf.float32),
            tf.zeros([self.max_time_steps - time_step], dtype=tf.float32)
        ], axis=0))

        # Return = discounted sum of rewards
        returns = self.discounted_cumulative_sums(r, self.gamma)  # shape (T,)
        self.return_buffer.assign(tf.concat([
            tf.cast(returns, tf.float32),
            tf.zeros([self.max_time_steps - time_step], dtype=tf.float32)
        ], axis=0))


    # new version using tensorflow instead of scipy.signal.lfilter making it computationally more efficient 
    # def discounted_cumulative_sums(self, x, discount):
    #     return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
    
    def discounted_cumulative_sums(self, x, discount):
        result = tf.TensorArray(dtype=tf.float32, size=tf.shape(x)[0])
        acc = tf.constant(0.0)
        for t in tf.range(tf.shape(x)[0] - 1, -1, -1):
            acc = x[t] + discount * acc
            result = result.write(t, acc)
        return result.stack()

    @tf.function
    def train_policy(self, states_buffer, action_buffer, logprobability_buffer, advantages_buffer):
        # Normalize advantages for better stability
        advantages_buffer = (advantages_buffer - tf.reduce_mean(advantages_buffer)) / (tf.math.reduce_std(advantages_buffer) + 1e-8)

        with tf.GradientTape() as tape:
            logits = self.actor_network(states_buffer)  # <— nur ein Forward-Pass
            entropy = -tf.reduce_sum(tf.nn.softmax(logits) * tf.nn.log_softmax(logits), axis=1)
            entropy_bonus = tf.reduce_mean(entropy)

            new_log_probs = tf.reduce_sum(
                tf.nn.log_softmax(logits) * tf.one_hot(action_buffer, self.action_size), axis=1
            )
            ratio = tf.exp(new_log_probs - logprobability_buffer)

            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages_buffer, clipped_ratio * advantages_buffer))

            # Add entropy regularization (encourages exploration)
            policy_loss -= self.entropy * entropy_bonus

        policy_grads = tape.gradient(policy_loss, self.actor_network.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_grads, self.actor_network.trainable_variables))



    @tf.function
    def train_value_function(self, states_buffer, return_buffer):
        # Setup value network's loss function
        with tf.GradientTape() as tape:
            value_loss = tf.reduce_mean(
                (return_buffer - self.critic_network(states_buffer)) ** 2)  # Train the value function by regression on mean-squared error

        # Use gradient based optimizer to optimize loss function and update weights
        value_grads = tape.gradient(value_loss, self.critic_network.trainable_variables)
        self.value_optimizer.apply_gradients(zip(value_grads, self.critic_network.trainable_variables))

    def save_model(self, avg_reward, base_dir='models'):
        os.makedirs(base_dir, exist_ok=True)

        existing = [f for f in os.listdir(base_dir) if f.startswith("ppo_agent_") and f.endswith(".keras")]

        index = len(existing)
        file_name = f"ppo_agent_{index}_reward{avg_reward:.2f}.keras"
        full_path = os.path.join(base_dir, file_name)

        while os.path.exists(full_path):
            index += 1
            file_name = f"ppo_agent_{index}_reward{avg_reward:.2f}.keras"
            full_path = os.path.join(base_dir, file_name)

        self.actor_network.save(full_path)
        print(f"PPO Agent saved: {file_name} in {full_path}")
        
        return full_path
