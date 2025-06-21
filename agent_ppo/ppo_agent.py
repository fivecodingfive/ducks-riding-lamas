from .model import build_critic_model, build_actor_model
from .buffers import initialize_buffers
from .config import ppo_config  

import scipy.signal 
import tensorflow as tf
import numpy as np
from environment import Environment

ppo_config = {
    "state_size": 10,
    "action_size": 5,

    "gamma": 0.99,
    "lam": 0.9,
    
    "clip_ratio": 0.2,
    "train_policy_epochs": 40,
    "train_value_function_epochs": 40,
    "policy_learning_rate": 0.0003,
    "value_learning_rate": 0.001,

    "n_episodes": 200
}


# Initialize agent with config
class PPO_Agent:
    def __init__(self, config=ppo_config):
        self.state_size = config["state_size"] 
        self.action_size = config["action_size"]

        # Load config
        self.gamma = config["gamma"]
        self.lam = config["lam"]
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
        

    
    def train_ppo(self, agent, env):
        rew, total_episodes = self.run_ppo(agent, training=True, env=env)
        return rew, total_episodes

    def validate_ppo(self, agent, env):
        rew, total_episodes = self.run_ppo(agent, training=False, env=env)
        return rew, total_episodes

    # Run loop
    def run_ppo(self, agent, training, env):
        number_episodes = self.no_episodes  # use the one from config
        reward_log = []
        total_steps = number_episodes*200

        if training is True:
            mode = 'training'

        for episode in range(1, number_episodes + 1):
            obs = env.reset(mode=mode)
            state = obs
            total_reward = 0
            # episodic step count
            step_count = 0
            done = False
            
            # STEP LOOP
            while not done: 
                encoded_state = tf.expand_dims(state, axis=0)
                logit, action = self.action_selection(encoded_state) # logit refers to the raw, unnormalized output of the actor neural network — before applying softmax. It is input into the logorbabilities function to calculate the log-probabilities of the actions taken

                # Get new transition
                reward, next_obs, done = env.step(action)
                next_state = next_obs
                
                if training:
                    value_from_critic = self.critic_network(encoded_state) # Get value of current state
                    logprobability = self.calc_logprobability(logit, action) # Transform output of NN into log-probability for numerical stability
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
                print(f"[{episode}/{number_episodes}] Average Reward: {avg_reward:.1f}")


        return reward_log, episode 
            # OLD RETURN: rew[:,:episode+1],episode
            

    # @tf.function
    def action_selection(self,state):
        logit = self.actor_network(state)
        action = int(tf.random.categorical(logit, 1)[0, 0])
        return logit, action
    
        # OLD
        """
        logit = self.actor_network(state)   # This returns a tensor of shape [1, action_dim], e.g. [[ 0.1, -0.3, 0.7, 1.2, -0.5]] if you have 5 actions.
        action = tf.random.categorical(logit, 1)
        action = tf.squeeze(action)  # shape: (), still a Tensor but scalar
        return logit, action
        """

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


    def discounted_cumulative_sums(self, x, discount):
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]



    @tf.function
    def train_policy(self, states_buffer, action_buffer, logprobability_buffer, advantages_buffer):
        # Setup of policy loss function
        with tf.GradientTape() as tape:
            ratio = tf.exp(self.calc_logprobability(self.actor_network(states_buffer), action_buffer) - logprobability_buffer)  # Calculate pi_new/pi_old
            #old version: calculates the ratio wrong in some cases says ChatGPT
            # clip_advantage = tf.where(advantages_buffer > 0, (1 + self.clip_ratio) * advantages_buffer,
            #                          (1 - self.clip_ratio) * advantages_buffer,)  # Apply clipping
            # policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages_buffer, clip_advantage))  # Setup loss function as mean of individual L_clip and negative sign, as tf minimizes by default

            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)   # Apply clipping
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages_buffer, clipped_ratio * advantages_buffer))  # Setup loss function as mean of individual L_clip and negative sign, as tf minimizes by default
        # Use gradient based optimizer to optimize loss function and update weights
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
