from .model import build_critic_model, build_actor_model
from .buffers import initialize_buffers
from .config import ppo_config  

# Initialize agent with config
class PPO_Agent:
    def __init__(self, config=ppo_config):
        self.state_size = config["state_size"] 
        self.action_size =  config["action_size"]

        # Load config
        self.gamma = config["gamma"]
        self.lam = config["lam"]
        self.clip_ratio = config["clip_ratio"]
        self.train_policy_epochs = config["train_policy_epochs"]
        self.train_value_function_epochs = config["train_value_function_epochs"]
        self.no_episodes = config["n_episodes"]

        # Initialize networks
        self.critic_network = build_critic_model(state_size, 1) # Create critic network
        self.actor_network = build_actor_model(state_size, action_size) # Create actor network
        
        # Define Optimizers
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=config["policy_learning_rate"])
        self.value_optimizer = tf.keras.optimizers.Adam(learning_rate=config["value_learning_rate"])

        # Initialize buffers
        self.initialize_buffers()
        

    # Train loop
    def trainPPO(self, agent, no_episodes):
            rew,total_episodes = self.runPPO(agent, no_episodes,training = True)
            return rew,total_episodes

    def validatePPO(self, agent, no_episodes):
            rew,total_episodes = self.runPPO(agent, no_episodes,training = False)
            return rew,total_episodes        


    # Run loop
    def run_ppo(self, agent, no_episodes,training = False):
        reward_log = []
        step_count = 0
        total_steps = no_episodes*200

        for episode in range(1, episodes + 1):
            obs = env.reset(mode=mode)
            state = obs
            total_reward = 0
            done = False
            
            # STEP LOOP
            while not done: 
                encoded_state = tf.convert_to_tensor([state.tolist()])
                action, logit = self.action_selection(state) # logit refers to the raw, unnormalized output of the actor neural network — before applying softmax. It is input into the logorbabilities function to calculate the log-probabilities of the actions taken

                # Get new transition
                reward, next_obs, done = env.step(action)
                next_state = next_obs
                
                if training:
                    value_from_critic = self.critic_network(encoded_state) # Get value of current state
                    logprobability = calc_logprobability(logit, action) # Transform output of NN into log-probability for numerical stability
                    self.store_transition(state, action, reward, value_from_critic,logprobability, step_count) # Record transition for usage in weight update
                
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
                print(f"[{episode}/{no_episodes}] Average Reward: {avg_reward:.1f}")

        return reward_log, episode 
            # OLD RETURN: rew[:,:episode+1],episode
            

    @tf.function
    def action_selection(self,state):
        # selection based on policy
        logit = self.actor_network(state)   # This returns a tensor of shape [1, action_dim], e.g. [[ 0.1, -0.3, 0.7, 1.2, -0.5]] if you have 5 actions.
        action = tf.squeeze(tf.random.categorical(logit, 1), axis=1)
        return logit, action

    def store_transition(self, state, action, reward, value_from_critic, logprobability, time_step):
        self.state_buffer[time_step,:] = state
        self.action_buffer[time_step] = action
        self.reward_buffer[time_step] = reward
        self.value_from_critic_buffer[time_step] = value_from_critic
        self.logprobability_buffer[time_step] = logprobability

     def prepare_training_data(self, step_count):
        states_buffer       = self.state_buffer[:step_count]
        action_buffer       = self.action_buffer[:step_count]
        logprobability_buffer = self.logprobability_buffer[:step_count]
        advantages_buffer   = self.advantages_buffer[:step_count]
        return_buffer       = self.return_buffer[:step_count]
        reward_buffer       = self.reward_buffer[:step_count]
        return reward_buffer, states_buffer, action_buffer, logprobability_buffer, advantages_buffer, return_buffer 

    # Compute the log-probabilities of taking actions a by using the outputs of actor NN, mainly done for numerical stability purposes
    def calc_logprobability(logit, action):
        logprobabilities_for_all_actions = tf.nn.log_softmax(logit) # Returns a tensor containing the log-probabilities of each action
        logprobability = tf.reduce_sum(tf.one_hot(action, self.action_size) * logprobabilities_for_all_actions, axis=1)
        return logprobability # Returns a tensor of shape [1] containing the log-probability of the action taken


    def calc_advantage(self, time_step):
        # δ = r + γV(s') - V(s)
        deltas = self.reward_buffer[:-1] + self.gamma * self.value_from_critic_buffer[1:] - self.value_from_critic_buffer[:-1]

        # Advantage = discounted sum of deltas
        self.advantages_buffer[:-1] = discounted_cumulative_sums(deltas, self.gamma * self.lam)

        # Return = discounted sum of rewards
        self.return_buffer[:time_step + 1] = discounted_cumulative_sums(self.reward_buffer[:time_step + 1], self.gamma)

    def discounted_cumulative_sums(x, discount):
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]



    @tf.function
    def train_policy(self, states_buffer, action_buffer, logprobability_buffer, advantages_buffer):
        # Setup of policy loss function
        with tf.GradientTape() as tape:
            ratio = tf.exp(calc_logprobability(self.actor_network(states_buffer), action_buffer) - logprobability_buffer)  # Calculate pi_new/pi_old
            clip_advantage = tf.where(advantages_buffer > 0, (1 + self.clip_ratio) * advantages_buffer,
                                     (1 - self.clip_ratio) * advantages_buffer,)  # Apply clipping
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages_buffer, clip_advantage))  # Setup loss function as mean of individual L_clip and negative sign, as tf minimizes by default

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
