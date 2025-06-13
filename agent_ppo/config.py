
class PPO_Agent:
    def __init__(self, no_of_states, no_of_actions):
        self.state_size = no_of_states
        self.action_size = no_of_actions

        # Hyperparameters
        self.gamma = 0.99  # discount rate
        self.lam = 0.9 # TD(lambda) weight
        self.clip_ratio = 0.2 # Clipping ratio based on initial PPO paper

        self.actors = 4 # No. of parallel actors
        self.max_time_steps = 10000  # Maximum time steps

        self.actor = self.nn_model(no_of_states, no_of_actions) # Create actor network
        self.train_policy_epochs = 40 # Define number of epochs for multiple weight update iterations within one episode
        self.critic = self.nn_model(no_of_states, 1) # Create critic network
        self.train_value_function_epochs = 40 # Define number of epochs for multiple weight update iterations within one episode

        policy_learning_rate = 0.0003
        value_function_learning_rate = 0.001
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=policy_learning_rate)
        self.value_optimizer = tf.keras.optimizers.Adam(learning_rate=value_function_learning_rate)