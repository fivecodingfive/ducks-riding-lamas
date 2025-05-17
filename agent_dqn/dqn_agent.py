import os
import numpy as np
import tensorflow as tf
from .model import build_q_network
from .replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(self, state_dim=11, action_dim=5, learning_rate=0.001,
                 gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.9999,
                 buffer_size=10000, batch_size=64):
        ## Check the state_dim in get_obs
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        self.q_network = build_q_network(state_dim, action_dim)
        self.target_network = build_q_network(state_dim, action_dim)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

        self.replay_buffer = ReplayBuffer(buffer_size)
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

    def train_iterate(self) -> None:
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        next_q = self.target_network(next_states)
        target_q = rewards + (1 - dones) * self.gamma * np.amax(next_q.numpy(), axis=1)

        with tf.GradientTape() as tape:
            q_values = self.q_network(states)
            q_pred = tf.reduce_sum(q_values * tf.one_hot(actions, self.action_dim), axis=1)
            loss = self.loss_fn(target_q, q_pred)

        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

        # Epsilon decay
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
    
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

        existing = [f for f in os.listdir(base_dir) if f.startswith("model_") and f.endswith(".keras")]

        index = len(existing)
        file_name = f"model_{index}_reward{avg_reward:.2f}.keras"
        full_path = os.path.join(base_dir, file_name)

        while os.path.exists(full_path):
            index += 1
            file_name = f"model_{index}_reward{avg_reward:.2f}.keras"
            full_path = os.path.join(base_dir, file_name)

        self.q_network.save(full_path)
        print(f"Model saved: {file_name} in {full_path}")
        
        return full_path
    
    def train(self, env, episodes=int, mode=str, target_update_freq=int, log_file=None) -> None:
        """
        Training process of the DQN agent and produce the log file of the training reward log if file path is given.

        Args:
            episodes (int, optional): episode number. Defaults to int.
            mode (str, optional): [training, validation, testing] Defaults to 'training'
            target_update_freq (int, optional): update frequency of target Q-network.
            log_file (str, optional): path to log file. 
        """
        reward_log = []

        for episode in range(1, episodes + 1):
            obs = env.reset(mode=mode)
            state = obs.numpy() if hasattr(obs, "numpy") else obs
            total_reward = 0
            done = False

            while not done:
                action = self.act(state)
                reward, next_obs, done = env.step(action)
                next_state = next_obs.numpy() if hasattr(next_obs, "numpy") else next_obs

                self.remember(state, action, reward, next_state, done)
                self.train_iterate()
                state = next_state
                total_reward += reward

            reward_log.append(total_reward)

            if episode % target_update_freq == 0:
                self.update_target_network()
                # print the avg reward of each target network update
                recent_avg = sum(reward_log[-target_update_freq:]) / target_update_freq
                print(f"[Training][Episode {episode}/{episodes}] Avg reward: {recent_avg:.2f} | Epsilon: {self.epsilon:.3f}")
                
        overall_avg = sum(reward_log) / len(reward_log)
        print(f"\n[Training Done] Overall Avg Reward: {overall_avg:.2f}")
        
        
        if log_file:
            import pandas as pd
            df = pd.DataFrame({
                'episode': list(range(1, episodes + 1)),
                'reward': reward_log
            })
            df.to_csv(log_file, index=False)
            print(f"[Log] Reward log saved to {log_file}")
        
        return self.save_model(overall_avg)
    
    def validate(self, env, model_path, episodes=100, mode='validation'):
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
        self.update_target_network()
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

            while not done:
                action = self.act(state)
                reward, next_obs, done = env.step(action)
                next_state = next_obs.numpy() if hasattr(next_obs, "numpy") else next_obs
                state = next_state
                total_reward += reward

            reward_log.append(total_reward)
            
            if episode % 5 == 0:
                self.update_target_network()
                recent_avg = sum(reward_log[-5:]) / 5
                print(f"[Validating][Episode {episode}/{episodes}] Avg reward: {recent_avg:.2f} | Epsilon: {self.epsilon:.3f}")

        self.epsilon = original_epsilon  # return to training exploring rate

        avg_reward = sum(reward_log) / len(reward_log)
        print(f"\n[Validation Done] Avg Reward: {avg_reward:.2f} | Model: {os.path.basename(model_path)}")