import os
import numpy as np
import tensorflow as tf
import ray
from .model import build_q_network
from .replay_buffer import ReplayBuffer

STATE_DIM = 9

class DQNAgent:
    def __init__(self, state_dim=STATE_DIM, action_dim=5, learning_rate=0.001,
                 gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.9999,
                 buffer_size=50000, batch_size=1024):
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

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, STATE_DIM], dtype=tf.float32),
            tf.TensorSpec(shape=[None], dtype=tf.int32),
            tf.TensorSpec(shape=[None], dtype=tf.float32),
            tf.TensorSpec(shape=[None, STATE_DIM], dtype=tf.float32),
            tf.TensorSpec(shape=[None], dtype=tf.float32),
        ],
        jit_compile=True
    )
    def train_iterate(self, s, a, r, s2, d):
        next_q   = self.target_network(s2)
        max_next = tf.reduce_max(next_q, axis=1)
        target_q = r + (1 - d) * tf.constant(self.gamma) * max_next

        with tf.GradientTape() as tape:
            q_vals = self.q_network(s)
            q_pred = tf.reduce_sum(q_vals * tf.one_hot(a, self.action_dim), axis=1)
            loss = self.loss_fn(target_q, q_pred)

        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

    
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
            state = obs
            total_reward = 0
            done = False

            step_count = 0
            while not done:
                action = self.act(state)
                reward, next_obs, done = env.step(action)
                next_state = next_obs
                self.remember(state, action, reward, next_state, done)
                step_count += 1

                if len(self.replay_buffer) >= self.batch_size and step_count % 4 == 0:
                    s, a, r, s2, d = self.replay_buffer.sample(self.batch_size)
                    self.train_iterate(s, a, r, s2, d)
                    
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

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



    # for parallel training


    def act_batch(self, states):
        # states: Tensor (n_envs, state_dim)
        # Rückgabe: ndarray (n_envs,)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim, size=states.shape[0])
        q_vals = self.q_network(states)  # (n_envs, action_dim)
        return tf.argmax(q_vals, axis=1).numpy()  # (n_envs,)

    def remember_batch(self, states, actions, rewards, next_states, dones):
        # alles als batch eintragen
        for s, a, r, s2, d in zip(states, actions, rewards, next_states, dones):
            self.replay_buffer.add(s.numpy() if hasattr(s, "numpy") else s,
                                int(a), float(r),
                                s2.numpy() if hasattr(s2, "numpy") else s2,
                                float(d))

    def parallel_train(self, vec_env, episodes=int, target_update_freq=int, log_file=None):
        """
        Paralleles Training mit vektorisierter Umgebung.
        Args:
            vec_env: VectorizedEnv-Instanz
            episodes: Episodenanzahl (pro Env)
            target_update_freq: Häufigkeit des Target-Network-Updates (in globalen Schritten)
            log_file: Optional: Log-Datei für Rewards
        """
        n_envs = vec_env.n_envs
        reward_log = []
        global_step = 0 # training step counter over all envs and episodes
        max_steps_per_episode = getattr(vec_env.envs[0], "episode_steps", 200)

        for episode in range(1, episodes + 1):
            states = vec_env.reset(mode="training")        # (n_envs, state_dim)
            total_rewards = np.zeros(n_envs)
            done_flags = np.zeros(n_envs, dtype=bool)
            step_in_episode = 0

            while not np.all(done_flags) and step_in_episode < max_steps_per_episode:
                actions = self.act_batch(states)
                rewards, next_states, dones = vec_env.step(actions)
                # "dones" gibt an, ob einzelne Env fertig ist

                # Rewards und Dones: ggf. auf numpy casten
                if hasattr(rewards, "numpy"): rewards = rewards.numpy()
                if hasattr(dones, "numpy"): dones = dones.numpy()

                for i, d in enumerate(dones):
                    if d:
                        next_states[i] = vec_env.envs[i].reset(mode="training")
                # Sammle Rewards nur für Envs, die noch laufen
                total_rewards += rewards * (~done_flags)
                # Setze Envs, die jetzt fertig sind, als "done"
                done_flags = np.logical_or(done_flags, dones.astype(bool))

                # ReplayBuffer befüllen
                self.remember_batch(states, actions, rewards, next_states, dones)

                # Training alle 4 globalen Schritte
                if len(self.replay_buffer) >= self.batch_size and global_step % 4 == 0:
                    s, a, r, s2, d = self.replay_buffer.sample(self.batch_size)
                    self.train_iterate(s, a, r, s2, d)

                # Epsilon-Decay pro Schritt
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

                # Target-Net-Update
                if global_step % target_update_freq == 0:
                    self.update_target_network()
                    recent_avg = np.mean(reward_log[-10:]) if len(reward_log) >= 10 else 0.0
                    #print(f"[Parallel][Step {global_step}] Recent avg reward: {recent_avg:.2f} | Epsilon: {self.epsilon:.3f}")

                states = next_states
                global_step += 1
                step_in_episode += 1

                # print(f"\n[Step {global_step}]")
                # print("Actions:", actions)
                # print("Rewards:", rewards)
                # print("Dones:", dones)
                # print("Done Flags:", done_flags)
                # print("States[0]:", states[0])         # erster Env-State
                # print("States[-1]:", states[-1])       # letzter Env-State


            # Episoden-Reward loggen (Mittelwert aller Envs)
            episode_avg = np.mean(total_rewards)
            reward_log.append(episode_avg)

            if episode % 5 == 0:
                print(f"[Parallel][Episode {episode}/{episodes}] Avg reward: {episode_avg:.2f} | Epsilon: {self.epsilon:.3f}")

        overall_avg = np.mean(reward_log)
        print(f"\n[Parallel Training Done] Overall Avg Reward: {overall_avg:.2f}")

        if log_file:
            import pandas as pd
            df = pd.DataFrame({
                'episode': list(range(1, episodes + 1)),
                'reward': reward_log
            })
            df.to_csv(log_file, index=False)
            print(f"[Log] Reward log saved to {log_file}")

        return self.save_model(overall_avg)


        # ray training

    def parallel_train_ray(self, n_envs=2, episodes=200, target_update_freq=5, log_file=None, env_kwargs=None):
        """
        Paralleles Training mit Ray.
        Args:
            n_envs: Anzahl paralleler Envs.
            episodes: Episodenanzahl pro Env.
            target_update_freq: Häufigkeit des Target-Network-Updates.
            log_file: Optional: Log-Datei für Rewards.
            env_kwargs: Dict mit Variant/Data-Dir/etc.
        """
        from ray_env_worker import EnvWorker

        ray.init(ignore_reinit_error=True)

        env_kwargs = env_kwargs or {}
        workers = [EnvWorker.remote(env_kwargs) for _ in range(n_envs)]

        # Reset alle Envs (zu Beginn)
        states = ray.get([w.reset.remote(mode="training") for w in workers])
        states = [tf.convert_to_tensor(s) if not isinstance(s, tf.Tensor) else s for s in states]
        states = tf.stack(states)

        reward_log = []
        total_rewards = [0 for _ in range(n_envs)]
        done_counts = [0 for _ in range(n_envs)]

        global_step = 0

        while max(done_counts) < episodes:
            actions = self.act_batch(states)
            # Schritte in allen Envs parallel ausführen
            futures = [w.step.remote(int(a)) for w, a in zip(workers, actions)]
            results = ray.get(futures) # Liste von (reward, next_state, done)

            rewards, next_states, dones = zip(*results)
            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
            dones = tf.convert_to_tensor(dones, dtype=tf.float32)
            next_states = [tf.convert_to_tensor(s) if not isinstance(s, tf.Tensor) else s for s in next_states]
            next_states = tf.stack(next_states)

            # Replay Buffer befüllen
            self.remember_batch(states, actions, rewards, next_states, dones)

            # Training
            if len(self.replay_buffer) >= self.batch_size and global_step % 4 == 0:
                s, a, r, s2, d = self.replay_buffer.sample(self.batch_size)
                self.train_iterate(s, a, r, s2, d)

            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            # Episoden-Fortschritt updaten
            for i in range(n_envs):
                total_rewards[i] += float(rewards[i])
                if dones[i]:
                    done_counts[i] += 1
                    reward_log.append(total_rewards[i])
                    total_rewards[i] = 0

            if global_step % target_update_freq == 0:
                self.update_target_network()
                if reward_log:
                    recent_avg = np.mean(reward_log[-10:]) if len(reward_log) >= 10 else np.mean(reward_log)
                    print(f"[Ray][Step {global_step}] Recent avg reward: {recent_avg:.2f} | Epsilon: {self.epsilon:.3f}")

            states = next_states
            global_step += 1

        overall_avg = np.mean(reward_log)
        print(f"\n[Ray Parallel Training Done] Overall Avg Reward: {overall_avg:.2f}")

        if log_file:
            import pandas as pd
            df = pd.DataFrame({
                'episode': list(range(1, len(reward_log)+1)),
                'reward': reward_log
            })
            df.to_csv(log_file, index=False)
            print(f"[Log] Reward log saved to {log_file}")

        ray.shutdown()
        return self.save_model(overall_avg)
