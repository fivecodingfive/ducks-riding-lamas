# ppo_agent.py  ·  unified PPO agent with roll-out buffers
# ------------------------------------------------------------------------------

from .model   import build_critic_model, build_actor_model
from .buffers import initialize_buffers
from .config  import ppo_config

import tensorflow as tf
import numpy  as np
import os, wandb
import socket
import getpass


class PPO_Agent:
    # ──────────────────────────────────────────────────────────────────────────
    # 1 · Init
    # ──────────────────────────────────────────────────────────────────────────
    def __init__(self, config: dict = ppo_config):
        # --- env / action spaces ------------------------------------------------
        self.state_size     = config["state_size"]
        self.action_size    = config["action_size"]
        self.rollout_steps  = config["rollout_steps"]
        self.max_time_steps = config["max_time_steps"]
        self.shaping        = config["shaping"]

        # --- hyper-parameters ---------------------------------------------------
        self.gamma   = config["gamma"]
        self.lam     = config["lam"]
        self.entropy = config["entropy"]
        self.entropy_decay = config["entropy_decay"]
        self.entropy_min   = config["entropy_min"]
        self.clip_ratio    = config["clip_ratio"]
        self.train_epochs = config["train_epochs"]
        self.no_episodes  = config["n_episodes"]

        # --- networks & optimizers ---------------------------------------------
        self.critic_network = build_critic_model(self.state_size, 1)
        self.actor_network  = build_actor_model(self.state_size, self.action_size)
        self.policy_optimizer = tf.keras.optimizers.Adam(config["policy_learning_rate"])
        self.value_optimizer  = tf.keras.optimizers.Adam(config["value_learning_rate"])

        # --- experience buffers -------------------------------------------------
        initialize_buffers(self)   # populates: state_buffer, reward_buffer, ...
    # ──────────────────────────────────────────────────────────────────────────


    # ──────────────────────────────────────────────────────────────────────────
    # 2 · Top-level train / validate wrappers
    # ──────────────────────────────────────────────────────────────────────────
    def train_ppo(self, env):
        reward_log, total_episodes = self._run_ppo(env, training=True)
        avg = float(np.mean(reward_log)) if reward_log else 0.0
        model_paths = self._save_model(avg)
        return reward_log, total_episodes, model_paths

    def validate_ppo(self, env, model_path):
        return self._run_ppo(env, training=False, model_path=model_path)
    # ──────────────────────────────────────────────────────────────────────────


    # ──────────────────────────────────────────────────────────────────────────
    # 3 · Main episode loop
    # ──────────────────────────────────────────────────────────────────────────
    def _run_ppo(self, env, *, training: bool, model_path=None):
        # -------- validation -- load weights -----------------------------------
        if not training:
            actor_path, critic_path = (model_path if isinstance(model_path, tuple)
                                       else self._infer_model_paths(model_path))
            if not os.path.exists(actor_path):
                print(f"[Err] PPO actor not found: {actor_path}")
                return [], 0
            self.actor_network  = tf.keras.models.load_model(actor_path)
            if os.path.exists(critic_path):
                self.critic_network = tf.keras.models.load_model(critic_path)
            self.entropy = self.entropy_decay = self.entropy_min = 0.0

        mode = "training" if training else "validation"
        reward_log = []

        # -------- episode loop --------------------------------------------------
        for ep in range(1, self.no_episodes + 1):
            state  = env.reset(mode=mode)
            total_reward, step_count, done = 0.0, 0, False

            while not done:
                encoded_state = tf.convert_to_tensor([state], tf.float32)
                logits, action = self._action_selection(encoded_state)

                # env-step
                reward, next_state, done = env.step(action, shaping=True if training else False)

                if training:
                    value = self.critic_network(encoded_state)
                    logp  = self._calc_logp(logits, action)
                    self._store_transition(state, action, reward, value, logp)
                    self._update_networks_if_ready(ep)          # may trigger roll-out update

                state = next_state
                total_reward += reward
                step_count   += 1

                if done and training:
                    self._finish_path()                        # GAE for last episode slice
                    self._update_networks_if_ready(ep)         # flush if buffer filled exactly

            reward_log.append(total_reward)

            # ---------- lightweight per-episode W&B log -------------------------
            if training:
                wandb.log({"episode": ep,
                           "reward": total_reward,
                           "episode_length": step_count,
                           "entropy_coef": self.entropy}, step=ep)

            # ---------- console output every 5 eps ------------------------------
            if ep % 5 == 0:
                avg5 = np.mean(reward_log[-5:])
                print(f"[{ep}/{self.no_episodes}] avg rew (5eps) = {avg5:.2f} | entropy {self.entropy:.3f}")
                


        # -------- final summary -------------------------------------------------
        overall = np.mean(reward_log) if reward_log else 0.0
        tag = "Training" if training else "Validation"
        print(f"\n[{tag} done] overall avg reward = {overall:.2f}")
        if self.shaping == True:
            self.entropy = self.entropy_decay = self.entropy_min = 0.0
            total_original_reward = 0.0
            original_reward_log = []
            for ep in range(1, 51):
                state = env.reset(mode="validation")
                total_original_reward, done = 0.0, False
                while not done:
                    encoded_state = tf.convert_to_tensor([state], tf.float32)
                    logits, action = self._action_selection(encoded_state)
                    original_reward, next_state, done = env.step(action, shaping=False)
                    state = next_state
                    total_original_reward += original_reward
                wandb.log({"original_reward": total_original_reward}, step=ep+self.no_episodes)
                print(f"[{ep}/{50}] original reward (no shaping) = {total_original_reward:.2f}")
                original_reward_log.append(total_original_reward)
            print(f"[{ep}/{50}] original reward (no shaping) = {np.mean(original_reward_log):.2f}")
            
        if training:
            wandb.summary["overall_avg_reward"] = overall
            wandb.summary["overall_original_reward"] = np.mean(original_reward_log)
        return reward_log, self.no_episodes
    # ──────────────────────────────────────────────────────────────────────────


    # ──────────────────────────────────────────────────────────────────────────
    # 4 ·  Interaction helpers
    # ──────────────────────────────────────────────────────────────────────────
    @tf.function
    def _action_selection(self, state):
        logits = self.actor_network(state)
        action = int(tf.random.categorical(logits, 1)[0, 0])
        return logits, action

    def _calc_logp(self, logits, action):
        logp_all = tf.nn.log_softmax(logits)
        logp     = tf.reduce_sum(logp_all * tf.one_hot(action, self.action_size), axis=1)
        return logp

    def _store_transition(self, s, a, r, v, logp):
        idx = int(self.ptr)
        self.state_buffer[idx].assign(s)
        self.action_buffer[idx].assign(a)
        self.reward_buffer[idx].assign(r)
        self.value_from_critic_buffer[idx].assign(v)
        self.logprobability_buffer[idx].assign(logp)
        self.ptr.assign_add(1)
    # ──────────────────────────────────────────────────────────────────────────


    # ──────────────────────────────────────────────────────────────────────────
    # 5 ·  Advantage / return calculation per episode
    # ──────────────────────────────────────────────────────────────────────────
    def _finish_path(self):
        start, end = int(self.path_start), int(self.ptr)
        r = self.reward_buffer[start:end]
        v = self.value_from_critic_buffer[start:end]

        v_next = tf.concat([v[1:], tf.zeros([1], v.dtype)], axis=0)
        deltas = r + self.gamma * v_next - v

        adv = self._discounted_cumsum(deltas, self.gamma * self.lam)
        ret = self._discounted_cumsum(r, self.gamma)

        self.adv_buffer[start:end].assign(adv)
        self.return_buffer[start:end].assign(ret)
        self.path_start.assign(self.ptr)
    # ──────────────────────────────────────────────────────────────────────────


    # ──────────────────────────────────────────────────────────────────────────
    # 6 ·  Roll-out flush (= network update)
    # ──────────────────────────────────────────────────────────────────────────
    def _update_networks_if_ready(self, episode):
        if int(self.ptr) < self.rollout_steps:
            return  # not enough transitions yet

        sl = slice(0, int(self.ptr))
        states  = self.state_buffer[sl]
        actions = self.action_buffer[sl]
        logp    = self.logprobability_buffer[sl]
        adv     = self.adv_buffer[sl]
        rets    = self.return_buffer[sl]

        # --- policy updates ----------------------------------------------------
        policy_losses, value_losses = [], []
        for _ in range(self.train_epochs):
            pl, ent = self._train_policy(states, actions, logp, adv)
            policy_losses.append(float(pl)); entropy_bonus = float(ent)

        # --- value-function updates -------------------------------------------
        for _ in range(self.train_epochs):
            vl = self._train_value_function(states, rets)
            value_losses.append(float(vl))

        # --- reset roll-out ----------------------------------------------------
        self.ptr.assign(0); self.path_start.assign(0)

        # ── decay exploration coeff --------------------------------------
        self.entropy = max(self.entropy * self.entropy_decay, self.entropy_min)

        # --- W&B log (once per flush) -----------------------------------------
        wandb.log({"episode": episode,
                   "rollout_policy_loss":  np.mean(policy_losses),
                   "rollout_value_loss":   np.mean(value_losses),
                   "rollout_entropy":      entropy_bonus}, step=episode)
    # ──────────────────────────────────────────────────────────────────────────


    # ──────────────────────────────────────────────────────────────────────────
    # 7 ·  Optimizers
    # ──────────────────────────────────────────────────────────────────────────
    @tf.function
    def _train_policy(self, states, actions, old_logp, advantages):
        advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)

        with tf.GradientTape() as tape:
            logits  = self.actor_network(states)
            entropy = -tf.reduce_sum(tf.nn.softmax(logits) * tf.nn.log_softmax(logits), axis=1)
            entropy_bonus = tf.reduce_mean(entropy)

            new_logp = tf.reduce_sum(tf.nn.log_softmax(logits) *
                                     tf.one_hot(actions, self.action_size), axis=1)
            ratio = tf.exp(new_logp - old_logp)
            clipped = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped * advantages))
            policy_loss -= self.entropy * entropy_bonus  # exploration bonus

        # 1) compute raw grads
        # grads = tape.gradient(policy_loss, self.actor_network.trainable_variables)
        # # 2) clip
        raw_grads = tape.gradient(policy_loss, self.actor_network.trainable_variables)
        clipped_grads, _ = tf.clip_by_global_norm(raw_grads, 0.5)
        self.policy_optimizer.apply_gradients(zip(clipped_grads,
                                          self.actor_network.trainable_variables))
        return policy_loss, entropy_bonus

    @tf.function
    def _train_value_function(self, states, returns):
        with tf.GradientTape() as tape:
            value_loss = tf.reduce_mean((returns - self.critic_network(states)) ** 2)
        raw_grads = tape.gradient(value_loss, self.critic_network.trainable_variables)
        clipped_grads, _ = tf.clip_by_global_norm(raw_grads, 0.5)
        self.value_optimizer.apply_gradients(zip(clipped_grads,
                                              self.critic_network.trainable_variables))
        return value_loss


    # 8 ·  Utilities
    # ──────────────────────────────────────────────────────────────────────────
    def _discounted_cumsum(self, x, discount):
        res = tf.TensorArray(tf.float32, size=tf.shape(x)[0])
        acc = tf.constant(0.0, tf.float32)
        for t in tf.range(tf.shape(x)[0] - 1, -1, -1):
            acc = x[t] + discount * acc
            res = res.write(t, acc)
        return res.stack()

    def _infer_model_paths(self, actor_path):
        if isinstance(actor_path, tuple):  # already split
            return actor_path
        if "_actor" in actor_path:
            return actor_path, actor_path.replace("_actor", "_critic")
        return actor_path, actor_path.replace(".keras", "_critic.keras")
      
    def _save_model(self, avg_reward, base='models'):
        os.makedirs(base, exist_ok=True)
        idx = len([f for f in os.listdir(base) if f.startswith("ppo_agent_") and f.endswith(".keras")])
        fn  = f"ppo_agent_{idx}_reward{avg_reward:.2f}"
        actor_path  = os.path.join(base, f"{fn}_actor.keras")
        critic_path = os.path.join(base, f"{fn}_critic.keras")
        while os.path.exists(actor_path) or os.path.exists(critic_path):
            idx += 1; fn = f"ppo_agent_{idx}_reward{avg_reward:.2f}"
            actor_path  = os.path.join(base, f"{fn}_actor.keras")
            critic_path = os.path.join(base, f"{fn}_critic.keras")

        self.actor_network.save(actor_path)
        self.critic_network.save(critic_path)
        print(f"[Model saved] {fn} → {base}")
        return actor_path, critic_path
