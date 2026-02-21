"""
# The agent is initialized
agent = PPO_Agent(no_of_states,no_of_actions)

# Train your agent
rew,total_episodes = environment.trainPPO(agent, no_episodes)

def trainPPO(self, agent, no_episodes):
    rew,total_episodes = self.runPPO(agent, no_episodes,training = True)


# RUN PPO
# Start training loop
for each episode:
    # Reset all rollout buffers (one per actor)
    initialize_rollout_buffers()

    for each actor:
        # Start a new episode
        state = env.reset()
        total_reward = 0

        for each timestep:
            # Select action using current policy network
            action, logit = actor_network(state)

            # Interact with environment
            next_state, reward, done = env.step(action)

            if training:
                # Estimate value of current state
                value = critic_network(state)

                # Compute log-probability of the taken action
                log_prob = compute_log_prob(logit, action)

                # Store the transition: state, action, reward, value, log_prob
                store_transition(state, action, reward, value, log_prob)

            total_reward += reward
            state = next_state

            # If episode ends, compute advantage (for this actor)
            if done:
                if training:
                    compute_advantage_for_actor()
                break

        # Store total reward for this actor and episode
        save_reward(actor_id, episode, total_reward)

    # ====== After all actors finish their rollouts ======
    if training:
        # Gather data from all actors into training batches
        prepare_training_data()

        # Policy network update: repeat multiple epochs
        for _ in policy_update_steps:
            update_policy(states, actions, old_log_probs, advantages)

        # Value network update: fit predicted values to returns
        for _ in value_update_steps:
            update_value_function(states, returns)

        # Optional: early stopping if recent performance is good
        if recent_reward_average > threshold:
            print("Training converged!")
            break

    # Log progress
    print(f"Episode {episode} reward: {average_reward}")

"""