from .model import build_critic_model, build_actor_model
from .buffers import initialize_buffers

# Initialize agent with config
# ...
self.critic_network = build_critic_model(no_of_states, 1) # Create critic network
self.actor_network = build_actor_model(no_of_states, no_of_actions) # Create actor network

# Initialize other things 
    # Env
    # actor 
    # critic
    # optimizer
    # buffers
self.initialize_buffers()
    # clip ratio
    # epochs
    # batch size


# Train loop
# ...


# Run loop
"""
def run_ac(self, agent, no_episodes,training = False):
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
            action, logit = self.action_selection(state)

            # Get new transition
            reward, next_obs, done = env.step(action)
            next_state = next_obs
            
            if training:
                value_from_critic = self.critic_network(encoded_state) # Get value of current state
                logprobability = logprobabilities(logit, action) # Transform output of NN into log-probability for numerical stability
                self.store_transitions(state, action, reward, value_from_critic,logprobability, step_count) # Record transition for usage in weight update
            
            state = next_state
            total_reward += reward
            step_count += 1

        if done: 
            if training:
                self.calc_advantage(step_count) # Calculate advantages
            break

        reward_log.append(total_reward)
    

    # WEIGHT UPDATE LOOP
    
    # START HERE IN NEXT SESSION
    
    #################################################################################################################
    if training:
        # Prepare data to be used in weight update
        reward_buffer,states_buffer,actions_buffer,logprobas_buffer,advantages_buffer,return_buffer = self.prepare_training_data()

        # Calculate weight updates for policy and value network
            for _ in range(self.train_policy_epochs):
                self.train_policy(states_buffer,actions_buffer,logprobas_buffer,advantages_buffer)
            for _ in range(self.train_value_function_epochs):
                self.train_value_function(states_buffer,return_buffer)

            if episode > 1:
                tot_rew_avg = np.average(rew[:,episode-2:episode+1]) # take average over 3 most recent episodes and all actors
                if tot_rew_avg > 400:
                    print("episode: {}/{} | score: {}".format(
                        episode + 1, no_episodes, np.average(rew[:, episode])))
                    print('Converged after ', episode + 1, ' episodes with mean total reward of ', tot_rew_avg)
                    break
            tot_rew_avg = np.average(rew[:, episode])
            print("episode: {}/{} | score: {}".format(
                episode + 1, no_episodes, tot_rew_avg))
        return rew[:,:episode+1],episode
    #################################################################################################################
        
              
        

def action_selection(self, state)
    # selection based on policy
    return action as int & logit (for clipping)

def store_transition(self, state, action, reward, value_from_critic, logprobability, time_step):
    self.state_buffer[time_step,:] = state
    self.action_buffer[time_step] = action
    self.reward_buffer[time_step] = reward
    self.value_from_critic_buffer[time_step] = value_from_critic
    self.logprobability_buffer[time_step] = logprobability
    
# Compute the log-probabilities of taking actions a by using the outputs of actor NN, mainly done for numerical stability purposes
def logprobabilities(logit, a):
    logprobabilities_all = tf.nn.log_softmax(logit)
    logprobability = tf.reduce_sum(tf.one_hot(a, 2) * logprobabilities_all, axis=1)
    return logprobability

def calc_advantage(self, time_step):
    # δ = r(s_t,a_t)+γV(s_{t+1})-V(s_t)
    deltas = self.reward_buffer[:-1] + self.gamma * self.value_from_critic_buffer[1:] - self.value_from_critic_buffer[:-1]
    
    # A(s_t,a_t) = Q(s_t,a_t)-V(s_t) = 𝔼[r(s_t,a_t)+γV(s_{t+1})|s_t,a] - V(s_t) ~ G^λ_t(s_t,a_t)-V̂(s_t) = Sum_{k=t}^{T} (γλ)^{k-t} δ_k
    # First two equalities from lecture on policy gradients and advanced policy gradients / last equality from exercise 5 in TD(λ) exercise sheet
    self.advantages_buffer[:-1] = discounted_cumulative_sums(deltas, self.gamma * self.lam)

    # Calculate total return (i.e., sum of discounted rewards) as target for value function update
    self.return_buffer[time_step + 1] = discounted_cumulative_sums(self.reward_buffer[time_step + 1], self.gamma)
    

"""