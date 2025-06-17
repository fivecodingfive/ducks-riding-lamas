import tensorflow as tf

# WIP - (WE CONVERTED FROM FROM NUMPY TO TF BASED ON DQN SETUP)
def initialize_buffers(self):
    self.advantages_buffer      = tf.zeros(self.max_time_steps, dtype=tf.float32)
    self.state_buffer           = tf.zeros([self.max_time_steps, self.state_size], dtype=tf.float32)
    self.action_buffer          = tf.zeros(self.max_time_steps, dtype=tf.int32)
    self.reward_buffer          = tf.zeros(self.max_time_steps, dtype=tf.float32)
    self.value_from_critic_buffer   = tf.zeros(self.max_time_steps, dtype=tf.float32)
    self.logprobability_buffer  = tf.zeros(self.max_time_steps, dtype=tf.float32)
    self.return_buffer          = tf.zeros(self.max_time_steps, dtype=tf.float32)

