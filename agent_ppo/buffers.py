import tensorflow as tf

# WIP - (MAYBE NEEDS TO BE CONVERTED FROM NUMPY TO TF)
def initialize_buffers(self):
    self.advantages_buffer      = tf.zeros(self.max_time_steps, dtype=tf.float32)
    self.state_buffer           = tf.zeros(self.max_time_steps, self.state_size, dtype=tf.float32)
    self.action_buffer          = tf.zeros(self.max_time_steps, dtype=tf.int32)
    self.reward_buffer          = tf.zeros(self.max_time_steps, dtype=tf.float32)
    self.value_from_critic_buffer           = tf.zeros(self.max_time_steps, dtype=tf.float32)
    self.logprobability_buffer  = tf.zeros(self.max_time_steps, dtype=tf.float32)
    # self.return_buffer          = tf.zeros(self.max_time_steps, dtype=tf.float32) (required?)
