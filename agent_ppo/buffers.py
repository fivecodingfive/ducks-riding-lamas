import tensorflow as tf

# WIP - (WE CONVERTED FROM FROM NUMPY TO TF BASED ON DQN SETUP)
def initialize_buffers(self):

    #NEW
    T = self.rollout_steps          # <— not episode_max_len
    self.state_buffer               = tf.Variable(tf.zeros([T, self.state_size], tf.float32))
    self.action_buffer              = tf.Variable(tf.zeros(T, tf.int32))
    self.reward_buffer              = tf.Variable(tf.zeros(T, tf.float32))
    self.value_from_critic_buffer   = tf.Variable(tf.zeros(T, tf.float32))
    self.logprobability_buffer      = tf.Variable(tf.zeros(T, tf.float32))
    self.adv_buffer                 = tf.Variable(tf.zeros(T, tf.float32))
    self.return_buffer              = tf.Variable(tf.zeros(T, tf.float32))
    self.ptr                        = tf.Variable(0, dtype=tf.int32)   # write-pointer
    self.path_start                 = tf.Variable(0, dtype=tf.int32)  # index where current episode bega


    # OLD
    """   self.advantages_buffer         = tf.Variable(tf.zeros(self.max_time_steps, dtype=tf.float32))
    self.state_buffer              = tf.Variable(tf.zeros([self.max_time_steps, self.state_size], dtype=tf.float32))
    self.action_buffer             = tf.Variable(tf.zeros(self.max_time_steps, dtype=tf.int32))
    self.reward_buffer             = tf.Variable(tf.zeros(self.max_time_steps, dtype=tf.float32))
    self.value_from_critic_buffer  = tf.Variable(tf.zeros(self.max_time_steps, dtype=tf.float32))
    self.logprobability_buffer     = tf.Variable(tf.zeros(self.max_time_steps, dtype=tf.float32))
    self.return_buffer             = tf.Variable(tf.zeros(self.max_time_steps, dtype=tf.float32))
    """