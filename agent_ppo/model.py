def build_actor_model(self, state_size, action_size):
    observation_input = tf.keras.Input(shape=(state_size,), dtype=tf.float32)
    dense1 = tf.keras.layers.Dense(units=64, activation=tf.tanh)(observation_input)
    dense2 = tf.keras.layers.Dense(units=64, activation=tf.tanh)(dense1)
    output = tf.keras.layers.Dense(units=action_size,activation = None)(dense2)
    if action_size == 1:
        output = tf.squeeze(output, axis=1)
    return tf.keras.Model(inputs=observation_input, outputs=output)

def build_critic_model(self, state_size, action_size):
    observation_input = tf.keras.Input(shape=(state_size,), dtype=tf.float32)
    dense1 = tf.keras.layers.Dense(units=64, activation=tf.tanh)(observation_input)
    dense2 = tf.keras.layers.Dense(units=64, activation=tf.tanh)(dense1)
    output = tf.keras.layers.Dense(units=action_size,activation = None)(dense2)
    if action_size == 1:
        output = tf.squeeze(output, axis=1)
    return tf.keras.Model(inputs=observation_input, outputs=output)    