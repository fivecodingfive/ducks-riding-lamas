import tensorflow as tf
from tensorflow.keras import layers, models
from keras.saving import register_keras_serializable

"""
def build_actor_model(state_size, action_size):
    observation_input = tf.keras.Input(shape=(state_size,), dtype=tf.float32)
    dense1 = tf.keras.layers.Dense(units=64, activation=tf.tanh)(observation_input)
    dense2 = tf.keras.layers.Dense(units=64, activation=tf.tanh)(dense1)
    output = tf.keras.layers.Dense(units=action_size,activation = None)(dense2)
    if action_size == 1:
        output = tf.squeeze(output, axis=1)
    return tf.keras.Model(inputs=observation_input, outputs=output)

def build_critic_model(state_size, action_size):
    observation_input = tf.keras.Input(shape=(state_size,), dtype=tf.float32)
    dense1 = tf.keras.layers.Dense(units=64, activation=tf.tanh)(observation_input)
    dense2 = tf.keras.layers.Dense(units=64, activation=tf.tanh)(dense1)
    output = tf.keras.layers.Dense(units=action_size,activation = None)(dense2)
    if action_size == 1:
        output = tf.squeeze(output, axis=1)
    return tf.keras.Model(inputs=observation_input, outputs=output)
    """


def build_actor_model(input_dim, output_dim):
    return models.Sequential([
        layers.Dense(128, activation='relu', kernel_initializer='he_normal', input_shape=(input_dim,)),
        layers.LayerNormalization(),  # Better than BN for RL
        layers.Dense(64, activation='relu', kernel_initializer='he_normal'),
        layers.Dense(output_dim)  # Logits output
    ], name='actor')

def build_critic_model(input_dim, output_dim=1):
    return models.Sequential([
        layers.Dense(128, activation='relu', kernel_initializer='he_normal', input_shape=(input_dim,)),
        layers.LayerNormalization(),
        layers.Dense(64, activation='relu', kernel_initializer='he_normal'),
        layers.Dense(output_dim, activation='linear')  # Must have linear activation!
    ], name='critic')