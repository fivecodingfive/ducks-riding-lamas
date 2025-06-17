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
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        
        layers.Dense(output_dim)
    ])
    return model

def build_critic_model(input_dim, output_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        
        layers.Dense(output_dim)
    ])
    return model