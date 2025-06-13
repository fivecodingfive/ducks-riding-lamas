import tensorflow as tf
from tensorflow.keras import layers, models

def build_actor_network(input_dim, output_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(output_dim, activation='softmax')
        #need to use tf.random.categorical to extract the action from the output
    ])
    return model

def build_critic_network(input_dim, output_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(output_dim)
    ])
    return model