import tensorflow as tf
from tensorflow.keras import layers, models

def build_q_network(input_dim, output_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(output_dim)
    ])
    return model
