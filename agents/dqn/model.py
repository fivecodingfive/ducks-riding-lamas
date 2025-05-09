# This file contains the network architecture
import tensorflow as tf
from tensorflow.keras import layers, models

def build_q_network(input_shape, num_actions):
    model = models.Sequential() # most simple keras model, layer by layer
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Dense(256, activation='relu')) # ReLU -> negative values = 0, positive values w/o change
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(num_actions))  # not Softmax, but Q-Values
    return model