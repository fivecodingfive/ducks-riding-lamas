
# MODEL.PY: BUILDS THE NEURAL NETWORK (2 DENSE LAYERS) THAT PREDICTS Q-VALUES FOR ACTIONS.

import tensorflow as tf

def build_q_network(input_shape, output_size):
    inputs = tf.keras.Input(shape=input_shape, dtype=tf.float32)
    x = tf.keras.layers.Dense(units=64, activation='relu')(inputs)
    x = tf.keras.layers.Dense(units=64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(units=output_size)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

model = build_q_network((4,), 4)
model.summary()