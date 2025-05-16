import tensorflow as tf

def build_q_network(input_shape, output_size):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),  # Remove extra parentheses
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(5),
        tf.keras.layers.Dense(output_size)
    ])

# For your environment's 76-element observation vector:
model = build_q_network((76,), 5)  # (observation_size,) , action_size
model.summary()