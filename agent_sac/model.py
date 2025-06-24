import tensorflow as tf
from tensorflow.keras import layers, models
from keras.saving import register_keras_serializable


def build_cnn_network(input_dim, output_dim, output_activation=None):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),               # e.g., input_dim = 100 for 5x5x4
        layers.Reshape((5, 5, 4)),

        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.1),

        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.1),

        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.1),

        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.1),

        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(output_dim, activation=output_activation)
    ])
    return model


def build_mlp_network(input_dim, output_dim, output_activation=None):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(output_dim, activation=output_activation)
    ])
    return model


def build_combine_network(input_dim, output_dim, output_activation=None):
    input_tensor = tf.keras.Input(shape=(input_dim,))  # total state vector

    # Split input for CNN and MLP branches
    cnn_flat, mlp_input = InputSplitter([100, 8])(input_tensor)
    cnn_input = tf.keras.layers.Reshape((5, 5, 4))(cnn_flat)

    # CNN branch
    x1 = layers.Conv2D(128, (3, 3), padding="same", activation='relu')(cnn_input)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Conv2D(128, (3, 3), padding="same", activation='relu')(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.MaxPooling2D((2, 2))(x1)
    x1 = layers.Dropout(0.3)(x1)
    x1 = layers.Flatten()(x1)

    # MLP branch
    x2 = layers.BatchNormalization()(mlp_input)
    x2 = layers.Dense(128, activation='relu')(x2)
    x2 = layers.Dropout(0.3)(x2)
    x2 = layers.Dense(64, activation='relu')(x2)

    # Combine
    combined = layers.Concatenate()([x1, x2])
    x = layers.Dense(128, activation='relu')(combined)
    x = layers.Dense(64, activation='relu')(x)
    output = layers.Dense(output_dim, activation=output_activation)(x)

    model = models.Model(inputs=input_tensor, outputs=output)
    return model


@register_keras_serializable()
class InputSplitter(tf.keras.layers.Layer):
    def __init__(self, split_sizes, **kwargs):
        super().__init__(**kwargs)
        self.split_sizes = split_sizes

    def call(self, inputs):
        return tf.split(inputs, self.split_sizes, axis=1)

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], size) for size in self.split_sizes]
