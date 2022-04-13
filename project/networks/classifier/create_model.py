import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model


def create_model(base_model, trainable, dense_1, dense_2, activation, dropout_rate, output_len):
    base_model.trainable = False
    # base_model.layers[-1].trainable = trainable

    x = base_model.output
    x = layers.GlobalAvgPool2D()(x)
    x = layers.Flatten(name='flatten')(x)
    if dense_1:
        x = layers.Dense(dense_1)(x)
        x = activation.get()(x)
        x = layers.Dropout(dropout_rate)(x)
    if dense_2:
        x = layers.Dense(dense_2)(x)
        x = activation.get()(x)
        x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(output_len, activation='softmax')(x)

    return Model(inputs=base_model.input, outputs=x)