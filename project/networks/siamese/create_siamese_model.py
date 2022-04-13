import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model


def _inject_suffix(model, suffix):
    for layer in model.layers:
        layer._name = layer.name + str(suffix)


def _create_base_model(base_model, image_size, trainable, dense_1, dense_2, activation, batch_norm, dropout_rate, suffix=''):
    model = base_model.get((*image_size, 3))
    model.trainable = trainable

    x = model.output
    x = layers.MaxPool2D(pool_size=(7, 7))(x)
    x = layers.Flatten()(x)

    if dense_1:
        x = layers.Dense(dense_1)(x)
        x = activation.get()(x)
        if batch_norm:
            x = layers.BatchNormalization()(x)
        if dropout_rate:
            x = layers.Dropout(dropout_rate)(x)
    if dense_2:
        x = layers.Dense(dense_2)(x)
        x = activation.get()(x)
        if batch_norm:
            x = layers.BatchNormalization()(x)
        if dropout_rate:
            x = layers.Dropout(dropout_rate)(x)

    _inject_suffix(model, suffix)

    return x, model.input


def create_siamese_model(base_model, image_size, trainable, dense_1, dense_2, activation, batch_norm, dropout_rate):
    output_left, input_left = _create_base_model(base_model, image_size, trainable, dense_1, dense_2, activation, batch_norm, dropout_rate, suffix="_left")
    output_right, input_right = _create_base_model(base_model, image_size, trainable, dense_1, dense_2, activation, batch_norm, dropout_rate, suffix="_right")

    distance = layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([output_left, output_right])
    
    outputs = layers.Dense(1, activation='sigmoid')(distance)
    
    return Model(inputs=[input_left, input_right], outputs=outputs)
