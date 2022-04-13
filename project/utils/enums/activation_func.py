import tensorflow as tf
from enum import Enum


class IActivationFunc:
    def get(self):
        pass


class ReLU(IActivationFunc):
    def get(self):
        return tf.keras.layers.ReLU()


class LeakyReLU(IActivationFunc):
    def get(self):
        return tf.keras.layers.LeakyReLU(alpha=0.2)


class ActivationFunctions(Enum):
    relu = ReLU()
    leaky_relu = LeakyReLU()
