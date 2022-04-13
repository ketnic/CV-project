import tensorflow as tf
from enum import Enum


class IOptimizer:
    def get(self, learning_rate):
        pass


class Adam(IOptimizer):
    def get(self, learning_rate):
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)


class Optimizers(Enum):
    adam = Adam()
