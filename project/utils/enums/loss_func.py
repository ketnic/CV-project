import tensorflow_addons as tfa
from tensorflow import keras
from enum import Enum


class ILossFunc:
    def get(self):
        pass


class CategoricalCrossentropyLoss(ILossFunc):
    def get(self):
        return 'categorical_crossentropy'


class KLDivergenceLoss(ILossFunc):
    def get(self):
        return keras.losses.kl_divergence


class BinaryCrossentropyLoss(ILossFunc):
    def get(self):
        return 'binary_crossentropy'


class ContrastiveLoss(ILossFunc):
    def get(self):
        return tfa.losses.contrastive_loss


class LossFunctions(Enum):
    categorical_crossentropy = CategoricalCrossentropyLoss()
    kl_divergence = KLDivergenceLoss()
    binary_crossentropy = BinaryCrossentropyLoss()
    contrastive_loss = ContrastiveLoss()
