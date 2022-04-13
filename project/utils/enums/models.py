import tensorflow
from enum import Enum


class IModel:
    def get(self, input_shape):
        pass


class VGG19(IModel):
    def get(self, input_shape):
        return tensorflow.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=input_shape)


class ResNet50(IModel):
    def get(self, input_shape):
        return tensorflow.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)


class ResNet101(IModel):
    def get(self, input_shape):
        return tensorflow.keras.applications.ResNet101(include_top=False, weights='imagenet', input_shape=input_shape)


class ResNet152(IModel):
    def get(self, input_shape):
        return tensorflow.keras.applications.ResNet152(include_top=False, weights='imagenet', input_shape=input_shape)


class Xception(IModel):
    def get(self, input_shape):
        return tensorflow.keras.applications.Xception(include_top=False, weights='imagenet', input_shape=input_shape)


class Models(Enum):
    vgg19 = VGG19()
    resnet50 = ResNet50()
    resnet101 = ResNet101()
    resnet152 = ResNet152()
    xception = Xception()
