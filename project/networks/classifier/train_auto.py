import os

from networks.classifier.config.Config import Config
from networks.classifier.train_manual import train

from utils.enums.models import Models
from utils.enums.loss_func import LossFunctions
from utils.enums.optimizers import Optimizers
from utils.enums.activation_func import ActivationFunctions


def autotrain():
    pass
    # configs = [
    #     [
    #         Config(BASE_MODEL=Models.vgg19.value,     DENSE_1=None, DENSE_2=None),
    #         Config(BASE_MODEL=Models.resnet50.value,  DENSE_1=None, DENSE_2=None),
    #         Config(BASE_MODEL=Models.resnet101.value, DENSE_1=None, DENSE_2=None),
    #         Config(BASE_MODEL=Models.resnet152.value, DENSE_1=None, DENSE_2=None),
    #         Config(BASE_MODEL=Models.xception.value,  DENSE_1=None, DENSE_2=None),
    #     ],
    #     [
    #         Config(BASE_MODEL=Models.vgg19.value,     DENSE_1=128, DENSE_2=None),
    #         Config(BASE_MODEL=Models.resnet50.value,  DENSE_1=128, DENSE_2=None),
    #         Config(BASE_MODEL=Models.resnet101.value, DENSE_1=128, DENSE_2=None),
    #         Config(BASE_MODEL=Models.resnet152.value, DENSE_1=128, DENSE_2=None),
    #         Config(BASE_MODEL=Models.xception.value,  DENSE_1=128, DENSE_2=None),
    #     ],
    #     [
    #         Config(BASE_MODEL=Models.vgg19.value,     DENSE_1=256, DENSE_2=None),
    #         Config(BASE_MODEL=Models.resnet50.value,  DENSE_1=256, DENSE_2=None),
    #         Config(BASE_MODEL=Models.resnet101.value, DENSE_1=256, DENSE_2=None),
    #         Config(BASE_MODEL=Models.resnet152.value, DENSE_1=256, DENSE_2=None),
    #         Config(BASE_MODEL=Models.xception.value,  DENSE_1=256, DENSE_2=None),
    #     ],
    #     [
    #         Config(BASE_MODEL=Models.vgg19.value,     DENSE_1=512, DENSE_2=None),
    #         Config(BASE_MODEL=Models.resnet50.value,  DENSE_1=512, DENSE_2=None),
    #         Config(BASE_MODEL=Models.resnet101.value, DENSE_1=512, DENSE_2=None),
    #         Config(BASE_MODEL=Models.resnet152.value, DENSE_1=512, DENSE_2=None),
    #         Config(BASE_MODEL=Models.xception.value,  DENSE_1=512, DENSE_2=None),
    #     ],
    #     [
    #         Config(BASE_MODEL=Models.vgg19.value,     DENSE_1=256, DENSE_2=128),
    #         Config(BASE_MODEL=Models.resnet50.value,  DENSE_1=256, DENSE_2=128),
    #         Config(BASE_MODEL=Models.resnet101.value, DENSE_1=256, DENSE_2=128),
    #         Config(BASE_MODEL=Models.resnet152.value, DENSE_1=256, DENSE_2=128),
    #         Config(BASE_MODEL=Models.xception.value,  DENSE_1=256, DENSE_2=128),
    #     ],
    #     [
    #         Config(BASE_MODEL=Models.vgg19.value,     DENSE_1=512, DENSE_2=256),
    #         Config(BASE_MODEL=Models.resnet50.value,  DENSE_1=512, DENSE_2=256),
    #         Config(BASE_MODEL=Models.resnet101.value, DENSE_1=512, DENSE_2=256),
    #         Config(BASE_MODEL=Models.resnet152.value, DENSE_1=512, DENSE_2=256),
    #         Config(BASE_MODEL=Models.xception.value,  DENSE_1=512, DENSE_2=256),
    #     ]
    # ]

    # _autotrain(configs)

    # configs = [
    #     [
    #         Config(
    #             BASE_MODEL=Models.resnet101.value,
    #             DENSE_1=256,
    #             DENSE_2=None,
    #             DROPOUT_RATE=0.2,
    #             ACTIVATION=ActivationFunctions.relu.value
    #         ),
    #         Config(
    #             BASE_MODEL=Models.resnet101.value,
    #             DENSE_1=256,
    #             DENSE_2=None,
    #             DROPOUT_RATE=0.2,
    #             ACTIVATION=ActivationFunctions.leaky_relu.value
    #         ),
    #     ],
    #     [
    #         Config(
    #             BASE_MODEL=Models.resnet101.value,
    #             DENSE_1=256,
    #             DENSE_2=None,
    #             DROPOUT_RATE=0.5,
    #             ACTIVATION=ActivationFunctions.relu.value
    #         ),
    #         Config(
    #             BASE_MODEL=Models.resnet101.value,
    #             DENSE_1=256,
    #             DENSE_2=None,
    #             DROPOUT_RATE=0.5,
    #             ACTIVATION=ActivationFunctions.leaky_relu.value
    #         ),
    #     ]
    # ]

    # _autotrain(configs)

    # configs = [
    #     [
    #         # Config(
    #         #     BASE_MODEL=Models.resnet101.value,
    #         #     DENSE_1=256,
    #         #     DENSE_2=None,
    #         #     DROPOUT_RATE=0.5,
    #         #     ACTIVATION=ActivationFunctions.relu.value,
    #         #     LOSS=LossFunctions.categorical_crossentropy.value,
    #         #     LEARNING_RATE=0.01
    #         # ),
    #         Config(
    #             BASE_MODEL=Models.resnet101.value,
    #             DENSE_1=256,
    #             DENSE_2=None,
    #             DROPOUT_RATE=0.5,
    #             ACTIVATION=ActivationFunctions.relu.value,
    #             LOSS=LossFunctions.kl_divergence.value,
    #             LEARNING_RATE=0.01
    #         )
    #     ],
    #     [
    #         Config(
    #             BASE_MODEL=Models.resnet101.value,
    #             DENSE_1=256,
    #             DENSE_2=None,
    #             DROPOUT_RATE=0.5,
    #             ACTIVATION=ActivationFunctions.relu.value,
    #             LOSS=LossFunctions.categorical_crossentropy.value,
    #             LEARNING_RATE=0.001
    #         ),
    #         Config(
    #             BASE_MODEL=Models.resnet101.value,
    #             DENSE_1=256,
    #             DENSE_2=None,
    #             DROPOUT_RATE=0.5,
    #             ACTIVATION=ActivationFunctions.relu.value,
    #             LOSS=LossFunctions.kl_divergence.value,
    #             LEARNING_RATE=0.001
    #         )
    #     ],
    #     [
    #         Config(
    #             BASE_MODEL=Models.resnet101.value,
    #             DENSE_1=256,
    #             DENSE_2=None,
    #             DROPOUT_RATE=0.5,
    #             ACTIVATION=ActivationFunctions.relu.value,
    #             LOSS=LossFunctions.categorical_crossentropy.value,
    #             LEARNING_RATE=0.0001
    #         ),
    #         Config(
    #             BASE_MODEL=Models.resnet101.value,
    #             DENSE_1=256,
    #             DENSE_2=None,
    #             DROPOUT_RATE=0.5,
    #             ACTIVATION=ActivationFunctions.relu.value,
    #             LOSS=LossFunctions.kl_divergence.value,
    #             LEARNING_RATE=0.0001
    #         )
    #     ]
    # ]

    # _autotrain(configs)


def _autotrain(configs):
    for i, arr in enumerate(configs):
        for j, config in enumerate(arr):
            config.OUTPUT_DIR = os.path.sep.join([config.OUTPUT_DIR, f'{i + 1}_{j + 1}'])
            config.SAVE_MODEL = False
            config.SAVE_BEST_MODEL_ONLY = True
            config.SAVE_MODEL_INTO_OUTPUT = True
            config.SAVE_METRICS = True
            config.CREATE_AND_SAVE_CONFUSION_MATRIX = True
            config.SAVE_HISTORY = True
            train(config)
