import os

from networks.siamese.config.Config import Config
from networks.siamese.train_manual import train

from utils.enums.models import Models
from utils.enums.loss_func import LossFunctions
from utils.enums.optimizers import Optimizers
from utils.enums.activation_func import ActivationFunctions


def autotrain():
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

    configs = [
        [
            # Config(BASE_MODEL=Models.vgg19.value,     DENSE_1=None, DENSE_2=None, BATCH_NORM=True),
            # Config(BASE_MODEL=Models.resnet50.value,  DENSE_1=None, DENSE_2=None, BATCH_NORM=True),
            # Config(BASE_MODEL=Models.resnet101.value, DENSE_1=None, DENSE_2=None, BATCH_NORM=True),
            # Config(BASE_MODEL=Models.resnet152.value, DENSE_1=None, DENSE_2=None, BATCH_NORM=True),
            # Config(BASE_MODEL=Models.xception.value,  DENSE_1=None, DENSE_2=None, BATCH_NORM=True),
        ],
        [
            Config(BASE_MODEL=Models.vgg19.value,     DENSE_1=128, DENSE_2=None, BATCH_NORM=True),
            Config(BASE_MODEL=Models.resnet50.value,  DENSE_1=128, DENSE_2=None, BATCH_NORM=True),
            Config(BASE_MODEL=Models.resnet101.value, DENSE_1=128, DENSE_2=None, BATCH_NORM=True),
            Config(BASE_MODEL=Models.resnet152.value, DENSE_1=128, DENSE_2=None, BATCH_NORM=True),
            Config(BASE_MODEL=Models.xception.value,  DENSE_1=128, DENSE_2=None, BATCH_NORM=True),
        ],
        [
            Config(BASE_MODEL=Models.vgg19.value,     DENSE_1=256, DENSE_2=None, BATCH_NORM=True),
            Config(BASE_MODEL=Models.resnet50.value,  DENSE_1=256, DENSE_2=None, BATCH_NORM=True),
            Config(BASE_MODEL=Models.resnet101.value, DENSE_1=256, DENSE_2=None, BATCH_NORM=True),
            Config(BASE_MODEL=Models.resnet152.value, DENSE_1=256, DENSE_2=None, BATCH_NORM=True),
            Config(BASE_MODEL=Models.xception.value,  DENSE_1=256, DENSE_2=None, BATCH_NORM=True),
        ],
        [
            Config(BASE_MODEL=Models.vgg19.value,     DENSE_1=512, DENSE_2=None, BATCH_NORM=True),
            Config(BASE_MODEL=Models.resnet50.value,  DENSE_1=512, DENSE_2=None, BATCH_NORM=True),
            Config(BASE_MODEL=Models.resnet101.value, DENSE_1=512, DENSE_2=None, BATCH_NORM=True),
            Config(BASE_MODEL=Models.resnet152.value, DENSE_1=512, DENSE_2=None, BATCH_NORM=True),
            Config(BASE_MODEL=Models.xception.value,  DENSE_1=512, DENSE_2=None, BATCH_NORM=True),
        ],
        [
            Config(BASE_MODEL=Models.vgg19.value,     DENSE_1=256, DENSE_2=128, BATCH_NORM=True),
            Config(BASE_MODEL=Models.resnet50.value,  DENSE_1=256, DENSE_2=128, BATCH_NORM=True),
            Config(BASE_MODEL=Models.resnet101.value, DENSE_1=256, DENSE_2=128, BATCH_NORM=True),
            Config(BASE_MODEL=Models.resnet152.value, DENSE_1=256, DENSE_2=128, BATCH_NORM=True),
            Config(BASE_MODEL=Models.xception.value,  DENSE_1=256, DENSE_2=128, BATCH_NORM=True),
        ],
        [
            Config(BASE_MODEL=Models.vgg19.value,     DENSE_1=512, DENSE_2=256, BATCH_NORM=True),
            Config(BASE_MODEL=Models.resnet50.value,  DENSE_1=512, DENSE_2=256, BATCH_NORM=True),
            Config(BASE_MODEL=Models.resnet101.value, DENSE_1=512, DENSE_2=256, BATCH_NORM=True),
            Config(BASE_MODEL=Models.resnet152.value, DENSE_1=512, DENSE_2=256, BATCH_NORM=True),
            Config(BASE_MODEL=Models.xception.value,  DENSE_1=512, DENSE_2=256, BATCH_NORM=True),
        ],
        [
            # Config(BASE_MODEL=Models.vgg19.value,     DENSE_1=None, DENSE_2=None, DROPOUT_RATE=None, BATCH_NORM=True),
            # Config(BASE_MODEL=Models.resnet50.value,  DENSE_1=None, DENSE_2=None, DROPOUT_RATE=None, BATCH_NORM=True),
            # Config(BASE_MODEL=Models.resnet101.value, DENSE_1=None, DENSE_2=None, DROPOUT_RATE=None, BATCH_NORM=True),
            # Config(BASE_MODEL=Models.resnet152.value, DENSE_1=None, DENSE_2=None, DROPOUT_RATE=None, BATCH_NORM=True),
            # Config(BASE_MODEL=Models.xception.value,  DENSE_1=None, DENSE_2=None, DROPOUT_RATE=None, BATCH_NORM=True),
        ],
        [
            Config(BASE_MODEL=Models.vgg19.value,     DENSE_1=128, DENSE_2=None, DROPOUT_RATE=None, BATCH_NORM=True),
            Config(BASE_MODEL=Models.resnet50.value,  DENSE_1=128, DENSE_2=None, DROPOUT_RATE=None, BATCH_NORM=True),
            Config(BASE_MODEL=Models.resnet101.value, DENSE_1=128, DENSE_2=None, DROPOUT_RATE=None, BATCH_NORM=True),
            Config(BASE_MODEL=Models.resnet152.value, DENSE_1=128, DENSE_2=None, DROPOUT_RATE=None, BATCH_NORM=True),
            Config(BASE_MODEL=Models.xception.value,  DENSE_1=128, DENSE_2=None, DROPOUT_RATE=None, BATCH_NORM=True),
        ],
        [
            Config(BASE_MODEL=Models.vgg19.value,     DENSE_1=256, DENSE_2=None, DROPOUT_RATE=None, BATCH_NORM=True),
            Config(BASE_MODEL=Models.resnet50.value,  DENSE_1=256, DENSE_2=None, DROPOUT_RATE=None, BATCH_NORM=True),
            Config(BASE_MODEL=Models.resnet101.value, DENSE_1=256, DENSE_2=None, DROPOUT_RATE=None, BATCH_NORM=True),
            Config(BASE_MODEL=Models.resnet152.value, DENSE_1=256, DENSE_2=None, DROPOUT_RATE=None, BATCH_NORM=True),
            Config(BASE_MODEL=Models.xception.value,  DENSE_1=256, DENSE_2=None, DROPOUT_RATE=None, BATCH_NORM=True),
        ],
        [
            Config(BASE_MODEL=Models.vgg19.value,     DENSE_1=512, DENSE_2=None, DROPOUT_RATE=None, BATCH_NORM=True),
            Config(BASE_MODEL=Models.resnet50.value,  DENSE_1=512, DENSE_2=None, DROPOUT_RATE=None, BATCH_NORM=True),
            Config(BASE_MODEL=Models.resnet101.value, DENSE_1=512, DENSE_2=None, DROPOUT_RATE=None, BATCH_NORM=True),
            Config(BASE_MODEL=Models.resnet152.value, DENSE_1=512, DENSE_2=None, DROPOUT_RATE=None, BATCH_NORM=True),
            Config(BASE_MODEL=Models.xception.value,  DENSE_1=512, DENSE_2=None, DROPOUT_RATE=None, BATCH_NORM=True),
        ],
        [
            Config(BASE_MODEL=Models.vgg19.value,     DENSE_1=256, DENSE_2=128, DROPOUT_RATE=None, BATCH_NORM=True),
            Config(BASE_MODEL=Models.resnet50.value,  DENSE_1=256, DENSE_2=128, DROPOUT_RATE=None, BATCH_NORM=True),
            Config(BASE_MODEL=Models.resnet101.value, DENSE_1=256, DENSE_2=128, DROPOUT_RATE=None, BATCH_NORM=True),
            Config(BASE_MODEL=Models.resnet152.value, DENSE_1=256, DENSE_2=128, DROPOUT_RATE=None, BATCH_NORM=True),
            Config(BASE_MODEL=Models.xception.value,  DENSE_1=256, DENSE_2=128, DROPOUT_RATE=None, BATCH_NORM=True),
        ],
        [
            Config(BASE_MODEL=Models.vgg19.value,     DENSE_1=512, DENSE_2=256, DROPOUT_RATE=None, BATCH_NORM=True),
            Config(BASE_MODEL=Models.resnet50.value,  DENSE_1=512, DENSE_2=256, DROPOUT_RATE=None, BATCH_NORM=True),
            Config(BASE_MODEL=Models.resnet101.value, DENSE_1=512, DENSE_2=256, DROPOUT_RATE=None, BATCH_NORM=True),
            Config(BASE_MODEL=Models.resnet152.value, DENSE_1=512, DENSE_2=256, DROPOUT_RATE=None, BATCH_NORM=True),
            Config(BASE_MODEL=Models.xception.value,  DENSE_1=512, DENSE_2=256, DROPOUT_RATE=None, BATCH_NORM=True),
        ]
    ]
    _autotrain(configs)

    # configs = [
    #     [
    #         # Config(BASE_MODEL=Models.resnet50.value, DENSE_1=512, DENSE_2=256, DROPOUT_RATE=None, BATCH_NORM=False),
    #         Config(BASE_MODEL=Models.resnet50.value, DENSE_1=512, DENSE_2=256, DROPOUT_RATE=None, BATCH_NORM=True)
    #     ],
    #     [
    #         Config(BASE_MODEL=Models.resnet50.value, DENSE_1=512, DENSE_2=256, DROPOUT_RATE=0.2, BATCH_NORM=False),
    #         Config(BASE_MODEL=Models.resnet50.value, DENSE_1=512, DENSE_2=256, DROPOUT_RATE=0.2, BATCH_NORM=True)
    #     ],
    #     [
    #         Config(BASE_MODEL=Models.resnet50.value, DENSE_1=512, DENSE_2=256, DROPOUT_RATE=0.5, BATCH_NORM=False),
    #         Config(BASE_MODEL=Models.resnet50.value, DENSE_1=512, DENSE_2=256, DROPOUT_RATE=0.5, BATCH_NORM=True)
    #     ]
    # ]
    # _autotrain(configs)

    # configs = [
    #     [
    #         Config(BASE_MODEL=Models.resnet50.value, DENSE_1=512, DENSE_2=256, DROPOUT_RATE=0.2, BATCH_NORM=True, LOSS=LossFunctions.binary_crossentropy.value, LEARNING_RATE=0.01),
    #         Config(BASE_MODEL=Models.resnet50.value, DENSE_1=512, DENSE_2=256, DROPOUT_RATE=None, BATCH_NORM=True, LOSS=LossFunctions.categorical_crossentropy.value, LEARNING_RATE=0.01)
    #     ],
    #     [
    #         Config(BASE_MODEL=Models.resnet50.value, DENSE_1=512, DENSE_2=256, DROPOUT_RATE=0.2, BATCH_NORM=True, LOSS=LossFunctions.binary_crossentropy.value, LEARNING_RATE=0.001),
    #         Config(BASE_MODEL=Models.resnet50.value, DENSE_1=512, DENSE_2=256, DROPOUT_RATE=None, BATCH_NORM=True, LOSS=LossFunctions.categorical_crossentropy.value, LEARNING_RATE=0.001)
    #     ],
    #     [
    #         Config(BASE_MODEL=Models.resnet50.value, DENSE_1=512, DENSE_2=256, DROPOUT_RATE=0.2, BATCH_NORM=True, LOSS=LossFunctions.binary_crossentropy.value, LEARNING_RATE=0.0001),
    #         Config(BASE_MODEL=Models.resnet50.value, DENSE_1=512, DENSE_2=256, DROPOUT_RATE=None, BATCH_NORM=True, LOSS=LossFunctions.categorical_crossentropy.value, LEARNING_RATE=0.0001)
    #     ]
    # ]
    # _autotrain(configs)


def _autotrain(configs):
    for i, arr in enumerate(configs):
        for j, config in enumerate(arr):
            config.OUTPUT_DIR = os.path.sep.join([config.OUTPUT_DIR, f'{i + 1}_{j + 1}'])
            config.WEIGHTS_PATH = os.path.sep.join([config.OUTPUT_DIR, 'weights.npy'])
            config.SAVE_MODEL_WEIGHTS = False
            config.SAVE_MODEL_INTO_OUTPUT = True
            config.SAVE_METRICS = True
            config.CREATE_AND_SAVE_CONFUSION_MATRIX = True
            config.SAVE_HISTORY = True
            train(config)
