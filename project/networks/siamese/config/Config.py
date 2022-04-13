import os

from config import (
    SIMILAR_PHOTOS, 
    MODELS,
    DATA as BASE_DATA, 
    SOURCE_PHOTOS
)

from utils.tools.convertible import join_to_path

from utils.enums.models import Models
from utils.enums.loss_func import LossFunctions
from utils.enums.optimizers import Optimizers
from utils.enums.activation_func import ActivationFunctions
from utils.enums.clusterizers import Сlusterizers


class Config:
    def __init__(self,
                 *,
                 PHOTOS=SOURCE_PHOTOS,
                 SIMILAR_PHOTOS=SIMILAR_PHOTOS,
                 DATA=join_to_path([BASE_DATA, 's_data']),
                 WEIGHTS_PATH=join_to_path([MODELS, 's_model_weights.npy']),

                 # DATA
                 BALANCED=True,
                 MAKE_ITSELF_PAIRS=True,
                 VALIDATION_SPLIT=0.2,
                 SEED=123,
                 DATASET_DIR=join_to_path([BASE_DATA, 'ds']),

                 # MODEL
                 BASE_MODEL=Models.resnet101.value,
                 DENSE_1=512,
                 DENSE_2=256,
                 ACTIVATION=ActivationFunctions.relu.value,
                 DROPOUT_RATE=None,
                 BATCH_NORM=True,
                 IMAGE_SHAPE=(224, 224, 3),
                 TRAINABLE=False,
                 LOSS=LossFunctions.binary_crossentropy.value,
                 OPTIMIZER=Optimizers.adam.value,
                 LEARNING_RATE=0.001,

                 # TRANING
                 BATCH_SIZE=32,
                 EPOCHS=100,
                 LOSS_LIMIT_ENABLED=False,
                 LOSS_LIMIT=0.38,
                 METRICS=[
                     'binary_accuracy',
                    #  keras.metrics.Precision(),
                    #  keras.metrics.Recall(),
                ],

                 # OUTPUT
                 OUTPUT_DIR=join_to_path([os.path.dirname(os.path.dirname(__file__)), 'output']),
                 SAVE_MODEL_WEIGHTS=True,
                 SAVE_MODEL_INTO_OUTPUT=True,
                 SAVE_METRICS=True,
                 CREATE_AND_SAVE_CONFUSION_MATRIX=True,
                 SAVE_HISTORY=True,

                 # CLUSTERING
                 CLUSTERER=Сlusterizers.dbscan.value,
                 DECISION_THRESHOLD=0.5
                 ):
        self.PHOTOS = PHOTOS
        self.SIMILAR_PHOTOS = SIMILAR_PHOTOS
        self.DATA = DATA
        self.WEIGHTS_PATH = WEIGHTS_PATH
        self.BALANCED = BALANCED
        self.MAKE_ITSELF_PAIRS = MAKE_ITSELF_PAIRS
        self.VALIDATION_SPLIT = VALIDATION_SPLIT
        self.SEED = SEED
        self.DATASET_DIR = DATASET_DIR
        self.BASE_MODEL = BASE_MODEL
        self.DENSE_1 = DENSE_1
        self.DENSE_2 = DENSE_2
        self.ACTIVATION = ACTIVATION
        self.DROPOUT_RATE = DROPOUT_RATE
        self.BATCH_NORM = BATCH_NORM
        self.IMAGE_SHAPE = IMAGE_SHAPE
        self.TRAINABLE = TRAINABLE
        self.LOSS = LOSS
        self.OPTIMIZER = OPTIMIZER
        self.LEARNING_RATE = LEARNING_RATE
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.LOSS_LIMIT_ENABLED = LOSS_LIMIT_ENABLED
        self.LOSS_LIMIT = LOSS_LIMIT
        self.METRICS = METRICS
        self.OUTPUT_DIR = OUTPUT_DIR
        self.SAVE_MODEL_WEIGHTS = SAVE_MODEL_WEIGHTS
        self.SAVE_MODEL_INTO_OUTPUT = SAVE_MODEL_INTO_OUTPUT
        self.SAVE_METRICS = SAVE_METRICS
        self.CREATE_AND_SAVE_CONFUSION_MATRIX = CREATE_AND_SAVE_CONFUSION_MATRIX
        self.SAVE_HISTORY = SAVE_HISTORY
        self.CLUSTERER = CLUSTERER
        self.DECISION_THRESHOLD = DECISION_THRESHOLD
