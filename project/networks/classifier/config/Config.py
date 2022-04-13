import os
from tensorflow import keras

from config import (
    SOURCE_PHOTOS, 
    FLAT_TYPES_CSV, 
    MODELS, 
    DATA as BASE_DATA
)

from utils.enums.models import Models
from utils.enums.loss_func import LossFunctions
from utils.enums.optimizers import Optimizers
from utils.enums.activation_func import ActivationFunctions

class Config:
    def __init__(self,
                 *,
                 PHOTOS=SOURCE_PHOTOS,
                 PHOTOS_CSV=FLAT_TYPES_CSV,
                 TRAIN_DATA=os.path.sep.join([BASE_DATA, 'c_data_train']),
                 VAL_DATA=os.path.sep.join([BASE_DATA, 'c_data_val']),
                 MODEL_PATH=os.path.sep.join([MODELS, 'c_model.h5']),
                 CLASSES = ['bathroom', 'bedroom', 'hallway', 'kitchen', 'livingroom'],

                 # DATA
                 VALIDATION_SPLIT=0.2,
                 BALANCE=True,
                 SEED=123,

                 # PREPROCESSIGN
                 ZCA_WHITENING=False,
                 ZCA_EPSILON=1e-06,
                 ROTATION_RANGE=0,  # in degree
                 SHIFT_RANGE=0.0,
                 SHEAR_RANGE=0,
                 ZOOM_RANGE=0.0,
                 HORIZONTAL_FLIP=False,
                 VERTICAL_FLIP=False,

                 # MODEL
                 IMAGE_SHAPE=(224, 224, 3),
                 TRAINABLE=False,
                 LOSS=LossFunctions.categorical_crossentropy.value,
                 OPTIMIZER=Optimizers.adam.value,
                 LEARNING_RATE=0.001,

                 # TRANING
                 BASE_MODEL=Models.vgg19.value,
                 DENSE_1=None,
                 DENSE_2=None,
                 ACTIVATION=ActivationFunctions.relu.value,
                 DROPOUT_RATE=0.2,
                 BATCH_SIZE=32,
                 EPOCHS=100,
                 LOSS_LIMIT_ENABLED=False,
                 LOSS_LIMIT=0.00001,
                 METRICS=[
                    'categorical_accuracy',
                    keras.metrics.Precision(),
                    keras.metrics.Recall()
                ],

                 # OUTPUT
                 OUTPUT_DIR=os.path.sep.join([os.path.dirname(os.path.dirname(__file__)), 'output']),
                 SAVE_MODEL=False,
                 SAVE_BEST_MODEL_ONLY=False,
                 SAVE_MODEL_INTO_OUTPUT=False,
                 SAVE_METRICS=True,
                 CREATE_AND_SAVE_CONFUSION_MATRIX=True,
                 SAVE_HISTORY=False
                 ):
        self.PHOTOS = PHOTOS
        self.PHOTOS_CSV = PHOTOS_CSV
        self.TRAIN_DATA = TRAIN_DATA
        self.VAL_DATA = VAL_DATA
        self.MODEL_PATH = MODEL_PATH
        self.CLASSES = CLASSES
        self.VALIDATION_SPLIT = VALIDATION_SPLIT
        self.BALANCE = BALANCE
        self.SEED = SEED
        self.ZCA_WHITENING = ZCA_WHITENING
        self.ZCA_EPSILON = ZCA_EPSILON
        self.ROTATION_RANGE = ROTATION_RANGE
        self.SHIFT_RANGE = SHIFT_RANGE
        self.SHEAR_RANGE = SHEAR_RANGE
        self.ZOOM_RANGE = ZOOM_RANGE
        self.HORIZONTAL_FLIP = HORIZONTAL_FLIP
        self.VERTICAL_FLIP = VERTICAL_FLIP
        self.IMAGE_SHAPE = IMAGE_SHAPE
        self.TRAINABLE = TRAINABLE
        self.LOSS = LOSS
        self.OPTIMIZER = OPTIMIZER
        self.LEARNING_RATE = LEARNING_RATE
        self.BASE_MODEL = BASE_MODEL
        self.DENSE_1 = DENSE_1
        self.DENSE_2 = DENSE_2
        self.ACTIVATION = ACTIVATION
        self.DROPOUT_RATE = DROPOUT_RATE
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.LOSS_LIMIT_ENABLED = LOSS_LIMIT_ENABLED
        self.LOSS_LIMIT = LOSS_LIMIT
        self.METRICS = METRICS
        self.OUTPUT_DIR = OUTPUT_DIR
        self.SAVE_MODEL = SAVE_MODEL
        self.SAVE_BEST_MODEL_ONLY = SAVE_BEST_MODEL_ONLY
        self.SAVE_MODEL_INTO_OUTPUT = SAVE_MODEL_INTO_OUTPUT
        self.SAVE_METRICS = SAVE_METRICS
        self.CREATE_AND_SAVE_CONFUSION_MATRIX = CREATE_AND_SAVE_CONFUSION_MATRIX
        self.SAVE_HISTORY = SAVE_HISTORY
