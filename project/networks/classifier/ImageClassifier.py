import numpy as np
import tensorflow as tf
from tensorflow import keras

from networks.classifier.config.test_config import test_config

from utils.tools.log import log
from utils.read_and_process_image import process_image


class ImageClassifier:
    model = None
    config = None

    def __init__(self, config=test_config) -> None:
        log('loading classifier model...')
        self.model = keras.models.load_model(config.MODEL_PATH)
        self.config = config

    def classify_images(self, images):
        log('image processing...')
        images = map(lambda image: process_image(image=image, img_size=self.config.IMAGE_SHAPE[:-1]), images)
        images = list(images)
        input = tf.convert_to_tensor(images, dtype=tf.float32)
        
        log('predicting...')
        result = self.model.predict(input)
        result = map(lambda y: np.argmax(y), result)
        result = map(lambda i: self.config.CLASSES[i], result)
        result = list(result)
        
        return result
