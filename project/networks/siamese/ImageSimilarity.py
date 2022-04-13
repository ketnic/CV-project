import numpy as np
import tensorflow as tf

from networks.siamese.config.test_config import test_config
from networks.siamese.create_siamese_model import create_siamese_model

from utils.tools.log import log
from utils.tools.inline_print import inline_print, complete_inline_print
from utils.read_and_process_image import process_image


class ImageSimilarity:
    config = None
    model = None

    def __init__(self, config=test_config):
        log('loading siamese model...')
        self.model = create_siamese_model(
            base_model=config.BASE_MODEL, 
            image_size=config.IMAGE_SHAPE[:-1],
            trainable=config.TRAINABLE,
            dense_1=config.DENSE_1,
            dense_2=config.DENSE_2,
            activation=config.ACTIVATION,
            batch_norm=config.BATCH_NORM,
            dropout_rate=config.DROPOUT_RATE
        )
        self.model.set_weights(np.load(config.WEIGHTS_PATH))
        self.config = config

    def compare_images(self, images):
        log('image processing...')
        images = map(lambda image: process_image(image=image, img_size=self.config.IMAGE_SHAPE[:-1]), images)
        images = map(lambda image: tf.convert_to_tensor(image, dtype=tf.float32), images)
        images = list(images)

        length = len(images)
        distance_matrix = [[0] * length] * length
        count = sum(range(length))
        counter = 1

        log('image comparing...')
        for i in range(length):
            img1 = images[i]
            input1 = tf.convert_to_tensor([img1], dtype=tf.float32)
            for j in range(i + 1, len(images)):
                img2 = images[j]
                input2 = tf.convert_to_tensor([img2], dtype=tf.float32)
                inline_print(f'{counter}/{count}')
                distance = 1 - self.model.predict((input1, input2))[0][0]
                distance_matrix[i][j] = distance
                distance_matrix[j][i] = distance
                # print(f'dist btw {images_basenames[i]} and {images_basenames[j]} = {distance}')
                # input()
                counter += 1
        complete_inline_print()

        return distance_matrix
