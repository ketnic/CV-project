import os
import numpy as np
from tensorflow import keras

from networks.classifier.config.test_config import test_config

from utils.tools.log import log
from utils.read_and_process_image import read_and_process_image, extend_image, read_image
from utils.grad_cam import make_gradcam_heatmap, make_gradcam_image
from utils.plots.image import plot_image

def test_grad_cam(config=test_config):
    log('loading model...')
    model = keras.models.load_model(config.MODEL_PATH)

    log('loading image...')
    img_path = r'https://remont-f.ru/upload/resize_cache/iblock/574/1000_800_1c90dcf07c205e9d57687f52ee182d65f/dizayn-interyera-5-komnatnoj-kvartiry-147-kv-m-foto-15-4097.jpg'
    image = read_and_process_image(img_path=img_path, img_size=config.IMAGE_SHAPE[:-1])
    image = extend_image(image)

    layers = map(lambda l: l.name, model.layers)
    layers = filter(lambda l_name: 'conv' in l_name, layers)
    layers = list(layers)
    last_conv_layer_name = layers[-1]

    log('heatmapping...')
    heatmap = make_gradcam_heatmap(image, model, last_conv_layer_name)
    image = make_gradcam_image(read_image(img_path=img_path), heatmap)

    if not os.path.exists(config.OUTPUT_DIR):
        log('create folder for plots')
        os.makedirs(config.OUTPUT_DIR)

    log('saving image...')
    image = keras.preprocessing.image.array_to_img(image)
    image.save(os.path.sep.join([config.OUTPUT_DIR, 'grad_cam.png']))
