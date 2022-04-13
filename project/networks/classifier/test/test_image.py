import tensorflow as tf
from tensorflow import keras

from networks.classifier.config.test_config import test_config

from utils.tools.log import log
from utils.read_and_process_image import read_and_process_image


def test_image(config=test_config):
    log('loading model...')
    model = keras.models.load_model(config.MODEL_PATH)

    log('loading image...')
    img_path = r'https://remont-f.ru/upload/resize_cache/iblock/574/1000_800_1c90dcf07c205e9d57687f52ee182d65f/dizayn-interyera-5-komnatnoj-kvartiry-147-kv-m-foto-15-4097.jpg'
    image = read_and_process_image(img_path, config.IMAGE_SHAPE[:2])
    ds = tf.data.Dataset.from_tensors(image).batch(1)

    log('predicting...')
    pred = model.predict(ds)[0]
    print(list(zip(config.CLASSES, pred)))
