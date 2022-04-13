import numpy as np
import tensorflow as tf
from tensorflow import keras

from networks.classifier.config.train_config import train_config

from utils.tools.log import log
from utils.plots.image import plot_image


def test_erroneous(config=train_config):
    log('loading model...')
    model = keras.models.load_model(config.MODEL_PATH)

    log('creating datasset...')
    ds = keras.preprocessing.image_dataset_from_directory(
        config.DATA,
        image_size=config.IMAGE_SHAPE[:-1],
        label_mode='categorical',
        batch_size=1,
        shuffle=False
    )
    class_names = ds.class_names

    skip = 0
    take = 6000
    ds = ds.skip(skip).take(take)

    x = ds.map(lambda x, _: x)
    y = ds.map(lambda _, y: y[0])

    log('predicting...')
    pred = model.predict(x)
    pred = tf.data.Dataset.from_tensor_slices(pred)

    result_ds = tf.data.Dataset.zip((x, (y, pred)))
    iter = result_ds.as_numpy_iterator()
    for image, (test, pred) in iter:
        class_name_true = class_names[np.argmax(test)]
        class_name_pred = class_names[np.argmax(pred)]
        if class_name_true != class_name_pred:
            image = image[0]
            plot = plot_image(image)
            plot.title(f'{class_name_true} => {class_name_pred} | {list(zip(class_names, pred))}')
            plot.show()
