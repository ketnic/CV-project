import os
import json
import operator
import tensorflow as tf
from tensorflow import keras

from networks.classifier.config.train_config import train_config
from networks.classifier.process_photos import process_photos
from networks.classifier.create_model import create_model
from networks.classifier.create_confusion_matrix import create_confusion_matrix

from utils.tools.log import log
from utils.EarlyStoppingByMetricValue import EarlyStoppingByMetricValue
from utils.plots.confusion_matrix import plot_confusion_matrix
from utils.plots.ggplot import ggplot


def train(config=train_config):
    log('process photos...')
    process_photos(
        source=config.PHOTOS,
        train=config.TRAIN_DATA,
        val=config.VAL_DATA,
        val_split=config.VALIDATION_SPLIT,
        csv=config.PHOTOS_CSV,
        image_size=config.IMAGE_SHAPE[:-1],
        balance=config.BALANCE
    )

    log('creating datasets...')
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        zca_whitening=config.ZCA_WHITENING,
        zca_epsilon=config.ZCA_EPSILON,
        rotation_range=config.ROTATION_RANGE,
        width_shift_range=config.SHIFT_RANGE,
        height_shift_range=config.SHIFT_RANGE,
        shear_range=config.SHEAR_RANGE,
        zoom_range=config.ZOOM_RANGE,
        horizontal_flip=config.HORIZONTAL_FLIP,
        vertical_flip=config.VERTICAL_FLIP,
        validation_split=config.VALIDATION_SPLIT
    )

    train_generator = datagen.flow_from_directory(
        config.TRAIN_DATA,
        target_size=config.IMAGE_SHAPE[:-1],
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        seed=config.SEED
    )

    val_ds = datagen.flow_from_directory(
        config.VAL_DATA,
        target_size=config.IMAGE_SHAPE[:-1],
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        seed=config.SEED
    )

    log('building model...')
    model = create_model(
        base_model=config.BASE_MODEL.get(config.IMAGE_SHAPE),
        dense_1=config.DENSE_1,
        dense_2=config.DENSE_2,
        activation=config.ACTIVATION,
        dropout_rate=config.DROPOUT_RATE,
        trainable=config.TRAINABLE,
        output_len=len(config.CLASSES)
    )

    log('compile model...')
    model.compile(
        loss=config.LOSS.get(),
        optimizer=config.OPTIMIZER.get(config.LEARNING_RATE),
        metrics=config.METRICS
    )

    log('training model...')
    callbacks=[]
    if config.SAVE_BEST_MODEL_ONLY:
        callbacks.append(keras.callbacks.ModelCheckpoint(
            filepath=config.MODEL_PATH,
            # save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True
        ))
    if config.LOSS_LIMIT_ENABLED:
        callbacks.append(EarlyStoppingByMetricValue(value=config.LOSS_LIMIT))
    history = model.fit(
        train_generator,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=callbacks
    )

    if not os.path.exists(config.OUTPUT_DIR):
        log('create output folder')
        os.makedirs(config.OUTPUT_DIR)

    if config.SAVE_MODEL and not config.SAVE_BEST_MODEL_ONLY:
        log('saving model...')
        if config.SAVE_MODEL_INTO_OUTPUT:
            model_path = os.path.sep.join([config.OUTPUT_DIR, os.path.basename(config.MODEL_PATH)])
            model.save(model_path)
        else:
            model.save(config.MODEL_PATH)

    if config.SAVE_METRICS:
        log('creating & saving metrics plots...')
        for metric in ['loss'] + config.METRICS:
            if not isinstance(metric, str):
                metric = type(metric).__name__.lower()
            plot_path = os.path.sep.join([config.OUTPUT_DIR, f'{metric}.png'])
            ggplot(history, metric).savefig(plot_path)

    if config.CREATE_AND_SAVE_CONFUSION_MATRIX:
        log('creating confusion matrix...')
        cm = create_confusion_matrix(model, val_ds)
        log('saving confusion matrix...')
        cm_plot_path = os.path.sep.join([config.OUTPUT_DIR, 'confusion_matrix.png'])
        plot_confusion_matrix(cm, config.CLASSES).savefig(cm_plot_path)

    if config.SAVE_HISTORY:
        log('saving history...')
        h = history.history
        index, value = min(enumerate(h['val_loss']), key=operator.itemgetter(1))
        h['min_val_loss'] = value
        h['min_val_loss_index'] = index
        h['min_val_categorical_accuracy'] = h['val_categorical_accuracy'][index]
        h['min_val_recall'] = h['val_recall'][index]
        h['min_val_precision'] = h['val_precision'][index]
        json_path = os.path.sep.join([config.OUTPUT_DIR, 'history.json'])
        with open(json_path, 'w+') as fp:
            json.dump(h, fp)

    return history
