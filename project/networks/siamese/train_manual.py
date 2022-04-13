import os
import json
import operator
import numpy as np

from networks.siamese.config.train_config import train_config
from networks.siamese.create_datesets import create_datasets
from networks.siamese.create_siamese_model import create_siamese_model
from networks.siamese.create_confusion_matrix import create_confusion_matrix

from utils.tools.log import log
from utils.EarlyStoppingByMetricValue import EarlyStoppingByMetricValue
from utils.plots.ggplot import ggplot
from utils.plots.confusion_matrix import plot_confusion_matrix


def train(config=train_config):
    log('creating datasets...')
    train_ds, val_ds = create_datasets(
        source=config.SIMILAR_PHOTOS, 
        dist=config.DATA, 
        balanced=config.BALANCED,
        image_size=config.IMAGE_SHAPE[:-1],
        dataset_dir=config.DATASET_DIR,
        validation_split=config.VALIDATION_SPLIT,
        paired_with_itself=config.MAKE_ITSELF_PAIRS
    )

    log("building model...")
    model = create_siamese_model(
        base_model=config.BASE_MODEL, 
        image_size=config.IMAGE_SHAPE[:-1],
        trainable=config.TRAINABLE,
        dense_1=config.DENSE_1,
        dense_2=config.DENSE_2,
        activation=config.ACTIVATION,
        batch_norm=config.BATCH_NORM,
        dropout_rate=config.DROPOUT_RATE
    )

    log('compiling model...')
    model.compile(
        loss=config.LOSS.get(),
        optimizer=config.OPTIMIZER.get(config.LEARNING_RATE),
        metrics=config.METRICS
    )

    log('training model...')
    callbacks=[]
    if config.LOSS_LIMIT_ENABLED:
        callbacks.append(EarlyStoppingByMetricValue(value=config.LOSS_LIMIT))
    history = model.fit(
        train_ds.batch(config.BATCH_SIZE),
        validation_data=val_ds.batch(config.BATCH_SIZE),
        epochs=config.EPOCHS,
        callbacks=callbacks
    )

    if not os.path.exists(config.OUTPUT_DIR):
        log('create output folder')
        os.makedirs(config.OUTPUT_DIR)
    
    if config.SAVE_MODEL_WEIGHTS:
        log('saving model weights...')
        if config.SAVE_MODEL_INTO_OUTPUT:
            model_path = os.path.sep.join([config.OUTPUT_DIR, os.path.basename(config.MODEL_PATH)])
            np.save(model_path, np.array(model.get_weights(), dtype='object'))
        else:
            if not os.path.exists(os.path.dirname(config.WEIGHTS_PATH)):
                os.mkdir(os.path.dirname(config.WEIGHTS_PATH))
            np.save(config.WEIGHTS_PATH, np.array(model.get_weights(), dtype='object'))

    if config.SAVE_METRICS:
        log('creating metrics plots...')
        for metric in ['loss'] + config.METRICS:
            if not isinstance(metric, str):
                metric = type(metric).__name__.lower()
            plot_path = os.path.sep.join([config.OUTPUT_DIR, f'{metric}.png'])
            ggplot(history, metric).savefig(plot_path)

    if config.CREATE_AND_SAVE_CONFUSION_MATRIX:
        log('creating confusion matrix...')
        cm = create_confusion_matrix(model, val_ds, threshold=config.DECISION_THRESHOLD)
        cm_plot_path = os.path.sep.join([config.OUTPUT_DIR, 'confusion_matrix.png'])
        plot_confusion_matrix(cm, [False, True]).savefig(cm_plot_path)
    
    if config.SAVE_HISTORY:
        h = history.history
        index, value = min(enumerate(h['val_loss']), key=operator.itemgetter(1))
        h['min_val_loss'] = value
        h['min_val_loss_index'] = index
        h['min_val_binary_accuracy'] = h['val_binary_accuracy'][index]
        # h['min_val_recall'] = h['val_recall'][index]
        # h['min_val_precision'] = h['val_precision'][index]
        json_path = os.path.sep.join([config.OUTPUT_DIR, 'history.json'])
        with open(json_path, 'w+') as fp:
            json.dump(h, fp)
    
    return history
