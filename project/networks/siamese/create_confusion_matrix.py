import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix


def create_confusion_matrix(model, val_ds, threshold):
    test_data = list(zip(*[(tf.expand_dims(l, axis=0), tf.expand_dims(r, axis=0), tf.expand_dims(y, axis=0)) for (l, r), y in val_ds]))
    x_left_test = np.concatenate(test_data[0])
    x_right_test = np.concatenate(test_data[1])
    y_test = np.concatenate(test_data[2])

    y_pred = model.predict((x_left_test, x_right_test))

    y_pred = y_pred > threshold

    return confusion_matrix(y_pred, y_test)
