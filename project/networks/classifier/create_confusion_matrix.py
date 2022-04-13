import numpy as np
from sklearn.metrics import confusion_matrix


def create_confusion_matrix(model, val_ds):
    test_data = list(zip(*[(x, y) for x, y in val_ds]))
    x_test = np.concatenate(test_data[0])
    y_test = np.concatenate(test_data[1])
    
    y_pred = model.predict(x_test)

    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)

    return confusion_matrix(y_pred, y_test)
