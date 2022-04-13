import tensorflow as tf
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as AUC


def create_auc_roc(model, dataset, batch_size=32, decision_threshold=0.5):
    y_test = dataset.map(lambda _, y: y)
    y_test = list(y_test.as_numpy_iterator())
    left_ds = dataset.map(lambda x, _: x[0])
    right_ds = dataset.map(lambda x, _: x[1])
    input = tf.data.Dataset.zip(((left_ds, right_ds), )).batch(batch_size)
    y_pred = model.predict(input).ravel()
    if isinstance(decision_threshold, float):
        y_pred = y_pred > decision_threshold
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        auc = AUC(fpr, tpr)
    elif isinstance(decision_threshold, list):
        auc_arr = []
        fpr_arr = []
        tpr_arr = []
        for d_t in decision_threshold:
            fpr, tpr, _ = roc_curve(y_test, y_pred > d_t)
            auc = AUC(fpr, tpr)
            auc_arr.append(auc)
            fpr_arr.append(fpr)
            tpr_arr.append(tpr)
        return auc_arr, fpr_arr, tpr_arr
