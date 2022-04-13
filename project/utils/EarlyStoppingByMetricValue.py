import tensorflow as tf
import warnings


class EarlyStoppingByMetricValue(tf.keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', value=0.001, verbose=0):
        super(tf.keras.callbacks.Callback, self).__init__()

        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn('Early stopping requires %s available!' %
                          self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print('Epoch %05d: early stopping THR' % epoch)
            self.model.stop_training = True
