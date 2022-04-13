import os
import sys


def disable_tf_logs():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def setup_np():
    import numpy as np
    np.load.__defaults__ = (None, True, True, 'ASCII')


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 0:
        setup_np()
        disable_tf_logs()
        from app.init import start_app
        start_app()
    elif 'dataset' in args:
        from merge_datasets import merge_datasets
        merge_datasets()
    if 'train' in args and '-c' in args:
        from networks.classifier.train_manual import train as train_c
        train_c()
    elif 'train' in args and '-s' in args:
        from networks.siamese.train_manual import train as train_s
        train_s()
#     elif 'autotrain' in args and '-c' in args:
#         disable_tf_logs()
#         from networks.classifier.train_auto import autotrain as start_c_autotrain
#         start_c_autotrain()
#     elif 'autotrain' in args and '-s' in args:
#         disable_tf_logs()
#         from networks.similarity.train_auto import autotrain as start_s_autotrain
#         start_s_autotrain()
    elif 'test' in args and '-c' in args:
        disable_tf_logs()
        from networks.classifier.test.test import test as test_c
        test_c()
    elif 'test' in args and '-cg' in args:
        disable_tf_logs()
        from networks.classifier.test.test_grad_cam import test_grad_cam
        test_grad_cam()
    elif 'test' in args and '-ce' in args:
        disable_tf_logs()
        from networks.classifier.test.test_erroneous import test_erroneous
        test_erroneous()
    elif 'test' in args and '-ci' in args:
        disable_tf_logs()
        from networks.classifier.test.test_image import test_image
        test_image()
    elif 'test' in args and '-s' in args:
        setup_np()
        disable_tf_logs()
        from networks.siamese.test.test import test as test_s
        test_s()
    elif 'test' in args and '-sar' in args:
        setup_np()
        disable_tf_logs()
        from networks.siamese.test.test_auc_roc import test_auc_roc
        test_auc_roc()
    elif 'test' in args and '-m' in args:
        setup_np()
        from networks.test_module import test_module
        test_module()
