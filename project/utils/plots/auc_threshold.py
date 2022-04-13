import matplotlib.pyplot as plt


def plot_auc_threshold(thresholds, auc):
    plt.clf()
    plt.style.use('default')
    plt.plot(thresholds, auc)
    plt.xlabel('Threshold')
    plt.ylabel('AUC')
    return plt
