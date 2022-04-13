import matplotlib.pyplot as plt


def ggplot(history, metrics):
    history = history.history
    plt.clf()
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(history[metrics], 'ro', label=f'train_{metrics}')
    plt.plot(history[f'val_{metrics}'], 'bo', label=f'val_{metrics}')
    plt.title(f'Training {metrics.title()}')
    plt.xlabel('Epoch #')
    plt.ylabel(metrics.title())
    plt.legend(loc='lower left')
    return plt
