import matplotlib.pyplot as plt


def plot_image(image):
    plt.clf()
    plt.style.use('default')
    plt.imshow(image.astype("uint8"))
    plt.axis("off")
    return plt
