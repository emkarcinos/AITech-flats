from datetime import timedelta

import matplotlib.pyplot as plt
import seaborn as sns

from metrics import Metrics


def plot_metrics(
        title: str,
        test_metrics: Metrics,
        train_metrics: Metrics,
        n_epochs: int,
        time: int,
        image_size: int,
        device: str = 'cpu'
):
    """
    Shows a plot from collected metrics
    :param title: Plot title
    :param test_metrics: A dict of keyed metric scores with arrays as values.
    Each metric should have the same # of items.
    Keys:
    - l - losses
    - a - accuracy scores
    - p - precision scores
    - r - recall scores
    - f - f scores
    :param train_metrics: A dict of keyed metric scores with arrays as values.
    See `test_metrics` for details
    :param n_epochs:
    :param time: Time taken to train the model in seconds
    :param image_size: A number corresponding to the size of images used to train the model
    :param device: What was used to train the model
    :return:
    """
    plt.style.use('classic')
    sns.set()

    fig, axis = plt.subplot_mosaic([['l', 'l'],
                                    ['a', 'p'],
                                    ['r', 'f']],
                                   constrained_layout=True, figsize=(10, 10))
    axis['l'].plot(test_metrics.loss)
    axis['l'].plot(train_metrics.loss)
    axis['l'].set_yscale('log')
    axis['l'].set_title("Loss")

    axis['a'].plot(test_metrics.accuracy)
    axis['a'].plot(train_metrics.accuracy)
    axis['a'].set_title("Accuracy")

    axis['p'].plot(test_metrics.precision)
    axis['p'].plot(train_metrics.precision)
    axis['p'].set_title("Precision")

    axis['r'].plot(test_metrics.recall)
    axis['r'].plot(train_metrics.recall)
    axis['r'].set_title("Recall")

    axis['f'].plot(test_metrics.f_score)
    axis['f'].plot(train_metrics.f_score)
    axis['f'].set_title("F-score")
    fig.tight_layout()
    fig.subplots_adjust(top=0.90, bottom=0.05)
    fig.suptitle(title, fontsize=24)
    fig.legend(axis, labels=['test', 'train'], loc="lower center")

    plt.text(0.30, 0.93,
             f'{device}, {image_size}x{image_size}, {n_epochs} iteracji, czas treningu: '
             f'{str(timedelta(seconds=time)).split(".", maxsplit=1)[0]}',
             fontsize=14,
             transform=plt.gcf().transFigure)
    plt.show()
