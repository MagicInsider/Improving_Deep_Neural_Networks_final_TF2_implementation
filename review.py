import os
import sys
import re
import _pickle as pickle
import matplotlib.pyplot as plt
from utilities import get_root_path


def plot_graph(path):
    """
    Plots models runs results graphs, extracting history objects stored by pickle
    Histories must stay in their respective model directories, placed in {path} folder
    parameter:
    path -- path to scan for models histories
    """

    not_model_dir_regex = r"\.log$"
    histories = {}

    with os.scandir(path) as entries:
        for entry in entries:
            try:
                with open(path + entry.name + '/history.pkl', 'rb') as source:
                    histories['/'.join(entry.name.split('_')[:3])] = pickle.load(source)
            except NotADirectoryError:
                pass
    plt.figure(figsize=(11, 6))
    for key, item in enumerate(histories):
        history = histories[item]
        metrics = list(history.keys())

        for i, metric in enumerate(metrics):
            plt.subplot(1, len(metrics), i + 1)
            plt.plot(history[metric])
            plt.ylabel(metric)
            plt.xlabel('epoch')

    plt.legend(list(histories.keys()), loc='best')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    root_path = get_root_path()
    review_path = os.path.join(root_path, 'review/')
    plot_graph(review_path)
    sys.exit(0)


