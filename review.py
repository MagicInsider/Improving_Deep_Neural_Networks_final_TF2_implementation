import os
import sys
import re
import _pickle as pickle
import matplotlib.pyplot as plt


def plot_graph(path):
    """
    Plots models runs results graphs, extracting history objects stored by pickle
    Histories must stay in their respective model directories, placed in {path} folder
    parameter:
    path -- path to scan for models histories
    """

    dir_regex = r"^mod"
    histories = {}

    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_dir and re.match(dir_regex, entry.name):
                with open(path + entry.name + '/history.pkl', 'rb') as source:
                    histories['/'.join(entry.name.split('_')[:3])] = pickle.load(source)
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
    root_path = os.path.dirname(sys.argv[0])
    review_path = os.path.join(root_path, 'review/')
    plot_graph(review_path)


