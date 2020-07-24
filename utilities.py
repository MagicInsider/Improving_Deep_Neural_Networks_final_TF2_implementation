from time import gmtime, strftime
from copy import copy
import os


def join_list_to_string(list_, divider, prefix='none'):
    local_list = copy(list_)
    if prefix != 'none':
        local_list.insert(0, prefix)
    string = divider.join(map(str, local_list))

    return string


def make_name_and_path(layers_string, drop_prob_layers, num_epochs, learning_rate, minibatch_size, root_path):
    """
    Composes name of the model and path to the model directory
    :parameters
    layers_string, drop_prob_layers, num_epochs, learning_rate, minibatch_size, -- model parameters
    root_path -- path to the python script
    :return:
    name -- string, in the name_mask variable format, separated by '_'
    path -- string
    """

    dropout_string = join_list_to_string(drop_prob_layers, '-')
    name_mask = [layers_string, dropout_string, learning_rate, minibatch_size, num_epochs, ]
    name = join_list_to_string(name_mask, '_')
    path = os.path.join(root_path, name)

    return name, path


def get_readable_runtime(tic, toc):
    """
    :parameters
    tic -- start time, system time
    toc -- finish time, system time
    :returns
    runtime - string in format 'MM min SS sec'
    """
    runtime_raw = toc - tic
    runtime = str(int(runtime_raw // 60)) + ' min ' + str(int(runtime_raw % 60)) + ' sec'
    return runtime


def get_readable_date(system_time):
    """
    :parameter
    time -- time, system time
    :returns
    date-string -- string in format YYYY-MM-DD_HH-MM-SS
    """

    date_string = strftime("%Y-%m-%d_%H-%M-%S", gmtime(system_time))
    return date_string


def make_directory(path):
    try:
        os.umask(0)
        os.mkdir(path, mode=0o777)
    except FileExistsError:
        pass
    return