from time import gmtime, strftime
from copy import copy
import os
import sys


def join_omnilist_to_string_with_prefix(list_, divider, prefix='none'):
    local_list = copy(list_)
    if prefix != 'none':
        local_list.insert(0, prefix)
    string = divider.join(map(str, local_list))

    return string


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


def get_root_path():
    root_path = os.path.dirname(sys.argv[0])
    print('root_path:', root_path)
    return root_path


def make_directory(path):
    try:
        os.umask(0)
        os.mkdir(path, mode=0o777)
    except FileExistsError:
        pass
    return