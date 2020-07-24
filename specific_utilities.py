import numpy as np
from h5py import File
import tensorflow as tf
import _pickle as pickle
import os
from utilities import join_omnilist_to_string_with_prefix


def load_data(root_path):
    """
    Loads datasets from local /datasets/train_signs.h5 and datasets/test_signs.h5
    Split into train and test parts
    Extracts classes -- list of classes for Softmax
    Flattens images into vectors, normalize vectors by maximum possible value for pixel intensity
    Converts train and test labels into one hot matrices
    :returns
    X_train, X_dev --
    Y_train, Y_dev --
    classes --
    """
    train_dataset_path = os.path.join(root_path, 'datasets/train_signs.h5')
    dev_dataset_path = os.path.join(root_path, 'datasets/test_signs.h5')

    train_dataset = File(train_dataset_path, "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # train set labels

    dev_dataset = File(dev_dataset_path, "r")
    dev_set_x_orig = np.array(dev_dataset["test_set_x"][:])  # dev set features
    dev_set_y_orig = np.array(dev_dataset["test_set_y"][:])  # dev set labels

    n_y = len(np.array(dev_dataset["list_classes"][:]))  # the number of Softmax classes

    # Flatten training and dev images into vectors
    X_train_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1)
    X_dev_flatten = dev_set_x_orig.reshape(dev_set_x_orig.shape[0], -1)

    # Normalize vectors, converting to float
    X_train = X_train_flatten / 255.
    X_dev = X_dev_flatten / 255.

    # Format training and dev labels as arrays
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    dev_set_y_orig = dev_set_y_orig.reshape((1, dev_set_y_orig.shape[0]))

    # Convert training and dev labels to one hot matrices with values [0..n_y], where n_y is the number of classes
    Y_train = convert_to_one_hot(train_set_y_orig, n_y)
    Y_dev = convert_to_one_hot(dev_set_y_orig, n_y)

    input_size = X_train.shape[1]

    return X_train, Y_train, X_dev, Y_dev, input_size


def convert_to_one_hot(Y, C):
    """
    Converts labels vector Y into one hot matrix Y
    :parameter
    Y -- vector of labels integers 0 to 5 shape (1, number of examples)
    C -- number of classes, integer
    :returns
    Y -- one hot matrix of shape (number of examples, number of classes)
    """

    Y = np.eye(C)[Y.reshape(-1)]

    return Y


def make_name_and_path(layers_string, drop_prob_layers, num_epochs, learning_rate, mini_batch_size, root_path):
    """
    Composes name of the model and path to the model directory
    :parameters
    layers_string, drop_prob_layers, num_epochs, learning_rate, minibatch_size, -- model parameters
    root_path -- path to the python script
    :return:
    name -- string, in the name_mask variable format, separated by '_'
    path -- string
    """

    dropout_string = join_omnilist_to_string_with_prefix(drop_prob_layers, '-')
    name_mask = [layers_string, dropout_string, learning_rate, mini_batch_size, num_epochs, ]
    name = join_omnilist_to_string_with_prefix(name_mask, '_')
    path = os.path.join(root_path, name)

    return name, path


def add_results_string(root_path, results, log_filename):
    """
    Procedure. Adds evaluation results to the log file name
    :parameters
    run_root_path - string
    model_name - string
    log_filename - string
    results - dictionary, results of the model evaluation
    """

    # composing results string, concatenating first literal of each metric and it's value
    log_string = [log_filename, ]
    for _, key in enumerate(results):
        log_string.append(key[:1].upper() + str(round(results[key], 3)))

    # compiling results strings
    log_filename_with_results = '_'.join(log_string)

    # adding results string to the log file and model directory names
    os.rename(os.path.join(root_path, (log_filename + '.log')),
              os.path.join(root_path, (log_filename_with_results + '.log'))
              )

    return


def save_model_and_history(model, model_path, model_training_history):
    model.save(
        model_path,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None,
    )
    print('Model has been saved to', model_path)

    try:
        with open(model_path + '/history.pkl', 'wb') as drop:
            drop.write(pickle.dumps(model_training_history.history))
            print('Model training history has been saved to ' + model_path + '/history.pkl')
    except IOError:
        print('Failed to save model training history as', model_path + '/history.pkl')

    return


def draw_model_diagram(model, run_root_path, layers_string):
    diagram_path_and_name = os.path.join(run_root_path, layers_string)
    print('diagram_path_and_name: ', diagram_path_and_name)
    tf.keras.utils.plot_model(model,
                              to_file=diagram_path_and_name + '.png',
                              show_shapes=True,
                              show_layer_names=True,
                              rankdir='TB',
                              expand_nested=False,
                              dpi=96
                              )
    return
