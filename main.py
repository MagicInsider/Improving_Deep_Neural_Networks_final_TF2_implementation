import tensorflow as tf
import os
import sys
from time import time
from utilities import join_omnilist_to_string_with_prefix, make_directory, get_readable_runtime, get_readable_date
from specific_utilities import load_data, make_name_and_path, add_results_string
from specific_utilities import save_model_and_history, draw_model_diagram
from utilities import get_root_path
from review import plot_graph


def make_blocks(layers_units, drop_prob):
    basic_blocks = {}
    for i in range(1, len(drop_prob)):
        layer_input = tf.keras.Input(shape=(layers_units[i - 1],))
        x = tf.keras.layers.Dense(units=layers_units[i],
                                  kernel_initializer='glorot_normal',
                                  bias_initializer='zeros',
                                  name='Z' + str(i))(layer_input)
        x = tf.keras.layers.Dropout(rate=drop_prob[i])(x)
        x = tf.keras.layers.ReLU(name='A'+str(i))(x)
        layer_output = tf.keras.layers.BatchNormalization()(x)
        basic_blocks[i] = tf.keras.Model(inputs=layer_input, outputs=layer_output)

    return basic_blocks


root_path = get_root_path()
X_train, Y_train, X_dev, Y_dev, input_size = load_data(root_path)

# setting model and experiments series hyper-parameters
# parameters to tinker around: layers - number of units of hidden layers !NB input counts as layer0
#                              drop_prob_layers - drop probabilities for layer0 (input), layer1 and layer2
#                              learning_rate_range - learning rate to try on
#                              num_epochs - number of epochs of training
#                              mini_batch_size - self-explanatory
# options: verbose - .fit method parameter, google for more info
#          stdout_to_file - set to True to send output to log file, False for standard command line output
#          review_results - plot graphs for the run results
#          draw_model - draw model diagram into file

layers = [input_size, 64, 32, 16, 6, ]
drop_prob_layers = [.05, .1, .1, ]
learning_rate_range = [.05, .01, .002, ]
num_epochs = 200
mini_batch_size = 32
verbose = 1
stdout_to_file = False
review_results = True
draw_model = False

layers_string = join_omnilist_to_string_with_prefix(layers[1:], '-', prefix='mod')
run_root_path = os.path.join(root_path, layers_string)
print('run_root_path:', run_root_path)
make_directory(run_root_path)

# business loop
for learning_rate in learning_rate_range:

    # compiling the name and the path of the model for current run
    model_name, model_path = make_name_and_path(layers_string,
                                                drop_prob_layers,
                                                num_epochs,
                                                learning_rate,
                                                mini_batch_size,
                                                run_root_path)

    # checking if such model has already been trained: there's a directory name and history.pkl in it
    if os.path.isdir(model_path) or os.path.isfile(model_path + '/history.pkl'):
        print('Model', model_name, 'has been trained already')

    else:  # shadow: there's no such directory or the run hasn't been finished

        log_start_message = 'Model ' + model_name + ' training started ' + get_readable_date(time())

        if stdout_to_file:
            sys.stdout = open(os.path.join(run_root_path, (model_name + '.log')), 'a')
        else:
            with open(os.path.join(run_root_path, (model_name + '.log')), 'a') as logfile:
                logfile.write(log_start_message)

        print(log_start_message)

        local_tic = time()

        blocks = make_blocks(layers, drop_prob_layers)
        Magus = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(layers[0],), name='A0'),
                tf.keras.layers.Dropout(drop_prob_layers[0]),
                blocks[1],
                blocks[2],
                tf.keras.layers.Dense(layers[3],
                                      kernel_initializer='glorot_normal',
                                      bias_initializer='zeros',
                                      name='Z4'),
                tf.keras.layers.ReLU(name='A4'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(layers[4],
                                      kernel_initializer='glorot_normal',
                                      bias_initializer='zeros',
                                      name='Z5'),
                tf.keras.layers.Softmax(name='output')
            ]
        )

        Magus.compile(
            optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=
            [
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()
            ]
        )

        model_training_history = Magus.fit(
            X_train,
            Y_train,
            batch_size=mini_batch_size,
            epochs=num_epochs,
            verbose=verbose,
            shuffle=True
        )

        eval_results = Magus.evaluate(X_dev, Y_dev, verbose=0)
        eval_results_dict = dict(zip(Magus.metrics_names, eval_results))

        #  calculating harmonic mean of precision and recall = F measure, adding it to zipped eval_results dict
        try:
            f_measure = 2 * (eval_results[1] * eval_results[2]) / (eval_results[1] + eval_results[2])
        except ZeroDivisionError:
            f_measure = 0
        eval_results_dict['F-measure'] = f_measure

        print('Model evaluation results:', eval_results_dict)

        save_model_and_history(Magus, model_path, model_training_history)  # Yup, Magus. Some homage to the Fowles :)

        local_toc = time()
        print('Model trained and saved in {}\n'.format(get_readable_runtime(local_tic, local_toc)))

        # closing stdout redirect to the log file
        if stdout_to_file:
            sys.stdout.close()

        add_results_string(run_root_path, eval_results_dict, model_name)

if review_results:
    plot_graph(run_root_path + '/')

if draw_model:
    try:
        draw_model_diagram(Magus, run_root_path, layers_string)
    except NameError:
        print('Draw error')
        pass





