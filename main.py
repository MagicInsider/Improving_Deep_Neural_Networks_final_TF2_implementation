import tensorflow as tf
import os
import sys
from time import time
from utilities import join_list_to_string, make_directory, get_readable_runtime, get_readable_date
from specific_utilities import load_data, make_name_and_path, add_results_string
from specific_utilities import save_model_and_history, draw_model_diagram


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


script_root_path = os.path.dirname(sys.argv[0])

X_train, Y_train, X_dev, Y_dev, input_size = load_data(script_root_path)

# setting model and experiments series hyper-parameters
# parameters to tinker around: layers - number of units of hidden layers !NB input counts as layer0
#                              drop_prob_layers - drop probabilities for layer0 (input), layer1 and layer2
#                              learning_rate_range - learning rate to try on
#                              num_epochs - number of epochs of training
#                              mini_batch_size - self-explanatory
# options: verbose - .fit method parameter, google for more info
#          stdout_to_file - set to True if you want to examine details of the run later, False for standard command line

layers = [input_size, 64, 32, 16, 6, ]
drop_prob_layers = [.1, .2, .1, ]
learning_rate_range = [.1, .01, .001, ]
num_epochs = 3
mini_batch_size = 32
verbose = 1
stdout_to_file = True

layers_string = join_list_to_string(layers[1:], '-', prefix='mod')
run_root_path = os.path.join(script_root_path, layers_string)
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

        log_filename = 'run_' + get_readable_date(time())
        log_start_message = '======== Model ' + model_name + ' training started ' + get_readable_date(time())

        if stdout_to_file:
            sys.stdout = open(os.path.join(run_root_path, (log_filename + '.log')), 'w')
        else:
            with open(os.path.join(run_root_path, (log_filename + '.log')), 'w') as logfile:
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
            shuffle=True)

        eval_results = Magus.evaluate(X_dev, Y_dev, verbose=0)
        eval_results_dict = dict(zip(Magus.metrics_names, eval_results))

        #  calculating harmonic mean of precision and recall = F measure, adding it to zipped eval_results dict
        f_measure = 2 * (eval_results[1] * eval_results[2]) / (eval_results[1] + eval_results[2])
        eval_results_dict['F-measure'] = f_measure

        print('Model evaluation results:', eval_results_dict)

        save_model_and_history(Magus, model_path, model_training_history)  # Some homage to Fowles :)

        local_toc = time()
        print('Model trained and saved in {}\n'.format(get_readable_runtime(local_tic, local_toc)))

        # closing stdout redirect to the run logfile
        if stdout_to_file:
            sys.stdout.close()

        add_results_string(run_root_path, model_name, eval_results_dict, log_filename)

try:
    draw_model_diagram(Magus, run_root_path)
except NameError:
    pass





