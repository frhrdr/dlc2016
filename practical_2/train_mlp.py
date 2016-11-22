"""
This module implements training and evaluation of a multi-layer perceptron.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import tensorflow as tf
import numpy as np
from mlp import MLP
from cifar10_utils import load_cifar10, preprocess_cifar10_data, dense_to_one_hot, DataSet

# The default parameters are the same parameters that you used during practical 1.
# With these parameters you should get similar results as in the Numpy exercise.
### --- BEGIN default constants ---
LEARNING_RATE_DEFAULT = 2e-3
WEIGHT_REGULARIZER_STRENGTH_DEFAULT = 0.
WEIGHT_INITIALIZATION_SCALE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 200
MAX_STEPS_DEFAULT = 1500
DROPOUT_RATE_DEFAULT = 0.
DNN_HIDDEN_UNITS_DEFAULT = '100'
WEIGHT_INITIALIZATION_DEFAULT = 'normal'
WEIGHT_REGULARIZER_DEFAULT = 'l2'
ACTIVATION_DEFAULT = 'relu'
OPTIMIZER_DEFAULT = 'sgd'  # 'sgd'
### --- END default constants---

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
# Directory for tensorflow logs
LOG_DIR_DEFAULT = './logs/cifar10'


# This is the list of options for command line arguments specified below using argparse.
# Make sure that all these options are available so we can automatically test your code
# through command line arguments.

# You can check the TensorFlow API at
# https://www.tensorflow.org/versions/r0.11/api_docs/python/contrib.layers.html#initializers
# https://www.tensorflow.org/versions/r0.11/api_docs/python/state_ops.html#sharing-variables
WEIGHT_INITIALIZATION_DICT = {'xavier': lambda x: tf.contrib.layers.xavier_initializer(),  # Xavier initialisation
                              'normal': lambda x: tf.random_normal_initializer(stddev=x),  # Initialization from a standard normal
                              'uniform': lambda x: tf.random_uniform_initializer(minval=- x / 2., maxval=x / 2.),  # Initialization from a uniform distribution
                             }

# You can check the TensorFlow API at
# https://www.tensorflow.org/versions/r0.11/api_docs/python/contrib.layers.html#regularizers
# https://www.tensorflow.org/versions/r0.11/api_docs/python/state_ops.html#sharing-variables
WEIGHT_REGULARIZER_DICT = {'none': lambda x: lambda y: tf.to_float(0),  # No regularization
                           'l1': tf.contrib.layers.l1_regularizer,  # L1 regularization
                           'l2': tf.contrib.layers.l2_regularizer  # L2 regularization
                          }

# You can check the TensorFlow API at
# https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#activation-functions
ACTIVATION_DICT = {'relu': tf.nn.relu,  # ReLU
                   'elu': tf.nn.elu,  # ELU
                   'tanh': tf.tanh,  # Tanh
                   'sigmoid': tf.sigmoid}  # Sigmoid

# You can check the TensorFlow API at
# https://www.tensorflow.org/versions/r0.11/api_docs/python/train.html#optimizers
OPTIMIZER_DICT = {'sgd': tf.train.GradientDescentOptimizer,  # Gradient Descent
                  'adadelta': tf.train.AdadeltaOptimizer,  # Adadelta
                  'adagrad': tf.train.AdagradOptimizer,  # Adagrad
                  'adam': tf.train.AdamOptimizer,  # Adam
                  'rmsprop': tf.train.RMSPropOptimizer  # RMSprop
                  }

FLAGS = None


def train():
  """
  Performs training and evaluation of MLP model. Evaluate your model each 100 iterations
  as you did in the practical 1. This time evaluate your model on the whole test set.
  """
  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  tf.set_random_seed(42)
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  # load and prep data
  n_classes = 10
  n_data_dims = 3072
  data = load_cifar10(FLAGS.data_dir)
  X_train, Y_train, X_test, Y_test = preprocess_cifar10_data(*data)
  # X_train, Y_train, X_test, Y_test = load_cifar10(FLAGS.data_dir)
  Y_train = dense_to_one_hot(Y_train, n_classes)
  Y_test = dense_to_one_hot(Y_test, n_classes)
  train_set = DataSet(X_train, Y_train)
  test_set = DataSet(X_test, Y_test)
  # build model
  with tf.Graph().as_default():

    x_pl = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, n_data_dims))
    y_pl = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, n_classes))

    model = MLP(n_hidden=dnn_hidden_units,
                n_classes=n_classes,
                is_training=tf.placeholder(tf.bool),
                activation_fn=ACTIVATION_DICT[FLAGS.activation],
                dropout_rate=FLAGS.dropout_rate,
                weight_initializer=WEIGHT_INITIALIZATION_DICT[
                    FLAGS.weight_init](FLAGS.weight_init_scale),
                weight_regularizer= WEIGHT_REGULARIZER_DICT[
                    FLAGS.weight_reg](FLAGS.weight_reg_strength)
                )
    logits = model.inference(x_pl)
    loss = model.loss(logits, y_pl)
    train_op = OPTIMIZER_DICT[FLAGS.optimizer](FLAGS.learning_rate).minimize(loss)

    acc = model.accuracy(logits, y_pl)

    summary_op = tf.merge_all_summaries()

    with tf.Session() as sess:

      train_summary_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/train', sess.graph)
      test_summary_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/test', sess.graph)
      init_op = tf.initialize_all_variables()
      sess.run(init_op)

      for step in range(FLAGS.max_steps):
        x_batch, y_batch = train_set.next_batch(FLAGS.batch_size)
        feed = {x_pl: np.reshape(x_batch, (FLAGS.batch_size, n_data_dims)),
                y_pl: y_batch,
                model.is_training: True}
        _, train_err, train_acc, summary_str = sess.run([train_op, loss, acc, summary_op], feed_dict=feed)

        train_summary_writer.add_summary(summary_str, step)
        train_summary_writer.flush()

        if step % 100 == 0:
          epoch = test_set.epochs_completed
          batch_count = 0.
          test_err = 0.
          test_acc = 0.
          while epoch == test_set.epochs_completed:
            x_batch, y_batch = test_set.next_batch(FLAGS.batch_size)
            feed = {x_pl: np.reshape(x_batch, (FLAGS.batch_size, n_data_dims)),
                    y_pl: y_batch,
                    model.is_training: False}

            batch_err, batch_acc = sess.run([loss, acc], feed_dict=feed)

            batch_count += 1.
            test_err += batch_err
            test_acc += batch_acc

          test_err /= batch_count
          test_acc /= batch_count
          print('iteration ' + str(step) +
                ' \n train error: ' + str(train_err) +
                ' train accuracy: ' + str(train_acc) +
                ' \n test error: ' + str(test_err) +
                ' test accuracy: ' + str(test_acc))

          summary_str = sess.run(summary_op, feed_dict=feed)
          test_summary_writer.add_summary(summary_str, step)
          test_summary_writer.flush()

  ########################
  # END OF YOUR CODE    #
  #######################


def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))


def main(_):
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  # Make directories if they do not exists yet
  if not tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.MakeDirs(FLAGS.log_dir)
  if not tf.gfile.Exists(FLAGS.data_dir):
    tf.gfile.MakeDirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--weight_init', type = str, default = WEIGHT_INITIALIZATION_DEFAULT,
                      help='Weight initialization type [xavier, normal, uniform].')
  parser.add_argument('--weight_init_scale', type = float, default = WEIGHT_INITIALIZATION_SCALE_DEFAULT,
                      help='Weight initialization scale (e.g. std of a Gaussian).')
  parser.add_argument('--weight_reg', type = str, default = WEIGHT_REGULARIZER_DEFAULT,
                      help='Regularizer type for weights of fully-connected layers [none, l1, l2].')
  parser.add_argument('--weight_reg_strength', type = float, default = WEIGHT_REGULARIZER_STRENGTH_DEFAULT,
                      help='Regularizer strength for weights of fully-connected layers.')
  parser.add_argument('--dropout_rate', type = float, default = DROPOUT_RATE_DEFAULT,
                      help='Dropout rate.')
  parser.add_argument('--activation', type = str, default = ACTIVATION_DEFAULT,
                      help='Activation function [relu, elu, tanh, sigmoid].')
  parser.add_argument('--optimizer', type = str, default = OPTIMIZER_DEFAULT,
                      help='Optimizer to use [sgd, adadelta, adagrad, adam, rmsprop].')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  parser.add_argument('--log_dir', type = str, default = LOG_DIR_DEFAULT,
                      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()

  tf.app.run()
