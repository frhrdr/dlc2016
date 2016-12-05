from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf
import numpy as np
import cifar10_utils
from convnet import ConvNet

LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 128
MAX_STEPS_DEFAULT = 15000
EVAL_FREQ_DEFAULT = 1000
CHECKPOINT_FREQ_DEFAULT = 5000
PRINT_FREQ_DEFAULT = 10
OPTIMIZER_DEFAULT = 'ADAM'

DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
LOG_DIR_DEFAULT = './logs/cifar10'
CHECKPOINT_DIR_DEFAULT = './checkpoints'

def train_step(loss):
    """
    Defines the ops to conduct an optimization step. You can set a learning
    rate scheduler or pick your favorite optimizer here. This set of operations
    should be applicable to both ConvNet() and Siamese() objects.

    Args:
        loss: scalar float Tensor, full loss = cross_entropy + reg_loss

    Returns:
        train_op: Ops for optimization.
    """
    ########################
    # PUT YOUR CODE HERE  #
    train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)
    ########################
    # END OF YOUR CODE    #
    ########################

    return train_op

def train():
    """
    Performs training and evaluation of ConvNet model.

    First define your graph using class ConvNet and its methods. Then define
    necessary operations such as trainer (train_step in this case), savers
    and summarizers. Finally, initialize your model within a tf.Session and
    do the training.

    ---------------------------
    How to evaluate your model:
    ---------------------------
    Evaluation on test set should be conducted over full batch, i.e. 10k images,
    while it is alright to do it over minibatch for train set.

    ---------------------------------
    How often to evaluate your model:
    ---------------------------------
    - on training set every print_freq iterations
    - on test set every eval_freq iterations

    ------------------------
    Additional requirements:
    ------------------------
    Also you are supposed to take snapshots of your model state (i.e. graph,
    weights and etc.) every checkpoint_freq iterations. For this, you should
    study TensorFlow's tf.train.Saver class. For more information, please
    checkout:
    [https://www.tensorflow.org/versions/r0.11/how_tos/variables/index.html]
    """

    # Set the random seeds for reproducibility. DO NOT CHANGE.
    tf.set_random_seed(42)
    np.random.seed(42)

    ########################
    # PUT YOUR CODE HERE  #
    ########################
    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
    cnn = ConvNet()
    data_dims = list(cifar10.train.images.shape[1:])
    with tf.Graph().as_default():
        x_pl = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size] + data_dims)
        y_pl = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, cnn.n_classes])

        logits = cnn.inference(x_pl)
        loss = cnn.loss(logits, y_pl)
        acc = cnn.accuracy(logits, y_pl)

        train_op = train_step(loss)
        summary_op = tf.merge_all_summaries()
        init_op = tf.initialize_all_variables()

        with tf.Session() as sess:
            saver = tf.train.Saver()
            sess.run(init_op)
            train_summary_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/train', sess.graph)
            test_summary_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/test', sess.graph)

            for step in range(FLAGS.max_steps):
                x, y = cifar10.train.next_batch(FLAGS.batch_size)
                feed = {x_pl: x, y_pl: y}
                train_loss, train_acc, summary_str, _ = sess.run([loss, acc, summary_op, train_op], feed_dict=feed)

                if step == 0 or (step + 1) % FLAGS.print_freq == 0 or step + 1 == FLAGS.max_steps:
                    print('TRAIN step: ', str(step), ' err: ', str(train_loss), ' acc: ', str(train_acc))
                    train_summary_writer.add_summary(summary_str, step)
                    train_summary_writer.flush()
                if step == 0 or (step + 1) % FLAGS.eval_freq == 0 or step + 1 == FLAGS.max_steps:
                    x, y = cifar10.test.images, cifar10.test.labels
                    num_batches = int(np.floor(x.shape[0] / FLAGS.batch_size))

                    test_err = 0.
                    test_acc = 0.
                    for idx in range(num_batches):

                        x_batch = x[idx * FLAGS.batch_size:(idx + 1) * FLAGS.batch_size, :, :, :]
                        y_batch = y[idx * FLAGS.batch_size:(idx + 1) * FLAGS.batch_size, :]
                        feed = {x_pl: x_batch, y_pl: y_batch}

                        batch_err, batch_acc = sess.run([loss, acc], feed_dict=feed)

                        test_err += batch_err
                        test_acc += batch_acc

                    test_err /= num_batches
                    test_acc /= num_batches
                    print('--- TEST --- step: ', str(step), ' err: ', str(train_loss), ' acc: ', str(train_acc))

                    summary_str = sess.run(summary_op, feed_dict=feed)  # possibly incorrect. should pool summaries
                    test_summary_writer.add_summary(summary_str, step)
                    test_summary_writer.flush()
                if (step + 1) % FLAGS.checkpoint_freq == 0 or step + 1 == FLAGS.max_steps:
                    checkpoint_file = os.path.join(FLAGS.checkpoint_dir, 'ckpt')
                    saver.save(sess, checkpoint_file, global_step=(step + 1))

    ########################
    # END OF YOUR CODE    #
    ########################


def train_siamese():
    """
    Performs training and evaluation of Siamese model.

    First define your graph using class Siamese and its methods. Then define
    necessary operations such as trainer (train_step in this case), savers
    and summarizers. Finally, initialize your model within a tf.Session and
    do the training.

    ---------------------------
    How to evaluate your model:
    ---------------------------
    On train set, it is fine to monitor loss over minibatches. On the other
    hand, in order to evaluate on test set you will need to create a fixed
    validation set using the data sampling function you implement for siamese
    architecture. What you need to do is to iterate over all minibatches in
    the validation set and calculate the average loss over all minibatches.

    ---------------------------------
    How often to evaluate your model:
    ---------------------------------
    - on training set every print_freq iterations
    - on test set every eval_freq iterations

    ------------------------
    Additional requirements:
    ------------------------
    Also you are supposed to take snapshots of your model state (i.e. graph,
    weights and etc.) every checkpoint_freq iterations. For this, you should
    study TensorFlow's tf.train.Saver class. For more information, please
    checkout:
    [https://www.tensorflow.org/versions/r0.11/how_tos/variables/index.html]
    """

    # Set the random seeds for reproducibility. DO NOT CHANGE.
    tf.set_random_seed(42)
    np.random.seed(42)

    ########################
    # PUT YOUR CODE HERE  #
    ########################
    raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    ########################


def feature_extraction(feature_op_name='d1_out', check_point_name='ckpt-15000'):
    """
    This method restores a TensorFlow checkpoint file (.ckpt) and rebuilds inference
    model with restored parameters. From then on you can basically use that model in
    any way you want, for instance, feature extraction, finetuning or as a submodule
    of a larger architecture. However, this method should extract features from a
    specified layer and store them in data files such as '.h5', '.npy'/'.npz'
    depending on your preference. You will use those files later in the assignment.

    Args:
        [optional]
    Returns:
        None
    """

    ########################
    # PUT YOUR CODE HERE  #
    ########################
    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
    x, y = cifar10.test.images, cifar10.test.labels
    cnn = ConvNet()
    data_dims = list(cifar10.train.images.shape[1:])

    with tf.Graph().as_default() as graph:

        x_pl = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size] + data_dims)
        cnn.inference(x_pl)
        print([n.name for n in tf.get_default_graph().as_graph_def().node])
        feature_op = graph.get_tensor_by_name(feature_op_name + ':0')

        num_samples = x.shape[0]
        assert num_samples % FLAGS.batch_size == 0, 'batch_size must be chosen to divide test set without rest'
        num_batches = num_samples / FLAGS.batch_size

        with tf.Session() as sess:

            saver = tf.train.Saver()

            saver.restore(sess, os.path.join(FLAGS.checkpoint_dir, check_point_name))
            feat_list = []

            for idx in range(num_batches):

                x_batch = x[idx * FLAGS.batch_size:(idx + 1) * FLAGS.batch_size, :, :, :]
                feed = {x_pl: x_batch}

                batch_features = sess.run([feature_op], feed_dict=feed)
                feat_list.append(batch_features)

            feat_x = np.concatenate(feat_list)
            file_name = feature_op_name + '_test_features'
            np.save(os.path.join(FLAGS.log_dir, file_name), feat_x)
    ########################
    # END OF YOUR CODE    #
    ########################


def initialize_folders():
    """
    Initializes all folders in FLAGS variable.
    """

    if not tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.MakeDirs(FLAGS.log_dir)

    if not tf.gfile.Exists(FLAGS.data_dir):
        tf.gfile.MakeDirs(FLAGS.data_dir)

    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))

def main(_):
    print_flags()

    initialize_folders()

    if eval(FLAGS.is_train):
        if FLAGS.train_model == 'linear':
            train()
        elif FLAGS.train_model == 'siamese':
            train_siamese()
        else:
            raise ValueError("--train_model argument can be linear or siamese")
    else:
        feature_extraction(FLAGS.extract_op)

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
    parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
    parser.add_argument('--print_freq', type = int, default = PRINT_FREQ_DEFAULT,
                      help='Frequency of evaluation on the train set')
    parser.add_argument('--eval_freq', type = int, default = EVAL_FREQ_DEFAULT,
                      help='Frequency of evaluation on the test set')
    parser.add_argument('--checkpoint_freq', type = int, default = CHECKPOINT_FREQ_DEFAULT,
                      help='Frequency with which the model state is saved.')
    parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
    parser.add_argument('--log_dir', type = str, default = LOG_DIR_DEFAULT,
                      help='Summaries log directory')
    parser.add_argument('--checkpoint_dir', type = str, default = CHECKPOINT_DIR_DEFAULT,
                      help='Checkpoint directory')
    parser.add_argument('--is_train', type = str, default = 'True',
                      help='Training or feature extraction')
    parser.add_argument('--train_model', type = str, default = 'linear',
                      help='Type of model. Possible options: linear and siamese')
    parser.add_argument('--extract_op', type = str, default = 'd1_out',                 # sorry, but this just
                      help='Name of operation for which features are extracted')        # makes things a lot cleaner
    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()
    ########################
