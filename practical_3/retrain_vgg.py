from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from vgg import load_pretrained_VGG16_pool5
import cifar10_utils
import tensorflow as tf
import numpy as np

LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 128
MAX_STEPS_DEFAULT = 15000
EVAL_FREQ_DEFAULT = 1000
CHECKPOINT_FREQ_DEFAULT = 5000
PRINT_FREQ_DEFAULT = 10
OPTIMIZER_DEFAULT = 'ADAM'
REFINE_AFTER_K_STEPS_DEFAULT = 0

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
    ########################
    train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)
    ########################
    # END OF YOUR CODE    #
    ########################

    return train_op


def fully_connected_layers(vgg_output):
    # dense layers
    with tf.name_scope('dense'):
        flat = tf.reshape(vgg_output, [vgg_output.get_shape()[0].value, -1], name='flat_out')

        xavier = tf.contrib.layers.xavier_initializer()
        const0 = tf.constant_initializer(0.)
        l2_reg = tf.contrib.layers.l2_regularizer(0.1)
        n_classes = 10

        with tf.name_scope('dense1'):
            w1 = tf.get_variable('w1', shape=[flat.get_shape()[1], 384], dtype=tf.float32,
                                 initializer=xavier, regularizer=l2_reg)
            b1 = tf.get_variable('b1', shape=[384], dtype=tf.float32,
                                 initializer=const0)
            fc1 = tf.nn.relu(tf.matmul(flat, w1) + b1, name='d1_out')

        # fc2	        Multiplication	[384, 192]
        #               ReLU
        with tf.name_scope('dense2'):
            w2 = tf.get_variable('w2', shape=[384, 192], dtype=tf.float32,
                                 initializer=xavier, regularizer=l2_reg)
            b2 = tf.get_variable('b2', shape=[192], dtype=tf.float32,
                                 initializer=const0)
            fc2 = tf.nn.relu(tf.matmul(fc1, w2) + b2, name='d2_out')

        # fc3	        Multiplication	[192, 10]
        with tf.name_scope('dense3'):
            w3 = tf.get_variable('w3', shape=[192, n_classes], dtype=tf.float32,
                                 initializer=xavier, regularizer=l2_reg)
            b3 = tf.get_variable('b3', shape=[n_classes], dtype=tf.float32,
                                 initializer=const0)
            fc3 = tf.matmul(fc2, w3) + b3
        return fc3


def vgg_loss(logits, labels):
    ce_loss = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
    ce_loss = tf.reduce_mean(ce_loss)
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_loss = tf.to_float(0.)
    if None not in reg_losses:  # this IS meant to switch while building the graph
        reg_loss = reduce(lambda x, y: tf.add(x, y), reg_losses)
    loss = ce_loss + reg_loss
    tf.scalar_summary('ce_loss', ce_loss)
    tf.scalar_summary('reg_loss', reg_loss)
    tf.scalar_summary('full_loss', loss)

    return loss


def accuracy(logits, labels):
        guesses = tf.argmax(logits, dimension=1)
        targets = tf.argmax(labels, dimension=1)
        score = tf.to_int32(tf.equal(guesses, targets))
        acc = tf.reduce_sum(score) / tf.size(score)

        tf.scalar_summary('accuracy', acc)
        return acc


def train():
    """
    Performs training and evaluation of your model.

    First define your graph using vgg.py with your fully connected layer.
    Then define necessary operations such as trainer (train_step in this case),
    savers and summarizers. Finally, initialize your model within a
    tf.Session and do the training.

    ---------------------------------
    How often to evaluate your model:
    ---------------------------------
    - on training set every PRINT_FREQ iterations
    - on test set every EVAL_FREQ iterations

    ---------------------------
    How to evaluate your model:
    ---------------------------
    Evaluation on test set should be conducted over full batch, i.e. 10k images,
    while it is alright to do it over minibatch for train set.
    """

    # Set the random seeds for reproducibility. DO NOT CHANGE.
    tf.set_random_seed(42)
    np.random.seed(42)

    ########################
    # PUT YOUR CODE HERE  #
    ########################
    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
    data_dims = list(cifar10.train.images.shape[1:])
    n_classes = 10
    with tf.Graph().as_default():
        x_pl = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size] + data_dims)
        y_pl = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, n_classes])
        stopgrads = tf.placeholder(dtype=tf.bool)
        pool5, assign_ops = load_pretrained_VGG16_pool5(x_pl, scope_name='vgg')
        pool5 = tf.cond(stopgrads, lambda: tf.stop_gradient(pool5), lambda: pool5)
        logits = fully_connected_layers(pool5)
        loss = vgg_loss(logits, y_pl)
        acc = accuracy(logits, y_pl)

        train_op = train_step(loss)
        summary_op = tf.merge_all_summaries()
        init_op = tf.initialize_all_variables()

        with tf.Session() as sess:
            saver = tf.train.Saver()
            sess.run(init_op)
            sess.run(assign_ops)
            train_summary_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/train', sess.graph)
            test_summary_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/test', sess.graph)

            for step in range(FLAGS.max_steps):
                x, y = cifar10.train.next_batch(FLAGS.batch_size)
                switch = True if step < FLAGS.refine_after_k else False
                feed = {x_pl: x, y_pl: y, stopgrads: switch}
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
                        feed = {x_pl: x_batch, y_pl: y_batch, stopgrads: True}

                        batch_err, batch_acc = sess.run([loss, acc], feed_dict=feed)

                        test_err += batch_err
                        test_acc += batch_acc
                        summary_str = sess.run(summary_op, feed_dict=feed)  # possibly incorrect. should pool summaries
                        test_summary_writer.add_summary(summary_str, step)
                        test_summary_writer.flush()
                    test_err /= num_batches
                    test_acc /= num_batches
                    print('--- TEST --- step: ', str(step), ' err: ', str(train_loss), ' acc: ', str(train_acc))

                    # summary_str = sess.run(summary_op, feed_dict=feed)  # possibly incorrect. should pool summaries
                    # test_summary_writer.add_summary(summary_str, step)
                    # test_summary_writer.flush()
                if (step + 1) % FLAGS.checkpoint_freq == 0 or step + 1 == FLAGS.max_steps:
                    checkpoint_file = os.path.join(FLAGS.checkpoint_dir, 'ckpt')
                    saver.save(sess, checkpoint_file, global_step=(step + 1))

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
    train()

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
    parser.add_argument('--refine_after_k', type = int, default = REFINE_AFTER_K_STEPS_DEFAULT,
                        help='Number of steps after which to refine VGG model parameters (default 0).')
    parser.add_argument('--checkpoint_freq', type = int, default = CHECKPOINT_FREQ_DEFAULT,
                        help='Frequency with which the model state is saved.')
    parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    parser.add_argument('--log_dir', type = str, default = LOG_DIR_DEFAULT,
                        help='Summaries log directory')
    parser.add_argument('--checkpoint_dir', type = str, default = CHECKPOINT_DIR_DEFAULT,
                        help='Checkpoint directory')

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()
