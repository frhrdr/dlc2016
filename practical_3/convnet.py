from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


class ConvNet(object):
    """
   This class implements a convolutional neural network in TensorFlow.
   It incorporates a certain graph model to be trained and to be used
   in inference.
    """

    def __init__(self, n_classes = 10):
        """
        Constructor for an ConvNet object. Default values should be used as hints for
        the usage of each parameter.
        Args:
          n_classes: int, number of classes of the classification problem.
                          This number is required in order to specify the
                          output dimensions of the ConvNet.
        """
        self.n_classes = n_classes

    def inference(self, x):
        """
        Performs inference given an input tensor. This is the central portion
        of the network where we describe the computation graph. Here an input
        tensor undergoes a series of convolution, pooling and nonlinear operations
        as defined in this method. For the details of the model, please
        see assignment file.

        Here we recommend you to consider using variable and name scopes in order
        to make your graph more intelligible for later references in TensorBoard
        and so on. You can define a name scope for the whole model or for each
        operator group (e.g. conv+pool+relu) individually to group them by name.
        Variable scopes are essential components in TensorFlow for parameter sharing.
        Although the model(s) which are within the scope of this class do not require
        parameter sharing it is a good practice to use variable scope to encapsulate
        model.

        Args:
          x: 4D float Tensor of size [batch_size, input_height, input_width, input_channels]

        Returns:
          logits: 2D float Tensor of size [batch_size, self.n_classes]. Returns
                  the logits outputs (before softmax transformation) of the
                  network. These logits can then be used with loss and accuracy
                  to evaluate the model.
        """
        with tf.variable_scope('ConvNet'):
            ########################
            # PUT YOUR CODE HERE  #
            ########################
            xavier = tf.contrib.layers.xavier_initializer()
            const0 = tf.constant_initializer(0.)
            l2_reg = tf.contrib.layers.l2_regularizer(0.1)
            pad_config = 'SAME'
            # Block name	Elements	Kernel Size	Filter depth	Output depth	Stride
            #               Convolution	[5, 5]	    3   	        64	            [1, 1]
            # conv1	        ReLU
            #               Max-pool	[3, 3]	    None   	        None	        [2, 2]
            with tf.name_scope('conv1'):
                f1 = tf.get_variable('f1', shape=[5, 5, 3, 64], dtype=tf.float32,
                                     initializer=xavier, regularizer=l2_reg)
                c1 = tf.nn.conv2d(x, f1, strides=[1, 1, 1, 1], padding=pad_config)
                r1 = tf.nn.relu(c1)
                o1 = tf.nn.max_pool(r1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding=pad_config)

            #               Convolution	[5, 5]	    64	            64	            [1, 1]
            # conv2	        ReLU
            #               Max-pool	[3, 3]	    None	        None	        [2, 2]
            with tf.name_scope('conv2'):
                f2 = tf.get_variable('f2', shape=[5, 5, 64, 64], dtype=tf.float32,
                                     initializer=xavier, regularizer=l2_reg)
                c2 = tf.nn.conv2d(o1, f2, strides=[1, 1, 1, 1], padding=pad_config)
                r2 = tf.nn.relu(c2)
                o2 = tf.nn.max_pool(r2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding=pad_config)

            # flatten	Flatten
            # o3 = tf.contrib.layers.flatten(o2)
            o3 = tf.reshape(o2, [tf.shape(o2)[0], sum(tf.shape(o2)[1:])], name='flat_out')
            # fc1	        Multiplication	[dim(conv2), 384]
            #               ReLU
            with tf.name_scope('dense1'):
                w1 = tf.get_variable('w1', shape=[o3.get_shape()[1], 384], dtype=tf.float32,
                                     initializer=xavier, regularizer=l2_reg)
                b1 = tf.get_variable('b1', shape=[384], dtype=tf.float32,
                                     initializer=const0)
                o4 = tf.nn.relu(tf.matmul(o3, w1) + b1, name='d1_out')

            # fc2	        Multiplication	[384, 192]
            #               ReLU
            with tf.name_scope('dense2'):
                w2 = tf.get_variable('w2', shape=[384, 192], dtype=tf.float32,
                                     initializer=xavier, regularizer=l2_reg)
                b2 = tf.get_variable('b2', shape=[192], dtype=tf.float32,
                                     initializer=const0)
                o5 = tf.nn.relu(tf.matmul(o4, w2) + b2, name='d2_out')

            # fc3	        Multiplication	[192, 10]
            with tf.name_scope('dense3'):
                w3 = tf.get_variable('w3', shape=[192, self.n_classes], dtype=tf.float32,
                                     initializer=xavier, regularizer=l2_reg)
                b3 = tf.get_variable('b3', shape=[self.n_classes], dtype=tf.float32,
                                     initializer=const0)
                o6 = tf.matmul(o5, w3) + b3

            #               Softmax
            logits = o6  # logits are made without actual softmaxing, right?
            ########################
            # END OF YOUR CODE    #
            ########################
        return logits

    def accuracy(self, logits, labels):
        """
        Calculate the prediction accuracy, i.e. the average correct predictions
        of the network.
        As in self.loss above, you can use tf.scalar_summary to save
        scalar summaries of accuracy for later use with the TensorBoard.

        Args:
          logits: 2D float Tensor of size [batch_size, self.n_classes].
                       The predictions returned through self.inference.
          labels: 2D int Tensor of size [batch_size, self.n_classes]
                     with one-hot encoding. Ground truth labels for
                     each observation in batch.

        Returns:
          accuracy: scalar float Tensor, the accuracy of predictions,
                    i.e. the average correct predictions over the whole batch.
        """
        ########################
        # PUT YOUR CODE HERE  #
        ########################
        guesses = tf.argmax(logits, dimension=1)
        targets = tf.argmax(labels, dimension=1)
        score = tf.to_int32(tf.equal(guesses, targets))
        accuracy = tf.reduce_sum(score) / tf.size(score)

        tf.scalar_summary('accuracy', accuracy)
        ########################
        # END OF YOUR CODE    #
        ########################

        return accuracy

    def loss(self, logits, labels):
        """
        Calculates the multiclass cross-entropy loss from the logits predictions and
        the ground truth labels. The function will also add the regularization
        loss from network weights to the total loss that is return.
        In order to implement this function you should have a look at
        tf.nn.softmax_cross_entropy_with_logits.
        You can use tf.scalar_summary to save scalar summaries of
        cross-entropy loss, regularization loss, and full loss (both summed)
        for use with TensorBoard. This will be useful for compiling your report.

        Args:
          logits: 2D float Tensor of size [batch_size, self.n_classes].
                       The predictions returned through self.inference.
          labels: 2D int Tensor of size [batch_size, self.n_classes]
                       with one-hot encoding. Ground truth labels for each
                       observation in batch.

        Returns:
          loss: scalar float Tensor, full loss = cross_entropy + reg_loss
        """
        ########################
        # PUT YOUR CODE HERE  #
        ########################
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
        ########################
        # END OF YOUR CODE    #
        ########################

        return loss
