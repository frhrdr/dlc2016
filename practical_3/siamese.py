from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


class Siamese(object):
    """
    This class implements a siamese convolutional neural network in
    TensorFlow. Term siamese is used to refer to architectures which
    incorporate two branches of convolutional networks parametrized
    identically (i.e. weights are shared). These graphs accept two
    input tensors and a label in general.
    """

    def inference(self, x, reuse = False):
        """
        Defines the model used for inference. Output of this model is fed to the
        objective (or loss) function defined for the task.

        Here we recommend you to consider using variable and name scopes in order
        to make your graph more intelligible for later references in TensorBoard
        and so on. You can define a name scope for the whole model or for each
        operator group (e.g. conv+pool+relu) individually to group them by name.
        Variable scopes are essential components in TensorFlow for parameter sharing.
        You can use the variable scope to activate/deactivate 'variable reuse'.

        Args:
           x: 4D float Tensor of size [batch_size, input_height, input_width, input_channels]
           reuse: Python bool to switch reusing on/off.

        Returns:
           l2_out: L2-normalized output tensor of shape [batch_size, 192]

        Hint: Parameter reuse indicates whether the inference graph should use
        parameter sharing or not. You can study how to implement parameter sharing
        in TensorFlow from the following sources:

        https://www.tensorflow.org/versions/r0.11/how_tos/variable_scope/index.html
        """
        with tf.variable_scope('ConvNet') as conv_scope:
            ########################
            # PUT YOUR CODE HERE  #
            ########################
            xavier = tf.contrib.layers.xavier_initializer()
            const0 = tf.constant_initializer(0.)
            l2_reg = tf.contrib.layers.l2_regularizer(0.1)
            pad_config = 'SAME'

            if reuse:
                conv_scope.reuse_variables()

            #               Convolution	[5, 5]	3	64	[1, 1]
            # conv1	        ReLU
            #               Max-pool	[3, 3]	None	None	[2, 2]
            with tf.name_scope('conv1'):
                f1 = tf.get_variable('f1', shape=[5, 5, 3, 64], dtype=tf.float32,
                                     initializer=xavier, regularizer=l2_reg)
                c1 = tf.nn.conv2d(x, f1, strides=[1, 1, 1, 1], padding=pad_config)
                r1 = tf.nn.relu(c1)
                o1 = tf.nn.max_pool(r1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding=pad_config)
            #               Convolution	[5, 5]	64	64	[1, 1]
            # conv2	        ReLU
            #               Max-pool	[3, 3]	None	None	[2, 2]
            with tf.name_scope('conv2'):
                f2 = tf.get_variable('f2', shape=[5, 5, 64, 64], dtype=tf.float32,
                                     initializer=xavier, regularizer=l2_reg)
                c2 = tf.nn.conv2d(o1, f2, strides=[1, 1, 1, 1], padding=pad_config)
                r2 = tf.nn.relu(c2)
                o2 = tf.nn.max_pool(r2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding=pad_config)

            # flatten	Flatten
            o3 = tf.reshape(o2, [o2.get_shape()[0].value, -1], name='flat_out')

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

            # L2-norm	L2-normalization
            l2_out = tf.nn.l2_normalize(o5, dim=1)
            ########################
            # END OF YOUR CODE    #
            ########################

        return l2_out

    def loss(self, channel_1, channel_2, label, margin):
        """
        Defines the contrastive loss. This loss ties the outputs of
        the branches to compute the following:

               L =  Y * d^2 + (1-Y) * max(margin - d^2, 0)

               where d is the L2 distance between the given
               input pair s.t. d = ||x_1 - x_2||_2 and Y is
               label associated with the pair of input tensors.
               Y is 1 if the inputs belong to the same class in
               CIFAR10 and is 0 otherwise.

               For more information please see:
               http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

        Args:
            channel_1: output of first channel (i.e. branch_1),
                              tensor of size [batch_size, 192]
            channel_2: output of second channel (i.e. branch_2),
                              tensor of size [batch_size, 192]
            label: Tensor of shape [batch_size]
            margin: Margin of the contrastive loss

        Returns:
            loss: scalar float Tensor
        """
        ########################
        # PUT YOUR CODE HERE  #
        ########################
        d = tf.sqrt(tf.reduce_sum((channel_1 - channel_2) * (channel_1 - channel_2), 1))
        Y = label
        d2 = d * d
        loss = Y * d2 + (1 - Y) * tf.maximum(margin - d2, 0)

        ########################
        # END OF YOUR CODE    #
        ########################

        return loss
