import tensorflow as tf
import tensorflow.contrib as tf_contrib
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Xavier : tf_contrib.layers.xavier_initializer()
# He : tf_contrib.layers.variance_scaling_initializer()
# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# l2_decay : tf_contrib.layers.l2_regularizer(0.0001)

weight_init = tf_contrib.layers.xavier_initializer()
weight_regularizer = tf_contrib.layers.l2_regularizer(0.0001)

##################################################################################
# Layer
##################################################################################


def conv(x, channels, kernel=4, stride=2, padding='SAME', use_bias=True, scope='conv_0'):
    with tf.variable_scope(scope):
        x = tf.layers.conv2d(inputs=x,
                             filters=channels,
                             kernel_size=kernel,
                             kernel_initializer=weight_init,
                             kernel_regularizer=weight_regularizer,
                             strides=stride,
                             use_bias=use_bias,
                             padding=padding,
                             data_format="channels_first")

        return x


def fully_conneted(x, units, use_bias=True, scope='fully_0'):
    with tf.variable_scope(scope):
        x = flatten(x)
        x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, use_bias=use_bias)

        return x


##################################################################################
# Sampling
##################################################################################


def flatten(x):
    return tf.layers.flatten(x)


def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    return gap


def avg_pooling(x):
    return tf.layers.average_pooling2d(x, pool_size=2, strides=2, padding='SAME')


##################################################################################
# Activation function
##################################################################################


def relu(x):
    return tf.nn.relu(x)


##################################################################################
# Normalization function
##################################################################################


def batch_norm(x, is_training=True, scope='batch_norm'):
    return tf_contrib.layers.batch_norm(x,
                                        decay=0.9,
                                        epsilon=1e-05,
                                        center=True,
                                        scale=True,
                                        updates_collections=None,
                                        is_training=is_training,
                                        scope=scope)


##################################################################################
# Loss function
##################################################################################


def classification_loss(logit_list, label_list, batch_size):
    # loss = tf.reduce_mean(
    #     tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit))
    # prediction = tf.equal(tf.argmax(logit, -1), tf.argmax(label, -1))
    # accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    # assert label_list.get_shape()[0] == logit_list.gei_shape()[0]
    total_loss = 0
    total_acc = [0] * 40
    attr_order = [[1, 3, 4, 5, 8, 9, 11, 12, 15, 17, 23, 28, 35], [7, 19, 27, 29, 30, 34], [6, 14, 16, 22, 21, 24, 36, 37, 38],
                  [0, 2, 10, 13, 18, 20, 25, 26, 32, 31, 33, 39]]
    for i in range(batch_size):
        label = label_list[i]
        # logit_list包含4个列表，每个列表是一个tsnet的输出logit
        for k in range(4):
            logit = logit_list[k][i]
            # print('======================================================')
            # print('[*] The shape of label is %s' % label.get_shape())
            # print('[*] The shape of logit is %s' % logit.get_shape())
            # print('[*] The shape of label_list is %s' % label_list.get_shape())
            # print('[*] The shape of logit_list is %s' % len(logit_list))
            # print('======================================================')
            order = attr_order[k]
            label_partial = tf.gather(label, order)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_partial, logits=logit))
            total_loss += loss
            # prediction = tf.equal(tf.argmax(logit, -1), tf.argmax(label, -1))

    # 以0.5为阈值
    logit = tf.concat([logit_list[0], logit_list[1], logit_list[2], logit_list[3]], 1)
    logit = tf.sigmoid(logit)
    judge = tf.where(logit >= 0.5 * tf.ones_like(logit), tf.ones_like(logit), tf.zeros_like(logit))
    # logit的shape应该是[batch_size,c_dim]
    # 而此处的acc为所有属性的acc
    # 因此需要将logit的每一列取出来，单独计算acc
    for j in range(len(order)):
        logit_col = judge[:, j]
        label_col = label_list[:, j]
        prediction = tf.equal(logit_col, label_col)
        accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
        total_acc[order[j]] = accuracy
    return total_loss, total_acc
