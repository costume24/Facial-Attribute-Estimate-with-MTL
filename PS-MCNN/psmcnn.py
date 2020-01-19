import tensorflow as tf
import numpy as np
import time
from ops import *
from utils import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class ps_mcnn(object):
    def __init__(self, sess, args):
        self.model_name = 'PS_MCNN'
        self.dataset_name = args.dataset
        self.img_height = 160
        self.img_width = 192
        self.c_dim = 3
        self.label_dim = 40
        self.label_dim_0 = 13
        self.label_dim_1 = 6
        self.label_dim_2 = 9
        self.label_dim_3 = 12

        # self.train_x, self.test_x, self.train_y, self.test_y = load_dataset()
        self.sess = sess
        self.checkpoint_dir = args.checkpoint_dir
        self.log_dir = args.log_dir
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.iteration = 162770 // self.batch_size

        self.init_lr = args.lr

    ##################################################################################
    # Generator
    ##################################################################################
    def network(self, x, is_training=True, reuse=False, use_bias=True):
        with tf.variable_scope("ps_mcnn", reuse=reuse):
            #x.shape  (64,3,160,192)

            #-------------conv0-------------#
            tsnet_0_0 = conv(x, 32, kernel=3, stride=1, use_bias=use_bias, scope="tsnet_0_0")
            tsnet_0_0 = batch_norm(tsnet_0_0, is_training, scope='tsnet_0_0/bn0')
            tsnet_0_0 = relu(tsnet_0_0)
            tsnet_0_0 = tf.layers.max_pooling2d(tsnet_0_0, 2, 2, data_format="channels_first")

            tsnet_1_0 = conv(x, 32, kernel=3, stride=1, use_bias=use_bias, scope="tsnet_1_0")
            tsnet_1_0 = batch_norm(tsnet_1_0, is_training, scope='tsnet_1_0/bn0')
            tsnet_1_0 = relu(tsnet_1_0)
            tsnet_1_0 = tf.layers.max_pooling2d(tsnet_1_0, 2, 2, data_format="channels_first")

            tsnet_2_0 = conv(x, 32, kernel=3, stride=1, use_bias=use_bias, scope="tsnet_2_0")
            tsnet_2_0 = batch_norm(tsnet_2_0, is_training, scope='tsnet_2_0/bn0')
            tsnet_2_0 = relu(tsnet_2_0)
            tsnet_2_0 = tf.layers.max_pooling2d(tsnet_2_0, 2, 2, data_format="channels_first")

            tsnet_3_0 = conv(x, 32, kernel=3, stride=1, use_bias=use_bias, scope="tsnet_3_0")
            tsnet_3_0 = batch_norm(tsnet_3_0, is_training, scope='tsnet_3_0/bn0')
            tsnet_3_0 = relu(tsnet_3_0)
            tsnet_3_0 = tf.layers.max_pooling2d(tsnet_3_0, 2, 2, data_format="channels_first")

            snet_0 = conv(x, 32, kernel=3, stride=1, use_bias=use_bias, scope="snet_0")
            snet_0 = batch_norm(snet_0, is_training, scope='snet_0/bn0')
            snet_0 = relu(snet_0)
            snet_0 = tf.layers.max_pooling2d(snet_0, 2, 2, data_format="channels_first")

            #-------------conv1-------------#
            tsnet_0_1 = conv(tf.concat([tsnet_0_0, snet_0], 1), 64, kernel=3, stride=1, use_bias=use_bias, scope="tsnet_0_1")
            tsnet_0_1 = batch_norm(tsnet_0_1, is_training, scope='tsnet_0_1/bn1')
            tsnet_0_1 = relu(tsnet_0_1)
            tsnet_0_1 = tf.layers.max_pooling2d(tsnet_0_1, 2, 2, data_format="channels_first")

            tsnet_1_1 = conv(tf.concat([tsnet_1_0, snet_0], 1), 64, kernel=3, stride=1, use_bias=use_bias, scope="tsnet_1_1")
            tsnet_1_1 = batch_norm(tsnet_1_1, is_training, scope='tsnet_1_1/bn1')
            tsnet_1_1 = relu(tsnet_1_1)
            tsnet_1_1 = tf.layers.max_pooling2d(tsnet_1_1, 2, 2, data_format="channels_first")

            tsnet_2_1 = conv(tf.concat([tsnet_2_0, snet_0], 1), 64, kernel=3, stride=1, use_bias=use_bias, scope="tsnet_2_1")
            tsnet_2_1 = batch_norm(tsnet_2_1, is_training, scope='tsnet_2_1/bn1')
            tsnet_2_1 = relu(tsnet_2_1)
            tsnet_2_1 = tf.layers.max_pooling2d(tsnet_2_1, 2, 2, data_format="channels_first")

            tsnet_3_1 = conv(tf.concat([tsnet_3_0, snet_0], 1), 64, kernel=3, stride=1, use_bias=use_bias, scope="tsnet_3_1")
            tsnet_3_1 = batch_norm(tsnet_3_1, is_training, scope='tsnet_3_1/bn1')
            tsnet_3_1 = relu(tsnet_3_1)
            tsnet_3_1 = tf.layers.max_pooling2d(tsnet_3_1, 2, 2, data_format="channels_first")

            tsnet_0_0_partial = tf.slice(tsnet_0_0, [0, 0, 0, 0], [-1, 32, -1, -1])
            tsnet_1_0_partial = tf.slice(tsnet_1_0, [0, 0, 0, 0], [-1, 32, -1, -1])
            tsnet_2_0_partial = tf.slice(tsnet_2_0, [0, 0, 0, 0], [-1, 32, -1, -1])
            tsnet_3_0_partial = tf.slice(tsnet_3_0, [0, 0, 0, 0], [-1, 32, -1, -1])
            # partial_0 = tsnet_0_0_partial + tsnet_1_0_partial + tsnet_2_0_partial + tsnet_3_0_partial+snet_0
            partial_0 = tf.concat([tsnet_0_0_partial, tsnet_1_0_partial, tsnet_2_0_partial, tsnet_3_0_partial, snet_0], 1)
            snet_1 = conv(partial_0, 64, kernel=3, stride=1, use_bias=use_bias, scope="snet_1")
            snet_1 = batch_norm(snet_1, is_training, scope='snet_1/bn1')
            snet_1 = relu(snet_1)
            snet_1 = tf.layers.max_pooling2d(snet_1, 2, 2, data_format="channels_first")

            #-------------conv2-------------#
            # tsnet_0_2 = Tnet(tsnet_0_1 + snet_1, scope="tsnet_0_2")
            # tsnet_1_2 = Tnet(tsnet_1_1 + snet_1, scope="tsnet_1_2")
            # tsnet_2_2 = Tnet(tsnet_2_1 + snet_1, scope="tsnet_2_2")
            # tsnet_3_2 = Tnet(tsnet_2_1 + snet_1, scope="tsnet_3_2")

            tsnet_0_2 = conv(tf.concat([tsnet_0_1, snet_1], 1), 128, kernel=3, stride=1, use_bias=use_bias, scope="tsnet_0_2")
            tsnet_0_2 = batch_norm(tsnet_0_2, is_training, scope='tsnet_0_2/bn2')
            tsnet_0_2 = relu(tsnet_0_2)
            tsnet_0_2 = tf.layers.max_pooling2d(tsnet_0_2, 2, 2, data_format="channels_first")

            tsnet_1_2 = conv(tf.concat([tsnet_1_1, snet_1], 1), 128, kernel=3, stride=1, use_bias=use_bias, scope="tsnet_1_2")
            tsnet_1_2 = batch_norm(tsnet_1_2, is_training, scope='tsnet_1_2/bn2')
            tsnet_1_2 = relu(tsnet_1_2)
            tsnet_1_2 = tf.layers.max_pooling2d(tsnet_1_2, 2, 2, data_format="channels_first")

            tsnet_2_2 = conv(tf.concat([tsnet_2_1, snet_1], 1), 128, kernel=3, stride=1, use_bias=use_bias, scope="tsnet_2_2")
            tsnet_2_2 = batch_norm(tsnet_2_2, is_training, scope='tsnet_2_2/bn2')
            tsnet_2_2 = relu(tsnet_2_2)
            tsnet_2_2 = tf.layers.max_pooling2d(tsnet_2_2, 2, 2, data_format="channels_first")

            tsnet_3_2 = conv(tf.concat([tsnet_3_1, snet_1], 1), 128, kernel=3, stride=1, use_bias=use_bias, scope="tsnet_3_2")
            tsnet_3_2 = batch_norm(tsnet_3_2, is_training, scope='tsnet_3_2/bn2')
            tsnet_3_2 = relu(tsnet_3_2)
            tsnet_3_2 = tf.layers.max_pooling2d(tsnet_3_2, 2, 2, data_format="channels_first")

            tsnet_0_1_partial = tf.slice(tsnet_0_1, [0, 0, 0, 0], [-1, 32, -1, -1])
            tsnet_1_1_partial = tf.slice(tsnet_1_1, [0, 0, 0, 0], [-1, 32, -1, -1])
            tsnet_2_1_partial = tf.slice(tsnet_2_1, [0, 0, 0, 0], [-1, 32, -1, -1])
            tsnet_3_1_partial = tf.slice(tsnet_3_1, [0, 0, 0, 0], [-1, 32, -1, -1])
            # partial_0 = tsnet_0_0_partial + tsnet_1_0_partial + tsnet_2_0_partial + tsnet_3_0_partial+snet_0
            partial_1 = tf.concat([tsnet_0_1_partial, tsnet_1_1_partial, tsnet_2_1_partial, tsnet_3_1_partial, snet_1], 1)
            snet_2 = conv(partial_1, 64, kernel=3, stride=1, use_bias=use_bias, scope="snet_2")
            snet_2 = batch_norm(snet_2, is_training, scope='snet_2/bn2')
            snet_2 = relu(snet_2)
            snet_2 = tf.layers.max_pooling2d(snet_2, 2, 2, data_format="channels_first")

            #-------------conv3-------------#
            # tsnet_0_3 = Tnet(tsnet_0_2 + snet_2, scope="tsnet_0_3")
            # tsnet_1_3 = Tnet(tsnet_1_2 + snet_2, scope="tsnet_1_3")
            # tsnet_2_3 = Tnet(tsnet_2_2 + snet_2, scope="tsnet_2_3")
            # tsnet_3_3 = Tnet(tsnet_3_2 + snet_2, scope="tsnet_3_3")

            tsnet_0_3 = conv(tf.concat([tsnet_0_2, snet_2], 1), 128, kernel=3, stride=1, use_bias=use_bias, scope="tsnet_0_3")
            tsnet_0_3 = batch_norm(tsnet_0_3, is_training, scope='tsnet_0_3/bn3')
            tsnet_0_3 = relu(tsnet_0_3)
            tsnet_0_3 = tf.layers.max_pooling2d(tsnet_0_3, 2, 2, data_format="channels_first")

            tsnet_1_3 = conv(tf.concat([tsnet_1_2, snet_2], 1), 128, kernel=3, stride=1, use_bias=use_bias, scope="tsnet_1_3")
            tsnet_1_3 = batch_norm(tsnet_1_3, is_training, scope='tsnet_1_3/bn3')
            tsnet_1_3 = relu(tsnet_1_3)
            tsnet_1_3 = tf.layers.max_pooling2d(tsnet_1_3, 2, 2, data_format="channels_first")

            tsnet_2_3 = conv(tf.concat([tsnet_2_2, snet_2], 1), 128, kernel=3, stride=1, use_bias=use_bias, scope="tsnet_2_3")
            tsnet_2_3 = batch_norm(tsnet_2_3, is_training, scope='tsnet_2_3/bn3')
            tsnet_2_3 = relu(tsnet_2_3)
            tsnet_2_3 = tf.layers.max_pooling2d(tsnet_2_3, 2, 2, data_format="channels_first")

            tsnet_3_3 = conv(tf.concat([tsnet_3_2, snet_2], 1), 128, kernel=3, stride=1, use_bias=use_bias, scope="tsnet_3_3")
            tsnet_3_3 = batch_norm(tsnet_3_3, is_training, scope='tsnet_3_3/bn3')
            tsnet_3_3 = relu(tsnet_3_3)
            tsnet_3_3 = tf.layers.max_pooling2d(tsnet_3_3, 2, 2, data_format="channels_first")

            tsnet_0_2_partial = tf.slice(tsnet_0_2, [0, 0, 0, 0], [-1, 32, -1, -1])
            tsnet_1_2_partial = tf.slice(tsnet_1_2, [0, 0, 0, 0], [-1, 32, -1, -1])
            tsnet_2_2_partial = tf.slice(tsnet_2_2, [0, 0, 0, 0], [-1, 32, -1, -1])
            tsnet_3_2_partial = tf.slice(tsnet_3_2, [0, 0, 0, 0], [-1, 32, -1, -1])
            # partial_0 = tsnet_0_0_partial + tsnet_1_0_partial + tsnet_2_0_partial + tsnet_3_0_partial+snet_0
            partial_2 = tf.concat([tsnet_0_2_partial, tsnet_1_2_partial, tsnet_2_2_partial, tsnet_3_2_partial, snet_2], 1)
            snet_3 = conv(partial_2, 64, kernel=3, stride=1, use_bias=use_bias, scope="snet_3")
            snet_3 = batch_norm(snet_3, is_training, scope='snet_3/bn3')
            snet_3 = relu(snet_3)
            snet_3 = tf.layers.max_pooling2d(snet_3, 2, 2, data_format="channels_first")
            #-------------conv4-------------#
            # tsnet_0_4 = Tnet(tsnet_0_3 + snet_3, scope="tsnet_0_4")
            # tsnet_1_4 = Tnet(tsnet_1_3 + snet_3, scope="tsnet_1_4")
            # tsnet_2_4 = Tnet(tsnet_2_3 + snet_3, scope="tsnet_2_4")
            # tsnet_3_4 = Tnet(tsnet_3_3 + snet_3, scope="tsnet_3_4")

            tsnet_0_4 = conv(tf.concat([tsnet_0_3, snet_3], 1), 128, kernel=3, stride=1, use_bias=use_bias, scope="tsnet_0_4")
            tsnet_0_4 = batch_norm(tsnet_0_4, is_training, scope='tsnet_0_4/bn4')
            tsnet_0_4 = relu(tsnet_0_4)
            tsnet_0_4 = tf.layers.max_pooling2d(tsnet_0_4, 2, 2, data_format="channels_first")

            tsnet_1_4 = conv(tf.concat([tsnet_1_3, snet_3], 1), 128, kernel=3, stride=1, use_bias=use_bias, scope="tsnet_1_4")
            tsnet_1_4 = batch_norm(tsnet_1_4, is_training, scope='tsnet_1_4/bn4')
            tsnet_1_4 = relu(tsnet_1_4)
            tsnet_1_4 = tf.layers.max_pooling2d(tsnet_1_4, 2, 2, data_format="channels_first")

            tsnet_2_4 = conv(tf.concat([tsnet_2_3, snet_3], 1), 128, kernel=3, stride=1, use_bias=use_bias, scope="tsnet_2_4")
            tsnet_2_4 = batch_norm(tsnet_2_4, is_training, scope='tsnet_2_4/bn4')
            tsnet_2_4 = relu(tsnet_2_4)
            tsnet_2_4 = tf.layers.max_pooling2d(tsnet_2_4, 2, 2, data_format="channels_first")

            tsnet_3_4 = conv(tf.concat([tsnet_3_3, snet_3], 1), 128, kernel=3, stride=1, use_bias=use_bias, scope="tsnet_3_4")
            tsnet_3_4 = batch_norm(tsnet_3_4, is_training, scope='tsnet_3_4/bn4')
            tsnet_3_4 = relu(tsnet_3_4)
            tsnet_3_4 = tf.layers.max_pooling2d(tsnet_3_4, 2, 2, data_format="channels_first")

            tsnet_0_3_partial = tf.slice(tsnet_0_3, [0, 0, 0, 0], [-1, 32, -1, -1])
            tsnet_1_3_partial = tf.slice(tsnet_1_3, [0, 0, 0, 0], [-1, 32, -1, -1])
            tsnet_2_3_partial = tf.slice(tsnet_2_3, [0, 0, 0, 0], [-1, 32, -1, -1])
            tsnet_3_3_partial = tf.slice(tsnet_3_3, [0, 0, 0, 0], [-1, 32, -1, -1])
            # partial_0 = tsnet_0_0_partial + tsnet_1_0_partial + tsnet_2_0_partial + tsnet_3_0_partial+snet_0
            partial_3 = tf.concat([tsnet_0_3_partial, tsnet_1_3_partial, tsnet_2_3_partial, tsnet_3_3_partial, snet_3], 1)
            snet_4 = conv(partial_3, 64, kernel=3, stride=1, use_bias=use_bias, scope="snet_4")
            snet_4 = batch_norm(snet_4, is_training, scope='snet_4/bn3')
            snet_4 = relu(snet_4)
            snet_4 = tf.layers.max_pooling2d(snet_4, 2, 2, data_format="channels_first")
            #-------------fc0-------------#
            fc_0_0 = fully_conneted(tsnet_0_4, units=512, scope='fc_0_0')
            fc_1_0 = fully_conneted(tsnet_1_4, units=512, scope='fc_1_0')
            fc_2_0 = fully_conneted(tsnet_2_4, units=512, scope='fc_2_0')
            fc_3_0 = fully_conneted(tsnet_3_4, units=512, scope='fc_3_0')
            fc_4_0 = fully_conneted(snet_4, units=512, scope='fc_4_0')  #fc_4代表snet通路

            #-------------fc1-------------#
            fc_0_1 = fully_conneted(fc_0_0, units=512, scope='fc_0_1')
            fc_1_1 = fully_conneted(fc_1_0, units=512, scope='fc_1_1')
            fc_2_1 = fully_conneted(fc_2_0, units=512, scope='fc_2_1')
            fc_3_1 = fully_conneted(fc_3_0, units=512, scope='fc_3_1')
            fc_4_1 = fully_conneted(fc_4_0, units=512, scope='fc_4_1')  #fc_4代表snet通路

            fc_0_1 = tf.concat([fc_0_1, fc_4_1], 1)
            fc_1_1 = tf.concat([fc_0_1, fc_4_1], 1)
            fc_2_1 = tf.concat([fc_0_1, fc_4_1], 1)
            fc_3_1 = tf.concat([fc_0_1, fc_4_1], 1)

            #-------------global average pooling-------------#
            # gap_0 = global_avg_pooling(fc_0_1)
            # gap_1 = global_avg_pooling(fc_1_1)
            # gap_2 = global_avg_pooling(fc_2_1)
            # gap_3 = global_avg_pooling(fc_3_1)

            # 这里的四个分支的输出维度由手工分组决定
            # 现在采用论文中的分组方式
            # 暂时写死，后续可以改为由输入参数决定
            output_0 = fully_conneted(fc_0_1, units=13, scope='logit_0')
            output_1 = fully_conneted(fc_1_1, units=6, scope='logit_1')
            output_2 = fully_conneted(fc_2_1, units=9, scope='logit_2')
            output_3 = fully_conneted(fc_3_1, units=12, scope='logit_3')

            return [output_0, output_1, output_2, output_3]

    def build_model(self):
        '''不再使用
        此方法适用于小数据集，可以一次性加载到内存中的情况
        '''
        """ Graph Input """
        print('[*] loading training images...')
        files = tf.train.match_filenames_once('/media/xuke/SoftWare/CelebA/celebA_train.tfrecords')
        filename_queue = tf.train.string_input_producer(files, shuffle=False)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'image': tf.FixedLenFeature((), tf.string),
                                               'label': tf.FixedLenFeature([40], tf.float32),
                                               'height': tf.FixedLenFeature([1], tf.int64),
                                               'width': tf.FixedLenFeature([1], tf.int64)
                                           })
        img, self.train_y = features['image'], features['label']
        height = tf.cast(features['height'], tf.int64)
        width = tf.cast(features['width'], tf.int64)

        decoded_image_train = tf.decode_raw(img, tf.uint8)
        decoded_image_train = tf.cast(decoded_image_train, tf.float32) * (1. / 255.0)
        self.train_x = tf.reshape(decoded_image_train, [3, 160, 192])
        self.train_x.set_shape([3, 160, 192])
        # decoded_image_train = tf.reshape(decoded_image_train, [3, height, width])

        print('[*] Finish loading training images')

        print('[*] loading test images...')

        files = tf.train.match_filenames_once('/media/xuke/SoftWare/CelebA/celebA_valid.tfrecords')
        filename_queue = tf.train.string_input_producer(files, shuffle=False)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'image': tf.FixedLenFeature([], tf.string),
                                               'label': tf.FixedLenFeature([40], tf.float32),
                                               'height': tf.FixedLenFeature([1], tf.int64),
                                               'width': tf.FixedLenFeature([1], tf.int64)
                                           })
        img, test_y = features['image'], features['label']
        height, width = features['height'], features['width']
        decoded_image_val = tf.decode_raw(img, tf.uint8)
        decoded_image_val = tf.cast(decoded_image_val, tf.float32) * (1. / 255.0)
        self.test_x = tf.reshape(decoded_image_val, [3, 160, 192])
        self.test_x.set_shape([3, 160, 192])
        # decoded_image_val = tf.reshape(decoded_image_val, [3, height, width])

        print('[*] Finish loading test images')

        # self.test_labels = tf.placeholder(tf.float32, [19867, self.label_dim], name='test_labels')
        """ Model """
        self.train_logits_0, self.train_logits_1, self.train_logits_2, self.train_logits_3 = self.network(self.train_inputs)
        self.test_logits_0, self.test_logits_1, self.test_logits_2, self.test_logits_3 = self.network(self.test_inputs, is_training=False, reuse=True)

        # accuracy为列表
        self.train_logits = [self.train_logits_0, self.train_logits_1, self.train_logits_2, self.train_logits_3]
        self.train_labels = [self.train_labels_0, self.train_labels_1, self.train_labels_2, self.train_labels_3]
        self.test_logits = [self.test_logits_0, self.test_logits_1, self.test_logits_2, self.test_logits_3]
        self.test_labels = [self.test_labels_0, self.test_labels_1, self.test_labels_2, self.test_labels_3]
        self.train_loss, self.train_accuracy = classification_loss(self.train_logits, self.train_labels)
        self.test_loss, self.test_accuracy = classification_loss(self.test_logits, self.test_labels)

        reg_loss = tf.losses.get_regularization_loss()
        self.train_loss += reg_loss
        self.test_loss += reg_loss
        """ Training """
        self.optim = tf.train.MomentumOptimizer(self.lr, momentum=0.9).minimize(self.train_loss)
        """" Summary """
        self.summary_train_loss = tf.summary.scalar("train_loss", self.train_loss)
        self.summary_train_accuracy = tf.summary.scalar("train_accuracy", tf.reduce_mean(self.train_accuracy))

        self.summary_test_loss = tf.summary.scalar("test_loss", self.test_loss)
        self.summary_test_accuracy = tf.summary.scalar("test_accuracy", tf.reduce_mean(self.test_accuracy))

        self.train_summary = tf.summary.merge([self.summary_train_loss, self.summary_train_accuracy])
        self.test_summary = tf.summary.merge([self.summary_test_loss, self.summary_test_accuracy])

    ##################################################################################
    # Train
    ##################################################################################

    # def train(self):
    #     # initialize all variables
    #     tf.global_variables_initializer().run()
    #     tf.local_variables_initializer().run()
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord=coord)
    #     # saver to save model
    #     self.saver = tf.train.Saver()

    #     # summary writer
    #     self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

    #     # restore check-point if it exits
    #     could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    #     if could_load:
    #         epoch_lr = self.init_lr
    #         start_epoch = (int)(checkpoint_counter / self.iteration)
    #         start_batch_id = checkpoint_counter - start_epoch * self.iteration
    #         counter = checkpoint_counter

    #         if start_epoch >= int(self.epoch * 0.75):
    #             epoch_lr = epoch_lr * 0.01
    #         elif start_epoch >= int(self.epoch * 0.5) and start_epoch < int(self.epoch * 0.75):
    #             epoch_lr = epoch_lr * 0.1
    #         print(" [*] Load SUCCESS")
    #     else:
    #         epoch_lr = self.init_lr
    #         start_epoch = 0
    #         start_batch_id = 0
    #         counter = 1
    #         print(" [!] Load failed...")

    #     min_after_dequeue = 1000
    #     capacity = min_after_dequeue + 3 * self.batch_size
    #     # loop for epoch
    #     start_time = time.time()
    #     for epoch in range(start_epoch, self.epoch):
    #         if epoch == int(self.epoch * 0.5) or epoch == int(self.epoch * 0.75):
    #             epoch_lr = epoch_lr * 0.1

    #         # get batch data
    #         for idx in range(start_batch_id, self.iteration):
    #             # batch_x = self.train_x[idx * self.batch_size:(idx + 1) * self.batch_size]
    #             # batch_y = self.train_y[idx * self.batch_size:(idx + 1) * self.batch_size]

    #             # batch_x = data_augmentation(batch_x, self.img_height, self.img_width,
    #             #                             self.dataset_name)
    #             batch_x, batch_y = tf.train.shuffle_batch([self.train_x, self.train_y],
    #                                                       batch_size=self.batch_size,
    #                                                       capacity=capacity,
    #                                                       min_after_dequeue=min_after_dequeue)
    #             train_feed_dict = {self.train_inputs: batch_x, self.train_labels: batch_y, self.lr: epoch_lr}
    #             test_x, test_y = tf.train.shuffle_batch([self.test_x, self.test_y],
    #                                                     batch_size=self.batch_size,
    #                                                     capacity=capacity,
    #                                                     min_after_dequeue=min_after_dequeue)
    #             test_feed_dict = {self.test_inputs: self.test_x, self.test_labels: self.test_y}

    #             # update network
    #             _, summary_str, train_loss, train_accuracy = self.sess.run([self.optim, self.train_summary, self.train_loss, self.train_accuracy],
    #                                                                        feed_dict=train_feed_dict)
    #             self.writer.add_summary(summary_str, counter)

    #             # test
    #             summary_str, test_loss, test_accuracy = self.sess.run([self.test_summary, self.test_loss, self.test_accuracy],
    #                                                                   feed_dict=test_feed_dict)
    #             self.writer.add_summary(summary_str, counter)

    #             # display training status
    #             counter += 1
    #             print("Epoch: [%2d] [%5d/%5d] time: %4.4f, train_accuracy: %.2f, test_accuracy: %.2f, learning_rate : %.4f" \
    #                   % (epoch, idx, self.iteration, time.time() - start_time, tf.reduce_mean(train_accuracy), tf.reduce_mean(test_accuracy), epoch_lr))

    #         # After an epoch, start_batch_id is set to zero
    #         # non-zero value is only for the first epoch after loading pre-trained model
    #         start_batch_id = 0

    #         # save model
    #         self.save(self.checkpoint_dir, counter)

    #     # save model for final step
    #     self.save(self.checkpoint_dir, counter)

    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        x_train, y_train = read_and_decode('/media/xuke/SoftWare/CelebA/celebA_train.tfrecords', is_train=True)
        x_test, y_test = read_and_decode('/media/xuke/SoftWare/CelebA/celebA_test.tfrecords', is_train=False)

        min_after_dequeue = 10000
        capacity = min_after_dequeue + 3 * args.batch_size
        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            epoch_lr = self.init_lr
            start_epoch = (int)(checkpoint_counter / self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter

            if start_epoch >= int(self.epoch * 0.75):
                epoch_lr = epoch_lr * 0.01
            elif start_epoch >= int(self.epoch * 0.5) and start_epoch < int(self.epoch * 0.75):
                epoch_lr = epoch_lr * 0.1
            print(" [*] Load SUCCESS")
        else:
            epoch_lr = self.init_lr
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):
            if epoch == int(self.epoch * 0.5) or epoch == int(self.epoch * 0.75):
                epoch_lr = epoch_lr * 0.1

            # get batch data
            for idx in range(start_batch_id, self.iteration):

                batch_x, batch_y = tf.train.shuffle_batch([x_train, y_train],
                                                          batch_size=args.batch_size,
                                                          capacity=capacity,
                                                          shapes=([3, 160, 192], [40]),
                                                          min_after_dequeue=min_after_dequeue)
                batch_x_test, batch_y_test = tf.train.batch([x_test, y_test],
                                                            batch_size=args.batch_size,
                                                            shapes=([3, 160, 192], [40]),
                                                            capacity=capacity,
                                                            num_threads=12)

                # update network
                _, summary_str, train_loss, train_accuracy = self.sess.run([self.optim, self.train_summary, self.train_loss, self.train_accuracy])
                self.writer.add_summary(summary_str, counter)

                # test
                summary_str, test_loss, test_accuracy = self.sess.run([self.test_summary, self.test_loss, self.test_accuracy])
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%5d/%5d] time: %4.4f, train_accuracy: %.2f, test_accuracy: %.2f, learning_rate : %.4f" \
                      % (epoch, idx, self.iteration, time.time() - start_time, tf.reduce_mean(train_accuracy), tf.reduce_mean(test_accuracy), epoch_lr))

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(self.model_name, self.dataset_name, self.batch_size, self.init_lr)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def test(self):
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        test_feed_dict = {self.test_inptus: self.test_x, self.test_labels: self.test_y}

        test_accuracy = self.sess.run(self.test_accuracy, feed_dict=test_feed_dict)
        print("test_accuracy: {}".format(test_accuracy))
