'''可以在此基础上改进
'''
import numpy as np
import tensorflow as tf

weight_decay = 1e-4


def relu(x, name='relu6'):
    return tf.nn.relu6(x, name)


def batch_norm(x, momentum=0.9, epsilon=1e-5, train=True, name='bn'):
    return tf.layers.batch_normalization(x, momentum=momentum, epsilon=epsilon, scale=True, training=train, name=name)


def conv2d(input_, output_dim, k_h, k_w, d_h, d_w, stddev=0.02, name='conv2d', bias=False):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        if bias:
            biases = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)

        return conv


def conv2d_block(input, out_dim, k, s, is_train, name):
    with tf.name_scope(name), tf.variable_scope(name):
        net = conv2d(input, out_dim, k, k, s, s, name='conv2d')
        net = batch_norm(net, train=is_train, name='bn')
        net = relu(net)
        return net


def conv_1x1(input, output_dim, name, bias=False):
    with tf.name_scope(name):
        return conv2d(input, output_dim, 1, 1, 1, 1, stddev=0.02, name=name, bias=bias)


def pwise_block(input, output_dim, is_train, name, bias=False):
    with tf.name_scope(name), tf.variable_scope(name):
        out = conv_1x1(input, output_dim, bias=bias, name='pwb')
        out = batch_norm(out, train=is_train, name='bn')
        out = relu(out)
        return out


def dwise_conv(input, k_h=3, k_w=3, channel_multiplier=1, strides=[1, 1, 1, 1], padding='SAME', stddev=0.02, name='dwise_conv', bias=False):
    with tf.variable_scope(name):
        in_channel = input.get_shape().as_list()[-1]
        w = tf.get_variable('w', [k_h, k_w, in_channel, channel_multiplier],
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.depthwise_conv2d(input, w, strides, padding, rate=None, name=None, data_format=None)
        if bias:
            biases = tf.get_variable('bias', [in_channel * channel_multiplier], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)

        return conv


def res_block(input, expansion_ratio, output_dim, stride, is_train, name, bias=False, shortcut=True):
    with tf.name_scope(name), tf.variable_scope(name):
        # pw
        bottleneck_dim = round(expansion_ratio * input.get_shape().as_list()[-1])
        net = conv_1x1(input, bottleneck_dim, name='pw', bias=bias)
        net = batch_norm(net, train=is_train, name='pw_bn')
        net = relu(net)
        # dw
        net = dwise_conv(net, strides=[1, stride, stride, 1], name='dw', bias=bias)
        net = batch_norm(net, train=is_train, name='dw_bn')
        net = relu(net)
        # pw & linear
        net = conv_1x1(net, output_dim, name='pw_linear', bias=bias)
        net = batch_norm(net, train=is_train, name='pw_linear_bn')

        # element wise add, only for stride==1
        if shortcut and stride == 1:
            in_dim = int(input.get_shape().as_list()[-1])
            if in_dim != output_dim:
                ins = conv_1x1(input, output_dim, name='ex_dim')
                net = ins + net
            else:
                net = input + net

        return net


def separable_conv(input, k_size, output_dim, stride, pad='SAME', channel_multiplier=1, name='sep_conv', bias=False):
    with tf.name_scope(name), tf.variable_scope(name):
        in_channel = input.get_shape().as_list()[-1]
        dwise_filter = tf.get_variable('dw', [k_size, k_size, in_channel, channel_multiplier],
                                       regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                       initializer=tf.truncated_normal_initializer(stddev=0.02))

        pwise_filter = tf.get_variable('pw', [1, 1, in_channel * channel_multiplier, output_dim],
                                       regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                       initializer=tf.truncated_normal_initializer(stddev=0.02))
        strides = [1, stride, stride, 1]

        conv = tf.nn.separable_conv2d(input, dwise_filter, pwise_filter, strides, padding=pad, name=name)
        if bias:
            biases = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)
        return conv


def global_avg(x):
    with tf.name_scope('global_avg'):
        net = tf.layers.average_pooling2d(x, x.get_shape()[1:-1], 1)
        return net


def flatten(x):
    # flattened = tf.reshape(input,[x.get_shape().as_list()[0], -1])  # or, tf.layers.flatten(x)
    return tf.contrib.layers.flatten(x)


def pad2d(inputs, pad=(0, 0), mode='CONSTANT'):
    paddings = [[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]]
    net = tf.pad(inputs, paddings, mode=mode)
    return net


def mobilenetv2(inputs, num_classes, is_train=True, reuse=False):

    exp = 6  # expansion ratio

    net = conv2d_block(inputs, 32, 3, 1, is_train, name='conv1_1')  # [64,128,96,32]

    net = res_block(net, 1, 16, 1, is_train, name='res1_1')  # [64,128,96,16]

    net = res_block(net, 1, 16, 2, is_train, name='res2_1')  # size/2 [64,64,48,16]

    net = res_block(net, exp, 32, 2, is_train, name='res3_1')  # size/4 [64,32,24,32]

    net = res_block(net, exp, 64, 2, is_train, name='res4_1')  # size/8 [64,16,12,64]

    net = res_block(net, exp, 128, 2, is_train, name='res5_1')  # size/16 [64,8,6,128]

    net = res_block(net, exp, 256, 2, is_train, name='res6_1')  # size/16 [64,8,6,128]

    net = pwise_block(net, 256, is_train, name='pw_conv7_1')  # [64,8,6,256]

    net = global_avg(net)  # [64,1,1,256]

    net = conv_1x1(net, num_classes, name='logits')  # [64,1,1,10]

    logits = flatten(net)  # [64,10]

    return logits


min_after_dequeue = 32
batch_size = 32
capacity = min_after_dequeue + 3 * batch_size

files = tf.train.match_filenames_once(r'/media/xuke/SoftWare/CelebA/celebA_train.tfrecords')

filename_queue = tf.train.string_input_producer(files, shuffle=True)

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(serialized_example,
                                   features={
                                       'img_raw': tf.FixedLenFeature([], tf.string),
                                       'label': tf.FixedLenFeature([], tf.float32)
                                   })

img = tf.decode_raw(features['img_raw'], tf.float32)
image = tf.reshape(img, [128, 96, 3])
l = tf.decode_raw(features['label'], tf.float32)
label = tf.reshape(l, [42, 2])

batch_x, batch_y = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)

vfiles = tf.train.match_filenames_once(r'/media/xuke/SoftWare/CelebA/celebA_test.tfrecords')
vfilename_queue = tf.train.string_input_producer(files, shuffle=False)

reader = tf.TFRecordReader()
_, vserialized_example = reader.read(vfilename_queue)
vfeatures = tf.parse_single_example(vserialized_example,
                                    features={
                                        'img_raw': tf.FixedLenFeature([], tf.string),
                                        'label': tf.FixedLenFeature([], tf.float32)
                                    })

vimg = tf.decode_raw(vfeatures['img_raw'], tf.float32)
vimage = tf.reshape(vimg, [128, 96, 3])
vl = tf.decode_raw(vfeatures['label'], tf.float32)
vlabel = tf.reshape(vl, [42, 2])
vbatch_x, vbatch_y = tf.train.shuffle_batch([vimage, vlabel], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)

w = 96
h = 128
c = 3
num_class = 80

n_step = 2001

model_path = r'/media/xuke/SoftWare/CelebA'
summary_path = r'/media/xuke/SoftWare/CelebA'

with tf.name_scope('input'):
    # 占位符
    x = tf.placeholder(tf.float32, shape=[None, h, w, c], name='x')
    y_ = tf.placeholder(tf.float32, shape=[None, 42, 2], name='y_')
    is_train = tf.placeholder(tf.bool)

logits = mobilenetv2(x, num_class, is_train)
pre = tf.reshape(logits, [-1, 42, 2])
res = tf.nn.softmax(pre, name='res')

b = tf.constant(value=1, dtype=tf.float32)
logits_eval = tf.multiply(pre, b, name='logits_eval')

global_step = tf.Variable(0)

with tf.name_scope('train'):
    with tf.name_scope('corss_entropy'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pre, labels=y_))
        tf.summary.scalar('corss_entropy', loss)
    with tf.name_scope('train_op'):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss, global_step=global_step)

with tf.name_scope('acc'):
    with tf.name_scope('correct_prediction'):
        max_idx_p = tf.argmax(res, 2)
        max_idx_l = tf.argmax(y_, 2)
        correct = tf.equal(max_idx_p, max_idx_l)
    with tf.name_scope('accuary'):
        acc = tf.reduce_mean(tf.cast(correct, tf.float32))
        tf.summary.scalar('accuary', acc)

merged = tf.summary.merge_all()
saver = tf.train.Saver(max_to_keep=10)

with tf.Session() as sess:
    # summary_writer
    train_writer = tf.summary.FileWriter(summary_path, tf.get_default_graph())
    # 变量初始化
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # 启动多线程
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    max_ac = 0
    # training
    for step in range(n_step):
        bx, by = sess.run([batch_x, batch_y])
        feed_dict = {x: bx, y_: by, is_train: True}
        train_sum, _, err, ac = sess.run([merged, train_op, loss, acc], feed_dict=feed_dict)
        print("step:{},train_loss:{},train_acc:{}".format(step, err, ac))
        train_writer.add_summary(train_sum, step)

        # validation
        vac = 0
        if step % 50 == 0:
            for vv in range(3):
                vbx, vby = sess.run([vbatch_x, vbatch_y])
                feed_dict = {x: vbx, y_: vby, is_train: False}
                v_ac = sess.run(acc, feed_dict=feed_dict)
                vac += v_ac
            print("val_acc:{}".format(vac / 3))
            if vac > max_ac:
                max_ac = vac
                saver.save(sess, model_path, global_step=global_step)

    train_writer.close()

    coord.request_stop()
    coord.join(threads)