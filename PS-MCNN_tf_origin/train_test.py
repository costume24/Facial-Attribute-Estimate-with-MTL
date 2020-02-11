import time
import tensorflow as tf
import argparse
import os
from psmcnn import *
from ops import *
from utils import *


def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args


"""parsing and configuration"""

desc = "Tensorflow implementation of PS-MCNN"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('--phase', type=str, default='train', help='train or test ?')
parser.add_argument('--dataset', type=str, default='celebA', help='[celebA,LFWA]')

parser.add_argument('--epoch', type=int, default=10, help='The number of epochs to run')
parser.add_argument('--batch_size', type=int, default=16, help='The size of batch per gpu')

parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', help='Directory name to save the checkpoints')
parser.add_argument('--log_dir', type=str, default='logs', help='Directory name to save training logs')

args = check_args(parser.parse_args())
"""checking arguments"""

print_freq = 1
epoch_lr = args.lr
print("[*] Start.")
with tf.device("/gpu:0"):
    '''导入数据
    '''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        x_train, y_train = read_and_decode(r'/media/xuke/SoftWare/CelebA/celebA_train.tfrecords', is_train=True)
        x_test, y_test = read_and_decode(r'/media/xuke/SoftWare/CelebA/celebA_test.tfrecords', is_train=False)
        # print('======================================================')
        # print('[*] The shape of y_train is %s' % y_train.get_shape())
        # [*] The shape of y_train is (40,)
        # print('======================================================')
        min_after_dequeue = 100
        capacity = min_after_dequeue + 3 * args.batch_size
        batch_x, batch_y = tf.train.shuffle_batch([x_train, y_train],
                                                  batch_size=args.batch_size,
                                                  capacity=capacity,
                                                  shapes=([3, 160, 192], [40]),
                                                  min_after_dequeue=min_after_dequeue)
        # print('======================================================')
        # print('[*] The shape of batch_y is %s' % batch_y.get_shape())
        # [*] The shape of batch_y is (64, 40)
        # print('======================================================')
        batch_x_test, batch_y_test = tf.train.shuffle_batch([x_test, y_test],
                                                            batch_size=args.batch_size,
                                                            shapes=([3, 160, 192], [40]),
                                                            capacity=capacity,
                                                            num_threads=12,
                                                            min_after_dequeue=min_after_dequeue)
        '''定义操作
        '''
        cnn = ps_mcnn(sess, args)
        train_logits_0, train_logits_1, train_logits_2, train_logits_3 = cnn.network(batch_x)
        test_logits_0, test_logits_1, test_logits_2, test_logits_3 = cnn.network(batch_x_test, is_training=False, reuse=True)

        # accuracy为列表
        train_logits = [train_logits_0, train_logits_1, train_logits_2, train_logits_3]
        test_logits = [test_logits_0, test_logits_1, test_logits_2, test_logits_3]

        train_loss, train_accuracy = classification_loss(train_logits, batch_y, args.batch_size)
        test_loss, test_accuracy = classification_loss(test_logits, batch_y_test, args.batch_size)

        #cost
        reg_loss = tf.losses.get_regularization_loss()
        train_loss += reg_loss
        test_loss += reg_loss
        """ Training """
        optim = tf.train.MomentumOptimizer(epoch_lr, momentum=0.9).minimize(train_loss)
        """" Summary """
        summary_train_loss = tf.summary.scalar("train_loss", train_loss)
        summary_train_accuracy = tf.summary.scalar("train_accuracy", tf.reduce_mean(train_accuracy))

        summary_test_loss = tf.summary.scalar("test_loss", test_loss)
        summary_test_accuracy = tf.summary.scalar("test_accuracy", tf.reduce_mean(test_accuracy))

        train_summary = tf.summary.merge([summary_train_loss, summary_train_accuracy])
        test_summary = tf.summary.merge([summary_test_loss, summary_test_accuracy])

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        print("[*] Start training ...")
        writer = tf.summary.FileWriter(args.log_dir + '/' + cnn.model_dir, sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = cnn.load(args.checkpoint_dir)
        if could_load:
            epoch_lr = args.init_lr
            start_epoch = (int)(checkpoint_counter / cnn.iteration)
            start_batch_id = checkpoint_counter - start_epoch * cnn.iteration
            counter = checkpoint_counter

            if start_epoch >= int(cnn.epoch * 0.75):
                epoch_lr = epoch_lr * 0.01
            elif start_epoch >= int(cnn.epoch * 0.5) and start_epoch < int(cnn.epoch * 0.75):
                epoch_lr = epoch_lr * 0.1
            print("[*] Load SUCCESS")
        else:
            epoch_lr = cnn.init_lr
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print("[!] Load failed...")

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, cnn.epoch):
            if epoch == int(cnn.epoch * 0.5) or epoch == int(cnn.epoch * 0.75):
                epoch_lr = epoch_lr * 0.1

            # get batch data
            for idx in range(start_batch_id, cnn.iteration):
                # update network
                # print('======================================================')
                # print(sess.run(train_loss))
                # print('======================================================')
                # _, summary_str, train_loss, train_accuracy = sess.run([optim, train_summary, train_loss, train_accuracy])
                _ = sess.run(optim)
                summary_str = sess.run(train_summary)
                train_loss = sess.run(train_loss)
                train_accuracy = sess.run(train_accuracy)

                writer.add_summary(summary_str, counter)

                # test
                summary_str, test_loss, test_accuracy = sess.run([test_summary, test_loss, test_accuracy])
                writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%5d/%5d] time: %4.4f, train_accuracy: %.2f, test_accuracy: %.2f, learning_rate : %.5f" \
                        % (epoch, idx, iteration, time.time() - start_time, tf.reduce_mean(train_accuracy), tf.reduce_mean(test_accuracy), epoch_lr))

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            cnn.save(cnn.checkpoint_dir, counter)

        # save model for final step
        cnn.save(cnn.checkpoint_dir, counter)

        coord.request_stop()
        coord.join(threads)
        sess.close()