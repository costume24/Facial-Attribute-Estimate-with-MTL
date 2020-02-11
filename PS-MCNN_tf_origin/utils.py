import tensorflow as tf
import tensorflow.contrib.slim as slim
from keras.datasets import cifar10, cifar100, mnist, fashion_mnist
from keras.utils import to_categorical
import numpy as np
import random
from scipy import misc
from data_process.train_test_split import load_split
import time
import os


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def show_all_variables():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def str2bool(x):
    return x.lower() in ('true')


def load_dataset():

    IMAGENET_MEAN = [123.68, 116.78, 103.94]

    print('[*] loading training images...')
    start = time.time()
    files = tf.train.match_filenames_once('/media/xuke/SoftWare/CelebA/celebA_train.tfrecords')
    filename_queue = tf.train.string_input_producer(files, shuffle=False)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'img_raw': tf.FixedLenFeature((), tf.string),
                                           'label': tf.FixedLenFeature((), tf.string),
                                           'height': tf.FixedLenFeature([1], tf.int64),
                                           'width': tf.FixedLenFeature([1], tf.int64)
                                       })
    img, label_train = features['image'], features['label']
    height = tf.cast(features['height'], tf.int64)
    width = tf.cast(features['width'], tf.int64)

    decoded_image_train = tf.decode_raw(img, tf.uint8)
    decoded_image_train = tf.cast(decoded_image_train, tf.float32)
    decoded_image_train = tf.reshape(decoded_image_train, [3, 160, 192])
    decoded_image_train.set_shape([3, 160, 192])
    # decoded_image_train = tf.reshape(decoded_image_train, [3, height, width])

    elapsed = time.time() - start
    print('[*] Finish loading training images')
    print('[*] Task finished after %f seconds' % elapsed)

    print('[*] loading test images...')
    start = time.time()

    files = tf.train.match_filenames_once('/media/xuke/SoftWare/CelebA/celebA_valid.tfrecords')
    filename_queue = tf.train.string_input_producer(files, shuffle=False)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'height': tf.FixedLenFeature([], tf.int64),
                                           'width': tf.FixedLenFeature([], tf.int64)
                                       })
    img, label_val = features['image'], features['label']
    height, width = features['height'], features['width']
    print('[*] type of label_val is %s' % type(label_val))
    decoded_image_val = tf.decode_raw(img, tf.uint8)
    decoded_image_val = tf.cast(decoded_image_val, tf.float32)
    decoded_image_val = tf.reshape(decoded_image_val, [3, 160, 192])
    decoded_image_val.set_shape([3, 160, 192])
    # decoded_image_val = tf.reshape(decoded_image_val, [3, height, width])

    elapsed = time.time() - start
    print('[*] Finish loading test images')
    print('[*] Task finished after %f seconds' % elapsed)

    # decoded_image_train, decoded_image_val = normalize(decoded_image_train, decoded_image_val)
    # 会报错，因为此时其维度为(3,?,?)，有两个维度缺失

    return decoded_image_train, decoded_image_val, label_train, label_val


def read_and_decode(filename, is_train):
    if is_train == True:
        files = tf.train.match_filenames_once(filename)
    else:
        files = tf.train.match_filenames_once(filename)
    filename_queue = tf.train.string_input_producer([files])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'img_raw': tf.FixedLenFeature((), tf.string),
                                           'label': tf.FixedLenFeature([40], tf.float32),
                                           'height': tf.FixedLenFeature([1], tf.int64),
                                           'width': tf.FixedLenFeature([1], tf.int64)
                                       })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [3, 160, 192])
    img = tf.cast(img, tf.float32) * (1. / 255)
    img = tf.reshape(img, [3, 160, 192])
    img.set_shape([3, 160, 192])

    label = tf.cast(features['label'], tf.float32)
    # attr_order = [[1, 3, 4, 5, 8, 9, 11, 12, 15, 17, 23, 28, 35], [7, 19, 27, 29, 30, 34], [6, 14, 16, 22, 21, 24, 36, 37, 38],
    #               [0, 2, 10, 13, 18, 20, 25, 26, 32, 31, 33, 39]]
    # label_0 = tf.gather(label, attr_order[0])
    # label_1 = tf.gather(label, attr_order[1])
    # label_2 = tf.gather(label, attr_order[2])
    # label_3 = tf.gather(label, attr_order[3])

    # labels = {
    #     "01": label_0,
    #     "02": label_1,
    #     "03": label_2,
    #     "04": label_3,
    # }

    return img, label


def normalize(X_train, X_test):

    mean = np.mean(X_train, axis=(0, 1, 2))
    std = np.std(X_train, axis=(0, 1, 2))

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test


def get_annotations_map():
    valAnnotationsPath = './tiny-imagenet-200/val/val_annotations.txt'
    valAnnotationsFile = open(valAnnotationsPath, 'r')
    valAnnotationsContents = valAnnotationsFile.read()
    valAnnotations = {}

    for line in valAnnotationsContents.splitlines():
        pieces = line.strip().split()
        valAnnotations[pieces[0]] = pieces[1]

    return valAnnotations


def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad, mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0], nw:nw + crop_shape[1]]
    return new_batch


def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch


def data_augmentation(batch, h, w, dataset_name):
    batch = _random_flip_leftright(batch)
    batch = _random_crop(batch, [w, h], 4)
    return batch