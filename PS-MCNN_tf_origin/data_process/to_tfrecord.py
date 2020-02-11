import tensorflow as tf
import numpy as np
from PIL import Image
import os
import argparse
import time

fpath = '/media/xuke/Files/Final/DataSet/'
label_path = '/media/xuke/SoftWare/BaiduNetdiskDownload/CelebA/Anno/list_attr_celeba.txt'


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_label(mType):
    label = []
    if mType == 'train':
        label = np.zeros([162770, 40], dtype='float32')
    elif mType == 'valid':
        label = np.zeros([19867, 40], dtype='float32')
    else:
        label = np.zeros([19962, 40], dtype='float32')
    with open(label_path, 'r') as f:
        raw_label = f.readlines()[2:]
        if mType == 'train':
            raw_label = raw_label[:162770]
        elif mType == 'valid':
            raw_label = raw_label[162770:162770 + 19867]
        else:
            raw_label = raw_label[162770 + 19867:]
        i = 0
        for line in raw_label:
            line = line.split()
            label[i] = line[1:]
            label[i] = list(map(int, label[i]))
            i += 1
        return label


def convert(mType):
    writer = tf.python_io.TFRecordWriter('/media/xuke/SoftWare/CelebA/celebA_%s.tfrecords' % mType)
    path = fpath + mType + '/'
    labels = get_label(mType)
    idx = 0
    for img_name in os.listdir(path):
        label = labels[idx]
        img_path = path + img_name
        # img = Image.open(img_path)
        # img = img.resize((160, 192), Image.ANTIALIAS)
        # img=np.array(img)
        # img_raw = img.tostring()
        fid = tf.gfile.GFile(img_path, 'rb')
        img = fid.read()
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'img_raw': _bytes_feature(img),
                'width': _int64_feature(192),
                'height': _int64_feature(160),
                'label': tf.train.Feature(float_list=tf.train.FloatList(value=label))
            }))
        writer.write(example.SerializeToString())  #序列化为字符串
        idx += 1
        fid.close()
        if idx % 1000 == 0:
            print('%d images has been processed' % idx)
    print('[*] Convert success')
    print('[*] Total numbers of examples: %d' % idx)
    writer.close()
    return idx


def single_convert():
    writer = tf.python_io.TFRecordWriter('/media/xuke/SoftWare/CelebA/io_test.tfrecords')
    path = fpath + 'train' + '/'

    def get_single_label(label_path):
        with open(label_path, 'r') as f:
            raw_label = f.readlines()[2:]
            line = raw_label[0]
            line = line.split()
            label = line[1:]
            label = list(map(int, label))
            return label

    label = get_single_label(label_path)
    idx = 1
    img_path = path + '000001.jpg'
    img = Image.open(img_path)
    img = img.resize((160, 192), Image.ANTIALIAS)
    img_raw = img.tobytes()
    example = tf.train.Example(features=tf.train.Features(
        feature={
            'img_raw': _bytes_feature(img_raw),
            'width': _int64_feature(192),
            'height': _int64_feature(160),
            'label': tf.train.Feature(float_list=tf.train.FloatList(value=label))
        }))
    writer.write(example.SerializeToString())  #序列化为字符串
    print('[*] Convert success')
    writer.close()
    return idx


def parse_args():
    desc = "Convert celebA to tfrecords"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--type', type=str, default='train', help='train or test ?')
    parser.add_argument('--test', type=str, default='False', help='test tfrecord')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print('[*] Converting %s images to TFRecords...' % args.type)
    try:
        start = time.time()
        if args.test == 'False':
            total = convert(args.type)
        elif args.test == 'True':
            total = single_convert()
        elapsed = (time.time() - start)
        print('Task finished after %f mins' % float((elapsed / 60)))
        print('Averaged %.5f seconds for each example' % float((elapsed / total)))
    except Exception as e:
        print('[!] Fail to convert')
        print(e)
