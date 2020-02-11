import os
import glob
import tensorflow as tf
from PIL import Image
import numpy as np


# check, if file exists, make link
def check_link(in_dir, basename, out_dir):
    in_file = os.path.join(in_dir, basename)
    if os.path.exists(in_file):
        link_file = os.path.join(out_dir, basename)
        rel_link = os.path.relpath(in_file, out_dir)  # from out_dir to in_file
        os.symlink(rel_link, link_file)


def add_splits(data_path):
    # images_path = os.path.join(data_path, 'Img/img_align_celeba')
    images_path = data_path
    out_path = '/media/xuke/Files/Final/DataSet'
    train_dir = os.path.join(out_path, 'train')
    valid_dir = os.path.join(out_path, 'valid')
    test_dir = os.path.join(out_path, 'test')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(valid_dir):
        os.makedirs(valid_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # these constants based on the standard CelebA splits
    NUM_EXAMPLES = 202599
    TRAIN_STOP = 162770
    VALID_STOP = 182637

    for i in range(0, TRAIN_STOP):
        basename = "{:06d}.jpg".format(i + 1)
        check_link(images_path, basename, train_dir)
    for i in range(TRAIN_STOP, VALID_STOP):
        basename = "{:06d}.jpg".format(i + 1)
        check_link(images_path, basename, valid_dir)
    for i in range(VALID_STOP, NUM_EXAMPLES):
        basename = "{:06d}.jpg".format(i + 1)
        check_link(images_path, basename, test_dir)


def load_split(mType):
    '''本方法不再使用，无法一次性加载到内存
    '''
    img_path = '/media/xuke/Files/Final/DataSet'
    label_path = '/media/xuke/SoftWare/BaiduNetdiskDownload/CelebA/Anno/list_attr_celeba.txt'
    dir = os.path.join(img_path, mType)
    paths = glob.glob("{}/*.{}".format(dir, 'jpg'))  # is a list
    raw_label = []
    res = []
    label = []
    if mType == 'train':
        res = np.zeros([162770, 3, 160, 192], dtype='float32')
        label = np.zeros([162770, 40], dtype='float32')
    elif mType == 'val':
        res = np.zeros([19867, 3, 160, 192], dtype='float32')
        label = np.zeros([19867, 40], dtype='float32')
    else:
        res = np.zeros([19962, 3, 160, 192], dtype='float32')
        label = np.zeros([19962, 40], dtype='float32')

    with open(label_path, 'r') as f:
        raw_label = f.readlines()[2:]
        if mType == 'train':
            raw_label = raw_label[:162770]
        elif mType == 'val':
            raw_label = raw_label[162770:162770 + 19867]
        else:
            raw_label = raw_label[162770 + 19867:]
        i = 0
        for line in raw_label:
            line = line.split()
            label[i] = line[1:]
            i += 1

    for i in range(len(paths)):
        with Image.open(paths[i]) as img:  # all images at the same size 178 x 218
            img = np.transpose(np.array(img), [2, 0, 1])
            res[i] = img
    return res, label


if __name__ == '__main__':
    # base_path = '/media/xuke/SoftWare/BaiduNetdiskDownload/CelebA/Img/img_align_celeba/img_align_celeba'
    # add_splits(base_path)
    # out_path = '/media/xuke/Files/Final/DataSet'
    # train_dir = os.path.join(out_path, 'train')
    # for ext in ["jpg", "png"]:
    #     paths = glob.glob("{}/*.{}".format(train_dir, ext))  # is a list
    #     if ext == "jpg":
    #         tf_decode = tf.image.decode_jpeg
    #     elif ext == "png":
    #         tf_decode = tf.image.decode_png

    #     if len(paths) != 0:
    #         break

    # with Image.open(paths[0]) as img:  # all images at the same size 178 x 218
    #     w, h = img.size
    #     shape = [h, w, 3]
    #     print(w, h)
    #     img = np.array(img)
    #     print(img.shape)
    #     img = np.transpose(img, [2, 0, 1])
    #     print(img.shape)
    train, label = load_split('val')
    print(label[0])