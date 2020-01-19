import tensorflow as tf


def load_dataset():
    print('loading training images...')
    files = tf.train.match_filenames_once('/media/xuke/SoftWare/CelebA/celebA_train.tfrecords')
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
    img, label_train = features['image'], features['label']
    height, width = features['height'], features['width']
    decoded_image_train = tf.decode_raw(img, tf.uint8)
    print(decoded_image_train.shape)
    decoded_image_train = np.array(decoded_image_train)

    print('finished loading training images')

    print('loading test images...')

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
    decoded_image_val = tf.decode_raw(img, tf.uint8)
    # decoded_image_val.set_shape([3, 160, 192])
    decoded_image_val = np.array(decoded_image_val)
    decoded_image_val = np.transpose(decoded_image_val, [0, 3, 1, 2])
    print('finished loading test images')

    decoded_image_train, decoded_image_val = normalize(decoded_image_train, decoded_image_val)

    return decoded_image_train, decoded_image_val, label_train, label_val