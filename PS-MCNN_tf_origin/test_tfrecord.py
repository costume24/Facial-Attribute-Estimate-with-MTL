import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

files = tf.train.match_filenames_once('/media/xuke/SoftWare/CelebA/io_test.tfrecords')
filename_queue = tf.train.string_input_producer(files, shuffle=False)
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(serialized_example,
                                   features={
                                       'img_raw': tf.FixedLenFeature((), tf.string),
                                       'label': tf.FixedLenFeature(([40]), tf.float32),
                                       'height': tf.FixedLenFeature([1], tf.int64),
                                       'width': tf.FixedLenFeature([1], tf.int64)
                                   })
img, train_y = features['img_raw'], features['label']
height = tf.cast(features['height'], tf.int64)
width = tf.cast(features['width'], tf.int64)

decoded_image_train = tf.decode_raw(img, tf.uint8)
decoded_image_train = tf.cast(decoded_image_train, tf.float32) * (1. / 255.0)
decoded_image_train = tf.reshape(decoded_image_train, [3, 160, 192])
decoded_image_train.set_shape([3, 160, 192])

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    example, label, decoded_image = sess.run([img, train_y, decoded_image_train])
    print(type(example))
    print(label.shape)
    print(decoded_image.shape)
    print(decoded_image)