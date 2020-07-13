import tensorflow as tf


def bytes_feature(value):
    """Converts value to byte feature for tfrecord."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    """Converts list of values to byte features for tfrecord."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def int64_feature(value):
    """Converts value to integer feature for tfrecord."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    """Converts list of values to integer fetures for tfrecord."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    """Converts value to float feature for tfrecord."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
    """Converts list of values to float fetures for tfrecord."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
