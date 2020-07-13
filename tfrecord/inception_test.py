import tensorflow as tf
from tfrecord import TARGET_DIMS
from .inception import decode_jpeg


def load_dataset(filenames,
                 category_count,
                 auto,
                 resolution=None,
                 crop=False,
                 free=False):
    """Create testing Tensorflow Dataset.
    If crop and free are both false, pads and resizes image to resolution.
    If only crop is false, uses raw image size.
    If only free is true pads and resizes bounding box crop of image.
    If crop and free are both true uses raw bounding box crop.

    :param filenames: List of tfrecord files containing raw data.
    :param category_count: Number of prediction categories.
    :param auto: Number of parallel calls to the data loader.
    :param resolution: Resolution to use for testing. Defaults to None to use the raw image dimensions.
    :param crop: Using bounding box crop for testing. Defaluts to False.
    :param free: Use raw image dimension for testing. Defaults to False.

    :return: Tensorflow Dataset of training or validation data.
    """
    records = tf.data.TFRecordDataset(filenames,
                                      num_parallel_reads=auto)
    if crop and free:
        return records.map(lambda r: parse_test_tfrecord(r, category_count, boxes=True),
                           num_parallel_calls=auto)
    elif crop:
        return records.map(lambda r: parse_test_tfrecord(r, category_count,
                                                         target_size=TARGET_DIMS[resolution],
                                                         boxes=True),
                           num_parallel_calls=auto)
    elif free:
        return records.map(lambda r: parse_test_tfrecord(r, category_count),
                           num_parallel_calls=auto)
    else:
        return records.map(lambda r: parse_test_tfrecord(r, category_count,
                                                         target_size=TARGET_DIMS[resolution]),
                           num_parallel_calls=auto)


def parse_test_tfrecord(example,
                        category_count,
                        target_size=None,
                        boxes=False):
    """Reads and pre-processes images from test data.

    :param example: Single example from tfrecord.
    :param category_count: Number of prediction categories.
    :param target_size: Final size of validation image. None if raw image size is used.
    :param boxes: Use bounding box crop for model testing. Defaluts to False.

    :return: Validation image and response variable.
    """
    if boxes:
        image, oh, bbox = decode_jpeg(example, category_count, boxes=boxes)
        width = tf.cast(tf.shape(image)[1], tf.float32)
        height = tf.cast(tf.shape(image)[0], tf.float32)

        ymin = tf.cast(bbox[0, 0, 0] * height, tf.int32)
        xmin = tf.cast(bbox[0, 0, 1] * width, tf.int32)
        ymax = tf.cast(bbox[0, 0, 2] * height, tf.int32)
        xmax = tf.cast(bbox[0, 0, 3] * width, tf.int32)
        image = tf.image.crop_to_bounding_box(image, ymin, xmin,
                                              ymax - ymin, xmax - xmin)
    else:
        image, oh = decode_jpeg(example, category_count)

    image = tf.keras.applications.imagenet_utils.preprocess_input(image * 255,
                                                                  mode='torch')

    if target_size:
        image = tf.image.resize_with_pad(image, target_size, target_size)

    return image, oh


def get_test_datasets(test_files,
                      category_count,
                      batch_size,
                      auto,
                      resolution=None,
                      crop=True,
                      free=True):
    """Returns test Tensorflow Datasets.

    :param test_files: List of validation data files.
    :param category_count: Number of prediction categories.
    :param batch_size: Batch size for training and validation.
    :param auto: Number of parallel calls to the data loader.
    :param resolution: Resolution to use for testing. Defaults to None to use the raw image dimensions.
    :param crop: Using bounding box crop for testing. Defaluts to False.
    :param free: Use raw image dimension for testing. Defaults to False.

    :return: Training and Validation Tensorflow datasets.
    """
    if crop and free:
        return load_dataset(test_files, category_count, auto,
                            crop=True, free=True).batch(1).prefetch(auto)
    elif crop:
        return load_dataset(test_files, category_count, auto,
                            resolution=resolution, crop=True).batch(batch_size).prefetch(auto)
    elif free:
        return load_dataset(test_files, category_count,
                            auto, free=True).batch(1).prefetch(auto)
    else:
        load_dataset(test_files, category_count,
                     auto, resolution=resolution).batch(batch_size).prefetch(auto)
