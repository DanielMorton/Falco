import tensorflow as tf
from tfrecord import TARGET_DIMS
from .imagenet import decode_jpeg


def load_dataset(filenames, category_count, auto, resolution=None, crop=False, free=False):
    # Read from TFRecords. For optimal performance, we interleave reads from multiple files.
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


def parse_test_tfrecord(example, category_count, target_size=None, boxes=False):
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


def get_test_datasets(test_files, batch_size, category_count, auto, resolution=None, crop=True, free=True):
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
