import tensorflow as tf
from tfrecord import TARGET_DIMS


def decode_jpeg(example,
                category_count,
                train=False):
    features = {
        "filename": tf.io.FixedLenFeature([], tf.string),

        "class_id": tf.io.FixedLenFeature([], tf.int64),
        "class_name": tf.io.FixedLenFeature([], tf.string),

        "name_id": tf.io.FixedLenFeature([], tf.int64),
        "name": tf.io.FixedLenFeature([], tf.string),

        "terminal_id": tf.io.FixedLenFeature([], tf.int64),
        "label_name": tf.io.FixedLenFeature([], tf.string),

        "xmin": tf.io.FixedLenFeature([], tf.float32),
        "ymin": tf.io.FixedLenFeature([], tf.float32),
        "xmax": tf.io.FixedLenFeature([], tf.float32),
        "ymax": tf.io.FixedLenFeature([], tf.float32),

        "image": tf.io.FixedLenFeature([], tf.string),
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(example, features)
    img_dim = (tf.cast(example['height'], tf.int32),
               tf.cast(example['width'], tf.int32), 3)

    decoded = tf.image.decode_jpeg(example['image'], channels=3)
    image = tf.reshape(tf.cast(decoded, tf.float32) / 255, img_dim)
    oh = tf.one_hot(example['terminal_id'], depth=category_count)

    if train:
        xmin = example['xmin']
        ymin = example['ymin']
        xmax = example['xmax']
        ymax = example['ymax']
        bbox = tf.expand_dims(tf.expand_dims(tf.stack([ymin, xmin, ymax, xmax]), 0), 0)
        return image, oh, tf.clip_by_value(bbox, 0.0, 1.0)
    else:
        return image, oh


@tf.function
def distort_color(image):
    image = tf.image.random_flip_left_right(image)
    color_ordering = tf.random.uniform([], maxval=4, dtype=tf.int32)
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    else:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    return tf.clip_by_value(image, 0.0, 1.0)


def distorted_bounding_box_crop(image,
                                bbox=tf.constant([0.0, 0.0, 1.0, 1.0],
                                                 dtype=tf.float32,
                                                 shape=[1, 1, 4]),
                                min_object_covered=0.5,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.1, 1.0),
                                max_attempts=100):
    bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)

    cropped_image = tf.slice(image, bbox_begin, bbox_size)
    return cropped_image


def parse_train_tfrecord(example, category_count, target_size):
    image, oh, bbox = decode_jpeg(example, category_count, train=True)
    image = distorted_bounding_box_crop(image, bbox)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, (target_size, target_size))

    image = distort_color(image)
    image = tf.keras.applications.imagenet_utils.preprocess_input(image * 255,
                                                                  mode='torch')

    return image, 0.9 * oh + 0.1 / category_count


def parse_val_tfrecord(example, category_count, target_size):
    image, oh = decode_jpeg(example, category_count)

    image = tf.keras.applications.imagenet_utils.preprocess_input(image * 255,
                                                                  mode='torch')
    image = tf.image.resize_with_pad(image, target_size, target_size)
    return image, 0.9 * oh + 0.1 / category_count


def load_dataset(filenames, category_count, c_size, auto, train=False):
    records = tf.data.TFRecordDataset(filenames,
                                      num_parallel_reads=auto)
    if train:
        return records.map(lambda r: parse_train_tfrecord(r, category_count, TARGET_DIMS[c_size]),
                           num_parallel_calls=auto)
    else:
        return records.map(lambda r: parse_val_tfrecord(r, category_count, TARGET_DIMS[c_size]),
                           num_parallel_calls=auto)


def get_datasets(train_files,
                 test_files,
                 category_count,
                 c_size,
                 batch_size,
                 auto,
                 shuffle=2048):
    train = load_dataset(train_files,
                         category_count,
                         c_size=c_size,
                         train=True,
                         auto=auto).repeat() \
        .shuffle(shuffle) \
        .batch(batch_size) \
        .prefetch(auto)

    val = load_dataset(test_files,
                       category_count,
                       c_size=c_size,
                       auto=auto).batch(batch_size).prefetch(auto)
    return train, val
