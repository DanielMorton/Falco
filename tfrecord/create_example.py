import tensorflow as tf
from .features import bytes_feature, float_feature, int64_feature


def create_tf_example(bird_dir, example):
    """Creates a single item to add to a tfrecord file."""
    height = example['img_height']  # Image height
    width = example['img_width']  # Image width
    filename = example['file']  # Filename of the image.

    # Encoded image bytes
    encoded_image_data = open(f'{bird_dir}/images/{filename}', 'rb').read()

    # Bounding Box
    xmin = example['x'] / example['img_width']
    xmax = (example['x'] + example['width']) / example['img_width']
    ymin = example['y'] / example['img_height']
    ymax = (example['y'] + example['height']) / example['img_height']

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'height': int64_feature(height),
        'width': int64_feature(width),
        'filename': bytes_feature(filename.encode()),
        'image': bytes_feature(encoded_image_data),

        'xmin': float_feature(xmin),
        'xmax': float_feature(xmax),
        'ymin': float_feature(ymin),
        'ymax': float_feature(ymax),

        'label_name': bytes_feature(example['label_name'].encode()),

        # ID of lowest level taxon class.
        'terminal_id': int64_feature(example['terminal_id']),

        # ID of Name level taxon class.
        'name_id': int64_feature(example['name_id']),
        'name': bytes_feature(example['name'].encode()),

        # ID of top level taxon class.
        'class_id': int64_feature(example['class_id']),
        'class_name': bytes_feature(example['class_name'].encode())
    }))
    return tf_example
