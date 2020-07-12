import contextlib2
import tensorflow as tf
from .create_example import create_tf_example
from tfrecord import TRAIN_DIR, TEST_DIR


def open_sharded_output_tfrecords(exit_stack, base_path, num_shards):
    tf_record_output_filenames = [
        '{}/tfrecord-{:05d}-of-{:05d}'.format(base_path, idx, num_shards)
        for idx in range(num_shards)
    ]

    tfrecords = [
        exit_stack.enter_context(tf.io.TFRecordWriter(file_name))
        for file_name in tf_record_output_filenames
    ]

    return tfrecords


def write_shards(data,
                 shards,
                 bird_dir,
                 output_dir,
                 log_progress=False):
    with contextlib2.ExitStack() as tf_record_close_stack:
        output_train = open_sharded_output_tfrecords(tf_record_close_stack,
                                                     base_path=f'{bird_dir}/{output_dir}',
                                                     num_shards=shards)
        for idx, example in data.iterrows():
            if idx % 1000 == 0 and log_progress:
                print(f'On image {idx} of {data.shape[0]}')
            output_train[idx % shards].write(create_tf_example(bird_dir, example=example).SerializeToString())


def write_records(train_data,
                  test_data,
                  bird_dir,
                  file_size,
                  log_progress=False):
    train_shards = train_data.shape[0] // file_size + 1
    test_shards = test_data.shape[0] // file_size + 1

    # Write Train Data Shards
    write_shards(data=train_data, shards=train_shards,
                 bird_dir=bird_dir,
                 output_dir=TRAIN_DIR,
                 log_progress=log_progress)

    # Write Test Data Shards
    write_shards(data=test_data, shards=test_shards,
                 bird_dir=bird_dir,
                 output_dir=TEST_DIR,
                 log_progress=log_progress)
