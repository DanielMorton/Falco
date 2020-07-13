from .create_example import create_tf_example
from .inception import get_datasets
from .inception_test import get_test_datasets

TARGET_DIMS = [224, 240, 260, 300, 380, 456, 528, 600]
TRAIN_DIR = 'tf_train'
TEST_DIR = 'tf_test'


