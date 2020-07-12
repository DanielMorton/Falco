import pandas as pd
from meta import TRAIN_TEST_SPLIT_FILE


def read_train_test(bird_dir):
    train_test = pd.read_table(f'{bird_dir}/{TRAIN_TEST_SPLIT_FILE}', sep=' ',
                               header=None)
    train_test.columns = ['image', 'is_train']
    return train_test
