import pandas as pd
from meta import SIZE_FILE


def read_sizes(bird_dir):
    sizes = pd.read_table(f'{bird_dir}/{SIZE_FILE}', sep=' ',
                          header=None)
    sizes.columns = ['image', 'img_width', 'img_height']
    return sizes
