import pandas as pd
from meta import IMAGE_FILE


def read_images(bird_dir):
    """Loads image table and converts to DataFrame
    :param bird_dir Directory containing Cornell Metadata.
    :return DataFrame of image file names.
    """
    images = pd.read_table(f'{bird_dir}/{IMAGE_FILE}', sep=' ',
                           header=None)
    images.columns = ['image', 'file']
    return images
