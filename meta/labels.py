import pandas as pd
from meta import CLASS_LABEL_FILE


def read_class_labels(bird_dir, top_levels, parent_map):
    def get_class(l):
        return l if l in top_levels else get_class(parent_map[l])

    class_labels = pd.read_table(f'{bird_dir}/{CLASS_LABEL_FILE}', sep=' ',
                                 header=None)
    class_labels.columns = ['image', 'id']
    class_labels['class_id'] = class_labels['id'].apply(get_class)

    return class_labels