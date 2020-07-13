from .boxes import read_boxes
from .classes import read_classes
from .hierarchy import read_hierarchy
from .images import read_images
from .labels import read_class_labels
from .sizes import read_sizes
from .train_test import read_train_test


def read_meta(bird_dir):
    """Loads all image meta data and performs joins to create train and test DataFrames."""
    hierarcy, parent_map, top_levels, terminal_levels = read_hierarchy(bird_dir=bird_dir)
    class_labels = read_class_labels(bird_dir=bird_dir,
                                     top_levels=top_levels,
                                     parent_map=parent_map)
    classes, terminal_classes = read_classes(bird_dir=bird_dir,
                                             terminal_levels=terminal_levels)

    meta = class_labels.merge(classes).merge(classes.rename(columns={'label_name': 'class_name',
                                                                     'id': 'class_id'})
                                             .drop(columns=['annotation', 'name']))
    name_map = {row['name']: idx for idx, row in meta[['name']].drop_duplicates()
                                                               .reset_index(drop=True)
                                                               .iterrows()}
    terminal_map = {row['label_name']: idx for idx, row in terminal_classes.iterrows()}
    meta['name_id'] = meta['name'].apply(lambda n: name_map[n])
    meta['terminal_id'] = meta['label_name'].apply(lambda n: terminal_map[n])

    images = read_images(bird_dir=bird_dir)
    boxes = read_boxes(bird_dir=bird_dir)
    sizes = read_sizes(bird_dir=bird_dir)
    train_test = read_train_test(bird_dir=bird_dir)
    train_test_meta = images.merge(meta).merge(boxes).merge(sizes).merge(train_test) \
        .sample(frac=1).reset_index(drop=True)
    train_meta = train_test_meta[train_test_meta['is_train'] == 1].drop(columns='is_train').reset_index(drop=True)
    test_meta = train_test_meta[train_test_meta['is_train'] == 0].drop(columns='is_train').reset_index(drop=True)
    return train_meta, test_meta
