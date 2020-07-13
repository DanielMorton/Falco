import pandas as pd
from meta import HIERARCHY_FILE


def read_hierarchy(bird_dir):
    """Loads table of class hierarchies. Returns hierarchy table
    parent-child class map, top class levels, and bottom class levels.
    """
    hierarchy = pd.read_table(f'{bird_dir}/{HIERARCHY_FILE}', sep=' ',
                              header=None)
    hierarchy.columns = ['child', 'parent']

    child_graph = {0: []}
    name_level = {0: 0}
    for _, row in hierarchy.iterrows():
        child_graph[row['parent']].append(row['child'])
        child_graph[row['child']] = []
        name_level[row['child']] = name_level[row['parent']] + 1

    terminal_levels = set()
    for key, value in name_level.items():
        if not child_graph[key]:
            terminal_levels.add(key)

    parent_map = {row['child']: row['parent'] for _, row in hierarchy.iterrows()}
    return hierarchy, parent_map, set(child_graph[0]), terminal_levels
