import numpy as np


def concatenating_map(semantic_data, map_data):
    rest = max([int(k) for k in map_data.keys()]) + 1
    max_key = max([int(k) for k in semantic_data.keys()])

    lookup_table = np.full((max_key + 1,), rest, dtype=np.int32)

    for key, value in semantic_data.items():
        mapped_value = next((int(map_key) for map_key, map_values in map_data.items() if value in map_values), rest)
        lookup_table[int(key)] = mapped_value

    return list(lookup_table)
