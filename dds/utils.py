"""Common utility functions."""

from collections.abc import MutableMapping

# Taken from https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
def flatten_nested_dict(nested_dict, parent_key="", sep="."):
    items = []
    for name, cfg in nested_dict.items():
        new_key = parent_key + sep + name if parent_key else name
        if isinstance(cfg, MutableMapping):
            items.extend(flatten_nested_dict(cfg, new_key, sep=sep).items())
        else:
            items.append((new_key, cfg))

    return dict(items)