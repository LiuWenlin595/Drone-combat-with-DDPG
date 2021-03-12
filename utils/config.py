import os
import yaml
import copy
from collections import OrderedDict

def load_ordered_yaml(
        stream,
        Loader=yaml.Loader,
        object_pairs_hook=OrderedDict):

    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)

def load_config(config_path):
    with open(config_path, "r") as fin:
        config = load_ordered_yaml(fin)

    return config