import yaml
from easydict import EasyDict as edict


def update_config(config_file):
    print(config_file)
    with open(config_file) as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))
        print(config)
        return config

