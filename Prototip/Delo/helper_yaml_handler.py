
import yaml


def read_yaml(file_path):
    with open(file_path, 'r') as file:
        dict = yaml.safe_load(file)
    
    return dict


def write_yaml(dict, file_path, sort_keys=False, indent=4):
    with open(file_path, 'w') as file:
        yaml.dump(dict, file, sort_keys=sort_keys, indent=indent)