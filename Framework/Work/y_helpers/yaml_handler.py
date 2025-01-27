
import yaml


def read_yaml(file_path):
    with open(file_path, 'r') as file:
        dict = yaml.safe_load(file)
    
    return dict


def write_yaml(dict, file_path, sort_keys=False, indent=4):
    with open(file_path, 'w') as file:
        yaml.dump(dict, file, sort_keys=sort_keys, indent=indent)


def get_readable_dict_str(dict, perline=5):

    final_str = ""
    cycling_ix = 0
        
    for key, val in dict.items():
        final_str += f"{key}: {val}      "
        cycling_ix += 1
        if cycling_ix >= perline:
            final_str += "\n"
            cycling_ix = 0
    
    return final_str