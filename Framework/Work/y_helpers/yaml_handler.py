
import yaml


def read_yaml(file_path):
    with open(file_path, 'r') as file:
        curr_dict = yaml.safe_load(file)
    
    return curr_dict


def write_yaml(inp_dict, file_path, sort_keys=False, indent=4):
    with open(file_path, 'w') as file:
        yaml.dump(inp_dict, file, sort_keys=sort_keys, indent=indent)


def get_readable_dict_str(inp_dict, perline=5):

    final_str = ""
    cycling_ix = 0
        
    for key, val in inp_dict.items():
        final_str += f"{key}: {val}      "
        cycling_ix += 1
        if cycling_ix >= perline:
            final_str += "\n"
            cycling_ix = 0
    
    return final_str
