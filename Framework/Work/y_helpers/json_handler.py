




import os
import os.path as osp

import orjson



def load(path):
    json_dict = None
    if osp.exists(path):
        with open(path, 'rb') as f:
            json_str = f.read()
            json_dict = orjson.loads(json_str)
    
    return json_dict

def dump(path, json_dict):
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        json_str = orjson.dumps(json_dict)
        f.write(json_str)



def dump_no_overwrite(path_no_suffix, json_dict, suffix=".json"):

    j_path = path_no_suffix + suffix
    
    if osp.exists(j_path):
        add_id = 1
        old_j_path = j_path
        j_path = path_no_suffix + f"_{add_id}{suffix}"
        while osp.exists(j_path):
            add_id += 1
            j_path = path_no_suffix + f"_{add_id}{suffix}"
        
        print(f"JSON file {old_j_path} already exists. We made {j_path} instead.")
    dump(j_path, json_dict)
            