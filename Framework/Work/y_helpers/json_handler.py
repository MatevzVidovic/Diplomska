






import orjson

import os.path as osp


def load(path):
    json_dict = None
    if osp.exists(path):
        with open(path, 'rb') as f:
            json_str = f.read()
            json_dict = orjson.loads(json_str)
    
    return json_dict

def dump(path, json_dict):
    with open(path, 'wb') as f:
        json_str = orjson.dumps(json_dict)
        f.write(json_str)
