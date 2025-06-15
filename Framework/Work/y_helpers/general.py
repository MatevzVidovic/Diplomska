

import logging
import yaml
import os.path as osp
import python_logger.log_helper as py_log_always_on

with open(f"{osp.join('pylog_configs', 'active_logging_config.txt')}", 'r') as f:
    cfg_name = f.read()
    yaml_path = osp.join('pylog_configs', cfg_name)

log_config_path = osp.join(yaml_path)
do_log = False
if osp.exists(yaml_path):
    with open(yaml_path, 'r') as stream:
        config = yaml.safe_load(stream)
        file_log_setting = config.get(osp.basename(__file__), False)
        if file_log_setting:
            do_log = True

print(f"{osp.basename(__file__)} do_log: {do_log}")
if do_log:
    import python_logger.log_helper as py_log
else:
    import python_logger.log_helper_off as py_log

MY_LOGGER = logging.getLogger("prototip") # or any string. Mind this: same string, same logger.
MY_LOGGER.setLevel(logging.DEBUG)




import os
from pathlib import Path


@py_log.autolog(passed_logger=MY_LOGGER)
def get_fresh_folder_basic(path_to_parent_folder):

    try:
        
        prefix = "" # will be k*"#" if overflow
        counter=999
        
        folder_path = Path(path_to_parent_folder) / f"{prefix}{counter}"
        while osp.exists(folder_path):
            counter -= 1
            folder_path = Path(path_to_parent_folder) / f"{prefix}{counter}"

            if counter <= 100:
                prefix += "#"
                counter = 999
    
        os.makedirs(folder_path, exist_ok=True)
        py_log.log_locals(MY_LOGGER)
        return Path(folder_path)
    
    except Exception as e:
        py_log.log_stack(MY_LOGGER)
        raise e