
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












from pydantic.dataclasses import dataclass
from pydantic import field_validator, Field
from pydantic import ConfigDict


from typing import Any, Union

import torch


config_dict = ConfigDict(
    arbitrary_types_allowed=True, # So we can work with general types, not just python natives
    validate_assignment = True # Validates values when attributes are set after model creation
)



@dataclass(config=config_dict)
class ModelWrapperParams:
    model_class: type
    input_example: torch.Tensor
    save_path: str
    device: str
    learning_rate: float
    optimizer_class: type
    is_resource_calc_ready: bool

@dataclass(config=config_dict)
class TrainingWrapperParams:
    device: str
    target: str
    loss_fn: torch.nn.Module
    zero_out_non_sclera_on_predictions: bool
    have_patchification: bool
    patchification_params: Union[dict, None]
    metrics_aggregation_fn: str
    num_classes: int
    





if __name__ == "__main__":
    training_params = TrainingWrapperParams(
        device="cpu", 
        target="target", 
        loss_fn=torch.nn.CrossEntropyLoss(), 
        zero_out_non_sclera_on_predictions=True, 
        have_patchification=False, 
        patchification_params={})
    
    print(training_params)
    training_params.device = "cuda"
    print(training_params)
    training_params.device = 123  # This will raise a TypeError
    print(training_params)
    training_params.loss_fn = "CrossEntropyLoss"  # This will raise a TypeError
    print(training_params)





