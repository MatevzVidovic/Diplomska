


from model_sorting import sort_tree_ixs

import logging
import python_logger.log_helper as py_log

MY_LOGGER = logging.getLogger("prototip")

class min_resource_percentage:

    @py_log.log(passed_logger=MY_LOGGER)
    def __init__(self, tree_ix_2_name_dict):
        self.tree_ix_2_name_dict = tree_ix_2_name_dict

        self.min_resource_percentage_dict = {}
        for tree_ix, _ in self.tree_ix_2_name_dict.items():
            self.min_resource_percentage_dict[tree_ix] = 0.0
    
    @py_log.log(passed_logger=MY_LOGGER)
    def set_by_name(self, name, percentage):
        
        applicable_tree_ixs = []
        for tree_ix, module_name in self.tree_ix_2_name_dict.items():
            if module_name == name:
                applicable_tree_ixs.append(tree_ix)
        
        assert len(applicable_tree_ixs) > 0, f"No module with name {name} found."

        for tree_ix in applicable_tree_ixs:
            self.min_resource_percentage_dict[tree_ix] = percentage
        
    
    @py_log.log(passed_logger=MY_LOGGER)
    def set_by_tree_ix_list(self, tree_ix_list, percentage):
        for tree_ix in tree_ix_list:
            assert tree_ix in self.tree_ix_2_name_dict, f"Tree ix {tree_ix} not found in tree_ix_2_name_dict."
            self.min_resource_percentage_dict[tree_ix] = percentage
    
    @py_log.log(passed_logger=MY_LOGGER)
    def set_by_tree_ix_dict(self, tree_ix_2_percentage_dict):
        for tree_ix, percentage in tree_ix_2_percentage_dict.items():
            assert tree_ix in self.tree_ix_2_name_dict, f"Tree ix {tree_ix} not found in tree_ix_2_name_dict."
            self.min_resource_percentage_dict[tree_ix] = percentage

    
    def get_ordered_list_of_tree_ixs_for_layer_name(self, layer_name):

        applicable_tree_ixs = []
        for tree_ix, module_name in self.tree_ix_2_name_dict.items():
            if module_name == layer_name:
                applicable_tree_ixs.append(tree_ix)

        assert len(applicable_tree_ixs) > 0, f"No module with name {layer_name} found."

        sorted_applicable_tree_ixs = sort_tree_ixs(applicable_tree_ixs)


        return sorted_applicable_tree_ixs
            


