
import pickle
import ConvResourceCalc


class pruner:



    def __init__(self, min_resource_percentage_dict, initial_conv_resource_calc: ConvResourceCalc, connection_lambda, filter_importance_lambda, conv_tree_ixs):
        self.initial_conv_resource_calc = initial_conv_resource_calc
        self.min_resource_percentage_dict = min_resource_percentage_dict
        self.connection_lambda = connection_lambda
        self.filter_importance_lambda = filter_importance_lambda
        self.conv_tree_ixs = conv_tree_ixs


        tree_ix_2_active_filters = {}
        for tree_ix in conv_tree_ixs:
            filter_num = initial_conv_resource_calc.module_tree_ix_2_weights_dimensions[tree_ix][1] # 1 is the number of filters of this layer
            tree_ix_2_active_filters[tree_ix] = list(range(filter_num)) 



        
    
    def binary_search(arr, target):
        left, right = 0, len(arr) - 1
        
        while left <= right:
            mid = (left + right) // 2
            
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return -1  # Target not found
    
    def prune(self, activations):

        # First we evaluate all the filters
        importance_dict = self.filter_importance_lambda(activations, self.conv_tree_ixs)

        # Then we sort them

        # Then we pick the first one which isn't below the FLOPs threshold

        pass
