
import pickle


class pruner:



    def __init__(self, wrapper_model, min_resource_percentage_dict, initial_conv_resource_calc, connection_lambda, filter_importance_lambda):
        self.wrapper_model = wrapper_model
        self.initial_conv_resource_calc = initial_conv_resource_calc
        self.min_resource_percentage_dict = min_resource_percentage_dict
        self.connection_lambda = connection_lambda
        self.filter_importance_lambda = filter_importance_lambda


        
    
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
    
    def prune(self):

        # First we evaluate all the filters

        # Then we sort them

        # Then we pick the first one which isn't below the FLOPs threshold

        pass
