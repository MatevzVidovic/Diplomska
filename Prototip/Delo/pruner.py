


class pruner:

    def __init__(self, wrapper_model, min_resource_percentage_dict, initial_resource_dict ,connection_lambda, filter_importance_lambda):
        self.wrapper_model = wrapper_model
        self.initial_resource_dict = initial_resource_dict
        self.min_resource_percentage_dict = min_resource_percentage_dict
        self.connection_lambda = connection_lambda
        self.filter_importance_lambda = filter_importance_lambda
    
    def prune(self):

        # First we evaluate all the filters

        # Then we sort them

        # Then we pick the first one which isn't below the FLOPs threshold

        pass
