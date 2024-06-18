

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F




# create object for calculating model's flops
class ConvResourceCalc():
    def __init__(self, wrapper_model, target_modules=None):
        self.wrapper_model = wrapper_model

        if target_modules is None:
            self.target_modules = (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.Dropout, nn.Upsample, nn.ConvTranspose2d)
        else:
            self.target_modules = target_modules



    def _get_len_of_generator(self, gen):
        return sum(1 for x in gen)

    def _is_leaf(self, model):
        return self._get_len_of_generator(model.children()) == 0



    def calculate_layer(self, layer, x, module):


        y = layer.old_forward(x)
        
        if isinstance(layer, nn.Conv2d):
            n_removed_filters = 0
            
            if hasattr(layer, 'number_of_removed_filters'):
                n_removed_filters = layer.number_of_removed_filters
                #print('layer has {0} removed filters'.format(n_removed_filters))

            
            # assert n_removed_filters == 0
            
            h = y.shape[2]
            w = y.shape[3]
            cur_flops = h * w * layer.weight.size(0) * (layer.weight.size(1) - n_removed_filters) * layer.weight.size(2) * layer.weight.size(3)

            self.module_resources_dict[module] = cur_flops
            self.cur_flops += h * w * layer.weight.size(0) * (layer.weight.size(1) - n_removed_filters) * layer.weight.size(2) * layer.weight.size(3)
            #self.original_flops += h * w * layer.weight.size(0) * layer.weight.size(1) * layer.weight.size(2) * layer.weight.size(3)
            #self.n_removed_filters += n_removed_filters

        elif isinstance(layer, nn.BatchNorm2d):
            # ne upostevas, ker ne uporabis pri inferenci ene slike, na podlagi katere delas flop count.
            # rezanje tega je samo posledica rezanja konvolucije, nikoli ne mores samo tega rezat, zato ga ne sestevam..
            pass

        



        return y



    def calculate_resources(self, input_example):
        # tale ubistvu spremeni forward tako, da poklice trace_layer na vsakem. V trace nardis dejansko forward, poleg tega pa se
        # izracunas stevilo flopov.
        #self.original_flops = 0
        self.cur_flops = 0
        #self.n_removed_filters = 0
        self.module_resources_dict = {}

        def modify_forward(model):
            model_name = type(model).__name__.lower()
                        
            for module in model.modules():
                if isinstance(module, self.target_modules):

                    def new_forward(layer):
                        def lambda_forward(x):
                            return self.calculate_layer(layer, x, module)

                        return lambda_forward

                    module.old_forward = module.forward
                    module.forward = new_forward(module)



        def restore_forward(model):
            
            model_name = type(model).__name__.lower()
            
            for module in model.modules():
                print(module)
                if isinstance(module, self.target_modules) and hasattr(module, 'old_forward'):
                    module.forward = module.old_forward
                    module.old_forward = None

            print(10*"\n" + "Children:")
            for child in model.children():
                print(child)

        modify_forward(self.wrapper_model.model)
        input_example = input_example.to(self.wrapper_model.device)
        y = self.wrapper_model.model.forward(input_example)
        restore_forward(self.wrapper_model.model)