
import os.path as osp
import logging
import python_logger.log_helper_off as py_log
import python_logger.log_helper as py_log_always_on


MY_LOGGER = logging.getLogger("prototip") # or any string. Mind this: same string, same logger.
MY_LOGGER.setLevel(logging.DEBUG)


import torch
from torch import nn





# In PyTorch, the automatic differentiation system, autograd, works by tracking operations on tensors to build a computational graph. 
# Each operation on a tensor creates a new node in this graph, and each node knows how to compute the gradient of the operation it represents. 
# This is facilitated by the grad_fn attribute, which points to a Function object that knows how to perform the backward pass for that operation.
# When you define a custom loss function in PyTorch, you typically use PyTorch tensor operations. 
# As long as you stick to these operations, PyTorch will automatically handle the differentiation for you. 
# This is because each PyTorch operation is designed to be differentiable and is part of the computational graph.


# Dice Loss is a smooth variation of IoU


class MultiClassDiceLoss(nn.Module):
    def __init__(self, smooth=1, background_adjustment=None):
        super(MultiClassDiceLoss, self).__init__()
        self.smooth = smooth
        self.background_adjustment = background_adjustment

    def forward(self, inputs, targets):

        try:

            if len(inputs.shape) == 3:
                inputs = inputs.unsqueeze(0)


            # Apply softmax to inputs if they are logits
            inputs = torch.softmax(inputs, dim=1) # Apply softmax across the class dimension


            # Initialize Dice Loss
            dice_loss = 0.0



            if self.background_adjustment is not None:
                is_background = (targets == 0).bool() # same size tensor of bools
                is_background_flat = is_background.reshape(-1)
            

            # Iterate over each class
            for c in range(inputs.shape[1]):
                
                # since our imgs are imbalanced (mostly background), i don't want the background to affect the loss too much
                # So in the return we also divide with one less.
                if c == 0:
                    continue

                input_to_be_flat = inputs[:,c,:,:].squeeze(1)
                input_flat = input_to_be_flat.reshape(-1)
                
                target_to_be_flat = (targets == c).float() # this will make the same sized tensor, just 1s where the target is c and 0s elsewhere
                target_flat = target_to_be_flat.reshape(-1)



                # Adjust background misprediction weights
                # Where the target is 0, we don't want the loss to be as severe as when the target is 1.
                # This is because the background is much more common than the other classes, so it is too safe of a bet to just predict background.
                # So the model doesn't want to move towards predicting classes, because it is too safe to just predict background.

                # We want to keep our loss between 0 and 1, so we can interpret it better and it is nicely graphable.
                # The reason dice loss is between 0 and 1 is because at each pixel, input_flat is between 0 and 1, and target_flat is 0 or 1.
                # This per pixel quality should stay the same.

                # We should simply pretend that our input_to_be_flat was closer to 0 than it actually was, when the target is 0.
                # This way, we don't discourage predictions of some actual class in the background spots.
                # We don't punish it as much.  
                
                if self.background_adjustment is not None:
                    background_distanes = input_flat.clone()
                    # The distances to zero are simply the values.
                    background_distanes[is_background_flat] = 0
                    # We then decide what percentage of these distances we would like to help the model with.
                    background_distanes = background_distanes * self.background_adjustment
                    # Then we actually go and adjust the input_flat, so the model will be penalised less for those.
                    input_flat = input_flat - background_distanes


                # Compute intersection
                intersection = (input_flat * target_flat)
                intersection = intersection.sum()
                
                # Compute Dice Coefficient for this class
                dice = (2. * intersection + self.smooth) / (input_flat.sum() + target_flat.sum() + self.smooth)
                
                # Accumulate Dice Loss
                dice_loss += 1 - dice
            
            # Average over all classes
            dice_loss =  dice_loss / (inputs.shape[1] - 1) # -1 because we skip the background class

            return dice_loss





        except Exception as e:
            py_log_always_on.log_stack(MY_LOGGER)
            raise e
        




class WeightedLosses(nn.Module):
    def __init__(self, losses_list, weights_list=[]):
        # super(MultiClassDiceLoss, self).__init__()
        self.losses_list = losses_list

        # if no weights are given, we will use equal weights
        if len(weights_list) == 0:
            self.weights_list = [(1/len(weights_list)) for _ in range(len(losses_list))]
        else:
            if len(weights_list) != len(losses_list):
                raise ValueError("The number of losses and weights must be the same.")
            self.weights_list = weights_list


    def forward(self, inputs, targets):

        try:
            
            computed_losses = [loss(inputs, targets) for loss in self.losses_list]
            weighted_losses = [computed_losses[i] * self.weights_list[i] for i in range(len(computed_losses))]
            total_loss = sum(weighted_losses)
            return total_loss

        except Exception as e:
            py_log_always_on.log_stack(MY_LOGGER)
            raise e



# we would like the loss to be weighted:
# Because there are many more 0s than 1s there is an imbalance and the model will be biased towards predicting 0s.
# We would like to give more importance to the 1s.
# So basically, we could just make a new criterion that is more focused on the 1s, and weight it with dice_loss.
# The aux loss should be between 0 and 1, so we can keep it in thiss range for graphing and interpretation and comperability across models.
