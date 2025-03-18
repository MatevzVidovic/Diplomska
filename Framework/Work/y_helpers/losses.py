
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






import torch
from torch import nn




# Look at https://arxiv.org/abs/2312.05391 to understand the losses.
# Focal Tversky loss is the goat!!!


# In PyTorch, the automatic differentiation system, autograd, works by tracking operations on tensors to build a computational graph. 
# Each operation on a tensor creates a new node in this graph, and each node knows how to compute the gradient of the operation it represents. 
# This is facilitated by the grad_fn attribute, which points to a Function object that knows how to perform the backward pass for that operation.
# When you define a custom loss function in PyTorch, you typically use PyTorch tensor operations. 
# As long as you stick to these operations, PyTorch will automatically handle the differentiation for you. 
# This is because each PyTorch operation is designed to be differentiable and is part of the computational graph.






class FocalTverskyLoss(nn.Module):
    def __init__(self, fp_imp=0.5, fn_imp=0.5, gamma=(4/3), smooth=1e-6, use_background=False, equalize=False):
        # fn_imp == fp_imp == 0.5 is the same as Dice Loss.
        # fn_imp == fp_imp == 1 is the same as Jaccard (IoU) Loss.
        # Generally, (fn_imp + fp_imp) == 1. Im not sure if scaling both by the same value even matters. 
        # Yes, the dependance of the loss on the number of correct examples is different, 
        # but idk if theres real impact on the derrivative, aside from needing a different learning rate.
        # Wouldn't really know.

        # https://arxiv.org/pdf/1810.07842
        # FTL = \sum_c (1 - TI_C)^gamma, TI_c being Tversky Index for class c
        # Gamma is suggested to be in the range of [1,3] in the paper. They used 4/3.
        # The loss had a problem with: over-suppression of the FTL when the class accuracy
        # is high, usually as the model is close to convergence
        # So maybe even just reducing the gamma in the end of training could be a good idea.



        super(FocalTverskyLoss, self).__init__()
        self.fp_imp = fp_imp
        self.fn_imp = fn_imp
        self.gamma = gamma
        self.smooth = smooth
        self.use_background = use_background
        self.equalize = equalize

    def forward(self, preds, targets):

        try:

            if len(preds.shape) == 3:
                preds = preds.unsqueeze(0)


            # Apply softmax to preds if they are logits
            preds = torch.softmax(preds, dim=1) # Apply softmax across the class dimension


            # Initialize Dice Loss
            tversky_loss = 0.0



            # Iterate over each class
            for c in range(preds.shape[1]):
                
                # since our imgs are imbalanced (mostly background), i don't want the background to affect the loss too much
                # So in the return we also divide with one less.
                if not self.use_background and c == 0:
                    continue

                pred_to_be_flat = preds[:,c,:,:].squeeze(1)
                pred_flat = pred_to_be_flat.reshape(-1)
                
                target_to_be_flat = (targets == c).float() # this will make the same sized tensor, just 1s where the target is c and 0s elsewhere
                target_flat = target_to_be_flat.reshape(-1)

                inv_target_flat = 1 - target_flat
                pos_num = target_flat.sum()
                neg_num = inv_target_flat.sum()



                # Calculate true positives, false positives, and false negatives

                if not self.equalize:
                    true_pos = (pred_flat * target_flat).sum()
                    false_neg = ((1 - pred_flat) * target_flat).sum()
                    # false_pos = (pred_flat * (1 - target_flat)).sum()
                    false_pos = (pred_flat * inv_target_flat).sum()


                else:
                    # I think this might completely mess up the loss, because then it becomes completely irrelevant how many of which case there is.
                    # but I could be wrong.

                    true_pos = (pred_flat * target_flat).sum() / pos_num
                    false_neg = ((1 - pred_flat) * target_flat).sum() / pos_num
                    # false_pos = (pred_flat * (1 - target_flat)).sum()
                    false_pos = (pred_flat * inv_target_flat).sum() / neg_num

                    # The division should put them on equal grounds - otherwise a huge imbalance will make it hard to set alpha and beta in a way 
                    # that we can be sure to make either recall or precision higher.



                # Calculate Tversky index
                tversky_index = (true_pos + self.smooth) / (true_pos + self.fp_imp * false_pos + self.fn_imp * false_neg + self.smooth)


                # Accumulate Tversky loss
                tversky_loss += (1 - tversky_index)**(1/self.gamma)




            # Average over all classes, just to make it easier to interpret.
            num_of_classes = preds.shape[1] - 1 if not self.use_background else preds.shape[1]
            tversky_loss =  1 - tversky_loss / num_of_classes
            
            return tversky_loss


        except Exception as e:
            py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
            raise e
        







class TverskyLoss(nn.Module):
    def __init__(self, fp_imp=0.5, fn_imp=0.5, equalize=False, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.fp_imp = fp_imp
        self.fn_imp = fn_imp
        self.equalize = equalize
        self.smooth = smooth

    def forward(self, preds, targets):

        try:

            if len(preds.shape) == 3:
                preds = preds.unsqueeze(0)


            # Apply softmax to preds if they are logits
            preds = torch.softmax(preds, dim=1) # Apply softmax across the class dimension


            # Initialize Dice Loss
            tversky_index_sum = 0.0



            # Iterate over each class
            for c in range(preds.shape[1]):
                
                # since our imgs are imbalanced (mostly background), i don't want the background to affect the loss too much
                # So in the return we also divide with one less.
                if c == 0:
                    continue

                pred_to_be_flat = preds[:,c,:,:].squeeze(1)
                pred_flat = pred_to_be_flat.reshape(-1)
                
                target_to_be_flat = (targets == c).float() # this will make the same sized tensor, just 1s where the target is c and 0s elsewhere
                target_flat = target_to_be_flat.reshape(-1)

                inv_target_flat = 1 - target_flat
                pos_num = target_flat.sum()
                neg_num = inv_target_flat.sum()



                # Calculate true positives, false positives, and false negatives

                if not self.equalize:
                    true_pos = (pred_flat * target_flat).sum()
                    false_neg = ((1 - pred_flat) * target_flat).sum()
                    # false_pos = (pred_flat * (1 - target_flat)).sum()
                    false_pos = (pred_flat * inv_target_flat).sum()


                else:
                    # I think this might completely mess up the loss, because then it becomes completely irrelevant how many of which case there is.
                    # but I could be wrong.

                    true_pos = (pred_flat * target_flat).sum() / pos_num
                    false_neg = ((1 - pred_flat) * target_flat).sum() / pos_num
                    # false_pos = (pred_flat * (1 - target_flat)).sum()
                    false_pos = (pred_flat * inv_target_flat).sum() / neg_num

                    # The division should put them on equal grounds - otherwise a huge imbalance will make it hard to set alpha and beta in a way 
                    # that we can be sure to make either recall or precision higher.



                # Calculate Tversky index
                tversky_index = (true_pos + self.smooth) / (true_pos + self.fp_imp * false_pos + self.fn_imp * false_neg + self.smooth)


                # Accumulate Tversky loss
                tversky_index_sum += tversky_index


            # Average over all classes
            tversky_loss =  1 - tversky_loss / (preds.shape[1] - 1) # -1 because we skip the background class
            
            return tversky_loss


        except Exception as e:
            py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
            raise e
        






class MultiClassDiceLoss(nn.Module):
    def __init__(self, smooth=1, background_adjustment=None):
        super(MultiClassDiceLoss, self).__init__()
        self.smooth = smooth
        self.background_adjustment = background_adjustment

    def forward(self, preds, targets):

        try:

            if len(preds.shape) == 3:
                preds = preds.unsqueeze(0)


            # Apply softmax to preds if they are logits
            preds = torch.softmax(preds, dim=1) # Apply softmax across the class dimension


            # Initialize Dice Loss
            dice_loss = 0.0



            if self.background_adjustment is not None:
                is_background = (targets == 0).bool() # same size tensor of bools
                is_background_flat = is_background.reshape(-1)
            

            # Iterate over each class
            for c in range(preds.shape[1]):
                
                # since our imgs are imbalanced (mostly background), i don't want the background to affect the loss too much
                # So in the return we also divide with one less.
                if c == 0:
                    continue

                pred_to_be_flat = preds[:,c,:,:].squeeze(1)
                pred_flat = pred_to_be_flat.reshape(-1)
                
                target_to_be_flat = (targets == c).float() # this will make the same sized tensor, just 1s where the target is c and 0s elsewhere
                target_flat = target_to_be_flat.reshape(-1)



                # Adjust background misprediction weights
                # Where the target is 0, we don't want the loss to be as severe as when the target is 1.
                # This is because the background is much more common than the other classes, so it is too safe of a bet to just predict background.
                # So the model doesn't want to move towards predicting classes, because it is too safe to just predict background.

                # We want to keep our loss between 0 and 1, so we can interpret it better and it is nicely graphable.
                # The reason dice loss is between 0 and 1 is because at each pixel, pred_flat is between 0 and 1, and target_flat is 0 or 1.
                # This per pixel quality should stay the same.

                # We should simply pretend that our pred_to_be_flat was closer to 0 than it actually was, when the target is 0.
                # This way, we don't discourage predictions of some actual class in the background spots.
                # We don't punish it as much.  
                
                if self.background_adjustment is not None:
                    background_distanes = pred_flat.clone()
                    # The distances to zero are simply the values.
                    background_distanes[is_background_flat] = 0
                    # We then decide what percentage of these distances we would like to help the model with.
                    background_distanes = background_distanes * self.background_adjustment
                    # Then we actually go and adjust the pred_flat, so the model will be penalised less for those.
                    pred_flat = pred_flat - background_distanes


                # Compute intersection
                intersection = (pred_flat * target_flat)
                intersection = intersection.sum()
                
                # Compute Dice Coefficient for this class
                dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
                
                # Accumulate Dice Loss
                dice_loss += 1 - dice
            
            # Average over all classes
            dice_loss =  dice_loss / (preds.shape[1] - 1) # -1 because we skip the background class

            return dice_loss





        except Exception as e:
            py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
            raise e
        




class JaccardLoss(nn.Module):
    def __init__(self, smooth=1, background_adjustment=None):
        super(JaccardLoss, self).__init__()
        self.smooth = smooth
        self.background_adjustment = background_adjustment

    def forward(self, inputs, targets):

        try:

            if len(inputs.shape) == 3:
                inputs = inputs.unsqueeze(0)


            # Apply softmax to inputs if they are logits
            inputs = torch.softmax(inputs, dim=1) # Apply softmax across the class dimension


            # Initialize jaccard Loss
            jaccard_index = 0.0



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

                union = input_flat.sum() + target_flat.sum() - intersection
                

                # Compute jaccard index
                jac = (intersection + self.smooth) / (union + self.smooth)
                
                # Accumulate Dice Loss
                jaccard_index += 1 - jac
            
            # Average over all classes
            jaccard_index = jaccard_index / (inputs.shape[1] - 1) # -1 because we skip the background class

            return jaccard_index





        except Exception as e:
            py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
            raise e










class WeightedLosses(nn.Module):

    def __init__(self, losses_list, weights_list=None):
        super(WeightedLosses, self).__init__()

        if weights_list is None:
            weights_list = []
        
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
            py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
            raise e



# we would like the loss to be weighted:
# Because there are many more 0s than 1s there is an imbalance and the model will be biased towards predicting 0s.
# We would like to give more importance to the 1s.
# So basically, we could just make a new criterion that is more focused on the 1s, and weight it with dice_loss.
# The aux loss should be between 0 and 1, so we can keep it in thiss range for graphing and interpretation and comperability across models.

