



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
import pickle

import y_helpers.json_handler as jh
import y_helpers.shared as shared

from y_framework.model_wrapper import ModelWrapper










def log_flops_and_weights(model_wrapper: ModelWrapper, main_save_path, unique_id):

    dict_with_all = {}
    dict_with_all["MODEL"] = shared.GLOBAL_DICT["MODEL"]

    # # This should only be used when manually making a new model instance to get the initial resources of a model with these architectural parameters.
    # # When I was doing this and changing the architectural params in the yaml, 
    # # the flops and weights weren't changing, because an initial_conv_resource_calc.pkl was already present.
    # # So this is a workaround one can use in such a scenario.
    # # Making sure the old resource calc is deleted (possibly of the old MODEL) 
    # # and a new one is created.
    # os.remove(osp.join(model_wrapper.save_path, "initial_conv_resource_calc.pkl"))
    # model_wrapper.init()

    import math

    def format_base2_scientific(v, precision=3):
        if v == 0:
            return f'0.{"0"*precision} x 2^0'
        exponent = math.floor(math.log2(abs(v)))
        mantissa = v / (2 ** exponent)
        return f'{mantissa:.{precision}f} x 2^{exponent}'


    # current resources:
    if model_wrapper.resource_calc is not None:
        model_wrapper.resource_calc.calculate_resources(model_wrapper.input_example)
        print("Flops and weights:")
        res_dict = model_wrapper.resource_calc.get_all_resources_of_whole_model()
        for k, v in res_dict.items():
            sci_format = f'{v:.3e}'
            base_2_format = format_base2_scientific(v)
            dict_with_all[k] = (base_2_format, sci_format, v)
            print(f"{k}: {dict_with_all[k]}")

    # initial resources:
    print("Initial flops and weights:")
    if model_wrapper.initial_resource_calc is not None:
        res_dict = model_wrapper.initial_resource_calc.get_all_resources_of_whole_model()
        for k, v in res_dict.items():
            sci_format = f'{v:.3e}'
            base_2_format = format_base2_scientific(v)
            k = f"initial_{k}"
            dict_with_all[k] = (base_2_format, sci_format, v)
            print(f"{k}: {dict_with_all[k]}")

    input_example_dims = str(model_wrapper.input_example.shape)
    dict_with_all["input_example_dims"] = input_example_dims
    
    j_path_no_suffix = osp.join(main_save_path, "flops_and_weights", f"{unique_id}_flops_and_weights")
    jh.dump_no_overwrite(j_path_no_suffix, dict_with_all)








@py_log.autolog(passed_logger=MY_LOGGER)
def is_previous_model(model_filename, model_wrapper):

    j_path = osp.join(model_wrapper.save_path, "previous_model_details.json")
    j_dict = jh.load(j_path)
    prev_filename = j_dict["previous_model_filename"]
    
    returner = model_filename == prev_filename

    return returner






class TrainingLogs:

    pickle_filename = "training_logs"

    @py_log.autolog(passed_logger=MY_LOGGER)
    def __init__(self, tl_main_save_path, number_of_epochs_per_training, cleaning_err_key, last_log=None, deleted_models_logs=None) -> None:
        
        if deleted_models_logs is None:
            deleted_models_logs = []

        self.tl_main_save_path = tl_main_save_path
        self.number_of_epochs_per_training = number_of_epochs_per_training
        self.last_log = last_log
        self.deleted_models_logs = deleted_models_logs

        self.cleaning_err_key = cleaning_err_key

        self.last_train_iter = None
        self.last_unique_id = None

        # of the form (val_error, test_error, train_iter, model_path)
        self.logs = []

    @py_log.autolog(passed_logger=MY_LOGGER)
    def add_log(self, log):
        if log is None:
            return
        
        # e.g. log = {"val_err": v, "test_err": t, "train_iter": ti, "model_filename": new_model_filename, "unique_id": unique_id, "is_not_automatic": True}
        self.last_log = log
        self.logs.append(log)
    
    @py_log.autolog(passed_logger=MY_LOGGER)
    def delete_log(self, log):
        self.logs.remove(log)
        self.deleted_models_logs.append(log)

    """
    The picking and loading is done in such a way, because if we change where we are running the proram (like go onto another computer)
    we want to just pass the main_save_path to TrainingLogs.load_or_create_training_logs() and it will load the training logs with the correct path,
    because the tl_main_save_path is has the correct path to main_save_path.
    But we also want to pickle to the new path, so we have to change the tl_main_save_path to the new path.
    """

    @staticmethod
    @py_log.autolog(passed_logger=MY_LOGGER)
    def load_or_create_training_logs(tl_main_save_path, number_of_epochs_per_training, cleaning_err_key, last_log=None, deleted_models_logs=None):

        if deleted_models_logs is None:
            deleted_models_logs = []

        os.makedirs(tl_main_save_path, exist_ok=True)
        j_path = osp.join(tl_main_save_path, "prev_training_logs_name.json")

        if osp.exists(j_path):

            j_dict = jh.load(j_path)
            tl_name = j_dict[f"prev_{TrainingLogs.pickle_filename}"]
            tl_path = osp.join(tl_main_save_path, "training_logs", tl_name)

            new_tl = pickle.load(open(tl_path, "rb"))
            new_tl.tl_main_save_path = tl_main_save_path
            return new_tl
            

        return TrainingLogs(tl_main_save_path, number_of_epochs_per_training, cleaning_err_key, last_log, deleted_models_logs)

    @py_log.autolog(passed_logger=MY_LOGGER)
    def pickle_training_logs(self, train_iter, unique_id):

        self.last_train_iter = train_iter
        self.last_unique_id = unique_id
        str_id = f"{train_iter}_{unique_id}"
        curr_name = f"{self.pickle_filename}_{str_id}.pkl"

        os.makedirs(osp.join(self.tl_main_save_path, "training_logs"), exist_ok=True)
        new_training_logs_path = osp.join(self.tl_main_save_path, "training_logs", curr_name)
        with open(new_training_logs_path, "wb") as f:
            pickle.dump(self, f)

        j_path = osp.join(self.tl_main_save_path, "prev_training_logs_name.json")
        new_j_dict = {f"prev_{self.pickle_filename}": curr_name}
        jh.dump(j_path, new_j_dict)

        os.makedirs(osp.join(self.tl_main_save_path, "old_tl_jsons"), exist_ok=True)
        j_path_for_copy = osp.join(self.tl_main_save_path, "old_tl_jsons", f"prev_training_logs_name_{str_id}.json")
        jh.dump(j_path_for_copy, new_j_dict)


    
    def __str__(self):
        returner = ""
        returner += f"Number of epochs per training: {self.number_of_epochs_per_training}\n"
        returner += f"Last train iteration: {self.last_log}\n"
        returner += f"logs: {self.logs}\n"
        return returner
    
    def __repr__(self):
        return self.__str__()




    # with the exception of keeping (k+1) models when one of the worse models is the last model we have 
    # (we have to keep it to continue training)
    @py_log.autolog(passed_logger=MY_LOGGER)
    def delete_all_but_best_k_models(self, k: int, model_wrapper: ModelWrapper):

        # I SUGGEST YOU ONLY DO THIS RIGHT BEFORE .perform_save()
        # THIS WAY YOU CAN BE SURE THAT THE LATEST MODEL WILL NOT BE DELETED.
        # I don't know how or why but there's some bug. I can't find it. I don't know what's going on.



        # Here, as long as we don't delete the last model, we are good.
        # We have to keep the last model, because we are going to continue training it.
        # Everything else is just for fun. Don't worry about it.

        
        # We want to have a comprehensive safety copy of models, so we can recreate them at any point in time of training.
        # But we don't want to keep all the models, because that would be a waste of space.


        # This is why, we will put create_safety_copy_of_existing_models() at all points of interest.
        # And after that, we will delete all but the best k models - because why keep more than that at that point - we have saved everything we have.
        
        # Before every perform_save() we will delete all but the best k models, so that they don't stay in the training logs and we have cleaned things up a bit.


        
        



        # We should do this cleaning before we pickle the training logs, which happens in the perform_save function.
        
        # The conceptual lifetime of training logs is created/loaded -> added to -> model_deletion -> saved
        # And then the process can repeat. Deletion can't be after saved, it makes no sense. Just think of doing just one iteration of it.
        
        
        
    

        # sort by validation log
        sorted_logs = sorted(self.logs, key = lambda x: x["val_err"][self.cleaning_err_key])

        to_delete = []

        while len(sorted_logs) > 0 and (len(self.logs) - len(to_delete)) > k:

            log = sorted_logs.pop() # pops last element
            
            model_filename = log["model_filename"]
            if is_previous_model(model_filename, model_wrapper):
                continue

            to_delete.append(log)


        for log in to_delete:
            model_filename = log["model_filename"]
            model_path = osp.join(model_wrapper.save_path, "models", model_filename)
            self.delete_log(log)
            try:
                os.remove(model_path)
                print(f"Deleting model {model_path}")
            except:
                print(f"Couldn't delete {model_path}. Probably doesn't exist.")




    

 

class PruningLogs:

    pickle_filename = "pruning_logs"

    @py_log.autolog(passed_logger=MY_LOGGER)
    def __init__(self, pl_main_save_path, pruning_logs=None) -> None:

        if pruning_logs is None:
            pruning_logs = []

        self.pl_main_save_path = pl_main_save_path
        self.pruning_logs = pruning_logs
        self.last_train_iter = None
        self.last_unique_id = None
    
    @staticmethod
    @py_log.autolog(passed_logger=MY_LOGGER)
    def load_or_create_pruning_logs(pl_main_save_path, pruning_logs=None):

        if pruning_logs is None:
            pruning_logs = []
        
        os.makedirs(pl_main_save_path, exist_ok=True)
        j_path = osp.join(pl_main_save_path, "prev_pruning_logs_name.json")

        if osp.exists(j_path):

            j_dict = jh.load(j_path)
            pl_name = j_dict[f"prev_{PruningLogs.pickle_filename}"]
            pl_path = osp.join(pl_main_save_path, "pruning_logs", pl_name)

            loaded_pl = pickle.load(open(pl_path, "rb"))
            loaded_pl.pl_main_save_path = pl_main_save_path

            return loaded_pl

        return PruningLogs(pl_main_save_path, pruning_logs)
    
    @py_log.autolog(passed_logger=MY_LOGGER)
    def pickle_pruning_logs(self, train_iter, unique_id):

        self.last_train_iter = train_iter
        self.last_unique_id = unique_id
        str_id = f"{train_iter}_{unique_id}"
        curr_name = f"{self.pickle_filename}_{str_id}.pkl" 

        os.makedirs(osp.join(self.pl_main_save_path, "pruning_logs"), exist_ok=True)
        new_pruning_logs_path = osp.join(self.pl_main_save_path, "pruning_logs", curr_name)
        with open(new_pruning_logs_path, "wb") as f:
            pickle.dump(self, f)
        

        new_j_dict = {f"prev_{self.pickle_filename}": curr_name}
        j_path = osp.join(self.pl_main_save_path, "prev_pruning_logs_name.json")
        jh.dump(j_path, new_j_dict)
        
        os.makedirs(osp.join(self.pl_main_save_path, "old_pl_jsons"), exist_ok=True)
        j_path_for_copy = osp.join(self.pl_main_save_path, "old_pl_jsons", f"previous_pruning_logs_name_{str_id}.json")
        jh.dump(j_path_for_copy, new_j_dict)


    # What we pruned is already saved in pruner_istance.pruning_logs
    # We just need to keep track of the corresponding train_iter to be able to know when which pruning happened.
    # That's what this function is for.
    @py_log.autolog(passed_logger=MY_LOGGER)
    def log_pruning_train_iter(self, train_iter, pickleable_conv_resource_calc):
        # Append the train_iter to the list of train_iters that correspond to prunings.
        # Second value is a flag that tells us if the model was actually saved. It is False to begin with. When we save it, we set it to True.

        self.pruning_logs.append((train_iter, False, pickleable_conv_resource_calc))



    # When we prune, we save the training iter of that pruning.
    # But what if we stop the training before that pruned model is actually saved?
    # This function sets the flag for the train_iter actually being confirmed.
    @py_log.autolog(passed_logger=MY_LOGGER)
    def confirm_last_pruning_train_iter(self):
        
                
        if len(self.pruning_logs) == 0:
            return
        
        self.pruning_logs[-1] = (self.pruning_logs[-1][0], True , self.pruning_logs[-1][2])

        if self.last_train_iter is not None and self.last_unique_id is not None:
            self.pickle_pruning_logs(self.last_train_iter, self.last_unique_id)



    # If we stop the training before the pruned model is saved, 
    # the last train iter would have turned to true in the next saving iteration,
    # despite the fact it was never saved and has no effect.
    # That's why we have to clean it up before training.
    @py_log.autolog(passed_logger=MY_LOGGER)
    def clean_up_pruning_train_iters(self):

        
        if len(self.pruning_logs) == 0:
            return
        
        if self.pruning_logs[-1][1] is False:
            self.pruning_logs = self.pruning_logs[:-1]

        if self.last_train_iter is not None and self.last_unique_id is not None:
            self.pickle_pruning_logs(self.last_train_iter, self.last_unique_id)


    def __str__(self):
        returner = ""
        if len(self.pruning_logs) == 0:
            returner += "No prunings have been done yet.\n"
        else:
            returner += f"pruning_logs[-1]: {self.pruning_logs[-1]}\n"
        return returner

    def __repr__(self):
        return self.__str__()