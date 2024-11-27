








import os
import os.path as osp
import logging
import python_logger.log_helper_off as py_log


MY_LOGGER = logging.getLogger("prototip") # or any string. Mind this: same string, same logger.
MY_LOGGER.setLevel(logging.DEBUG)




import matplotlib.pyplot as plt
import pickle

import json_handler as jh

from ModelWrapper import ModelWrapper

from model_eval_graphs import resource_graph, show_results















def is_previous_model(model_path, model_wrapper, get_last_model_path = False):

    j_path = osp.join(model_wrapper.save_path, "previous_model_details.json")
    j_dict = jh.load(j_path)
    prev_filename = j_dict["previous_model_filename"]
    prev_model_path = osp.join(model_wrapper.save_path, prev_filename)
    
    returner = model_path == prev_model_path

    if get_last_model_path:
        returner = (returner, prev_model_path)

    return returner






class TrainingLogs:

    pickle_filename = "training_logs"

    def __init__(self, tl_main_save_path, number_of_epochs_per_training, cleaning_err_ix, last_error=None, deleted_models_errors=[]) -> None:
        
        self.tl_main_save_path = tl_main_save_path
        self.number_of_epochs_per_training = number_of_epochs_per_training
        self.last_error = last_error
        self.deleted_models_errors = deleted_models_errors

        self.cleaning_err_ix = cleaning_err_ix

        self.last_train_iter = None
        self.last_unique_id = None

        # of the form (val_error, test_error, train_iter, model_path)
        self.errors = []

    def add_error(self, error):
        if error is None:
            return
        # (val_error, test_error, train_iter, model_path, unique_id, is_not_automatic)
        self.last_error = error
        self.errors.append(error)
    
    def delete_error(self, error):
        self.errors.remove(error)
        self.deleted_models_errors.append(error)

    """
    The picking and loading is done in such a way, because if we change where we are running the proram (like go onto another computer)
    we want to just pass the main_save_path to TrainingLogs.load_or_create_training_logs() and it will load the training logs with the correct path,
    because the tl_main_save_path is has the correct path to main_save_path.
    But we also want to pickle to the new path, so we have to change the tl_main_save_path to the new path.
    """

    @staticmethod
    def load_or_create_training_logs(tl_main_save_path, number_of_epochs_per_training, cleaning_err_ix, last_error=None, deleted_models_errors=[]):

        os.makedirs(tl_main_save_path, exist_ok=True)
        j_path = osp.join(tl_main_save_path, "prev_training_logs_name.json")

        if osp.exists(j_path):

            j_dict = jh.load(j_path)
            tl_name = j_dict[f"prev_{TrainingLogs.pickle_filename}"]
            tl_path = osp.join(tl_main_save_path, tl_name)

            new_tl = pickle.load(open(tl_path, "rb"))
            new_tl.tl_main_save_path = tl_main_save_path
            return new_tl
            

        return TrainingLogs(tl_main_save_path, number_of_epochs_per_training, cleaning_err_ix, last_error, deleted_models_errors)

    @py_log.log(passed_logger=MY_LOGGER)
    def pickle_training_logs(self, train_iter, unique_id):

        self.last_train_iter = train_iter
        self.last_unique_id = unique_id
        str_id = f"{train_iter}_{unique_id}"
        curr_name = f"{self.pickle_filename}_{str_id}.pkl"

        new_training_logs_path = osp.join(self.tl_main_save_path, curr_name)
        with open(new_training_logs_path, "wb") as f:
            pickle.dump(self, f)

        j_path = osp.join(self.tl_main_save_path, "prev_training_logs_name.json")
        new_j_dict = {f"prev_{self.pickle_filename}": curr_name}
        jh.dump(j_path, new_j_dict)

        os.makedirs(osp.join(self.tl_main_save_path, "copies"), exist_ok=True)
        j_path_for_copy = osp.join(self.tl_main_save_path, "copies", f"prev_training_logs_name_{str_id}.json")
        jh.dump(j_path_for_copy, new_j_dict)


    
    def __str__(self):
        returner = ""
        returner += f"Number of epochs per training: {self.number_of_epochs_per_training}\n"
        returner += f"Last train iteration: {self.last_error}\n"
        returner += f"Errors: {self.errors}\n"
        returner += f"Deleted models errors: {self.deleted_models_errors}\n"
        return returner
    
    def __repr__(self):
        # Generate a string representation of the object
        items = (f"{key}={value!r}" for key, value in self.__dict__.items())
        return f"<{self.__class__.__name__}({', '.join(items)})>"




    # with the exception of keeping (k+1) models when one of the worse models is the last model we have 
    # (we have to keep it to continue training)
    def delete_all_but_best_k_models(self, k: int, model_wrapper: ModelWrapper):

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
        
        
        
    

        # sort by validation error
        sorted_errors = sorted(self.errors, key = lambda x: x[0][self.cleaning_err_ix])

        to_delete = []

        while len(sorted_errors) > 0 and (len(self.errors) - len(to_delete)) > k:

            error = sorted_errors.pop() # pops last element
            
            model_path = error[3]
            if is_previous_model(model_path, model_wrapper):
                continue

            to_delete.append(error)


        for error in to_delete:
            model_filename = error[3]
            model_path = osp.join(model_wrapper.save_path, model_filename)
            self.delete_error(error)
            try:
                os.remove(model_path)
                print(f"Deleting model {model_path}")
            except:
                print(f"Couldn't delete {model_path}. Probably doesn't exist.")




    

 

class PruningLogs:

    pickle_filename = "pruning_logs"

    def __init__(self, pl_main_save_path, pruning_logs=[]) -> None:
        self.pl_main_save_path = pl_main_save_path
        self.pruning_logs = pruning_logs
        self.last_train_iter = None
        self.last_unique_id = None
    
    @staticmethod
    def load_or_create_pruning_logs(pl_main_save_path, pruning_logs=[]):
        
        os.makedirs(pl_main_save_path, exist_ok=True)
        j_path = osp.join(pl_main_save_path, "prev_pruning_logs_name.json")

        if osp.exists(j_path):

            j_dict = jh.load(j_path)
            pl_name = j_dict[f"prev_{PruningLogs.pickle_filename}"]
            pl_path = osp.join(pl_main_save_path, pl_name)

            loaded_pl = pickle.load(open(pl_path, "rb"))
            loaded_pl.pl_main_save_path = pl_main_save_path

            return loaded_pl

        return PruningLogs(pl_main_save_path, pruning_logs)
    
    @py_log.log(passed_logger=MY_LOGGER)
    def pickle_pruning_logs(self, train_iter, unique_id):

        self.last_train_iter = train_iter
        self.last_unique_id = unique_id
        str_id = f"{train_iter}_{unique_id}"
        curr_name = f"{self.pickle_filename}_{str_id}.pkl" 

        new_pruning_logs_path = osp.join(self.pl_main_save_path, curr_name)
        with open(new_pruning_logs_path, "wb") as f:
            pickle.dump(self, f)
        

        new_j_dict = {f"prev_{self.pickle_filename}": curr_name}
        j_path = osp.join(self.pl_main_save_path, "prev_pruning_logs_name.json")
        jh.dump(j_path, new_j_dict)
        
        os.makedirs(osp.join(self.pl_main_save_path, "copies"), exist_ok=True)
        j_path_for_copy = osp.join(self.pl_main_save_path, "copies", f"previous_pruning_logs_name_{str_id}.json")
        jh.dump(j_path_for_copy, new_j_dict)


    # What we pruned is already saved in pruner_istance.pruning_logs
    # We just need to keep track of the corresponding train_iter to be able to know when which pruning happened.
    # That's what this function is for.
    def log_pruning_train_iter(self, train_iter, pickleable_conv_resource_calc):
        # Append the train_iter to the list of train_iters that correspond to prunings.
        # Second value is a flag that tells us if the model was actually saved. It is False to begin with. When we save it, we set it to True.

        self.pruning_logs.append((train_iter, False, pickleable_conv_resource_calc))



    # When we prune, we save the training iter of that pruning.
    # But what if we stop the training before that pruned model is actually saved?
    # This function sets the flag for the train_iter actually being confirmed.
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
    def clean_up_pruning_train_iters(self):

        
        if len(self.pruning_logs) == 0:
            return
        
        if self.pruning_logs[-1][1] == False:
            self.pruning_logs = self.pruning_logs[:-1]

        if self.last_train_iter is not None and self.last_unique_id is not None:
            self.pickle_pruning_logs(self.last_train_iter, self.last_unique_id)


    def __str__(self):
        returner = ""
        returner += f"Pruning logs: {self.pruning_logs}\n"
        return returner

    def __repr__(self):
        # Generate a string representation of the object
        items = (f"{key}={value!r}" for key, value in self.__dict__.items())
        return f"<{self.__class__.__name__}({', '.join(items)})>"





def perform_save(model_wrapper: ModelWrapper, training_logs: TrainingLogs, pruning_logs: PruningLogs, train_iter, unique_id, val_error=None, test_error=None):

    new_model_filename, _ = model_wrapper.save(f"{train_iter}_{unique_id}")
    pruning_logs.confirm_last_pruning_train_iter()

    if val_error is None or test_error is None:
        new_error = None
        # this only happens if we do a manual save before any training even took place
        # or maybe if we prune before any training took place
        if training_logs.last_error is not None:
            v = training_logs.last_error[0]
            t = training_logs.last_error[1]
            ti = training_logs.last_error[2]
            new_error = (v, t, ti, new_model_filename, unique_id, True)

        training_logs.add_error(new_error)
    else:
        training_logs.add_error((val_error, test_error, train_iter, new_model_filename, str(train_iter), False))

    training_logs.pickle_training_logs(train_iter, unique_id)
    pruning_logs.pickle_pruning_logs(train_iter, unique_id)
    

    return training_logs, pruning_logs









def train_automatically(model_wrapper: ModelWrapper, main_save_path, val_stop_fn, max_training_iters=1e9, max_auto_prunings=1e9, train_iter_possible_stop=5, pruning_phase=False, cleaning_err_ix=1, cleanup_k=3, num_of_epochs_per_training=1, pruning_kwargs_dict={}):





    
    os.makedirs(main_save_path, exist_ok=True)






    training_logs = TrainingLogs.load_or_create_training_logs(main_save_path, num_of_epochs_per_training, cleaning_err_ix)

    pruning_logs = PruningLogs.load_or_create_pruning_logs(main_save_path)
    
    # We now save pruning every time we prune, so we don't need to clean up the pruning logs.
    # (The confirming flags will still exist, but who cares.)
    # pruning_logs.clean_up_pruning_train_iters()
        


    if training_logs.last_error is not None:
        train_iter = training_logs.last_error[2]
    else:
        train_iter = 0
    
    initial_train_iter = train_iter







    j_path = osp.join(main_save_path, "initial_train_iters.json")
    j_dict = jh.load(j_path)
    if j_dict is not None:
        j_dict["initial_train_iters"].append(initial_train_iter)
    else:
        j_dict = {"initial_train_iters" : [initial_train_iter]}
    jh.dump(j_path, j_dict)











    num_of_auto_prunings = 0

    while True:



        # Implement the stopping by hand. We need this for debugging.
        
        if train_iter_possible_stop == 0 or ( (train_iter - initial_train_iter) >= train_iter_possible_stop and  (train_iter - initial_train_iter) % train_iter_possible_stop == 0 ):
            
            inp = input(f"""{train_iter_possible_stop} trainings have been done without error stopping.
                        Best k models are kept. (possibly (k+1) models are kept if one of the worse models is the last model we have).
                        Enter "resource_graph" to trigger resource_graph() and re-ask for input.
                        Enter s to save the model and re-ask for input.
                        Enter g to show the graph of the model and re-ask for input.
                        Enter r to trigger show_results() and re-ask for input.
                        Enter a number to reset in how many trainings we ask you this again, and re-ask for input.
                        Enter p to prune anyways (in production code, that is commented out, so the program will simply stop).
                        Press Enter to continue training.
                        Enter any other key to stop.\n""")
            
            if inp == "resource_graph":
                fig, _, res_dict = resource_graph(main_save_path, model_wrapper.save_path)
                fig.savefig(osp.join(main_save_path, f"{train_iter}_resource_graph.png"))
                with open(osp.join(main_save_path, f"{train_iter}_resource_graph.pkl"), "wb") as f:
                    pickle.dump(fig, f)
                with open(osp.join(main_save_path, f"{train_iter}_resource_dict.pkl"), "wb") as f:
                    pickle.dump(res_dict, f)
                inp = input(f"""
                        Enter s to save the model and re-ask for input.
                        Enter g to show the graph of the model and re-ask for input.
                        Enter r to trigger show_results() and re-ask for input.
                        Enter a number to reset in how many trainings we ask you this again, and re-ask for input.
                        Enter p to prune anyways (in production code, that is commented out, so the program will simply stop).
                        Press Enter to continue training.
                        Enter any other key to stop.\n""")
            
            if inp == "s":
                # saving model and reasking for input


                training_logs.delete_all_but_best_k_models(cleanup_k, model_wrapper)
                training_logs, pruning_logs = perform_save(model_wrapper, training_logs, pruning_logs, train_iter, "special_save")
                model_wrapper.create_safety_copy_of_existing_models(f"{train_iter}_special_save")
                
                inp = input(f"""
                        Enter g to show the graph of the model and re-ask for input.
                        Enter r to trigger show_results() and re-ask for input.
                        Enter a number to reset in how many trainings we ask you this again, and re-ask for input.
                        Enter p to prune anyways (in production code, that is commented out, so the program will simply stop).
                        Press Enter to continue training.
                        Enter any other key to stop.\n""")
            
            
            if inp == "g":
                fig, _ = model_wrapper.model_graph()
                fig.savefig(osp.join(main_save_path, f"{train_iter}_model_graph.png"))
                with open(osp.join(main_save_path, f"{train_iter}_model_graph.pkl"), "wb") as f:
                    pickle.dump(fig, f)
                inp = input("""
                        Enter r to trigger show_results() and re-ask for input.
                        Enter a number to reset in how many trainings we ask you this again, and re-ask for input.
                        Enter p to prune anyways (in production code, that is commented out, so the program will simply stop).
                        Press Enter to continue training.
                        Enter any other key to stop.\n""")
            
            if inp == "r":
                fig, _ = show_results(main_save_path)
                if fig is not None:
                    fig.savefig(osp.join(main_save_path, f"{train_iter}_show_results.png"))
                    with open(osp.join(main_save_path, f"{train_iter}_show_results.pkl"), "wb") as f:
                        pickle.dump(fig, f)
                inp = input("""
                        Enter a number to reset in how many trainings we ask you this again, and re-ask for input.
                        Enter p to prune anyways (in production code, that is commented out, so the program will simply stop).
                        Press Enter to continue training.
                        Enter any other key to stop.\n""")
            
            
            try:
                train_iter_possible_stop = int(inp)
                print(f"New trainings before stopping: {train_iter_possible_stop}")
                inp = input("""
                        Enter p to prune anyways (in production code, that is commented out, so the program will simply stop).
                        Press Enter to continue training.
                        Enter any other key to stop.\n""")
            except ValueError:
                pass


            if inp == "p":
                
                # This will ensure I have the best k models from every pruning phase.

                curr_pickleable_conv_res_calc = model_wrapper.resource_calc.get_copy_for_pickle()

                model_wrapper.create_safety_copy_of_existing_models(f"{train_iter}_before_pruning")

                # And this makes even less sense:
                # pruning_logs.log_pruning_train_iter(train_iter, curr_pickleable_conv_res_calc)

                # Makes no sense to also save the model before pruning - it is literally the same model we saved at the end of the previous while.
                # training_logs, pruning_logs = perform_save(model_wrapper, training_logs, pruning_logs, train_iter, "before_pruning")


                curr_pickleable_conv_res_calc, _ = model_wrapper.prune(**pruning_kwargs_dict)

                training_logs.delete_all_but_best_k_models(cleanup_k, model_wrapper)
                pruning_logs.log_pruning_train_iter(train_iter, curr_pickleable_conv_res_calc)
                training_logs, pruning_logs = perform_save(model_wrapper, training_logs, pruning_logs, train_iter, "after_pruning")
                model_wrapper.create_safety_copy_of_existing_models(f"{train_iter}_after_pruning")
                training_logs.delete_all_but_best_k_models(cleanup_k, model_wrapper)


                inp = input("""
                        Enter g to show the graph of the model and re-ask for input.
                        Press Enter to continue training.
                        Enter any other key to stop.\n""")

            if inp == "g":
                fig, _ = model_wrapper.model_graph()
                fig.savefig(osp.join(main_save_path, f"{train_iter}_model_graph_later.png"))
                with open(osp.join(main_save_path, f"{train_iter}_model_graph_later.pkl"), "wb") as f:
                    pickle.dump(fig, f)
                inp = input("""
                        Press Enter to continue training.
                        Enter any other key to stop.\n""")
            

            if inp != "":
                break



        # The pruning mechanism

        if pruning_phase and val_stop_fn(training_logs, pruning_logs, train_iter, initial_train_iter):
            
            curr_pickleable_conv_res_calc = model_wrapper.resource_calc.get_copy_for_pickle()

            # This will ensure I have the best k models from every pruning phase.
            model_wrapper.create_safety_copy_of_existing_models(f"{train_iter}_before_pruning")

            # And this makes even less sense:
            # pruning_logs.log_pruning_train_iter(train_iter, curr_pickleable_conv_res_calc)

            # Makes no sense to also save the model before pruning - it is literally the same model we saved at the end of the previous while.
            # training_logs, pruning_logs = perform_save(model_wrapper, training_logs, pruning_logs, train_iter, "before_pruning")

            
            curr_pickleable_conv_res_calc, are_there_more_to_prune_in_the_future = model_wrapper.prune(**pruning_kwargs_dict)

            num_of_auto_prunings += 1

            # This will ensure I have the best k models from every pruning phase.
            training_logs.delete_all_but_best_k_models(cleanup_k, model_wrapper)
            pruning_logs.log_pruning_train_iter(train_iter, curr_pickleable_conv_res_calc)
            training_logs, pruning_logs = perform_save(model_wrapper, training_logs, pruning_logs, train_iter, "after_pruning")
            model_wrapper.create_safety_copy_of_existing_models(f"{train_iter}_after_pruning")
            training_logs.delete_all_but_best_k_models(cleanup_k, model_wrapper)


            if not are_there_more_to_prune_in_the_future:
                print("There are no more kernels that could be pruned in the future.")
                break
        
        if (train_iter - initial_train_iter) >= max_training_iters:
            break

        if num_of_auto_prunings >= max_auto_prunings:
            break
    









        
        model_wrapper.train(num_of_epochs_per_training)

        # print(f"Hooks: {model_wrapper.tree_ix_2_hook_handle}")

        val_error = model_wrapper.validation()
        test_error = model_wrapper.test()


        train_iter += 1 # this reflects how many trainings we have done



        training_logs.delete_all_but_best_k_models(cleanup_k, model_wrapper)
        training_logs, pruning_logs = perform_save(model_wrapper, training_logs, pruning_logs, train_iter, "", val_error, test_error)





    # After the while loop is broken out of:
    model_wrapper.create_safety_copy_of_existing_models(f"{train_iter}_ending_save")
        


