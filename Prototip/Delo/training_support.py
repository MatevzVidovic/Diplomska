








import os
import logging
import python_logger.log_helper_off as py_log


MY_LOGGER = logging.getLogger("prototip") # or any string. Mind this: same string, same logger.
MY_LOGGER.setLevel(logging.DEBUG)





import matplotlib.pyplot as plt
import pandas as pd
import pickle

from ModelWrapper import ModelWrapper















def is_previous_model(model_path, model_wrapper, get_last_model_path = False):

    prev_model_details = pd.read_csv(os.path.join(model_wrapper.save_path, "previous_model_details.csv"))
    prev_model_path = prev_model_details["previous_model_path"][0]

    returner = model_path == prev_model_path

    if get_last_model_path:
        returner = (returner, prev_model_path)

    return returner


def delete_old_model(model_path, model_wrapper):
   
    is_prev, prev_model_path = is_previous_model(model_path, model_wrapper, get_last_model_path=True)
    
    if is_prev:
        print("The model you are trying to delete is the last model that was saved. You can't delete it.")
        return False
    
    os.remove(prev_model_path)
    return True





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
        tl_csv_path = os.path.join(tl_main_save_path, "prev_training_logs_name.csv")

        if os.path.exists(tl_csv_path):
            
            prev_training_logs_name = pd.read_csv(tl_csv_path)
            pl_name = prev_training_logs_name[f"prev_{TrainingLogs.pickle_filename}"][0]
            training_logs_path = os.path.join(tl_main_save_path, pl_name)

            new_tl = pickle.load(open(training_logs_path, "rb"))
            new_tl.tl_main_save_path = tl_main_save_path
            return new_tl

        return TrainingLogs(tl_main_save_path, number_of_epochs_per_training, cleaning_err_ix, last_error, deleted_models_errors)

    @py_log.log(passed_logger=MY_LOGGER)
    def pickle_training_logs(self, train_iter, unique_id):

        self.last_train_iter = train_iter
        self.last_unique_id = unique_id
        curr_name = f"{self.pickle_filename}_{train_iter}_{unique_id}.pkl"

        new_training_logs_path = os.path.join(self.tl_main_save_path, curr_name)
        with open(new_training_logs_path, "wb") as f:
            pickle.dump(self, f)
        
        new_df = pd.DataFrame({f"prev_{self.pickle_filename}": [curr_name]})
        new_df.to_csv(os.path.join(self.tl_main_save_path, "prev_training_logs_name.csv"))


    
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
            model_path = error[3]
            self.delete_error(error)
            os.remove(model_path)




    

 

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
        pl_csv_path = os.path.join(pl_main_save_path, "prev_pruning_logs_name.csv")

        if os.path.exists(pl_csv_path):

            prev_pruning_logs_name = pd.read_csv(pl_csv_path)
            pl_name = prev_pruning_logs_name[f"prev_{PruningLogs.pickle_filename}"][0]

            pruning_logs_path = os.path.join(pl_main_save_path, pl_name)
            loaded_pl = pickle.load(open(pruning_logs_path, "rb"))
            loaded_pl.pl_main_save_path = pl_main_save_path
            return loaded_pl

        return PruningLogs(pl_main_save_path, pruning_logs)
    
    @py_log.log(passed_logger=MY_LOGGER)
    def pickle_pruning_logs(self, train_iter, unique_id):

        self.last_train_iter = train_iter
        self.last_unique_id = unique_id
        curr_name = f"{self.pickle_filename}_{train_iter}_{unique_id}.pkl" 

        new_pruning_logs_path = os.path.join(self.pl_main_save_path, curr_name)
        with open(new_pruning_logs_path, "wb") as f:
            pickle.dump(self, f)
        
        new_df = pd.DataFrame({f"prev_{self.pickle_filename}": [curr_name]})
        new_df.to_csv(os.path.join(self.pl_main_save_path, "prev_pruning_logs_name.csv"))



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





def perform_save(model_wrapper: ModelWrapper, training_logs: TrainingLogs, pruning_logs: PruningLogs, main_save_path, train_iter, unique_id, curr_training_phase_serial_num, cleanup_k, val_error=None, test_error=None):

    new_model_path, _ = model_wrapper.save(f"{train_iter}_{unique_id}")
    pruning_logs.confirm_last_pruning_train_iter()

    if val_error is None or test_error is None:
        new_error = None
        # this only happens if we do a manual save before any training even took place
        # or maybe if we prune before any training took place
        if training_logs.last_error is not None:
            v = training_logs.last_error[0]
            t = training_logs.last_error[1]
            ti = training_logs.last_error[2]
            new_error = (v, t, ti, new_model_path, unique_id, True)

        training_logs.add_error(new_error)
    else:
        training_logs.add_error((val_error, test_error, train_iter, new_model_path, str(train_iter), False))


    # In the sense of automatic saving:
    # This cleanup has to be done before saving the training logs.
    # Otherwise we wil load a training_logs that will still have something in its errors that it has actually deleted.
    # e.g. Errors: [(0.6350785493850708, 0.6345304846763611, 0, 6), (0.6335894465446472, 0.6331750154495239, 1, 7), (0.6319190859794617, 0.6316145658493042, 2, 8), (0.630038321018219, 0.6299036741256714, 3, 9)]
    # But model 6 has actually been already deleted: [(0.6350785493850708, 0.6345304846763611, 0, 6)]
    
    # The conceptual lifetime of training logs is created/loaded -> added to -> model_deletion -> saved
    # And then the process can repeat. Deletion can't be after saved, it makes no sense. Just think of doing just one iteration of it.

    training_logs.delete_all_but_best_k_models(cleanup_k, model_wrapper)

    training_logs.pickle_training_logs(train_iter, unique_id)
    pruning_logs.pickle_pruning_logs(train_iter, unique_id)
    
    new_df = pd.DataFrame({"previous_serial_num": [curr_training_phase_serial_num]})
    new_df.to_csv(os.path.join(main_save_path, "previous_training_phase_details.csv"))

    return training_logs, pruning_logs









def train_automatically(model_wrapper: ModelWrapper, main_save_path, val_stop_fn, max_training_iters=1e9, max_auto_prunings=1e9, train_iter_possible_stop=5, pruning_phase=False, cleaning_err_ix=1, cleanup_k=3, num_of_epochs_per_training=1, pruning_kwargs_dict={}):





    
    os.makedirs(main_save_path, exist_ok=True)




    previous_training_phase_details_path = os.path.join(main_save_path, "previous_training_phase_details.csv")

    if os.path.exists(previous_training_phase_details_path):
        prev_training_phase_details = pd.read_csv(previous_training_phase_details_path)
        prev_training_phase_serial_num = prev_training_phase_details["previous_serial_num"][0]
    else:
        prev_training_phase_serial_num = None
    

    if prev_training_phase_serial_num is None:
        curr_training_phase_serial_num = 0
    else:
        curr_training_phase_serial_num = prev_training_phase_serial_num + 1


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
                fig, _, res_dict = resource_graph(model_wrapper.initial_resource_calc, pruning_logs)
                fig.savefig(os.path.join(main_save_path, f"{train_iter}_resource_graph.png"))
                with open(os.path.join(main_save_path, f"{train_iter}_resource_graph.pkl"), "wb") as f:
                    pickle.dump(fig, f)
                with open(os.path.join(main_save_path, f"{train_iter}_resource_dict.pkl"), "wb") as f:
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
                model_wrapper.create_safety_copy_of_existing_models(f"{train_iter}_special_save")
                training_logs, pruning_logs = perform_save(model_wrapper, training_logs, pruning_logs, main_save_path, train_iter, "special_save", curr_training_phase_serial_num, cleanup_k)
                inp = input(f"""
                        Enter g to show the graph of the model and re-ask for input.
                        Enter r to trigger show_results() and re-ask for input.
                        Enter a number to reset in how many trainings we ask you this again, and re-ask for input.
                        Enter p to prune anyways (in production code, that is commented out, so the program will simply stop).
                        Press Enter to continue training.
                        Enter any other key to stop.\n""")
            
            
            if inp == "g":
                fig, _ = model_wrapper.model_graph()
                fig.savefig(os.path.join(main_save_path, f"{train_iter}_model_graph.png"))
                with open(os.path.join(main_save_path, f"{train_iter}_model_graph.pkl"), "wb") as f:
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
                    fig.savefig(os.path.join(main_save_path, f"{train_iter}_show_results.png"))
                    with open(os.path.join(main_save_path, f"{train_iter}_show_results.pkl"), "wb") as f:
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
                pruning_logs.log_pruning_train_iter(train_iter, curr_pickleable_conv_res_calc)
                training_logs, pruning_logs = perform_save(model_wrapper, training_logs, pruning_logs, main_save_path, train_iter, "before_pruning", curr_training_phase_serial_num, cleanup_k)


                curr_pickleable_conv_res_calc, _ = model_wrapper.prune(**pruning_kwargs_dict)


                model_wrapper.create_safety_copy_of_existing_models(f"{train_iter}_after_pruning")
                pruning_logs.log_pruning_train_iter(train_iter, curr_pickleable_conv_res_calc)
                training_logs, pruning_logs = perform_save(model_wrapper, training_logs, pruning_logs, main_save_path, train_iter, "after_pruning", curr_training_phase_serial_num, cleanup_k)

                inp = input("""
                        Enter g to show the graph of the model and re-ask for input.
                        Press Enter to continue training.
                        Enter any other key to stop.\n""")

            if inp == "g":
                fig, _ = model_wrapper.model_graph()
                fig.savefig(os.path.join(main_save_path, f"{train_iter}_model_graph_later.png"))
                with open(os.path.join(main_save_path, f"{train_iter}_model_graph_later.pkl"), "wb") as f:
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
            pruning_logs.log_pruning_train_iter(train_iter, curr_pickleable_conv_res_calc)

            training_logs, pruning_logs = perform_save(model_wrapper, training_logs, pruning_logs, main_save_path, train_iter, "before_pruning", curr_training_phase_serial_num, cleanup_k)

            
            curr_pickleable_conv_res_calc, are_there_more_to_prune_in_the_future = model_wrapper.prune(**pruning_kwargs_dict)

            num_of_auto_prunings += 1

            # This will ensure I have the best k models from every pruning phase.
            model_wrapper.create_safety_copy_of_existing_models(f"{train_iter}_after_pruning")
            pruning_logs.log_pruning_train_iter(train_iter, curr_pickleable_conv_res_calc)

            training_logs, pruning_logs = perform_save(model_wrapper, training_logs, pruning_logs, main_save_path, train_iter, "after_pruning", curr_training_phase_serial_num, cleanup_k)

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




        training_logs, pruning_logs = perform_save(model_wrapper, training_logs, pruning_logs, main_save_path, train_iter, "", curr_training_phase_serial_num, cleanup_k, val_error, test_error)





    # After the while loop is broken out of:
    model_wrapper.create_safety_copy_of_existing_models(f"{train_iter}_ending_save")
    training_logs, pruning_logs = perform_save(model_wrapper, training_logs, pruning_logs, main_save_path, train_iter, "ending_save", curr_training_phase_serial_num, cleanup_k)


        




def show_results(main_save_path):

    try:

        # inp = input("Enter what comes between pruner_ and .pkl: ")
        # pruner_instance = pickle.load(open("./saved/pruner_" + inp + ".pkl", "rb"))
        # pruning_logs_dict = pruner_instance.pruning_logs


        tl_csv_path = os.path.join(main_save_path, "prev_training_logs_name.csv")
        pl_csv_path = os.path.join(main_save_path, "prev_pruning_logs_name.csv")

        if not os.path.exists(tl_csv_path) or not os.path.exists(pl_csv_path):
            print("The logs don't exist.")
            return

        training_logs_name = pd.read_csv(tl_csv_path)
        pl_name = pd.read_csv(pl_csv_path)

        training_logs_path = os.path.join(main_save_path, training_logs_name[f"prev_{TrainingLogs.pickle_filename}"][0])
        pruning_logs_path = os.path.join(main_save_path, pl_name[f"prev_{PruningLogs.pickle_filename}"][0])

        training_logs = pickle.load(open(training_logs_path, "rb"))
        pruning_logs = pickle.load(open(pruning_logs_path, "rb"))
        
        
        
        pruning_moments = [i[0] for i in pruning_logs.pruning_logs]


        
        print(training_logs)
        print(pruning_logs)

        model_errors = training_logs.errors + training_logs.deleted_models_errors
        model_errors = [log for log in model_errors if log is not None]
        model_errors = [log for log in model_errors if not log[5]]
        model_errors.sort(key = lambda x: x[2])

        val_errors = [error[0] for error in model_errors]
        test_errors = [error[1] for error in model_errors]


        val_losses = [error[0] for error in val_errors]
        # error[1] is approx IoU
        val_1minus_F1s = [1 - error[2] for error in val_errors]
        val_1minus_IoUs = [1 - error[3] for error in val_errors]

        

        test_losses = [error[1] for error in test_errors]
        # error[1] is approx IoU
        test_1minus_F1s = [1 - error[2] for error in test_errors]
        test_1minus_IoUs = [1 - error[3] for error in test_errors]


        # make a plot of the val errors and test errors over time
        # make vertical lines for when pruning happened (in pruning moments)

        # make new plt
        fig, ax = plt.subplots()


        plt.plot(val_losses, label="avg cross entropy (validation)")
        plt.plot(test_losses, label="avg cross entropy (test)")
        plt.plot(val_1minus_F1s, label="1-F1 (validation)")
        plt.plot(test_1minus_F1s, label="1-F1 (test)")
        plt.plot(val_1minus_IoUs, label="1-IoU (validation)")
        plt.plot(test_1minus_IoUs, label="1-IoU (test)")

        # plt.ylim(0, 1)

        plt.ylabel("Error")
        plt.xlabel("training iterations")

        for moment in pruning_moments:
            plt.axvline(x=moment, color="blue", linestyle="--")
            




        from matplotlib.patches import Patch

        # Create the initial legend
        plt.legend()

        # Get current handles and labels
        handles, labels = plt.gca().get_legend_handles_labels()

        # Create a dummy handle for the arbitrary string
        extra = Patch(facecolor='none', edgecolor='none', label='Arbitrary String')

        # Append the new handle and label
        # handles.append(extra)
        # labels.append("Green line: 10 filters pruned")
        handles.append(extra)
        labels.append("Blue line: 1 pruning")

        # Update the legend with the new entries
        plt.legend(handles=handles, labels=labels)

        plt.show(block=False)

        return fig, ax

    except Exception as e:
        # Print the exception message
        print("An exception occurred in show_results():", e)
        # Print the type of the exception
        print("Exception type:", type(e).__name__)
        print("Continuing operation.")
        return None, None



def resource_graph(initial_resource_calc, pruning_logs):

    try:

        initial_flops = initial_resource_calc.get_resource_of_whole_model("flops_num")
        initial_weights = initial_resource_calc.get_resource_of_whole_model("weights_num")

        pruning_moments = [i[0] for i in pruning_logs.pruning_logs]
        resource_calcs = [i[2] for i in pruning_logs.pruning_logs]
        flops = [i.get_resource_of_whole_model("flops_num") for i in resource_calcs]
        weights = [i.get_resource_of_whole_model("weights_num") for i in resource_calcs]

        flops_percents = [(i) / initial_flops for i in flops]
        weights_percents = [(i) / initial_weights for i in weights]

        fig, ax = plt.subplots()
        plt.plot(pruning_moments, flops_percents, label="FLOPs")
        plt.plot(pruning_moments, weights_percents, label="Weights")

        plt.ylabel("Resource percent")
        plt.xlabel("training iterations")

        plt.legend()


        plt.show(block=False)

        to_pkl = []
        for i in range(len(pruning_moments)):
            to_pkl.append((pruning_moments[i], flops_percents[i], weights_percents[i]))

        res_dict = {"(pruning_moment, flops_percent, weights_percent": to_pkl}

        return fig, ax, res_dict



    except Exception as e:

        # raise e

        # Print the exception message
        print("An exception occurred in resource_graph():", e)
        # Print the type of the exception
        print("Exception type:", type(e).__name__)
        print("Continuing operation.")
        return None, None, None

