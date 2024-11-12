








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

    pickle_fileaname = "training_logs.pkl"

    def __init__(self, tl_main_save_path, number_of_epochs_per_training, last_train_iter=None, deleted_models_errors=[]) -> None:
        
        self.tl_main_save_path = tl_main_save_path

        self.number_of_epochs_per_training = number_of_epochs_per_training
        self.last_train_iter = last_train_iter
        self.deleted_models_errors = deleted_models_errors

        # of the form (val_error, test_error, train_iter, model_path)
        self.errors = []

    def add_error(self, val_error, test_error, train_iter, model_path, unique_id, is_just_after_pruning):
        self.errors.append((val_error, test_error, train_iter, model_path, unique_id, is_just_after_pruning))
        self.last_train_iter = train_iter

    """
    The picking and loading is done in such a way, because if we change where we are running the proram (like go onto another computer)
    we want to just pass the main_save_path to TrainingLogs.load_or_create_training_logs() and it will load the training logs with the correct path,
    because the tl_main_save_path is has the correct path to main_save_path.
    But we also want to pickle to the new path, so we have to change the tl_main_save_path to the new path.
    """

    @staticmethod
    def load_or_create_training_logs(tl_main_save_path, number_of_epochs_per_training, last_train_iter=None, deleted_models_errors=[]):
        os.makedirs(tl_main_save_path, exist_ok=True)
        training_logs_path = os.path.join(tl_main_save_path, TrainingLogs.pickle_fileaname)
        if os.path.exists(training_logs_path):
            new_tl = pickle.load(open(training_logs_path, "rb"))
            new_tl.tl_main_save_path = tl_main_save_path
            return new_tl
        return TrainingLogs(tl_main_save_path, number_of_epochs_per_training, last_train_iter, deleted_models_errors)

    @py_log.log(passed_logger=MY_LOGGER)
    def pickle_training_logs(self):
        new_training_logs_path = os.path.join(self.tl_main_save_path, self.pickle_fileaname)
        with open(new_training_logs_path, "wb") as f:
            pickle.dump(self, f)

    
    def __str__(self):
        returner = ""
        returner += f"Number of epochs per training: {self.number_of_epochs_per_training}\n"
        returner += f"Last train iteration: {self.last_train_iter}\n"
        returner += f"Errors: {self.errors}\n"
        returner += f"Deleted models errors: {self.deleted_models_errors}\n"
        return returner
    
    def __repr__(self):
        # Generate a string representation of the object
        items = (f"{key}={value!r}" for key, value in self.__dict__.items())
        return f"<{self.__class__.__name__}({', '.join(items)})>"


# with the exception of keeping (k+1) models when one of the worse models is the last model we have 
# (we have to keep it to continue training)
def delete_all_but_best_k_models(k: int, training_logs: TrainingLogs, model_wrapper: ModelWrapper):

    # sort by validation error
    sorted_errors = sorted(training_logs.errors, key = lambda x: x[0], reverse=True)

    to_delete = []

    while len(sorted_errors) > 0 and (len(training_logs.errors) - len(to_delete)) > k:

        error = sorted_errors.pop(0)
        model_path = error[3]

        if is_previous_model(model_path, model_wrapper):
            continue

        to_delete.append(error)
    


    for error in to_delete:
        model_path = error[3]
        os.remove(model_path)
    

    to_keep = [error for error in training_logs.errors if error not in to_delete]
    new_training_logs = TrainingLogs(training_logs.tl_main_save_path, training_logs.number_of_epochs_per_training, training_logs.last_train_iter, training_logs.deleted_models_errors)
    for error in to_keep:
        new_training_logs.add_error(*error)

    new_training_logs.deleted_models_errors.extend(to_delete)
    
    return new_training_logs

 

class PrunigLogs:

    pickle_filename = "pruning_logs.pkl"

    def __init__(self, pl_main_save_path, pruning_logs=[]) -> None:
        self.pl_main_save_path = pl_main_save_path
        self.pruning_logs = pruning_logs
    
    @staticmethod
    def load_or_create_pruning_logs(pl_main_save_path, pruning_logs=[]):
        os.makedirs(pl_main_save_path, exist_ok=True)
        pruning_logs_path = os.path.join(pl_main_save_path, PrunigLogs.pickle_filename)
        if os.path.exists(pruning_logs_path):
            new_pl = pickle.load(open(pruning_logs_path, "rb"))
            new_pl.pl_main_save_path = pl_main_save_path
            return new_pl
        return PrunigLogs(pl_main_save_path, pruning_logs)
    
    @py_log.log(passed_logger=MY_LOGGER)
    def pickle_pruning_logs(self):
        new_pruning_logs_path = os.path.join(self.pl_main_save_path, self.pickle_filename)
        with open(new_pruning_logs_path, "wb") as f:
            pickle.dump(self, f)


    # What we pruned is already saved in pruner_istance.pruning_logs
    # We just need to keep track of the corresponding train_iter to be able to know when which pruning happened.
    # That's what this function is for.
    def log_pruning_train_iter(self, train_iter, pickleable_conv_resource_calc):
        # Append the train_iter to the list of train_iters that correspond to prunings.
        # Second value is a flag that tells us if the model was actually saved. It is False to begin with. When we save it, we set it to True.

        self.pruning_logs.append((train_iter, False, pickleable_conv_resource_calc))

        self.pickle_pruning_logs()


    # When we prune, we save the training iter of that pruning.
    # But what if we stop the training before that pruned model is actually saved?
    # This function sets the flag for the train_iter actually being confirmed.
    def confirm_last_pruning_train_iter(self):
        
                
        if len(self.pruning_logs) == 0:
            return
        
        self.pruning_logs[-1] = (self.pruning_logs[-1][0], True , self.pruning_logs[-1][2])

        self.pickle_pruning_logs()


    # If we stop the training before the pruned model is saved, 
    # the last train iter would have turned to true in the next saving iteration,
    # despite the fact it was never saved and has no effect.
    # That's why we have to clean it up before training.
    def clean_up_pruning_train_iters(self):

        
        if len(self.pruning_logs) == 0:
            return
        
        if self.pruning_logs[-1][1] == False:
            self.pruning_logs = self.pruning_logs[:-1]

        self.pickle_pruning_logs()


    def __str__(self):
        returner = ""
        returner += f"Pruning logs: {self.pruning_logs}\n"
        return returner

    def __repr__(self):
        # Generate a string representation of the object
        items = (f"{key}={value!r}" for key, value in self.__dict__.items())
        return f"<{self.__class__.__name__}({', '.join(items)})>"


        



def perform_save(model_wrapper: ModelWrapper, training_logs: TrainingLogs, pruning_logs: PrunigLogs, main_save_path, val_error, test_error, train_iter, unique_id, is_just_after_pruning, curr_training_phase_serial_num):

    new_model_path, _ = model_wrapper.save(unique_id)
    pruning_logs.confirm_last_pruning_train_iter()

    training_logs.add_error(val_error, test_error, train_iter, new_model_path, unique_id, is_just_after_pruning)

    # This has to be done before saving the training logs.
    # Otherwise we wil load a training_logs that will still have something in its errors that it has actually deleted.
    # e.g. Errors: [(0.6350785493850708, 0.6345304846763611, 0, 6), (0.6335894465446472, 0.6331750154495239, 1, 7), (0.6319190859794617, 0.6316145658493042, 2, 8), (0.630038321018219, 0.6299036741256714, 3, 9)]
    # But model 6 has actually been already deleted: [(0.6350785493850708, 0.6345304846763611, 0, 6)]
    
    # The conceptual lifetime of training logs is created/loaded -> added to -> model_deletion -> saved
    # And then the process can repeat. Deletion can't be after saved, it makes no sense. Just think of doing just one iteration of it.
    training_logs = delete_all_but_best_k_models(3, training_logs, model_wrapper)

    training_logs.pickle_training_logs()
    pruning_logs.pickle_pruning_logs()
    
    new_df = pd.DataFrame({"previous_serial_num": [curr_training_phase_serial_num]})
    new_df.to_csv(os.path.join(main_save_path, "previous_training_phase_details.csv"))

    return training_logs, pruning_logs









def train_automatically(model_wrapper: ModelWrapper, main_save_path, val_stop_fn, max_training_iters=1e9, max_auto_prunings=1e9, train_iter_possible_stop=5, pruning_phase=False, error_ix=1, num_of_epochs_per_training=1, pruning_kwargs_dict={}):





    
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


    training_logs = TrainingLogs.load_or_create_training_logs(main_save_path, num_of_epochs_per_training)

    pruning_logs = PrunigLogs.load_or_create_pruning_logs(main_save_path)
    
    # We now save pruning every time we prune, so we don't need to clean up the pruning logs.
    # (The confirming flags will still exist, but who cares.)
    # pruning_logs.clean_up_pruning_train_iters()
        










    train_iter = training_logs.last_train_iter
    
    if train_iter is None:
        train_iter = 0
    

    initial_train_iter = train_iter


    num_of_auto_prunings = 0

    while True:
        
        model_wrapper.train(num_of_epochs_per_training)

        # print(f"Hooks: {model_wrapper.tree_ix_2_hook_handle}")

        val_error = model_wrapper.validation()[error_ix]
        test_error = model_wrapper.test()[error_ix]

        if error_ix in [1, 2, 3]:
            val_error = 1 - val_error
            test_error = 1 - test_error


        train_iter += 1 # this reflects how many trainings we have done




        training_logs, pruning_logs = perform_save(model_wrapper, training_logs, pruning_logs, main_save_path, val_error, test_error, train_iter, str(train_iter), False, curr_training_phase_serial_num)
        










        # Implement the stopping by hand. We need this for debugging.
        
        if (train_iter - initial_train_iter) % train_iter_possible_stop == 0:
            
            inp = input(f"""{train_iter_possible_stop} trainings have been done error stopping.
                        Best k models are kept. (possibly (k+1) models are kept if one of the worse models is the last model we have).
                        
                        Enter g to show the graph of the model and re-ask for input.
                        Enter r to trigger show_results() and re-ask for input.
                        Enter a number to reset in how many trainings we ask you this again, and re-ask for input.
                        Enter p to prune anyways (in production code, that is commented out, so the program will simply stop).
                        Press Enter to continue training.
                        Enter any other key to stop.\n""")
            
            
            if inp == "g":
                model_wrapper.model_graph()
                inp = input("""Enter r to trigger show_results() and re-ask for input.
                        Enter a number to reset in how many trainings we ask you this again, and re-ask for input.
                        Enter p to prune anyways (in production code, that is commented out, so the program will simply stop).
                        Press Enter to continue training.
                        Enter any other key to stop.\n""")
            
            if inp == "r":
                show_results(main_save_path)
                inp = input("""Enter a number to reset in how many trainings we ask you this again, and re-ask for input.
                        Enter p to prune anyways (in production code, that is commented out, so the program will simply stop).
                        Press Enter to continue training.
                        Enter any other key to stop.\n""")
            
            
            try:
                train_iter_possible_stop = int(inp)
                print(f"New trainings before stopping: {train_iter_possible_stop}")
                inp = input("""Enter p to prune anyways (in production code, that is commented out, so the program will simply stop).
                        Press Enter to continue training.
                        Enter any other key to stop.\n""")
            except ValueError:
                pass


            if inp == "p":
                
                curr_pickleable_conv_res_calc, _ = model_wrapper.prune(**pruning_kwargs_dict)

                # This will ensure I have the best k models from every pruning phase.
                model_wrapper.create_safety_copy_of_existing_models(f"{train_iter}_just_pruned")

                pruning_logs.log_pruning_train_iter(train_iter, curr_pickleable_conv_res_calc)
        
                training_logs, pruning_logs = perform_save(model_wrapper, training_logs, pruning_logs, main_save_path, val_error, test_error, train_iter, f"{train_iter}_just_pruned", True, curr_training_phase_serial_num)

                inp = input("""Enter g to show the graph of the model and re-ask for input.
                        Press Enter to continue training.
                        Enter any other key to stop.\n""")

            if inp == "g":
                model_wrapper.model_graph()
                inp = input("""Press Enter to continue training.
                        Enter any other key to stop.\n""")

            if inp != "":
                break


        
        if pruning_phase and val_stop_fn(training_logs, pruning_logs, train_iter, initial_train_iter):
            
            curr_pickleable_conv_res_calc, are_there_more_to_prune_in_the_future = model_wrapper.prune(**pruning_kwargs_dict)

            num_of_auto_prunings += 1

            # This will ensure I have the best k models from every pruning phase.
            model_wrapper.create_safety_copy_of_existing_models(f"{train_iter}_just_pruned")
            pruning_logs.log_pruning_train_iter(train_iter, curr_pickleable_conv_res_calc)

            training_logs, pruning_logs = perform_save(model_wrapper, training_logs, pruning_logs, main_save_path, val_error, test_error, train_iter, f"{train_iter}_just_pruned", True, curr_training_phase_serial_num)

            if not are_there_more_to_prune_in_the_future:
                print("There are no more kernels that could be pruned in the future.")
                break
        
        if (train_iter - initial_train_iter) >= max_training_iters:
            break

        if num_of_auto_prunings >= max_auto_prunings:
            break



def show_results(main_save_path):

    try:

        # inp = input("Enter what comes between pruner_ and .pkl: ")
        # pruner_instance = pickle.load(open("./saved/pruner_" + inp + ".pkl", "rb"))
        # pruning_logs_dict = pruner_instance.pruning_logs


        pruning_logs = pickle.load(open(os.path.join(main_save_path,"pruning_logs.pkl"), "rb"))
        pruning_moments = [i[0] for i in pruning_logs.pruning_logs]


        training_logs = pickle.load(open(os.path.join(main_save_path, "training_logs.pkl"), "rb"))

        model_errors = training_logs.errors + training_logs.deleted_models_errors
        model_errors = [error for error in model_errors if error[5]]
        model_errors.sort(key = lambda x: x[2])

        val_errors = [error[0] for error in model_errors]
        test_errors = [error[1] for error in model_errors]


        # make a plot of the val errors and test errors over time
        # make vertical lines for when pruning happened (in pruning moments)

        # make new plt
        plt.figure()

        plt.plot(val_errors, label="Validation errors")
        plt.plot(test_errors, label="Test errors")

        # plt.ylim(0, 1)

        plt.ylabel("1 - IoU")
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

    except Exception as e:
        # Print the exception message
        print("An exception occurred in show_results():", e)
        # Print the type of the exception
        print("Exception type:", type(e).__name__)
        print("Continuin operation.")


