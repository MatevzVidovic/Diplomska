

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

import torch
from torch.utils.data import DataLoader


import y_helpers.json_handler as jh
from y_helpers.img_and_fig_tools import show_image, save_plt_fig, smart_conversion, save_img
from y_helpers.model_eval_graphs import resource_graph, show_results

from y_framework.model_wrapper import ModelWrapper

from y_framework.log_handlers import TrainingLogs, PruningLogs



















@py_log.autolog(passed_logger=MY_LOGGER)
def perform_save(model_wrapper: ModelWrapper, training_logs: TrainingLogs, pruning_logs: PruningLogs, train_iter, unique_id, val_error=None, test_error=None):

    new_model_filename, _ = model_wrapper.save(f"{train_iter}_{unique_id}")
    pruning_logs.confirm_last_pruning_train_iter()

    if val_error is None or test_error is None:
        new_log = None
        # this only happens if we do a manual save before any training even took place
        # or maybe if we prune before any training took place
        if training_logs.last_log is not None:
            v = training_logs.last_log["val_err"]
            t = training_logs.last_log["test_err"]
            ti = training_logs.last_log["train_iter"]
            # new_log = (v, t, ti, new_model_filename, unique_id, True)
            new_log = ({"val_err": v, "test_err": t, "train_iter": ti, "model_filename": new_model_filename, "unique_id": unique_id, "is_not_automatic": True})

        training_logs.add_log(new_log)
    else:
        new_log = {"val_err": val_error, "test_err": test_error, "train_iter": train_iter, "model_filename": new_model_filename, "unique_id": unique_id, "is_not_automatic": False}
        training_logs.add_log(new_log)


    training_logs.pickle_training_logs(train_iter, unique_id)
    pruning_logs.pickle_pruning_logs(train_iter, unique_id)
    

    return training_logs, pruning_logs








@py_log.autolog(passed_logger=MY_LOGGER)
def train_automatically(model_wrapper: ModelWrapper, main_save_path, val_stop_fn=None, max_training_iters=1e9, max_total_training_iters=1e9,
                        max_auto_prunings=1e9, train_iter_possible_stop=5, pruning_phase=False, cleaning_err_key="loss", 
                        cleanup_k=3, num_of_epochs_per_training=1, pruning_kwargs_dict=None, model_graph_breakup_param=0.05, one_big_svg_width=500):
    
    if pruning_kwargs_dict is None:
        pruning_kwargs_dict = {}


    # to prevent an error I had, where even the last model would somehow get deleted (which is another error on top of that, because that should never happen)
    if cleanup_k < 1:
        raise ValueError("cleanup_k must be at least 1.")

    
    os.makedirs(main_save_path, exist_ok=True)






    training_logs = TrainingLogs.load_or_create_training_logs(main_save_path, num_of_epochs_per_training, cleaning_err_key)

    pruning_logs = PruningLogs.load_or_create_pruning_logs(main_save_path)
    
    # We now save pruning every time we prune, so we don't need to clean up the pruning logs.
    # (The confirming flags will still exist, but who cares.)
    # pruning_logs.clean_up_pruning_train_iters()
        


    if training_logs.last_log is not None:
        train_iter = training_logs.last_log["train_iter"]
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
                        Enter bst to do a batch_size_train and re-ask for input.
                        Enter bse to do a batch_size_eval and re-ask for input.
                        Enter sp to do a save_preds from the data_path/save_preds directory, and re-ask for input.
                        Enter da to do a data augmentation showcase and re-ask for input.
                        Enter ts to do a test showcase of the model and re-ask for input.
                        Enter "resource_graph" to trigger resource_graph() and re-ask for input.
                        Enter s to save the model and re-ask for input.
                        Enter g to show the graph of the model and re-ask for input.
                        Enter r to trigger show_results() and re-ask for input.
                        Enter a number to reset in how many trainings we ask you this again, and re-ask for input.
                        Enter p to prune anyways (in production code, that is commented out, so the program will simply stop).
                        Press Enter to continue training.
                        Enter any other key to stop.\n""")
            

            if inp == "bst":
                model_wrapper.training_wrapper.batch_size_train()

                inp = input(f"""{train_iter_possible_stop} trainings have been done without error stopping.
                            Best k models are kept. (possibly (k+1) models are kept if one of the worse models is the last model we have).
                            Enter bse to do a batch_size_eval and re-ask for input.
                            Enter sp to do a save_preds from the data_path/save_preds directory, and re-ask for input.
                            Enter da to do a data augmentation showcase and re-ask for input.
                            Enter ts to do a test showcase of the model and re-ask for input.
                            Enter "resource_graph" to trigger resource_graph() and re-ask for input.
                            Enter s to save the model and re-ask for input.
                            Enter g to show the graph of the model and re-ask for input.
                            Enter r to trigger show_results() and re-ask for input.
                            Enter a number to reset in how many trainings we ask you this again, and re-ask for input.
                            Enter p to prune anyways (in production code, that is commented out, so the program will simply stop).
                            Press Enter to continue training.
                            Enter any other key to stop.\n""")          

            if inp == "bse":
                model_wrapper.training_wrapper.batch_size_eval()

                inp = input(f"""{train_iter_possible_stop} trainings have been done without error stopping.
                            Best k models are kept. (possibly (k+1) models are kept if one of the worse models is the last model we have).
                            Enter sp to do a save_preds from the data_path/save_preds directory, and re-ask for input.
                            Enter da to do a data augmentation showcase and re-ask for input.
                            Enter ts to do a test showcase of the model and re-ask for input.
                            Enter "resource_graph" to trigger resource_graph() and re-ask for input.
                            Enter s to save the model and re-ask for input.
                            Enter g to show the graph of the model and re-ask for input.
                            Enter r to trigger show_results() and re-ask for input.
                            Enter a number to reset in how many trainings we ask you this again, and re-ask for input.
                            Enter p to prune anyways (in production code, that is commented out, so the program will simply stop).
                            Press Enter to continue training.
                            Enter any other key to stop.\n""")
            
            if inp == "sp":
                model_wrapper.training_wrapper.save_preds(path_to_save_to=osp.join(main_save_path, f"{train_iter}_save_preds"))

                inp = input(f"""
                            Enter da to do a data augmentation showcase and re-ask for input.
                            Enter ts to do a test showcase of the model and re-ask for input.
                            Enter "resource_graph" to trigger resource_graph() and re-ask for input.
                            Enter s to save the model and re-ask for input.
                            Enter g to show the graph of the model and re-ask for input.
                            Enter r to trigger show_results() and re-ask for input.
                            Enter a number to reset in how many trainings we ask you this again, and re-ask for input.
                            Enter p to prune anyways (in production code, that is commented out, so the program will simply stop).
                            Press Enter to continue training.
                            Enter any other key to stop.\n""")

            if inp == "da":
                
                inp = ""
                curr_dataset = model_wrapper.training_wrapper.dataloaders_dict["train"].dataset
                target_name = model_wrapper.training_wrapper.params.target
                da_dataloader = DataLoader(curr_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
                img_ix = 0

                save_path = osp.join(main_save_path, "data_aug")
                quick_figs_counter = 0
                while inp == "":
                    
                    ix = 0
                    for data_dict in da_dataloader:
                        
                        # so we actually get to the image we want to see
                        if ix < img_ix:
                            ix += 1
                            continue


                        X = data_dict["images"]
                        y = data_dict[target_name]
                        # img_names = data_dict["img_names"]

                        # curr_img = X[0, :3].squeeze()
                        curr_img = X[0].squeeze()[:3]

                        curr_target = y[0].squeeze()
                        break

                    combined_img = curr_img * (1 - curr_target)   # This will make all vein pixels blacked out.
                    combined_img_2 = curr_img * curr_target       # This will make all non-vein pixels blacked out.
                    save_img(combined_img, save_path, f"da_{quick_figs_counter}_img_vein_blacked_out.png")
                    save_img(combined_img_2, save_path, f"da_{quick_figs_counter}_img_non_vein_blacked_out.png")
                    
                    
                    
                    # fig, _ = show_image([curr_img, curr_target])
                    # save_plt_fig(fig, save_path, f"da_{quick_figs_counter}")

                    curr_img = smart_conversion(curr_img, "ndarray", "uint8")
                    # save_img(curr_img, save_path, f"da_{quick_figs_counter}_img.png")
                    

                    # mask is int64, because torch likes it like that. Lets make it float, because the vals are only 0s and 1s, and so smart conversion in save_img()
                    # will make it 0s and 255s.
                    curr_target = curr_target.to(torch.float32)
                    # save_img(curr_target, save_path, f"da_{quick_figs_counter}_target.png")




                    quick_figs_counter += 1

                    inp = input("""Press Enter to get the next data augmentation. Enter a number to swith to img with that ix in the dataset as the subject of this data augmentation test.
                                Enter anything to stop with the data augmentation testing.\n""")
                    
                    if inp.isdigit():
                        img_ix = int(inp)
                        inp = ""
                    
                    

                inp = input(f"""
                        Enter ts to do a test showcase of the model and re-ask for input.
                        Enter "resource_graph" to trigger resource_graph() and re-ask for input.
                        Enter s to save the model and re-ask for input.
                        Enter g to show the graph of the model and re-ask for input.
                        Enter r to trigger show_results() and re-ask for input.
                        Enter a number to reset in how many trainings we ask you this again, and re-ask for input.
                        Enter p to prune anyways (in production code, that is commented out, so the program will simply stop).
                        Press Enter to continue training.
                        Enter any other key to stop.\n""")
            
            
            
            if inp == "ts":
                model_wrapper.training_wrapper.test_showcase(path_to_save_to=osp.join(main_save_path, f"{train_iter}_test_showcase"))
                inp = input(f"""
                        Enter "resource_graph" to trigger resource_graph() and re-ask for input.
                        Enter s to save the model and re-ask for input.
                        Enter g to show the graph of the model and re-ask for input.
                        Enter r to trigger show_results() and re-ask for input.
                        Enter a number to reset in how many trainings we ask you this again, and re-ask for input.
                        Enter p to prune anyways (in production code, that is commented out, so the program will simply stop).
                        Press Enter to continue training.
                        Enter any other key to stop.\n""")
            

            
            if inp == "resource_graph":
                res = resource_graph(main_save_path, model_wrapper.save_path)
                
                if res is not None:
                    fig, _, res_dict = res
                    graph_save_path = osp.join(main_save_path, "graphs")
                    os.makedirs(graph_save_path, exist_ok=True)
                    save_plt_fig(fig, graph_save_path, f"{train_iter}_resource_graph")
                    with open(osp.join(graph_save_path, f"{train_iter}_resource_dict.pkl"), "wb") as f:
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

                graph_save_path = osp.join(main_save_path, f"graphs")

                # make the plt pickle verion (for viewing thee moddel graph by opening the picke and interacting through plt)
                model_graph_args_dict = {
                    "width": 10,
                }
                list_of_fig_ax_id_tuples = model_wrapper.model_graph(model_graph_args_dict)
                fig = list_of_fig_ax_id_tuples[0][0]
                fig.show()
                input("wait")
                save_plt_fig(fig, graph_save_path, f"{train_iter}_model_graph", formats={"pkl"})


                # one big svg
                model_graph_args_dict = {
                    "width": one_big_svg_width,
                    "for_img_dict" : {
                        "min_child_width_limit": 0.0
                    }
                }
                list_of_fig_ax_id_tuples = model_wrapper.model_graph(model_graph_args_dict)
                fig = list_of_fig_ax_id_tuples[0][0]
                save_plt_fig(fig, graph_save_path, f"{train_iter}_model_graph", formats={"svg"})



                # broken up svgs
                # change up min_child_width_limit until you find a nice value for where the graphs split up 
                model_graph_args_dict = {
                    "width": 100,
                    "for_img_dict" : {
                        "min_child_width_limit": model_graph_breakup_param
                    }
                }
                list_of_fig_ax_id_tuples = model_wrapper.model_graph(model_graph_args_dict)
                save_path_for_collection = osp.join(graph_save_path, f"{train_iter}_broken_up_svgs")
                
                count = 1
                while osp.exists(save_path_for_collection):    # to make sure we don't overwrite anything, and we keep things readable
                    save_path_for_collection = osp.join(graph_save_path, f"{train_iter}_broken_up_svgs_{count}")
                    count += 1
                
                for fig, _, curr_id in list_of_fig_ax_id_tuples:
                    save_plt_fig(fig, save_path_for_collection, f"{train_iter}_model_graph_{curr_id}", formats={"svg"})






                
                inp = input("""
                        Enter r to trigger show_results() and re-ask for input.
                        Enter a number to reset in how many trainings we ask you this again, and re-ask for input.
                        Enter p to prune anyways (in production code, that is commented out, so the program will simply stop).
                        Press Enter to continue training.
                        Enter any other key to stop.\n""")
            
            if inp == "r":
                res = show_results(main_save_path)
                if res is not None:
                    fig, _ = res
                    graph_save_path = osp.join(main_save_path, "graphs")
                    save_plt_fig(fig, graph_save_path, f"{train_iter}_show_results")
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


                inp = input("""
                        Press Enter to continue training.
                        Enter any other key to stop.\n""")

            # if inp == "g":

            #     graph_save_path = osp.join(main_save_path, f"graphs")

            #     # make the plt pickle verion (for viewing thee moddel graph by opening the picke and interacting through plt)
            #     model_graph_args_dict = {
            #         "width": 10,
            #     }
            #     list_of_fig_ax_id_tuples = model_wrapper.model_graph(model_graph_args_dict)
            #     fig = list_of_fig_ax_id_tuples[0][0]
            #     fig.show()
            #     input("wait")
            #     save_plt_fig(fig, graph_save_path, f"{train_iter}_model_graph", formats={"pkl"})


            #     # one big svg
            #     model_graph_args_dict = {
            #         "width": 800,
            #         "for_img_dict" : {
            #             "min_child_width_limit": 0.0
            #         }
            #     }
            #     list_of_fig_ax_id_tuples = model_wrapper.model_graph(model_graph_args_dict)
            #     fig = list_of_fig_ax_id_tuples[0][0]
            #     save_plt_fig(fig, graph_save_path, f"{train_iter}_model_graph", formats={"svg"})



            #     # broken up svgs
            #     # change up min_child_width_limit until you find a nice value for where the graphs split up 
            #     model_graph_args_dict = {
            #         "width": 100,
            #         "for_img_dict" : {
            #             "min_child_width_limit": 0.08
            #         }
            #     }
            #     list_of_fig_ax_id_tuples = model_wrapper.model_graph(model_graph_args_dict)
            #     save_path_for_collection = osp.join(graph_save_path, f"{train_iter}_broken_up_svgs")
            #     for fig, _, curr_id in list_of_fig_ax_id_tuples:
            #         save_plt_fig(fig, save_path_for_collection, f"{train_iter}_model_graph_{curr_id}", formats={"svg"})


            #     inp = input("""
            #             Press Enter to continue training.
            #             Enter any other key to stop.\n""")
            

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
            # training_logs.delete_all_but_best_k_models(cleanup_k, model_wrapper)
            model_wrapper.create_safety_copy_of_existing_models(f"{train_iter}_after_pruning")



            if not are_there_more_to_prune_in_the_future:
                print("There are no more kernels that could be pruned in the future.")
                break



        if train_iter >= max_total_training_iters:
            print(f"Max total training iterations reached: {max_total_training_iters}. Train_iter: {train_iter}")
            break
        
        if (train_iter - initial_train_iter) >= max_training_iters:
            print(f"Max training iterations reached: {max_training_iters}. Train_iter: {train_iter}, Initial_train_iter: {initial_train_iter}")
            break

        if num_of_auto_prunings >= max_auto_prunings:
            print(f"Max auto prunings reached: {max_auto_prunings}. num_of_auto_prunings: {num_of_auto_prunings}")
            break
    









        
        model_wrapper.train(num_of_epochs_per_training)

        # print(f"Hooks: {model_wrapper.tree_ix_2_hook_handle}")

        val_error = model_wrapper.validation()
        test_error = model_wrapper.test()


        train_iter += 1 # this reflects how many trainings we have done

        print(f"We have finished training iteration {train_iter}")



        training_logs.delete_all_but_best_k_models(cleanup_k, model_wrapper)
        training_logs, pruning_logs = perform_save(model_wrapper, training_logs, pruning_logs, train_iter, "", val_error, test_error)
        # training_logs.delete_all_but_best_k_models(cleanup_k, model_wrapper)






    # After the while loop is broken out of:
    model_wrapper.create_safety_copy_of_existing_models(f"{train_iter}_ending_save")
        


