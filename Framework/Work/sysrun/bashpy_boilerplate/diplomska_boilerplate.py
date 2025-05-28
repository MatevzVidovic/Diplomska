






from pathlib import Path
import io
import re

from sysrun.helpers.help import get_yaml, write_yaml, run
import sysrun.helpers.bashpy_help as bh



out_and_err_ix = 0
def get_novel_out_and_err(out_folder_path):
    global out_and_err_ix
    returner = {
        "stdout_path": out_folder_path / "outs" / f"out_{out_and_err_ix}.txt",
        "stderr_path": out_folder_path / "errs" / f"err_{out_and_err_ix}.txt"
    }
    out_and_err_ix += 1
    return returner


# Helper fn for run_me
def _get_returning_std(returned_std, path_to_std):
    """
    If Popen gets a file to write to (stdout or stderr), it will return None in .communicate().
    Otherwise it returns the output as a string.
    We handle both these cases, so that we can append to the file with all stdouts and stderrs (nicer for ctrl+F).
    """
    if returned_std is None:
        with open(path_to_std, "r") as std_file:
            return std_file.read()
    elif isinstance(returned_std, str):
        return returned_std
    else:
        return f"Error: {returned_std} is not a string or None. It is of type {type(returned_std)}."

# Helper fn for run_me
def _get_stdin(stdin):
    """
    If stdin is a file, we will return its contents.
    If it is a string, we simply return it.
    """
    if isinstance(stdin, io.IOBase):
        with open(stdin.name, "r") as stdin_file:
            return stdin_file.read()
    elif isinstance(stdin, str):
        return stdin
    else:
        return f"stdin was {stdin} (of type {type(stdin)})."

def parse_nested_sysrun_stdout(stdout_text):
    
    # sysrun does:
    # print(f" sysrun_runner_outputs_out_folder_path:[{out_folder_path}] \n")

    match = re.search(r"sysrun_runner_outputs_out_folder_path:\[(.*?)\]", stdout_text)
    if match:
        return match.group(1)
    else:
        return "Pattern not found."

def merge_nested_sysrun_exit_codes(nested_exit_codes_yaml, main_exit_codes_yaml, out_and_err_ix):
    """
    We get a yaml of exit codes from a nested sysrun.
    The sysrun itself was run successfully (sbatch submitted), so we already have:
    {out_and_err_ix: exit_code} in the main exit codes yaml.
    
    Now we want to add the contents of the nested exit codes yaml to the main exit codes yaml.
    
    The nested exit codes yaml has the same structure as the main exit codes:
    it can only have str keys (if no nesting happened inside the nested call, obviously strs.
    But if it did, we used this merging procedure, so the keys are also bound to be strings.


    We concat the out_and_err_ix and the nested_exit_codes key into a string:
    '{out_and_err_ix},{nested_exit_codes_key}'.
    e.g. '0,3,1'

    And again, we simply add it to the bad codes, if it is a bad code.
    """
    nested_exit_codes = nested_exit_codes_yaml["exit_codes"]
    
    for key, value in nested_exit_codes.items():

        # we know the "exit_codes" and "0_bad_codes" keys exist in main_exit_codes_yaml,
        # because this happens in run_me() beforehand. So we can safely add to them.

        new_key = f"{out_and_err_ix},{key}"
        main_exit_codes_yaml["exit_codes"][new_key] = value
        if value != 0:
            main_exit_codes_yaml["0_bad_codes"][new_key] = value




def run_me(command: list, out_folder_path, sysrun_out_folder_path, stdin=None, shell=False, terminal_inp=False, is_nested_sysrun=False):

    result = get_novel_out_and_err(out_folder_path)
    stdout_path = result["stdout_path"]
    stderr_path = result["stderr_path"]
    stdout, stderr, exit_code = run(command, stdout_path, stderr_path, stdin=stdin, shell=shell, terminal_inp=terminal_inp)
    
    global out_and_err_ix
    # Building all_outs_and_errs_concated.txt
    all_outs_and_errs_concated_path = Path(out_folder_path) / "0_all_outs_and_errs_concated.txt"
    stdout_text = _get_returning_std(stdout, stdout_path) # needed for nested sysruns
    with open(all_outs_and_errs_concated_path, "a") as all_outs_and_errs_concated:
        a = all_outs_and_errs_concated
        a.write(8*"\n" + 16*"=")
        a.write(f"Command_{out_and_err_ix}:\n" + str(command)  + 2*"\n")
        a.write(f"stdin_{out_and_err_ix}:\n" + _get_stdin(stdin) + 2*"\n")
        a.write(f"stdout_{out_and_err_ix}:\n" + stdout_text + 2*"\n")
        a.write(f"\nstderr_{out_and_err_ix}:\n" + _get_returning_std(stderr, stderr_path) + 2*"\n")
        a.write(f"\nexit_code_{out_and_err_ix}: " + str(exit_code) + 2*"\n")
        a.write(16*"=" + 8*"\n")

    # Keeping track of all exit codes in one file.
    exit_codes_yaml_path = Path(out_folder_path) / "0_exit_codes.yaml"
    exit_codes_yaml = get_yaml(exit_codes_yaml_path)
    if "0_bad_codes" not in exit_codes_yaml: exit_codes_yaml["0_bad_codes"] = {}
    if exit_code != 0: exit_codes_yaml["0_bad_codes"][f"{out_and_err_ix}"] = exit_code
    if "1_nested_sysruns" not in exit_codes_yaml: exit_codes_yaml["1_nested_sysruns"] = {}
    if "exit_codes" not in exit_codes_yaml: exit_codes_yaml["exit_codes"] = {}
    exit_codes_yaml["exit_codes"][f"{out_and_err_ix}"] = exit_code
    if is_nested_sysrun:
        # this is quite spaghetti, but how else would you do it, yk
        nested_sysrun_folder = parse_nested_sysrun_stdout(stdout_text)
        exit_codes_yaml["1_nested_sysruns"][f"{out_and_err_ix}"] = {}
        exit_codes_yaml["1_nested_sysruns"][f"{out_and_err_ix}"]["sysrun_runner_out_folder"] = nested_sysrun_folder
        try:
            # again, quite spaghetti
            nested_sysrun_folder_path = Path(sysrun_out_folder_path).parent / nested_sysrun_folder
            nested_exit_codes = get_yaml(nested_sysrun_folder_path / "0_exit_codes.yaml")
            merge_nested_sysrun_exit_codes(nested_exit_codes, exit_codes_yaml, out_and_err_ix)
        except Exception as e:
            # raise e
            pass

    write_yaml(exit_codes_yaml, exit_codes_yaml_path)



    # Adding all of this into sysrun_runner_outputs for much easier use.
    # (not creating symlinks, rather copying, for better permanence)

    sysrun_all_outs_and_errs_concated_path = Path(sysrun_out_folder_path) / "0_all_outs_and_errs_concated.txt"
    with open(sysrun_all_outs_and_errs_concated_path, "a") as sysrun_all_outs_and_errs_concated:
        with open(all_outs_and_errs_concated_path, "r") as all_outs_and_errs_concated:
            sysrun_all_outs_and_errs_concated.write(all_outs_and_errs_concated.read())

    sysrun_exit_codes_yaml_path = Path(sysrun_out_folder_path) / "0_exit_codes.yaml"
    write_yaml(exit_codes_yaml, sysrun_exit_codes_yaml_path)



# # Tempfiles aren't needed, because we can also pass strings to run() and it works nicely.
# import tempfile
# with tempfile.NamedTemporaryFile(delete=False) as temp_file:
#     print(f'Temporary file created: {temp_file.name}')
#     temp_file.write(b'Temporary data.')
# # The file remains after closing because delete=False
# os.remove(temp_file.name)  # Clean up the file if needed


temp_file_strs = {
    "save_and_stop": "s\nstop\n",
    "results_and_stop": "r\nstop\n",
    "graph_and_stop": "g\n\nstop\n",
    "resource_graph_and_stop": "resource_graph\nstop\n",
    "test_showcase": "ts\nall\nstop\nstop\n",
    "data_aug": "da\n\n\n\n\n1\n\n\n\n\n2\n\n\n\n\nstop\nstop\n",
    "save_preds": "sp\nstop\nstop\n",
    "batch_size_train": "bst\nstop\nstop\n",
    "batch_size_eval": "bse\nstop\nstop\n",
    "flops_and_weights": "fw\nstop\nstop\n",
}

def boilerplate(path_to_yaml, module_path_to_this_file):

    whole_yaml = get_yaml(path_to_yaml)
    main_name = whole_yaml["oth"]["bashpy_args"]["main_name"]

    main_yaml_path, out_folder_path, sd_path = bh.folder_and_yaml_setup(path_to_yaml, module_path_to_this_file)

    sysrun_runner_outputs_path = whole_yaml["sysrun_info"]["sysrun_runner_outputs_out_folder_path"]

    returner_dict = {
        "whole_yaml": whole_yaml,
        "out_folder_path": out_folder_path,
        "sysrun_runner_outputs_path": sysrun_runner_outputs_path,
        "main_name": main_name,
        "sd_path": sd_path,
        "main_yaml_path": main_yaml_path,
    }
    return returner_dict
    




