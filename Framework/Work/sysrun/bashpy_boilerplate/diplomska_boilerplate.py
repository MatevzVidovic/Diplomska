






from pathlib import Path
import io

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

def run_me(command: list, out_folder_path, stdin=None, shell=False, terminal_inp=False):

    result = get_novel_out_and_err(out_folder_path)
    stdout_path = result["stdout_path"]
    stderr_path = result["stderr_path"]
    stdout, stderr, exit_code = run(command, stdout_path, stderr_path, stdin=stdin, shell=shell, terminal_inp=terminal_inp)
    
    global out_and_err_ix
    # Building all_outs_and_errs_concated.txt
    all_outs_and_errs_concated_path = Path(out_folder_path) / "0_all_outs_and_errs_concated.txt"
    with open(all_outs_and_errs_concated_path, "a") as all_outs_and_errs_concated:
        a = all_outs_and_errs_concated
        a.write(8*"\n" + 16*"=")
        a.write(f"Command_{out_and_err_ix}:\n" + str(command)  + 2*"\n")
        a.write(f"stdin_{out_and_err_ix}:\n" + _get_stdin(stdin) + 2*"\n")
        a.write(f"stdout_{out_and_err_ix}:\n" + _get_returning_std(stdout, stdout_path) + 2*"\n")
        a.write(f"\nstderr_{out_and_err_ix}:\n" + _get_returning_std(stderr, stderr_path) + 2*"\n")
        a.write(f"\nexit_code_{out_and_err_ix}: " + str(exit_code) + 2*"\n")
        a.write(16*"=" + 8*"\n")

    # Keeping track of all exit codes in one file.
    exit_codes_yaml_path = Path(out_folder_path) / "0_exit_codes.yaml"
    exit_codes_yaml = get_yaml(exit_codes_yaml_path)
    if "0_bad_codes" not in exit_codes_yaml: exit_codes_yaml["0_bad_codes"] = {}
    if exit_code != 0: exit_codes_yaml["0_bad_codes"][f"{out_and_err_ix}"] = exit_code
    if "exit_codes" not in exit_codes_yaml: exit_codes_yaml["exit_codes"] = {}
    exit_codes_yaml["exit_codes"][f"{out_and_err_ix}"] = exit_code
    write_yaml(exit_codes_yaml, exit_codes_yaml_path)



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

    returner_dict = {
        "whole_yaml": whole_yaml,
        "out_folder_path": out_folder_path,
        "main_name": main_name,
        "sd_path": sd_path,
        "main_yaml_path": main_yaml_path,
    }
    return returner_dict
    




