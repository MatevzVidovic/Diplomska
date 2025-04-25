






from pathlib import Path

from sysrun.helpers.help import get_yaml, write_yaml



out_and_err_ix = 0
def get_novel_out_and_err(out_folder_path):
    global out_and_err_ix
    returner = {
        "stdout_path": out_folder_path / f"out_{out_and_err_ix}.txt",
        "stderr_path": out_folder_path / f"err_{out_and_err_ix}.txt"
    }
    out_and_err_ix += 1
    return returner

def boilerplate(path_to_yaml, module_path_to_this_file):


    whole_yaml = get_yaml(path_to_yaml)
    YD = whole_yaml["oth"]

    oth_yaml_path = Path("sysrun") / "bashpy_temp" / "oth.yaml"
    write_yaml(YD, oth_yaml_path)

    YD1 = YD["bashpy_args"]
    YD2 = YD["added_auto_main_args"]

    import sysrun.helpers.bashpy_help as bh
    main_yaml_path, out_folder_path, sd_path = bh.main(path_to_yaml, module_path_to_this_file)


    main_name = YD1["main_name"]

    # this is how mti gets added for example
    auto_main_args = []
    for key, value in YD2.items():
        auto_main_args.extend([str(key), str(value)])
    
    returner_dict = {
        "main_name": main_name,
        "auto_main_args": auto_main_args,
        "sd_path": sd_path,
        "main_yaml_path": main_yaml_path,
        "out_folder_path": out_folder_path
    }
    return returner_dict
    

