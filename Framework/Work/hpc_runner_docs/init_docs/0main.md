



How to add to your project:

- Clone repo just so you get the code.
- cp -r sysrun/ path_inside_your_repo
- add what is in this .gitignore to the .gitignore at: path_inside_your_repo
- possibly copy hpc_runner_test to test it out for your project, but not really needed


How to run:

matevz@DESKTOP-L7B8USF:~/project_root_dir/

python3 -m sysrun.sysrun hpc_runner_test/a/b/c/run1.py --yamls a/yamls/whatevs.yaml --args oth:added_auto_main_args:arg1:2 a:9

python3 -m sysrun.sysrun hpc_runner_test/a/b/c/y_train.py --um


Sysrun will:
- build the yaml dict
- write it to ~/project_root_dir/sysrun/temp.yaml
- run:    
        python3 a/b/c/run1.py --yaml_path ~/project_root_dir/sysrun/temp.yaml  
        (since it has the abspath to the yaml, it can read it and use it however it likes.)


The above run only happens, if --um is present in the args (unconstrained mode).
But most of the time, we rather run run1.py through sbatch.
<!-- { Skip reading this if first time reading:
Most of the time when we write a file like run1.py, we want it to be called with sbatch already, so we don't have to deal with that stuff. Because when run with sbatch, we can then use simple popen bash runnings and we (mostly) do stuff sequentially (we can still do parallel bash commands, but they share the compute nodes comp resources).
Unconstrained mode only makes sense, if we would like to do parallel sbatch commands (which can speeed up stuff). But in most cases we rather just run more different files in constrained mode.
} -->

To run with sbatch, we do:
- make ~/project_root_dir/sysrun/temp.sbatch:
    --SBATCH sys_arg_1=val_1
    --SBATCH sys_arg_2=val_2

    python3 a/b/c/run1.py --yaml_path ~/project_root_dir/sysrun/temp.yaml
- run: sbatch ~/project_root_dir/sysrun/temp.sbatch





How do we build the yaml dict:


We start with an empty dict, and gradually add stuff to it. We do 4 steps:
- dirdef.yaml (directory default yaml)
- samename.yaml
- --yamls
- --args

Example yaml dict:
sys:
  hey: 5
  gpu: Analyze
oth:
  added_auto_main_args:
    arg1: this
    arg2: that
  main_yaml:
    name: main
    who: 4
    ntibp: 2
    model:
      name: unet
      expansion: 2
yamls:
  - here.yaml
  - there.yaml



- dirdef.yaml (directory default yaml)
    The run_path is   ./a/b/c/run1.py
    We look for ./dirdef.yaml  
    We add what it contains to curr dict.
    We look for ./a/dirdef.yaml
    We add what it contains to our curr dict.
    Same for ./a/b/dirdef.yaml  ./a/b/c/dirdef.yaml

    How do we overwrite same keys?
    See example:
    ./dirdef.yaml:
    sys:
        hey: 5
        wazzap: 7
    oth_1: 7
    oth_2:
        - 1
        - 2

    ./a/dirdef.yaml:
    sys:
        hey: 2
        yo: 5
    oth_1: 3
    oth_2:
        - 4
    oth_3: Great

    We start with empty dict as curr dict.
    We add ./dirdef.yaml so essentially it becomes the new curr dict.
    We then add ./a/dirdef.yaml to the curr dict.

    resulting curr dict:
    sys:
        hey: 2
        wazzap: 7
        yo: 5
    oth_1: 3
    oth_2:
        - 4
    oth_3: Great
    
    We can see:
        - oth_3 is simply added. if a key wasn't in the previous curr dict, it is simply added
        - oth_1 and oth_2 got overwritten. But sys wasn't overwritten, it was merged, because   wazzap: 7
        Why?
        If a key is in the old and in the new dict, and the values of that key in both dicts are of type dict(),
        we recursively perform the same updating operation we are doing, but on the value dicts. This way we get
        a nice recursive merging.
        oth_1 is just a val, so it gets overwritten. oth_2 is a list and also gets overwritten.


- samename.yaml
    The run_path is   ./a/b/c/run1.py
    We also go look for ./a/b/c/run1.yaml
    If it exists, we do the same updating operation we show above.
- --yamls
    In the --yamls arg we can specify a list of paths to yamls. Paths relative to the project root.
    The same updating step as described happens.
- --args
    As an arg we can give a list like:   oth:added_auto_main_args:arg1:2 a:9
    This then gets parsed along : and it is made into a small dict.
    Then the same updating operation happens.






But there is another feature we did not mention. We can see an example dict:
sys:
  gpu: Analyze
oth:
  added_auto_main_args:
    arg1: this
    arg2: that
yamls:
  - here/whatever.yaml
  - a/b/there.yaml

What is this yamls key? It is a special feature.
When a dirdef.yaml contains the key yamls, we don't process it like we otherwise would.
We take remove it from the dict before we update to prev curr dict.
Then we look for yamls in the paths from this list of yaml paths we got in yamls.
We first look at the first one.
We try to find it relative to the curr dit.
Lets say the run_path is   ./a/b/c/run1.py
We would first look at the relative path ./a/b/c/ + here/whatever.yaml
If it exists, we add it by the procedure we showed above.
Otherwise, we look at the abspath from the root:
./ + /here/whatever.yaml
If even that doesn't exist, we warn you.
We do this for every element in the yamls list.

This isn't only true for dirdef. It's the same for
dirdef.yaml, samename.yaml, and --yamls






How to test that your constructed yaml contains all necessary fields?

python3 -m sysrun.sysrun hpc_runner_test/a/b/c/run1.py --yamls a/yamls/whatevs.yaml --args oth:added_auto_main_args:arg1:2 a:9 --test_yaml sysrun/template_yaml.yaml

parser.add_argument("--test_yaml", type=str, help="Pass path from the root of the project to a template yaml. If any field, \
                    which is present in the template yaml, isn't present in the constructed yaml, we tell you about it. \
                    Either way, we stop before running anything.", default=None)

This is a nice check, but to make sure you cover all fields you'd have to have a template yaml for each runner file.
But just having the general template yaml file with the general fields can help you somewhat too.

