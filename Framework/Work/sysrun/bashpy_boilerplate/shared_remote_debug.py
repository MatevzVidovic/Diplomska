

# ---------- Remote debugging ----------

# Avoid from shared import GLOBAL_DICT
# Using this form of import creates a local reference to the dictionary in the importing module. 
# If you reassign GLOBAL_DICT in that module, it won't affect the original dictionary in shared_data. 
# Always use import shared and access the dictionary as shared.GLOBAL_DICT 
# to maintain a single shared object.

# you are allowed to do:
# import y_helpers.shared as shared


# Do:

# import y_helpers.shared_debug as debug
# debug.start_debug()

# although you could do:
# from y_helpers.shared_debug import start_debug
# start_debug()
# And this would work fine, because while there is now a local reference to the function,
# the function itself does not create a local reference to the dictionary GLOBAL_DICT - that reference is fused as finction definition.
# But you know, stay safe by just doing the import as above.

DEBUG = True #True

GLOBAL_DICT = {
    "debug_connection_already_open": False,
}

DEBUG_PORT = 8765


# !!!!!!!!!!!!!!!!
# For my intended use this doesn't work as I expected!!!
# It still works, just be careful.
# I import this in the run.py file, which then uses Popen to run segnet_main.py
# These are two completely different processes. Their imports aren't shared.
# So the GLOBAL_DICT is not shared between them. Nothing is.
# So run.py opens connection on port 5678, changes the GLOBAL_DICT
# and then segnet_main.py is run, which has its own GLOBAL_DICT (so debug_connection_already_open is False again)
# So the connection on 5678 is already open, and the debugger has an exception.
# But we just catch it and move on, and the debugging then still works 
# (because remote debugging nicely works in python subprocesses) (look at launch.json for more info on this).


def start_remote_debug():
    if DEBUG and not GLOBAL_DICT["debug_connection_already_open"]:
        import debugpy
        try:
            debugpy.listen(("localhost", DEBUG_PORT))
            print("Waiting for debugger attach...")
            GLOBAL_DICT["debug_connection_already_open"] = True
            debugpy.wait_for_client()
        except Exception as e:
            print(f"Debugpy connection failed: {e}")
            print(f"Probs debugpy connection already open.")

