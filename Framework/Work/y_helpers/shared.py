


# Avoid from shared import GLOBAL_DICT
# Using this form of import creates a local reference to the dictionary in the importing module. 
# If you reassign GLOBAL_DICT in that module, it won't affect the original dictionary in shared_data. 
# Always use import shared and access the dictionary as shared.GLOBAL_DICT 
# to maintain a single shared object.

# you are allowed to do:
# import y_helpers.shared as shared


GLOBAL_DICT = {}




# When plt can't do X-forwarding (displaying the plot as a GUI), a C-level exception happens.
# This doesn't raise a python exception, so the program goes on normally.
# Also, you can't just try-catch it and make it go away.
# But still, the exit code of the program is 1.
# (exit of python is 0, but exit of of the C-level thingy is 1, so the overall exit code is 1).
# This bugs me, because I have automatic recording of exit codes, which allows me to quickly see if something went wrong.
# So I want to easily disable all plt.show() calls when I do wsl or ssh stuff and can't get a GUI.
# And this is what this is for.

# The problem reallyisn't with plt.show() commands, but at the moment you import matplotplib.pyplot.
# You have to use Agg as a sidplay server, which doesn't have GUI display, just for saving to file.
# You need to make this happen before importing pyplot:
# import y_helpers.shared as shared
# if not shared.PLT_SHOW: # For more info, see shared.py
#     import matplotlib
#     matplotlib.use("Agg", force=True)
# import matplotlib.pyplot as plt
#
# Then plt.show() are generally no-operation, but maybe they trigger some errors?
# Just for safety, they will also go under an if shared.PLT_SHOW clause.
#
# Also, explicit .ion() for interactive mode could then cause problems, I think.
# So it also goes in an if shared.PLT_SHOW clause.
PLT_SHOW = False # True