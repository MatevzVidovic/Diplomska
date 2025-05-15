


# Avoid from shared import GLOBAL_DICT
# Using this form of import creates a local reference to the dictionary in the importing module. 
# If you reassign GLOBAL_DICT in that module, it won't affect the original dictionary in shared_data. 
# Always use import shared and access the dictionary as shared.GLOBAL_DICT 
# to maintain a single shared object.

# you are allowed to do:
# import y_helpers.shared as shared


GLOBAL_DICT = {}