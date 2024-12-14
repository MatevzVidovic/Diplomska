


"""
NEW IDEA:
I will make an ordered list where there are only tree_ixs of the lowest level modules - these are the onew who actually get pruned.
These are the ones we are actually concerned with when writing the conections lambda.
The user can then check the list and check the corresponding names.
This should make working with the tree_ixs easier.

If it doesn't work, the user can always still type the tree_ixs manually.
It isn't hard - they would mostly be copying and pasting them and only changing a numebr or two.

However, it is kind of hard to know what is the ordinal number of the lowest level module.
This is why lower we make a better method.
"""



def denest_tuple(tup):
    returning_list = []
    for item in tup:
        if isinstance(item, tuple):
            returning_list.extend(denest_tuple(item))
        elif isinstance(item, int):
            returning_list.append(item)
    
    return returning_list


def renest_tuple(lst):
    curr_tuple = (lst[0],)
    for item in lst[1:]:
        curr_tuple = (curr_tuple, item)
    
    return curr_tuple



def sort_tree_ixs(list_of_tree_ixs):
    
    denested_list = [denest_tuple(tree_ix) for tree_ix in list_of_tree_ixs]
    
    sorted_denested_list = sorted(denested_list)

    sorted_list = [renest_tuple(lst) for lst in sorted_denested_list]
    
    return sorted_list