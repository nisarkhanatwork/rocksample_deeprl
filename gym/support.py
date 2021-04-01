from copy import deepcopy
from sys import argv
import glob
import os
import numpy as np

DEBUG = True

def differ(cur_input, prev_input):
    from copy import deepcopy
    pli = prev_input.ravel().tolist()
    cli = cur_input.ravel().tolist()
    found = False
    try:
        ic = cli.index(5.)
        c_found5 = True
    except:
        pass
    try:
        i5 = pli.index(5.)
        found5 = True
    except:
        pass
    res = deepcopy(cur_input)
    res = res.astype(np.float).ravel()
    if found5 is True:
        res[i5] = 7.
    else:
        pass
    if c_found5 == True & (ic == i5):
        res[i5] = 5.
    
    return res

# https://stackoverflow.com/questions/39327032/how-to-get-the-latest-file-in-a-folder-using-python
def get_latest_file():
    print(argv)
    list_of_files = glob.glob('rock_my*') # * means all if need specific format then *.csv
    latest_file = None
    if list_of_files != []:
        latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def get_list_of_files():
    list_of_files = glob.glob('rock_my*') # * means all if need specific format then *.csv
    return list_of_files 

def print_map(x):
    count = 0
    x = x.tolist()
    for i in x:
        print("{0:>7}".format(float(i)),end=' ')
        count += 1
        if(count % 7 == 0):
            print()

def print_map_orig(x):
    count = 0
    x = x.tolist()
    for i in x:
        print("{0:>7}".format(float(i)),end=' ')
        count += 1
        if(count % 10 == 0):
            print()
def print_debug(a):
    if DEBUG == True:
        print(a)

# get_pos; feature extraction...but will this work..?
# when the model is "encoded" this function gets called
def get_pos(x):
    x = x.tolist()
    try:
        prev = x.index(7.)
    except:
        prev = 255. 
    try:
        curr = x.index(5.)
    except:
        curr = -prev 
    if(sum(x)) == 0.:
        curr = 255.
        prev = 255.
    return [curr, prev, 
            18. if x[18] == 3. else -18., 
            25. if x[25] == 3. else -25., 
            35. if x[35] == 3. else -35., 
            45. if x[45] == 3. else -45.]

def get_rover_pos(x, r_tup, e_tup, rover_poss):
   
    y = np.array(x.tolist()).reshape(7, 8)

    y = y.T
    x = y.ravel().tolist()
    r_on, e_on, v_on = False, False, False
    pos = -1
    try:
        pos = x.index(5.) 
        pos_x = int(pos % 7)
        pos_y = int(pos / 7)
    except:
        pos_x = -1
        pos_y = -1
    
    # visited
    for p in rover_poss:
        x[p[0] * 7 + p[1]] = 1.

    for p in rover_poss:
        if(p in r_tup or p in e_tup):
            x[p[0] * 7 + p[1]] = -7.

    """
    if ((pos_y, pos_x) in r_tup):
        r_on = True
    elif ((pos_y, pos_x) in e_tup):
        #print("Found POS ============================================================================", (pos_y, pos_x)) 
        e_on = True
    elif((pos_y, pos_x) in rover_poss):
        v_on = True
    if(r_on or e_on or v_on):
        x[pos] = 5.
    else:
        x[pos] = 0.
    """
    x[pos] = 5.0
    #x.append(pos_y)
    #x.append(pos_x)
    return x, (pos_y, pos_x)

# dummy to return the same 
def get_same(x):
    return x
