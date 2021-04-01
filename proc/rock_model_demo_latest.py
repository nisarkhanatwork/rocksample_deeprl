#!/usr/bin/env python
# coding: utf-8

import time

#https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder
import sys
sys.path.insert(0,'../gym')

import numpy as np
from support import *
from model import *

def run_exper(model, steps, get_features, pre_proc_features):
    from environment import SIMULATOR

    # initializing our environment
    my_sim = SIMULATOR()    

    # beginning of an episode
    state_temp = my_sim.reset()
    observation = my_sim.state_to_tensor(state_temp)
    r_tup, e_tup, rover_poss = [], [], []
    # main loop
    prev_input = None
    total_moves = 0
    MAX_MOVES = 25
    for i in range(steps):
        total_moves += 1
        start = time.perf_counter()
        cur_input = observation
        x = cur_input.astype(np.float).ravel() if prev_input is not None else np.zeros(70)
        x = x[10:80] if prev_input is not None else x
        x = np.array([x[i] for i in range(len(x)) if not (i%10 == 0)])
        x = np.array([x[i] for i in range(len(x)) if not ((i - 8 )% 9 == 0)])

        x , rov_pos = get_rover_pos(x, r_tup, e_tup, rover_poss)
        x = np.array(x)
        rover_poss.append(rov_pos)
        """
        x = x[x != 0]
        if(len(x) == 1):
            x = np.zeros(4)
            x = x.tolist()
            x.append(-7.)
            x = np.array(x)
        """
        #print_map(x)
        x_t = pre_proc_features.fit_transform(x.reshape(-1, 1))
        x_t = x_t.reshape(1, INPUT_SIZE)[0]
        print("Shape = ", x_t.shape)
        prev_input = cur_input

        # forward the policy network and sample action according to the proba distribution
        #print_map(x)
        proba = model.predict(np.expand_dims(x_t, axis=1).T)
        end = time.perf_counter()
        action = proba[0].argmax()
        print("Time taken = ", end - start)

        #run one step
        state_temp, reward, done, r_tup, e_tup = my_sim.step(action)
        observation = my_sim.state_to_tensor(state_temp)
        my_sim.render()
        time.sleep(1)

        if total_moves == MAX_MOVES:
            total_moves = 0
            done = True
        # if episode is over, reset to beginning
        if done:
            state_temp = my_sim.reset()
            observation = my_sim.state_to_tensor(state_temp)
            my_sim.render()    
            rover_poss = []


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', default = 'sparse', 
            type=str,  help = 'Choose between encoded or sparse')
    parser.add_argument('--n_steps', default = 30,
            type=int,  help = 'Choose a number.')
    parser.add_argument('--demo_file', default = '', 
            type=str,  help = 'File for demo.')
    args = parser.parse_args()
    data_type = args.data_type
    steps = args.n_steps
    latest_file = args.demo_file
    model = get_model(data_type)
    get_features, pre_proc_features = get_pre_proc_info(data_type)

    if(len(latest_file) == 0):
        latest_file = get_latest_file()

    if latest_file != None and latest_file[0:13] == "rock_my_model":
        print("===>", latest_file)
        model.load_weights(latest_file)
    else:
        print("Model not found: Exiting...")
        sys.exit(0)
    
    run_exper(model, steps, get_features, pre_proc_features)
