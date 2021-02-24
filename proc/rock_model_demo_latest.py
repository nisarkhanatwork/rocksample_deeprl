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

    # main loop
    prev_input = None
    for i in range(steps):
        cur_input = observation
        x = differ(cur_input, prev_input) if prev_input is not None else np.zeros(70)
        x = x[10:80] if prev_input is not None else x
        prev_input = cur_input
        x = np.array(get_features(x))

        # forward the policy network and sample action according to the proba distribution
        start = time.perf_counter()
        x_shape = x.shape
        x_t = pre_proc_features.fit_transform(x.reshape(-1, 1))
        x_t = x_t.reshape(x_shape)
        #print_map(x)
        proba = model.predict(np.expand_dims(x_t, axis=1).T)
        end = time.perf_counter()
        action = proba[0].argmax()
        print("Time taken = ", end - start)

        #run one step
        state_temp, reward, done, _, _ = my_sim.step(action)
        observation = my_sim.state_to_tensor(state_temp)
        my_sim.render()
        time.sleep(1)

        # if episode is over, reset to beginning
        if done:
            state_temp = my_sim.reset()
            observation = my_sim.state_to_tensor(state_temp)
            my_sim.render()    


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
