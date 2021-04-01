#!/usr/bin/env python
# coding: utf-8

#https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder
import sys
sys.path.insert(0,'../gym')
import numpy as np
from support import *
from model import *
MAX_STEPS = 25 

def run_exper(model, steps, get_features, pre_proc_features):
    r_tup, e_tup = [], []
    rover_poss = []
    total_stats = {'total':0, 'good':0}

    from environment import SIMULATOR

    # initializing our environment
    my_sim = SIMULATOR()    

    # beginning of an episode
    state_temp = my_sim.reset()
    observation = my_sim.state_to_tensor(state_temp)
    state_obs = observation
    total_moves = 0

    # main loop
    prev_input = None
    for i in range(steps):
        # preprocess the observation, set input as difference between images
        cur_input = observation

        x = cur_input.astype(np.float).ravel() if prev_input is not None else np.zeros(70)
        x = x[10:80] if prev_input is not None else x
        x = np.array([x[i] for i in range(len(x)) if not (i%10 == 0)])
        x = np.array([x[i] for i in range(len(x)) if not ((i - 8 )% 9 == 0)])


        prev_input = cur_input

        x, rover_pos = get_rover_pos(x, r_tup, e_tup, rover_poss)
        rover_poss.append(rover_pos)
        x = np.array(x)
        """
        x = x[x != 0]
        if(len(x) == 1):
            x = np.zeros(4)
            x = x.tolist()
            x.append(-7.)
            x = np.array(x)
        """

        x_t = pre_proc_features.fit_transform(x.reshape(-1, 1))
        x_t = x_t.reshape(1, INPUT_SIZE)[0]
        # forward the policy network and sample action according to the proba distribution
        proba = model.predict(np.expand_dims(x_t, axis=1).T)
        action = proba.argmax() 

        #run one step
        state_temp, reward, done, r_tup, e_tup = my_sim.step(action)
        observation = my_sim.state_to_tensor(state_temp)
        #my_sim.render()
        total_moves += 1
        if(total_moves == MAX_STEPS):
            done = True
            total_moves = 0

        # if episode is over, reset to beginning
        if done:
            total_stats['total'] += 1
            so = np.asarray(state_obs).ravel().tolist()
            o = np.asarray(observation).ravel().tolist()
            #print("state obs ===============")
            #print(state_obs)
            #print("obs ===============")
            #print(observation)
            try:
                index_obs = so.index(7.0)
            except ValueError:
                index_obs = -1
            try:
                index_curr = o.index(7.0)
            except ValueError:
                index_curr = -1

            if(index_obs != -1 and index_curr == -1):
                #print("Good Game")
                #print(so)
                #print(o)
                total_stats['good'] += 1
            state_temp = my_sim.reset()
            observation = my_sim.state_to_tensor(state_temp)
            state_obs = observation
            rover_poss = []
            #my_sim.render()    

    return total_stats

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', default = 'sparse', 
            type=str,  help = 'Choose between encoded or sparse')
    parser.add_argument('--n_steps', required = True, 
            type = int,  help = 'Choose a number to test numtiple maps.')
    args = parser.parse_args()
    data_type = args.data_type
    steps = args.n_steps

    model = get_model(data_type)
    get_features, pre_proc_features = get_pre_proc_info(data_type)
    list_of_files = get_list_of_files()
    for i in range(len(list_of_files)):
        latest_file = max(list_of_files, key=os.path.getctime)
        model.load_weights(latest_file)
        print(latest_file, ":", run_exper(model, steps, get_features, pre_proc_features))    
        list_of_files.remove(latest_file)
